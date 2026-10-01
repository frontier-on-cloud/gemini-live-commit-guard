"""commit_guard: client-side handling of in-flight side-effecting tool calls on the
Gemini Live API.

A reference pattern, not a package. The application keeps its own receive loop and
passes every server message to `CommitGuard.observe()`. For the tools it guards, the
guard runs the job on a two-phase service (prepare, then commit or cancel) and decides
when the irreversible commit may happen. Four behaviors, each switchable in
`GuardConfig` so they can be measured separately:

1. hold     Hold on speech onset. After `prepare()`, the commit waits `grace_s`. It is
            released only if no user speech onset (`voiceActivity ACTIVITY_START`) and no
            `interrupted` is open. If speech starts while a job is pending, the guard
            waits for that utterance's input transcript and runs a keyword intent check:
            stop words cancel the job, anything else lets it continue. The transcript of
            the utterance that triggered the call is checked too (a stop spoken right
            after the request can arrive merged with it, before the call exists).
2. dedupe   Deduplicate by business key: tool name plus canonical arguments (sorted
            JSON after the tool's own normalisation) within `dedupe_window_s`. A repeat
            with a new call id is answered from the existing job and never executed.
3. inject   Authoritative status injection. After a stop or cancel decision, and after
            a commit that happened despite a stop, send the model a short
            "System note: ..." user turn with `send_client_content`, so what it says
            next is grounded in the real state.
4. abandon  Abandoned BLOCKING calls. If `interrupted` arrives while a BLOCKING call is
            unanswered, the server has dropped that call. The guard holds its commit,
            does not answer the dead id, and answers a re-issued call for the same key
            (found by behavior 2) with the existing job's status.

`TruthCheck` compares what the model said with the service state afterwards.

Usage:

    guard = CommitGuard(session, service, [GuardedTool("book_slot")], GuardConfig())
    async for msg in session.receive():
        for fc in guard.observe(msg):   # function calls the guard does not own
            ...
    await guard.aclose()
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Protocol

from google.genai import types

Intent = Literal["stop", "other", "unknown"]
Claim = Literal["claims_booked", "claims_not_booked", "neither"]
Decision = Literal["commit", "cancel"]

FINAL_STATES = ("committed", "cancelled", "failed")


# --------------------------------------------------------------- service API --


class TwoPhaseService(Protocol):
    """A backend wrapped as prepare-then-commit.

    `prepare` does the slow, reversible part (check availability, place a hold).
    `commit` is the irreversible side effect. `cancel` releases a job that was not
    committed; it may be called while `prepare` is still running (the guard cancels
    the prepare task first)."""

    async def prepare(self, job_id: str, tool: str, args: dict[str, Any]) -> None: ...

    async def commit(self, job_id: str) -> dict[str, Any]: ...

    async def cancel(self, job_id: str) -> None: ...


def sorted_json(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class GuardedTool:
    """A side-effecting tool the guard owns.

    `canonical` maps the model's arguments to the business fields that identify the
    request (the dedupe key is the tool name plus their sorted JSON). `describe` turns
    arguments and, once committed, the service result into words for status notes."""

    name: str
    blocking: bool = False
    canonical: Callable[[dict[str, Any]], dict[str, Any]] = dict
    describe: Callable[[dict[str, Any], dict[str, Any] | None], str] | None = None

    def key(self, args: dict[str, Any]) -> str:
        return f"{self.name}:{sorted_json(self.canonical(args))}"

    def what(self, args: dict[str, Any], result: dict[str, Any] | None) -> str:
        if self.describe is not None:
            return self.describe(args, result)
        return f"{self.name} request {sorted_json(args)}"


# ------------------------------------------------------------ intent / truth --


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").lower()).strip()


@dataclass(frozen=True)
class IntentCheck:
    """Deterministic keyword rules: any stop phrase in the utterance means "stop"."""

    phrases: tuple[str, ...] = ("stop", "cancel", "don't", "do not", "wait",
                                "never mind", "nevermind", "hold on")

    def matches(self, text: str) -> list[str]:
        t = _norm(text)
        return [p for p in self.phrases
                if re.search(r"(?<![\w'])" + re.escape(p).replace(r"\ ", r"\s+") + r"(?![\w'])", t)]

    def classify(self, text: str) -> Intent:
        if not text.strip():
            return "unknown"
        return "stop" if self.matches(text) else "other"


_CLAUSE_SPLIT = re.compile(r"[.;:!?,]|\b(?:and|but|so|because|although|however)\b")
_NEGATION = re.compile(r"\b(?:not|no|never|nothing|cannot)\b|n't\b")
_BOOK_WORD = re.compile(r"\b(?:book|booked|made|make|schedule|scheduled|reserve|reserved|"
                        r"confirm|confirmed)\b")
_CANCEL_STATE = re.compile(r"\b(?:cancell?ed|stopped|called off)\b")
_BOOKED_STATE = re.compile(r"\b(?:booked|made|scheduled|reserved|confirmed)\b")
_SUBORDINATE = re.compile(r"\b(?:before|after|when|until|since|once)\b")


@dataclass(frozen=True)
class TruthResult:
    claim: Claim
    actual: Literal["booked", "not_booked"]
    consistent: bool
    statement: str


class TruthCheck:
    """Classify what the model told the user and compare it with the service state.

    Rules, per clause (split on punctuation and and/but/so/...), applied to the part
    before any before/after/when/until/since/once (the subordinate part only counts
    if the main part has no claim):
      negation + booking word ("was not booked", "nothing is scheduled") -> not booked
      cancel state without negation ("has been cancelled", "I've stopped") -> not booked
      booked state without negation ("was already made", "is confirmed") -> booked
    Offers ("would you like me to cancel it?") use base forms and match nothing.
    The last clause with a claim wins, so "it was booked, but I cancelled it" counts
    as not booked. "neither" is recorded as not consistent: the user was not told.
    This classifies the transcript, not what a listener heard (see guard_test.py
    `note_cuts` for replies cut by the guard's own note)."""

    @staticmethod
    def _claim(part: str) -> Claim:
        negated = bool(_NEGATION.search(part))
        if negated and _BOOK_WORD.search(part):
            return "claims_not_booked"
        if not negated and _CANCEL_STATE.search(part):
            return "claims_not_booked"
        if not negated and _BOOKED_STATE.search(part):
            return "claims_booked"
        return "neither"

    def classify(self, text: str) -> Claim:
        claim: Claim = "neither"
        for clause in _CLAUSE_SPLIT.split(_norm(text)):
            if not clause or not clause.strip():
                continue
            # The main clause decides: "was cancelled before it was made" is not booked,
            # "was already confirmed before you cancelled" is booked.
            sub = _SUBORDINATE.search(clause)
            c = self._claim(clause[:sub.start()] if sub else clause)
            if c == "neither" and sub:
                c = self._claim(clause[sub.start():])
            if c != "neither":
                claim = c
        return claim

    def check(self, statements: list[str], committed: bool) -> TruthResult:
        """`statements`: the model's turns after the stop, in order. They are joined
        before classifying, because a turn cut by an interruption continues in the next
        one ("I have not " / "booked that slot for you."). The last claim wins."""
        actual: Literal["booked", "not_booked"] = "booked" if committed else "not_booked"
        said = re.sub(r"\s+", " ", " ".join(statements)).strip()
        claim = self.classify(said)
        consistent = (claim == "claims_booked") == committed and claim != "neither"
        return TruthResult(claim, actual, consistent, said)


# --------------------------------------------------------------------- guard --


@dataclass
class GuardConfig:
    hold: bool = True       # behavior 1
    dedupe: bool = True     # behavior 2
    inject: bool = True     # behavior 3
    abandon: bool = True    # behavior 4
    grace_s: float = 1.5
    # How long a ready commit may wait for the transcript of an open utterance. Counted
    # from the later of the speech onset and the moment the commit became due, because
    # the transcript only arrives ~1.2-1.3 s after speech ends (measured), which is
    # ~3.3 s after the onset of a 2.2 s "stop" utterance.
    transcript_timeout_s: float = 3.0
    on_timeout: Decision = "cancel"   # no transcript in time: fail closed by default
    dedupe_window_s: float = 30.0
    trigger_lookback_s: float = 2.0   # stop in the utterance that ended <= this before the call
    own_interrupt_s: float = 1.0      # `interrupted` this soon after our own note is ours
    intent: IntentCheck = field(default_factory=IntentCheck)
    note_cancelled: str = ("System note: the {what} was cancelled before it was committed. "
                           "Nothing is scheduled.")
    note_cancelled_unclear: str = ("System note: the {what} was cancelled before it was "
                                   "committed because the user spoke and the request could "
                                   "not be confirmed. Nothing is scheduled. Ask the user "
                                   "whether they still want it.")
    note_committed_after_stop: str = ("System note: {what} is confirmed. The user asked to "
                                      "stop after it was committed. Tell them it is confirmed "
                                      "and offer to cancel it.")
    note_confirmed: str = "System note: {what} is confirmed."

    @classmethod
    def off(cls, **kw: Any) -> "GuardConfig":
        return cls(hold=False, dedupe=False, inject=False, abandon=False, **kw)


@dataclass
class Job:
    job_id: str
    tool: GuardedTool
    args: dict[str, Any]
    key: str
    call_id: str
    created_at: float
    state: str = "preparing"   # preparing -> held -> committed | cancelled | failed
    prepared_at: float | None = None
    abandoned: bool = False
    decision: Decision | None = None
    reason: str = ""
    decided_at: float | None = None
    committed_at: float | None = None
    result: dict[str, Any] | None = None
    duplicates: list[str] = field(default_factory=list)
    responded: set[str] = field(default_factory=set)
    speech_while_pending: list[str] = field(default_factory=list)
    stop_heard: bool = False   # a stop resolved while pending but the job was not held
    noted: bool = False
    decided: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Utterance:
    opened_at: float
    source: str
    text: str = ""
    ended: bool = False


@dataclass(frozen=True)
class _Resolved:
    at: float
    opened_at: float
    text: str
    intent: Intent
    how: str


class CommitGuard:
    """Owns the guarded tool calls of one Live session. See the module docstring."""

    def __init__(self, session: Any, service: TwoPhaseService, tools: list[GuardedTool],
                 config: GuardConfig | None = None,
                 log: Callable[..., None] | None = None,
                 now: Callable[[], float] = time.monotonic) -> None:
        self.session = session
        self.service = service
        self.tools = {t.name: t for t in tools}
        self.cfg = config or GuardConfig()
        self.now = now
        self._log_cb = log
        self.jobs: list[Job] = []
        self.decisions: list[dict[str, Any]] = []
        self.duplicates: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.notes: list[dict[str, Any]] = []
        self.utterances: list[_Resolved] = []
        self._utt: _Utterance | None = None
        self._last: _Resolved | None = None
        self._vad_seen = False
        self._changed = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._send_lock = asyncio.Lock()
        self._n = 0
        self._closing = False

    # -- public ------------------------------------------------------------

    def observe(self, msg: types.LiveServerMessage) -> list[types.FunctionCall]:
        """Feed one server message. Returns the function calls the guard does not own."""
        va = msg.voice_activity
        kind = None
        if va is not None and va.voice_activity_type is not None:
            self._vad_seen = True
            kind = str(getattr(va.voice_activity_type, "value", va.voice_activity_type))
        if kind == "ACTIVITY_START":
            self._speech_start("voice_activity")
        sc = msg.server_content
        if sc is not None and sc.input_transcription and sc.input_transcription.text:
            self._transcript(sc.input_transcription.text)
        if kind == "ACTIVITY_END":
            self._speech_end()
        if sc is not None and sc.interrupted:
            self._interrupted()
        passthrough: list[types.FunctionCall] = []
        if msg.tool_call:
            for fc in msg.tool_call.function_calls or []:
                tool = self.tools.get(fc.name or "")
                if tool is None:
                    passthrough.append(fc)
                else:
                    self._on_call(tool, fc)
        return passthrough

    def pending(self) -> bool:
        return any(j.state not in FINAL_STATES for j in self.jobs) or any(
            not t.done() for t in self._tasks)

    async def aclose(self) -> None:
        """Stop all guard tasks. Jobs that were not committed are cancelled."""
        self._closing = True
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for job in self.jobs:
            if job.state not in FINAL_STATES:
                job.state = "cancelled"
                job.reason = job.reason or "session ended"
                try:
                    await self.service.cancel(job.job_id)
                except Exception:
                    pass
                self._log("job_aborted_at_close", job_id=job.job_id)

    # -- speech tracking ---------------------------------------------------

    def _speech_start(self, source: str) -> None:
        if self._utt is None:
            self._utt = _Utterance(self.now(), source)
            self._log("speech_onset", source=source,
                      pending_jobs=[j.job_id for j in self.jobs if j.state not in FINAL_STATES])
        self._wake()

    def _transcript(self, text: str) -> None:
        if self._utt is None:  # transcript with no open onset: treat it as complete
            self._utt = _Utterance(self.now(), "transcript", ended=True)
        self._utt.text += text
        if self._utt.ended or not self._vad_seen or self.cfg.intent.classify(self._utt.text) == "stop":
            self._resolve("transcript")
        self._wake()

    def _speech_end(self) -> None:
        if self._utt is None:
            return
        self._utt.ended = True
        if self._utt.text.strip():
            self._resolve("activity_end")
        self._wake()

    def _interrupted(self) -> None:
        t = self.now()
        own = self.notes and t - self.notes[-1]["at"] <= self.cfg.own_interrupt_s
        self._log("interrupted_seen", caused_by_own_note=bool(own))
        if own:
            return
        if not self._vad_seen:   # no voiceActivity in this session: interrupted is the onset
            self._speech_start("interrupted")
        if self.cfg.abandon:
            for job in self.jobs:
                if (job.tool.blocking and job.state not in FINAL_STATES and not job.abandoned
                        and job.call_id not in job.responded):
                    job.abandoned = True
                    self._log("call_abandoned", job_id=job.job_id, call_id=job.call_id)
        self._wake()

    def _resolve(self, how: str) -> None:
        utt, self._utt = self._utt, None
        if utt is None:
            return
        text = utt.text.strip()
        intent = self.cfg.intent.classify(text)
        r = _Resolved(self.now(), utt.opened_at, text, intent, how)
        self._last = r
        self.utterances.append(r)
        self._log("utterance_resolved", intent=intent, how=how, text=text,
                  matched=self.cfg.intent.matches(text), onset_s=round(utt.opened_at, 3))
        for job in self.jobs:
            if job.state == "committed" or (job.decision == "commit" and job.state != "failed"):
                # The stop came after the commit (or while it was running): say so.
                if (intent == "stop" and not job.noted
                        and r.at - job.created_at <= self.cfg.dedupe_window_s):
                    job.stop_heard = True
                    if job.state == "committed":
                        self._spawn(self._note(job))
                continue
            if job.state in FINAL_STATES or job.decision is not None:
                continue
            job.speech_while_pending.append(text)
            if intent == "stop":
                if self._held(job):
                    self._decide(job, "cancel", f"stop intent in user speech: {text!r}")
                else:
                    job.stop_heard = True
            elif intent == "unknown" and self._held(job) and self.cfg.on_timeout == "cancel":
                self._decide(job, "cancel", "user spoke; no transcript before the timeout")
        self._wake()

    # -- calls and jobs ----------------------------------------------------

    def _on_call(self, tool: GuardedTool, fc: types.FunctionCall) -> None:
        args = dict(fc.args or {})
        key = tool.key(args)
        call_id = fc.id or f"noid-{len(self.responses)}"
        t = self.now()
        dup = self._find_duplicate(key, t) if self.cfg.dedupe else None
        if dup is not None:
            dup.duplicates.append(call_id)
            self.duplicates.append({"at": t, "call_id": call_id, "job_id": dup.job_id,
                                    "original_call_id": dup.call_id, "job_state": dup.state})
            self._log("duplicate_call", call_id=call_id, job_id=dup.job_id,
                      original_call_id=dup.call_id, job_state=dup.state, key=key)
            if dup.state in FINAL_STATES:
                self._spawn(self._respond(dup, call_id))
            return
        self._n += 1
        job = Job(f"job-{self._n}", tool, args, key, call_id, t)
        self.jobs.append(job)
        self._log("job_started", job_id=job.job_id, call_id=call_id, key=key,
                  blocking=tool.blocking)
        last = self._last
        if (self.cfg.hold and last is not None and last.intent == "stop"
                and t - last.at <= self.cfg.trigger_lookback_s):
            self._decide(job, "cancel",
                         f"stop intent in the utterance that triggered the call: {last.text!r}")
        self._spawn(self._run(job))

    def _find_duplicate(self, key: str, t: float) -> Job | None:
        for job in reversed(self.jobs):
            if job.key != key or t - job.created_at > self.cfg.dedupe_window_s:
                continue
            if job.state == "failed":
                return None
            if (job.state == "cancelled" and self._last is not None
                    and job.decided_at is not None and self._last.at > job.decided_at
                    and self._last.intent == "other"):
                return None   # the user asked again after the cancel: a new request
            return job
        return None

    def _held(self, job: Job) -> bool:
        return self.cfg.hold or (self.cfg.abandon and job.abandoned)

    def _decide(self, job: Job, decision: Decision, reason: str) -> Decision:
        job.decision, job.reason, job.decided_at = decision, reason, self.now()
        self.decisions.append({"at": job.decided_at, "job_id": job.job_id,
                               "call_id": job.call_id, "decision": decision,
                               "reason": reason, "state": job.state})
        self._log("decision", job_id=job.job_id, call_id=job.call_id, decision=decision,
                  reason=reason, state=job.state, abandoned=job.abandoned)
        job.decided.set()
        self._wake()
        return decision

    async def _run(self, job: Job) -> None:
        prep = asyncio.create_task(self.service.prepare(job.job_id, job.tool.name, job.args))
        dec = asyncio.create_task(job.decided.wait())
        try:
            await asyncio.wait({prep, dec}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            dec.cancel()
        if not prep.done():   # cancelled while preparing
            prep.cancel()
            await asyncio.gather(prep, return_exceptions=True)
        else:
            try:
                prep.result()
            except Exception as exc:
                job.state, job.reason = "failed", f"prepare failed: {exc}"
                self._log("job_failed", job_id=job.job_id, error=str(exc)[:200])
                await self._finish(job)
                return
            job.state, job.prepared_at = "held", self.now()
            self._log("prepared", job_id=job.job_id)
        decision = job.decision if job.decision is not None else await self._gate(job)
        if decision == "commit":
            try:
                job.result = await self.service.commit(job.job_id)
            except Exception as exc:
                job.state, job.reason = "failed", f"commit failed: {exc}"
                self._log("job_failed", job_id=job.job_id, error=str(exc)[:200])
                await self._finish(job)
                return
            job.state, job.committed_at = "committed", self.now()
            self._log("committed", job_id=job.job_id, result=job.result)
        else:
            await self.service.cancel(job.job_id)
            job.state = "cancelled"
            self._log("cancelled", job_id=job.job_id, reason=job.reason)
        await self._finish(job)

    async def _gate(self, job: Job) -> Decision:
        assert job.prepared_at is not None
        due = job.prepared_at + (self.cfg.grace_s if self.cfg.hold else 0.0)
        while True:
            if job.decision is not None:
                return job.decision
            if not self._held(job):
                return self._decide(job, "commit", "guard hold off")
            t = self.now()
            if self._utt is not None:
                deadline = max(self._utt.opened_at, due) + self.cfg.transcript_timeout_s
                if t >= deadline:
                    self._resolve("timeout")
                    if job.decision is None and self.cfg.on_timeout == "commit":
                        return self._decide(job, "commit", "transcript timeout, policy commit")
                    continue
                until = deadline
            elif t >= due:
                heard = "; speech while pending had no stop intent" if job.speech_while_pending else (
                    "; no speech onset or interruption while pending")
                return self._decide(job, "commit", f"grace window passed{heard}")
            else:
                until = due
            ev = self._changed
            try:
                await asyncio.wait_for(ev.wait(), max(0.0, until - t))
            except TimeoutError:
                pass

    # -- responses and notes -----------------------------------------------

    def _payload(self, job: Job, call_id: str) -> dict[str, Any]:
        if job.state == "committed":
            out = dict(job.result or {})
        elif job.state == "cancelled":
            why = ("the user spoke and the request could not be confirmed"
                   if job.reason.startswith("user spoke; no transcript")
                   else "the user asked to stop")
            out = {"status": "cancelled",
                   "detail": f"Cancelled before commit because {why}. Nothing was committed."}
        else:
            out = {"status": "failed", "detail": job.reason}
        if call_id != job.call_id:
            out["duplicate_of"] = job.call_id
            out["note"] = "Same request as an earlier call; it was not executed again."
        return out

    async def _respond(self, job: Job, call_id: str) -> None:
        if call_id in job.responded or self._closing:
            return
        job.responded.add(call_id)
        if call_id == job.call_id and job.abandoned and self.cfg.abandon:
            self._log("response_skipped", call_id=call_id, job_id=job.job_id,
                      reason="call abandoned by the server after interrupted")
            return
        payload = self._payload(job, call_id)
        fr = types.FunctionResponse(id=call_id, name=job.tool.name, response=payload)
        try:
            async with self._send_lock:
                await self.session.send_tool_response(function_responses=[fr])
        except Exception as exc:
            self._log("send_error", where="send_tool_response", error=str(exc)[:300])
            return
        self.responses.append({"at": self.now(), "call_id": call_id, "job_id": job.job_id,
                               "status": payload.get("status"), "payload": payload})
        self._log("tool_response_sent", call_id=call_id, job_id=job.job_id, response=payload)

    async def _finish(self, job: Job) -> None:
        for call_id in [job.call_id, *job.duplicates]:
            await self._respond(job, call_id)
        await self._note(job)
        self._wake()

    def _note_text(self, job: Job) -> str | None:
        what = job.tool.what(job.args, job.result)
        if job.state == "cancelled" and job.decision == "cancel":
            unclear = job.reason.startswith("user spoke; no transcript")
            return (self.cfg.note_cancelled_unclear if unclear else self.cfg.note_cancelled).format(what=what)
        if job.state == "committed" and job.stop_heard:
            return self.cfg.note_committed_after_stop.format(what=what)
        live = [c for c in job.duplicates if c in job.responded]
        if job.state == "committed" and job.abandoned and self.cfg.abandon and not live:
            return self.cfg.note_confirmed.format(what=what)   # no live id got the result
        return None

    async def _note(self, job: Job) -> None:
        if not self.cfg.inject or job.noted or self._closing:
            return
        text = self._note_text(job)
        if text is None:
            return
        job.noted = True
        turn = types.Content(role="user", parts=[types.Part(text=text)])
        try:
            async with self._send_lock:
                await self.session.send_client_content(turns=turn, turn_complete=True)
        except Exception as exc:
            self._log("send_error", where="send_client_content", error=str(exc)[:300])
            return
        self.notes.append({"at": self.now(), "job_id": job.job_id, "text": text})
        self._log("status_note_sent", job_id=job.job_id, text=text)

    # -- plumbing ----------------------------------------------------------

    def _wake(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def _spawn(self, coro: Awaitable[None]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self._log("guard_task_error", error=repr(task.exception())[:300])

    def _log(self, event: str, **fields: Any) -> None:
        if self._log_cb is not None:
            self._log_cb(event, **fields)
