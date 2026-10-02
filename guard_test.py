"""Gemini Live API: does a client-side commit guard keep "stop" from becoming a booking?

One run = one Live session with speech input (adapted from ../gemini-live-stop-test/
stop_test.py, audio mode only):

  1. stream assets/audio/book.wav ("Book me the 3pm slot tomorrow, please.") as 16 kHz
     16-bit mono PCM, 100 ms per send_realtime_input(audio=...) call, paced in real
     time, with silence between utterances like an open microphone (server VAD default)
  2. book_slot calls go to CommitGuard (commit_guard.py), which runs them on a fake
     two-phase booking service: prepare takes --latency s, commit is instant.
     --guard off: all four behaviors off (prepare, then commit at once: the stop-test
     baseline). --guard on: all on. --no-hold / --no-dedupe / --no-inject /
     --no-abandon switch one behavior off.
  3. the stop clip ("Actually, stop. Don't book it.") starts --stop-after s after the
     tool call, or --stop-after-request s after the last chunk of the booking clip, or
     --stop-after-model-speech s after the first model audio chunk that follows the
     call (scenario G2, with --followup-audio streamed after the call)
  4. every server event and every guard decision is written to results/<name>.jsonl
     with ms since session start; after the run, TruthCheck compares the model's
     post-stop statement with the fake service. aggregate.py rebuilds
     results/summary.md from the JSONL files.
  5. --save-audio (as in the stop test) writes the model's output audio of each run as a
     WAV plus a JSON sidecar with chunk arrival times, copies of the user clips with
     their send times, and the guard's decisions (results/audio_out/).

The API key is read only from GEMINI_API_KEY (.env here or the environment) and is
never printed. Exit status 3 means a quota or billing error stopped the runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
import wave
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from commit_guard import CommitGuard, GuardConfig, GuardedTool, TruthCheck, TruthResult

try:  # websockets is a google-genai dependency
    from websockets.exceptions import ConnectionClosed
except Exception:  # pragma: no cover
    ConnectionClosed = ()  # type: ignore[assignment]

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "gemini-3.8-live"
SDK_VERSION = pkg_version("google-genai")

# Same system prompt as the stop-test harness, so guard off/on and the harness compare.
SYSTEM_INSTRUCTION = (
    "You are a scheduling assistant. When the user asks to book a slot, call "
    "book_slot immediately, then tell the user you are booking it. Keep replies "
    "to one or two short sentences. If the user tells you to stop or cancel, "
    "tell them plainly whether the booking was already made or not."
)
AUDIO_DIR = HERE / "assets" / "audio"
AUDIO_RATE = 16000
AUDIO_MIME = f"audio/pcm;rate={AUDIO_RATE}"
CHUNK_S = 0.1
MODEL_AUDIO_RATE = 24000  # Hz; used for --save-audio only if the mime type has no rate

# ---------------------------------------------------------------- redaction --

_SECRETS: list[str] = []
_KEY_RE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")
_KEY_PARAM_RE = re.compile(r"((?:api[_-]?)?key=)[^&\s'\"]+", re.IGNORECASE)


def redact(text: str) -> str:
    for secret in _SECRETS:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _KEY_RE.sub("[REDACTED]", text)
    return _KEY_PARAM_RE.sub(r"\1[REDACTED]", text)


def err_text(exc: BaseException) -> str:
    return redact(f"{type(exc).__name__}: {exc}")[:500]


# ------------------------------------------------------- booking (app side) --

_DAYS = r"today|tonight|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday"


def canonical_slot(args: dict[str, Any]) -> dict[str, Any]:
    """Business key of a booking: day + 24 h time. The model re-issued 'tomorrow 3pm'
    as 'tomorrow at 3pm' in the stop test, so raw sorted JSON would not match."""
    s = re.sub(r"\s+", " ", str(args.get("slot", "")).lower()).strip()
    day = re.search(rf"\b({_DAYS})\b", s)
    tm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\b", s)
    if day and tm:
        hour = int(tm.group(1)) % 12 + (12 if tm.group(3) == "p" else 0)
        return {"slot": f"{day.group(1)} {hour:02d}:{int(tm.group(2) or 0):02d}"}
    return {"slot": " ".join(w for w in re.split(r"[\s,]+", s) if w not in ("at", "the", "on"))}


def describe_booking(args: dict[str, Any], result: dict[str, Any] | None) -> str:
    slot = str(args.get("slot", "the requested slot"))
    ref = (result or {}).get("confirmation_id")
    return f"booking {ref} for {slot}" if ref else f"booking for {slot}"


class FakeBookingService:
    """Two-phase fake backend. prepare() is the slow, reversible part (--latency s);
    commit() is instant and irreversible; cancel() releases a job not yet committed.
    Its lists are the ground truth the metrics use."""

    def __init__(self, latency_s: float, ms: Callable[[], int]) -> None:
        self.latency_s = latency_s
        self.ms = ms
        self.state: dict[str, str] = {}
        self.args: dict[str, dict[str, Any]] = {}
        self.prepared: list[dict[str, Any]] = []
        self.committed: list[dict[str, Any]] = []
        self.cancelled: list[dict[str, Any]] = []
        self._ref = 1000

    async def prepare(self, job_id: str, tool: str, args: dict[str, Any]) -> None:
        self.state[job_id], self.args[job_id] = "preparing", dict(args)
        await asyncio.sleep(self.latency_s)
        self.state[job_id] = "prepared"
        self.prepared.append({"job_id": job_id, "at_ms": self.ms()})

    async def commit(self, job_id: str) -> dict[str, Any]:
        if self.state.get(job_id) != "prepared":
            raise RuntimeError(f"cannot commit {job_id} in state {self.state.get(job_id)}")
        self._ref += 1
        record = {"status": "booked", "slot": self.args[job_id].get("slot"),
                  "confirmation_id": f"BK-{self._ref}"}
        self.state[job_id] = "committed"
        self.committed.append(record | {"job_id": job_id, "at_ms": self.ms()})
        return record

    async def cancel(self, job_id: str) -> None:
        if self.state.get(job_id) == "committed":
            raise RuntimeError(f"{job_id} is already committed")
        before = self.state.get(job_id)
        self.state[job_id] = "cancelled"
        self.cancelled.append({"job_id": job_id, "at_ms": self.ms(), "was": before})


# ------------------------------------------------------------------ helpers --


class Clock:
    """Milliseconds since session start, from time.monotonic()."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()

    def ms(self) -> int:
        return int(round((time.monotonic() - self.t0) * 1000))

    def s(self) -> float:
        return time.monotonic() - self.t0


class JsonlWriter:
    def __init__(self, path: Path, scenario: str, verbose: bool) -> None:
        self.scenario = scenario
        self.verbose = verbose
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def write(self, run: int, t_ms: int, event: str, **fields: Any) -> None:
        obj = {"scenario": self.scenario, "run": run, "t_ms": t_ms, "event": event} | fields
        line = redact(json.dumps(obj, ensure_ascii=False, default=str))
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.verbose:
            print("   ", line[:300])

    def close(self) -> None:
        self._fh.close()


def load_pcm(path: Path) -> bytes:
    """Raw PCM frames of a 16 kHz, 16-bit, mono, uncompressed WAV file."""
    with wave.open(str(path), "rb") as w:
        fmt = (w.getframerate(), w.getsampwidth() * 8, w.getnchannels(), w.getcomptype())
        if fmt != (AUDIO_RATE, 16, 1, "NONE"):
            raise ValueError(f"{path.name}: expected (16000 Hz, 16 bit, 1 ch, NONE), got {fmt}")
        return w.readframes(w.getnframes())


def pcm_seconds(pcm: bytes) -> float:
    return len(pcm) / (AUDIO_RATE * 2)


@dataclass
class Utterance:
    label: str
    chunks: list[bytes]
    started: asyncio.Future  # ms of the first chunk sent, or None if sending failed
    ended: asyncio.Future    # ms of the last chunk sent, or None if sending failed


class Mic:
    """Simulated open microphone (same as the stop-test harness): CHUNK_S of PCM per
    send_realtime_input call against an absolute real-time schedule, silence while no
    utterance is queued, and a queued utterance starts on the very next send."""

    def __init__(self, session: Any, st: "RunState") -> None:
        self.session = session
        self.st = st
        self.chunk_bytes = int(AUDIO_RATE * CHUNK_S) * 2
        self.silence = bytes(self.chunk_bytes)
        self.queue: deque[Utterance] = deque()
        self.wake = asyncio.Event()

    def say(self, label: str, pcm: bytes) -> Utterance:
        loop = asyncio.get_running_loop()
        chunks = [pcm[i:i + self.chunk_bytes] for i in range(0, len(pcm), self.chunk_bytes)]
        utt = Utterance(label, chunks, loop.create_future(), loop.create_future())
        self.queue.append(utt)
        self.wake.set()
        return utt

    def _fail_pending(self, current: Utterance | None) -> None:
        for utt in ([current] if current else []) + list(self.queue):
            for fut in (utt.started, utt.ended):
                if not fut.done():
                    fut.set_result(None)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        st = self.st
        current: Utterance | None = None
        idx = 0
        next_t = loop.time()
        while not (st.closed or st.ending):
            self.wake.clear()
            if current is None and self.queue:
                current, idx = self.queue.popleft(), 0
            data = current.chunks[idx] if current else self.silence
            try:
                await self.session.send_realtime_input(
                    audio=types.Blob(data=data, mime_type=AUDIO_MIME))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not (st.closed or st.ending):
                    st.error("send_audio", err_text(exc))
                self._fail_pending(current)
                return
            if current is not None:
                if idx == 0:
                    t = st.emit("user_audio_start", label=current.label,
                                audio_s=round(pcm_seconds(b"".join(current.chunks)), 3))
                    st.utterances.append({"label": current.label, "start_ms": t})
                    current.started.set_result(t)
                idx += 1
                if idx == len(current.chunks):
                    t = st.emit("user_audio_end", label=current.label)
                    st.utterances[-1]["end_ms"] = t
                    current.ended.set_result(t)
                    current = None
            next_t += len(data) / (AUDIO_RATE * 2)
            now = loop.time()
            if next_t < now - 0.5:  # event loop stalled: resync instead of bursting
                next_t = now
            delay = next_t - now
            if current is None and not self.queue:
                try:
                    await asyncio.wait_for(self.wake.wait(), max(0.0, delay))
                    next_t = loop.time()  # utterance queued while idle: start now
                except TimeoutError:
                    pass
            elif delay > 0:
                await asyncio.sleep(delay)


class WsTap:
    """Wraps the SDK's websocket to see raw key names and in-band errors."""

    KNOWN_TOP = {"setupComplete", "serverContent", "toolCall", "toolCallCancellation",
                 "goAway", "sessionResumptionUpdate", "usageMetadata", "voiceActivity",
                 "voiceActivityDetectionSignal", "error"}

    def __init__(self, ws: Any, st: "RunState") -> None:
        self._ws = ws
        self._st = st

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)

    async def recv(self, *args: Any, **kwargs: Any) -> Any:
        try:
            raw = await self._ws.recv(*args, **kwargs)
        except ConnectionClosed as exc:  # type: ignore[misc]
            self._st.on_ws_closed(exc)
            raise
        try:
            data = json.loads(raw) if raw else {}
        except Exception:
            return raw
        if isinstance(data, dict):
            if isinstance(data.get("error"), dict):
                e = data["error"]
                self._st.error("server_error_message", f"code={e.get('code')} status="
                               f"{e.get('status')} message={redact(str(e.get('message')))[:300]}")
            unknown = sorted(set(data) - self.KNOWN_TOP)
            if unknown:
                self._st.emit("raw_unknown_keys", keys=unknown)
        return raw


# -------------------------------------------------------------------- state --


@dataclass
class RunState:
    args: argparse.Namespace
    run: int
    clock: Clock
    out: JsonlWriter
    service: FakeBookingService = field(init=False)
    guard: CommitGuard | None = None
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_at_ms: int | None = None
    stop_sent_at_ms: int | None = None    # first chunk of the stop clip
    stop_audio_end_ms: int | None = None  # last chunk of the stop clip
    stop_skipped: str | None = None
    stop_skipped_ms: int | None = None
    no_tool_call: bool = False
    done_reason: str = "unknown"
    closed: bool = False
    ending: bool = False
    texts: list[tuple[int, int, str]] = field(default_factory=list)  # (ms, turn, text)
    turn_idx: int = 0
    last_output_ms: int = -1
    turn_complete_ms: list[int] = field(default_factory=list)
    generation_complete_ms: list[int] = field(default_factory=list)
    interrupted_ms: list[int] = field(default_factory=list)
    cancellations: list[dict] = field(default_factory=list)
    vad_events: list[dict] = field(default_factory=list)
    input_texts: list[tuple[int, str]] = field(default_factory=list)
    utterances: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    audio_seg: dict | None = None
    speech_after_call_ms: int | None = None
    tool_call_event: asyncio.Event = field(default_factory=asyncio.Event)
    model_speech_event: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: list[asyncio.Task] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # everything emitted, for note_cuts()
    audio_out: list[dict] = field(default_factory=list)  # --save-audio: per-chunk metadata
    audio_data: list[bytes] = field(default_factory=list)  # --save-audio: chunk bytes
    wall: str = ""

    def __post_init__(self) -> None:
        self.service = FakeBookingService(self.args.latency, self.clock.ms)

    def emit(self, event: str, **fields: Any) -> int:
        t = self.clock.ms()
        self.out.write(self.run, t, event, **fields)
        self.events.append({"t_ms": t, "event": event} | fields)
        return t

    def error(self, where: str, detail: str) -> None:
        self.errors.append(f"{where}: {detail}")
        self.emit("error", where=where, detail=detail)

    def on_ws_closed(self, exc: BaseException) -> None:
        if self.closed:
            return
        self.closed = True
        rcvd = getattr(exc, "rcvd", None)
        code = getattr(rcvd, "code", None)
        reason = redact(str(getattr(rcvd, "reason", "") or ""))[:300]
        self.emit("ws_closed", code=code, reason=reason, during_run_end=self.ending)
        if not self.ending:
            self.errors.append(f"ws_closed code={code} {reason}".strip())

    def audio_chunk(self, data: bytes = b"", mime: str = "") -> None:
        t = self.clock.ms()
        if getattr(self.args, "save_audio", False):
            self.audio_out.append({"t_ms": t, "bytes": len(data), "mime_type": mime,
                                   "turn": self.turn_idx})
            self.audio_data.append(data)
        if self.audio_seg is None:
            self.audio_seg = {"first_ms": t, "last_ms": t, "chunks": 0}
            self.emit("audio_start")
        self.audio_seg["last_ms"] = t
        self.audio_seg["chunks"] += 1
        self.last_output_ms = t
        if self.tool_call_at_ms is not None and self.speech_after_call_ms is None:
            self.speech_after_call_ms = t
            self.emit("model_speech_after_tool_call", since_tool_call_ms=t - self.tool_call_at_ms)
            self.model_speech_event.set()

    def flush_audio(self, reason: str) -> None:
        if self.audio_seg is not None:
            self.emit("audio_segment", closed_by=reason, **self.audio_seg)
            self.audio_seg = None

    def ms(self, s: float | None) -> int | None:
        return None if s is None else int(round(s * 1000))

    def sends_ms(self) -> list[int]:
        g = self.guard
        return [self.ms(x["at"]) for x in (g.responses + g.notes)] if g else []

    def settled(self, now: int) -> bool:
        """The model answered after the stop clip, every job is final, the model is not
        mid-turn, and any tool response or note sent after its last turn got a reply or
        --quiet s of silence. A short tail catches a re-issued call."""
        ref = self.stop_audio_end_ms
        if ref is None or now - ref < self.args.min_post_stop * 1000:
            return False
        if self.guard is None or self.guard.pending():
            return False
        if self.last_output_ms < ref:
            return False
        last_tc = max(self.turn_complete_ms, default=-1)
        if last_tc < ref or self.last_output_ms > last_tc:
            return False
        last_send = max(self.sends_ms(), default=-1)
        if last_send > last_tc:
            return now - max(last_send, self.last_output_ms) >= self.args.quiet * 1000
        return now - last_tc >= 1500


# ------------------------------------------------------------------- config --


def build_config(args: argparse.Namespace) -> types.LiveConnectConfig:
    decl: dict[str, Any] = dict(
        name="book_slot",
        description="Book an appointment slot for the user. Returns a confirmation.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"slot": types.Schema(
                type=types.Type.STRING,
                description="The slot to book, e.g. 'tomorrow 3pm'.")},
            required=["slot"],
        ),
    )
    if args.behavior != "unset":
        decl["behavior"] = types.Behavior(args.behavior)
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_INSTRUCTION)]),
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(**decl)])],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        input_audio_transcription=types.AudioTranscriptionConfig(),
    )


def guard_config(args: argparse.Namespace) -> GuardConfig:
    on = args.guard == "on"
    return GuardConfig(
        hold=on and not args.no_hold, dedupe=on and not args.no_dedupe,
        inject=on and not args.no_inject, abandon=on and not args.no_abandon,
        grace_s=args.grace, transcript_timeout_s=args.transcript_timeout,
        on_timeout=args.on_timeout, dedupe_window_s=args.dedupe_window)


# ------------------------------------------------------------- run pieces ----


def enum_str(v: Any) -> str | None:
    return None if v is None else str(getattr(v, "value", v))


def handle_server_content(st: RunState, sc: types.LiveServerContent) -> None:
    if sc.model_turn and sc.model_turn.parts:
        for part in sc.model_turn.parts:
            if part.inline_data is not None and (part.inline_data.mime_type or "").startswith("audio"):
                st.audio_chunk(part.inline_data.data or b"", part.inline_data.mime_type or "")
            elif part.text and not part.thought:
                t = st.emit("model_text", text=part.text)
                st.texts.append((t, st.turn_idx, part.text))
                st.last_output_ms = t
    if sc.output_transcription and sc.output_transcription.text:
        t = st.emit("model_transcript", text=sc.output_transcription.text)
        st.texts.append((t, st.turn_idx, sc.output_transcription.text))
        st.last_output_ms = t
    if sc.input_transcription and sc.input_transcription.text:
        t = st.emit("input_transcript", text=sc.input_transcription.text)
        st.input_texts.append((t, sc.input_transcription.text))
    if sc.interrupted:
        st.flush_audio("interrupted")
        st.interrupted_ms.append(st.emit("interrupted"))
        st.turn_idx += 1
    if sc.generation_complete:
        st.flush_audio("generation_complete")
        st.generation_complete_ms.append(st.emit("generation_complete"))
    if sc.turn_complete:
        st.flush_audio("turn_complete")
        st.turn_complete_ms.append(st.emit("turn_complete"))
        st.turn_idx += 1


def log_message(session: Any, st: RunState, msg: types.LiveServerMessage) -> None:
    if msg.voice_activity:
        kind = enum_str(msg.voice_activity.voice_activity_type)
        t = st.emit("voice_activity", voice_activity_type=kind,
                    audio_offset=msg.voice_activity.audio_offset)
        st.vad_events.append({"at_ms": t, "type": kind})
    if msg.server_content:
        handle_server_content(st, msg.server_content)
    if msg.tool_call:
        for fc in msg.tool_call.function_calls or []:
            first = st.tool_call_at_ms is None and fc.name == "book_slot"
            t = st.emit("tool_call_received", call_id=fc.id, name=fc.name,
                        args=dict(fc.args or {}), first=first)
            st.tool_calls.append({"id": fc.id, "name": fc.name, "at_ms": t,
                                  "args": dict(fc.args or {})})
            if first:
                st.tool_call_at_ms = t
                st.tool_call_event.set()
    if msg.tool_call_cancellation:
        ids = list(msg.tool_call_cancellation.ids or [])
        st.cancellations.append({"at_ms": st.emit("tool_call_cancellation", ids=ids), "ids": ids})
    if msg.go_away:
        st.emit("go_away", time_left=str(msg.go_away.time_left))


async def receiver(session: Any, st: RunState) -> None:
    consecutive_errors = 0
    while not st.closed:
        try:
            # session.receive() stops after each completed turn, so loop around it.
            async for msg in session.receive():
                consecutive_errors = 0
                log_message(session, st, msg)
                assert st.guard is not None
                for fc in st.guard.observe(msg):   # calls the guard does not own
                    fr = types.FunctionResponse(id=fc.id, name=fc.name,
                                                response={"error": f"unknown function {fc.name}"})
                    st.tasks.append(asyncio.create_task(
                        session.send_tool_response(function_responses=[fr])))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            if isinstance(exc, errors.APIError) and isinstance(code, int) and 1000 <= code < 5000:
                if not st.closed:
                    st.closed = True
                    st.emit("ws_closed", code=code, reason=err_text(exc), during_run_end=st.ending)
                    if not st.ending:
                        st.errors.append(f"ws_closed {err_text(exc)[:200]}")
                return
            st.error("receive", err_text(exc))
            consecutive_errors += 1
            if st.closed or consecutive_errors >= 5:
                return
            await asyncio.sleep(0.05)


def skip_stop(st: RunState, reason: str) -> None:
    st.stop_skipped = reason
    st.stop_skipped_ms = st.emit("stop_skipped", reason=reason)


async def user_script(session: Any, st: RunState) -> None:
    """--stop-after counts from the tool call to the FIRST chunk of the stop clip.
    --stop-after-request counts from the LAST chunk of the booking clip.
    --stop-after-model-speech counts from the first model audio after the call."""
    a = st.args
    mic = Mic(session, st)
    st.tasks.append(asyncio.create_task(mic.run()))
    book = mic.say("book_request", a.clips["book"])
    book_end = await book.ended
    if book_end is None:
        st.no_tool_call = True
        return
    if a.stop_after_request is not None:
        target = book_end + a.stop_after_request * 1000
    else:
        try:
            await asyncio.wait_for(st.tool_call_event.wait(), a.tool_call_wait)
        except TimeoutError:
            pass
        if st.tool_call_at_ms is None:
            st.no_tool_call = True
            st.emit("no_tool_call", waited_s=a.tool_call_wait)
            return
        if a.followup_audio is not None:
            target = st.tool_call_at_ms + a.followup_after_tool_call * 1000
            await asyncio.sleep(max(0.0, (target - st.clock.ms()) / 1000))
            mic.say("followup", a.clips["followup"])
            st.emit("followup_queued", since_tool_call_ms=st.clock.ms() - st.tool_call_at_ms)
        if a.stop_after_model_speech is not None:
            # Anchor: first model audio after the call, only while no tool response went out.
            if st.speech_after_call_ms is None:
                try:
                    await asyncio.wait_for(st.model_speech_event.wait(), a.tool_call_wait)
                except TimeoutError:
                    pass
            if st.speech_after_call_ms is None or (st.guard and st.guard.responses):
                skip_stop(st, "no model speech while the call was pending")
                return
            target = st.speech_after_call_ms + a.stop_after_model_speech * 1000
        else:
            target = st.tool_call_at_ms + a.stop_after * 1000
    await asyncio.sleep(max(0.0, (target - st.clock.ms()) / 1000))
    if a.stop_after_model_speech is not None and st.guard and st.guard.responses:
        skip_stop(st, "tool response sent before the stop")
        return
    stop = mic.say("stop", a.clips["stop"])
    started = await stop.started
    if started is None:
        st.stop_sent_at_ms = st.stop_audio_end_ms = st.clock.ms()
        return
    st.stop_sent_at_ms = started
    st.emit("stop_sent", since_tool_call_ms=(started - st.tool_call_at_ms
                                             if st.tool_call_at_ms is not None else None),
            since_book_audio_end_ms=started - book_end)
    ended = await stop.ended
    st.stop_audio_end_ms = ended if ended is not None else st.clock.ms()


async def wait_until_done(st: RunState) -> str:
    while True:
        now = st.clock.ms()
        if st.closed:
            return "ws_closed"
        if st.no_tool_call:
            return "no_tool_call"
        if st.stop_skipped is not None and now - st.stop_skipped_ms >= st.args.post_stop_window * 1000:
            return "stop_skipped_window_elapsed"
        ref = st.stop_audio_end_ms
        if ref is not None:
            if now - ref >= st.args.post_stop_window * 1000:
                return "post_stop_window_elapsed"
            if st.settled(now):
                return "settled"
        await asyncio.sleep(0.05)


QUOTA_RE = re.compile(r"quota|billing|RESOURCE_EXHAUSTED|exceeded your current|prepay|"
                      r"credits?\b|payment|\b429\b", re.IGNORECASE)
EXIT_QUOTA = 3


async def run_once(args: argparse.Namespace, run: int, out: JsonlWriter,
                   connect: Callable[..., Any]) -> RunState:
    clock = Clock()
    st = RunState(args=args, run=run, clock=clock, out=out)
    st.wall = datetime.now().isoformat(timespec="seconds")
    cfg = guard_config(args)
    st.emit("run_start", wall=st.wall, model=args.model, sdk=SDK_VERSION,
            behavior=args.behavior, guard=args.guard,
            behaviors={"hold": cfg.hold, "dedupe": cfg.dedupe, "inject": cfg.inject,
                       "abandon": cfg.abandon},
            grace_s=cfg.grace_s, transcript_timeout_s=cfg.transcript_timeout_s,
            on_timeout=cfg.on_timeout, dedupe_window_s=cfg.dedupe_window_s,
            stop_after_s=args.stop_after, stop_after_request_s=args.stop_after_request,
            stop_after_model_speech_s=args.stop_after_model_speech,
            followup=Path(args.followup_audio).name if args.followup_audio else None,
            latency_s=args.latency,
            clips_s={k: round(pcm_seconds(v), 3) for k, v in args.clips.items()})
    done_reason = "unknown"
    tool = GuardedTool("book_slot", blocking=args.behavior == "BLOCKING",
                       canonical=canonical_slot, describe=describe_booking)
    try:
        async with asyncio.timeout(args.run_timeout):
            async with connect(model=args.model, config=build_config(args)) as session:
                st.emit("setup_complete")
                if hasattr(session, "_ws"):
                    session._ws = WsTap(session._ws, st)
                st.guard = CommitGuard(
                    session, st.service, [tool], cfg, now=clock.s,
                    log=lambda event, **f: st.emit("guard_" + event, **f))
                recv_task = asyncio.create_task(receiver(session, st))
                script_task = asyncio.create_task(user_script(session, st))
                st.tasks += [recv_task, script_task]
                done_reason = await wait_until_done(st)
                st.emit("run_stop_condition", reason=done_reason)
                st.ending = True
                for t in st.tasks:
                    t.cancel()
                await asyncio.gather(*st.tasks, return_exceptions=True)
                await st.guard.aclose()
        st.emit("session_closed")
    except TimeoutError:
        done_reason = "run_timeout"
        st.error("run_timeout", f"hard timeout {args.run_timeout}s")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        done_reason = "exception"
        st.error("session", err_text(exc))
    finally:
        st.ending = True
        if st.guard is not None:
            await st.guard.aclose()
        for t in st.tasks:
            t.cancel()
        await asyncio.gather(*st.tasks, return_exceptions=True)
        st.flush_audio("run_end")
    st.done_reason = done_reason
    return st


# ----------------------------------------------------------------- summary --


def md(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).replace("|", "\\|").strip()


def turns_with_times(chunks: list[tuple[int, int, str]]) -> list[dict]:
    """One entry per model turn: time of its first text chunk and the joined text."""
    turns: dict[int, dict] = {}
    for t, turn, txt in chunks:
        d = turns.setdefault(turn, {"start_ms": t, "text": ""})
        d["text"] += txt
    return [dict(d, text=re.sub(r"\s+", " ", d["text"]).strip())
            for d in turns.values() if d["text"].strip()]


def post_stop_truth(turns: list[dict], stop_end: Any, committed: bool) -> TruthResult:
    """TruthCheck on the model turns that start after the last chunk of the stop clip."""
    after = [x["text"] for x in turns if isinstance(stop_end, int) and x["start_ms"] >= stop_end]
    return TruthCheck().check(after, committed=committed)


def first_claim(turns: list[dict], stop_end: Any, committed: bool) -> tuple[str, bool, str]:
    """The model's FIRST claim after the stop clip: the shortest run of post-stop turns
    (joined, so a cut sentence still reads whole) that makes a claim. Returns (claim,
    consistent, text). TruthCheck itself reports the LAST claim."""
    after = [x["text"] for x in turns if isinstance(stop_end, int) and x["start_ms"] >= stop_end]
    for i in range(1, len(after) + 1):
        r = TruthCheck().check(after[:i], committed=committed)
        if r.claim != "neither":
            return r.claim, r.consistent, r.statement
    return "neither", False, ""


def note_cuts(events: list[dict]) -> list[dict]:
    """Model turns cut by an `interrupted` that the guard's own status note caused
    (send_client_content with turn_complete=True interrupts generation). A client that
    drops queued audio on `interrupted` plays at most `audio_received_ms` of the cut
    text, so the listener may hear only the continuation in the next turn."""
    cuts: list[dict] = []
    cur, cut_text, seg_first = "", "", None
    for e in events:
        ev = e["event"]
        if ev in ("model_transcript", "model_text"):
            cur += e["text"]
        elif ev == "audio_segment" and e.get("closed_by") == "interrupted":
            seg_first = e.get("first_ms")
        elif ev == "interrupted":
            cut_text, cur = cur, ""
        elif ev == "turn_complete":
            cur = ""
        elif ev == "guard_interrupted_seen" and e.get("caused_by_own_note") and cut_text.strip():
            cuts.append({"at_ms": e["t_ms"], "text": cut_text.strip(),
                         "audio_received_ms": e["t_ms"] - seg_first if seg_first is not None else None})
            cut_text = ""
    return cuts


def user_interrupt_ms(events: list[dict], stop: Any) -> int | None:
    """First `interrupted` at or after the stop that the guard did not cause itself."""
    if not isinstance(stop, int):
        return None
    return next((e["t_ms"] for e in events if e["event"] == "guard_interrupted_seen"
                 and not e.get("caused_by_own_note") and e["t_ms"] >= stop), None)


def fmt_cuts(cuts: list[dict]) -> str:
    return "; ".join(f"{c['text']!r} ({c['audio_received_ms']} ms of its audio received)"
                     for c in cuts) or "no"


def summarize(st: RunState) -> dict:
    a, g, svc = st.args, st.guard, st.service
    stop, stop_end = st.stop_sent_at_ms, st.stop_audio_end_ms
    turns = turns_with_times(st.texts)
    after = [x for x in turns if stop_end is not None and x["start_ms"] >= stop_end]
    commits = svc.committed
    keys = {j.job_id: j.key for j in (g.jobs if g else [])}
    per_key = Counter(keys.get(c["job_id"]) for c in commits)
    unwanted = [c for c in commits if stop is not None and c["at_ms"] > stop]
    truth = post_stop_truth(turns, stop_end, bool(commits))
    first = first_claim(turns, stop_end, bool(commits))
    cuts = note_cuts(st.events)
    decisions = g.decisions if g else []
    first_dec = decisions[0] if decisions else None
    guard_cancels = [j for j in (g.jobs if g else []) if j.state == "cancelled" and j.decision == "cancel"]
    onset = next((e["at_ms"] for e in st.vad_events
                  if e["type"] == "ACTIVITY_START" and stop is not None and e["at_ms"] >= stop), None)
    intr = user_interrupt_ms(st.events, stop)
    stop_utt = next((u for u in (g.utterances if g else []) if u.intent == "stop"), None)
    notes = g.notes if g else []
    first_note = st.ms(notes[0]["at"]) if notes else None
    service = [f"committed {c['confirmation_id']} @{c['at_ms']}" for c in commits]
    service += [f"cancelled {c['job_id']} @{c['at_ms']} (was {c['was']})" for c in svc.cancelled]
    row = {
        "scenario": a.scenario or a.name, "run": st.run, "guard": a.guard,
        "behaviors": "".join(k[0].upper() if getattr(g.cfg, k) else "-"
                             for k in ("hold", "dedupe", "inject", "abandon")) if g else "-",
        "behavior": a.behavior, "latency_s": a.latency,
        "n_tool_calls": len(st.tool_calls),
        "tool_call_at_ms": st.tool_call_at_ms if st.tool_call_at_ms is not None else "no_tool_call",
        "stop_sent_at_ms": stop if stop is not None else f"skipped ({st.stop_skipped})",
        "stop_audio_end_ms": stop_end if stop_end is not None else "-",
        "stop_onset_ms": onset if onset is not None else "-",
        "interrupted_ms": intr if intr is not None else "no",
        "stop_transcript_ms": st.ms(stop_utt.at) if stop_utt else "-",
        "decision": (f"{first_dec['decision']} @{st.ms(first_dec['at'])}: {first_dec['reason'][:70]}"
                     if first_dec else "-"),
        "time_from_stop_to_decision_ms": (st.ms(first_dec["at"]) - stop
                                          if first_dec and stop is not None else "-"),
        "service": "; ".join(service) or "nothing",
        "cancellation_before_commit": "yes" if guard_cancels and not commits else "no",
        "unwanted_commit": "yes" if unwanted else "no",
        "double_booking": "yes" if any(n > 1 for n in per_key.values()) else "no",
        "duplicates_suppressed": len(g.duplicates) if g else 0,
        "tool_responses": ", ".join(f"{r['status']} @{st.ms(r['at'])}" for r in (g.responses if g else []))
                          or "none",
        "notes_ms": ", ".join(str(st.ms(n["at"])) for n in notes) or "none",
        "note_cut_reply": fmt_cuts(cuts),
        "model_after_stop": " / ".join(x["text"] for x in after)[:220] or "-",
        "claim": truth.claim, "actual": truth.actual,
        "model_statement_consistent": str(truth.consistent).lower(),
        "first_claim": first[0], "first_statement_consistent": str(first[1]).lower(),
        "errors": "; ".join(e[:120] for e in st.errors) or "-",
        # JSONL only
        "done_reason": st.done_reason,
        "tool_calls": st.tool_calls,
        "model_turns": turns,
        "statement_used": truth.statement,
        "note_cuts": cuts,
        "reply_after_note": [x for x in turns if first_note is not None and x["start_ms"] >= first_note],
        "notes": [{"at_ms": st.ms(n["at"]), "text": n["text"]} for n in notes],
        "decisions": [d | {"at_ms": st.ms(d["at"])} for d in decisions],
        "duplicates": [d | {"at_ms": st.ms(d["at"])} for d in (g.duplicates if g else [])],
        "responses": [{"at_ms": st.ms(r["at"]), "call_id": r["call_id"], "payload": r["payload"]}
                      for r in (g.responses if g else [])],
        "utterances": [{"onset_ms": st.ms(u.opened_at), "resolved_ms": st.ms(u.at),
                        "intent": u.intent, "how": u.how, "text": u.text}
                       for u in (g.utterances if g else [])],
        "vad_events": st.vad_events, "user_audio": st.utterances,
        "input_transcripts": [{"at_ms": t, "text": x} for t, x in st.input_texts],
        "commits": commits, "cancelled": svc.cancelled,
        "interrupted_all_ms": st.interrupted_ms, "turn_complete_ms": st.turn_complete_ms,
        "tool_call_cancellations": st.cancellations,
    }
    return row


COLUMNS = ["run", "guard", "behaviors", "n_tool_calls", "tool_call_at_ms", "stop_sent_at_ms",
           "stop_audio_end_ms", "stop_onset_ms", "interrupted_ms", "stop_transcript_ms",
           "decision", "time_from_stop_to_decision_ms", "service",
           "cancellation_before_commit", "unwanted_commit", "double_booking",
           "duplicates_suppressed", "tool_responses", "notes_ms", "note_cut_reply",
           "model_after_stop",
           "claim", "model_statement_consistent", "first_claim",
           "first_statement_consistent", "errors"]


def scenario_meta(args: argparse.Namespace) -> str:
    """One line describing a scenario's settings, for the summary table header."""
    cfg = guard_config(args)
    if args.stop_after_model_speech is not None:
        stop = (f"follow-up {Path(args.followup_audio).name} {args.followup_after_tool_call} s "
                f"after the call, stop clip {args.stop_after_model_speech} s after the first "
                "model audio that follows the call")
    elif args.stop_after_request is not None:
        stop = f"stop clip {args.stop_after_request} s after the last chunk of the booking clip"
    else:
        stop = f"stop clip {args.stop_after} s after the tool call"
    return (f"model `{args.model}`, google-genai {SDK_VERSION}, "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}, guard={args.guard} "
            f"(hold={cfg.hold}, dedupe={cfg.dedupe}, inject={cfg.inject}, abandon={cfg.abandon}; "
            f"grace {cfg.grace_s} s, transcript timeout {cfg.transcript_timeout_s} s, "
            f"on timeout {cfg.on_timeout}, dedupe window {cfg.dedupe_window_s} s), "
            f"behavior={args.behavior}, {stop}, service prepare {args.latency} s then instant "
            "commit"
            + (", model output audio saved (--save-audio) to results/audio_out/"
               f"{args.name}_run<N>_model.wav" if getattr(args, "save_audio", False) else "")
            + ". Times are ms since session start; `behaviors` is H/D/I/A for "
            "hold/dedupe/inject/abandon.")


def scenario_table(name: str, meta: str, rows: list[dict]) -> str:
    lines = [f"### {name}", "", meta, "",
             "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(md(r.get(c, "-")) for c in COLUMNS) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- save audio --


def mime_rate(mime: str) -> int | None:
    m = re.search(r"rate=(\d+)", mime or "")
    return int(m.group(1)) if m else None


def save_run_audio(st: RunState, out_dir: Path) -> dict:
    """--save-audio, same file layout as the stop test's harness:
    <name>_run<N>_model.wav (model output, chunks concatenated in arrival order),
    <name>_run<N>_user_<label>.wav (copies of the clips sent), and
    <name>_run<N>_audio.json (chunk arrival times, clip send times, key events and the
    guard's decisions)."""
    a, g, svc = st.args, st.guard, st.service
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.name}_run{st.run}"
    mimes = sorted({c["mime_type"] for c in st.audio_out})
    rates = {mime_rate(m) for m in mimes}
    if len(rates) == 1 and None not in rates:
        rate, rate_source = rates.pop(), "mime type"
    else:
        rate, rate_source = MODEL_AUDIO_RATE, f"default (mime types: {mimes or 'none'})"
    chunks, offset = [], 0
    for meta, data in zip(st.audio_out, st.audio_data):
        chunks.append({"t_ms": meta["t_ms"], "turn": meta["turn"], "bytes": len(data),
                       "wav_offset_ms": round(offset / 2 / rate * 1000, 1),
                       "duration_ms": round(len(data) / 2 / rate * 1000, 1)})
        offset += len(data)
    pcm = b"".join(st.audio_data)
    if len(pcm) % 2:
        pcm = pcm[:-1]
    wav_path = out_dir / f"{stem}_model.wav"
    if pcm:
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm)
    user = []
    for u in st.utterances:
        src = Path(a.audio_paths[u["label"]])
        dst = out_dir / f"{stem}_user_{u['label']}.wav"
        shutil.copyfile(src, dst)
        user.append({"label": u["label"], "file": dst.name, "source": src.name,
                     "sent_start_ms": u.get("start_ms"), "sent_end_ms": u.get("end_ms"),
                     "duration_ms": round(pcm_seconds(load_pcm(src)) * 1000, 1),
                     "sample_rate": AUDIO_RATE})
    sidecar = {
        "scenario": a.name, "run": st.run, "wall": st.wall, "model": a.model,
        "guard": a.guard,
        "time_base": "ms since session start (time.monotonic() when the run started). "
                     "t_ms of a model chunk is when the harness received it; "
                     "sent_start_ms / sent_end_ms of a user clip are when its first / last "
                     "100 ms chunk was sent.",
        "model_audio": {
            "file": wav_path.name if pcm else None,
            "mime_types": mimes, "sample_rate": rate, "sample_rate_from": rate_source,
            "channels": 1, "sample_width_bits": 16,
            "duration_s": round(len(pcm) / 2 / rate, 3), "chunks": len(chunks),
            "first_chunk_ms": chunks[0]["t_ms"] if chunks else None,
            "last_chunk_ms": chunks[-1]["t_ms"] if chunks else None,
            "note": "Chunks are concatenated in arrival order with no gaps. The server "
                    "sends audio faster than real time, so wav_offset_ms is not the "
                    "arrival time: place each chunk at its t_ms or later. A client that "
                    "plays audio drops what is still queued when `interrupted` arrives; "
                    "this file keeps everything received.",
        },
        "chunks": chunks,
        "user_clips": user,
        "events": {
            "tool_calls": st.tool_calls,
            "tool_responses": [{"at_ms": st.ms(r["at"]), "call_id": r["call_id"],
                                "status": r["status"], "payload": r["payload"]}
                               for r in (g.responses if g else [])],
            "service": {"prepared": svc.prepared, "committed": svc.committed,
                        "cancelled": svc.cancelled},
            "guard_decisions": [d | {"at_ms": st.ms(d["at"])} for d in (g.decisions if g else [])],
            "status_notes": [{"at_ms": st.ms(n["at"]), "text": n["text"]}
                             for n in (g.notes if g else [])],
            "cancellations": st.cancellations,
            "interrupted_ms": st.interrupted_ms,
            "generation_complete_ms": st.generation_complete_ms,
            "turn_complete_ms": st.turn_complete_ms,
            "voice_activity": st.vad_events,
            "stop_sent_at_ms": st.stop_sent_at_ms,
            "stop_audio_end_ms": st.stop_audio_end_ms,
            "model_transcript": [{"t_ms": t, "turn": turn, "text": x} for t, turn, x in st.texts],
            "input_transcript": [{"t_ms": t, "text": x} for t, x in st.input_texts],
        },
    }
    side_path = out_dir / f"{stem}_audio.json"
    side_path.write_text(redact(json.dumps(sidecar, indent=1, ensure_ascii=False, default=str)),
                         encoding="utf-8")
    return {"model_wav": wav_path.name if pcm else None, "sidecar": side_path.name,
            "model_audio_s": sidecar["model_audio"]["duration_s"], "sample_rate": rate,
            "user_clips": [u["file"] for u in user]}


# -------------------------------------------------------------------- main --


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gemini Live API: tool-call commit guard vs. user 'stop' (see README.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--name", required=True, help="results go to results/<name>.jsonl")
    p.add_argument("--scenario", default=None, help="scenario label for aggregation (A, C, F, G2)")
    p.add_argument("-n", "--runs", type=int, default=3)
    p.add_argument("--first-run", type=int, default=1,
                   help="number of the first run (to append runs to an existing JSONL)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--guard", choices=["on", "off"], default="on")
    p.add_argument("--no-hold", action="store_true", help="behavior 1 off")
    p.add_argument("--no-dedupe", action="store_true", help="behavior 2 off")
    p.add_argument("--no-inject", action="store_true", help="behavior 3 off")
    p.add_argument("--no-abandon", action="store_true", help="behavior 4 off")
    p.add_argument("--grace", type=float, default=1.5)
    p.add_argument("--transcript-timeout", type=float, default=3.0)
    p.add_argument("--on-timeout", choices=["cancel", "commit"], default="cancel")
    p.add_argument("--dedupe-window", type=float, default=30.0)
    p.add_argument("--behavior", choices=["unset", "BLOCKING", "NON_BLOCKING"], default="unset",
                   help="FunctionDeclaration.behavior (unset = model default, NON_BLOCKING)")
    p.add_argument("--book-audio", default=str(AUDIO_DIR / "book.wav"))
    p.add_argument("--stop-audio", default=str(AUDIO_DIR / "stop.wav"))
    p.add_argument("--stop-after", type=float, default=1.0,
                   help="seconds from the tool call to the first chunk of the stop clip")
    p.add_argument("--stop-after-request", type=float, default=None,
                   help="seconds from the last chunk of the booking clip to the stop clip")
    p.add_argument("--stop-after-model-speech", type=float, default=None,
                   help="seconds from the first model audio after the call to the stop clip")
    p.add_argument("--followup-audio", default=None)
    p.add_argument("--followup-after-tool-call", type=float, default=0.5)
    p.add_argument("--latency", type=float, default=4.0, help="seconds the prepare phase takes")
    p.add_argument("--min-post-stop", type=float, default=0.0)
    p.add_argument("--tool-call-wait", type=float, default=15.0)
    p.add_argument("--post-stop-window", type=float, default=15.0)
    p.add_argument("--quiet", type=float, default=3.0)
    p.add_argument("--run-timeout", type=float, default=50.0)
    p.add_argument("--between-runs", type=float, default=2.0)
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--save-audio", action="store_true",
                   help="write the model's output audio per run to results/audio_out/"
                        "<name>_run<N>_model.wav plus a JSON sidecar with chunk arrival "
                        "times, and copies of the user clips with their send times")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.stop_after_model_speech is not None and args.stop_after_request is not None:
        p.error("--stop-after-model-speech and --stop-after-request are exclusive")
    args.clips = {}
    args.audio_paths = {"book_request": args.book_audio, "stop": args.stop_audio}
    if args.followup_audio is not None:
        args.audio_paths["followup"] = args.followup_audio
    return args


async def amain(args: argparse.Namespace, connect: Callable[..., Any] | None = None) -> int:
    try:
        args.clips = {"book": load_pcm(Path(args.book_audio)), "stop": load_pcm(Path(args.stop_audio))}
        if args.followup_audio is not None:
            args.clips["followup"] = load_pcm(Path(args.followup_audio))
    except Exception as exc:
        print(f"cannot load audio clips: {exc}", file=sys.stderr)
        return 2
    if connect is None:
        load_dotenv(HERE / ".env")
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            print("GEMINI_API_KEY is not set (put it in .env next to this script).", file=sys.stderr)
            return 2
        _SECRETS.append(key)
        connect = genai.Client(api_key=key, vertexai=False).aio.live.connect
    results = Path(args.results_dir)
    out = JsonlWriter(results / f"{args.name}.jsonl", args.name, args.verbose)
    meta = scenario_meta(args)
    out.write(0, 0, "scenario_meta" if args.first_run == 1 else "scenario_meta_append",
              meta=meta, scenario=args.scenario or args.name, guard=args.guard,
              argv=sys.argv[1:])
    rows: list[dict] = []
    quota = None
    try:
        last = args.first_run + args.runs - 1
        for run in range(args.first_run, last + 1):
            try:
                st = await run_once(args, run, out, connect)
                row = summarize(st)
                if args.save_audio:
                    try:
                        row["saved_audio"] = save_run_audio(st, results / "audio_out")
                    except Exception as exc:
                        row["saved_audio"] = {"error": err_text(exc)}
                        out.write(run, -1, "save_audio_failed", detail=err_text(exc))
                quota = next((e for e in st.errors if QUOTA_RE.search(e)), None)
            except Exception as exc:  # never let one run kill the scenario
                quota = err_text(exc) if QUOTA_RE.search(err_text(exc)) else None
                out.write(run, -1, "run_crashed", detail=err_text(exc))
                row = {"scenario": args.scenario or args.name, "run": run, "guard": args.guard,
                       "errors": err_text(exc)}
            out.write(run, -1, "run_end", summary=row)
            rows.append(row)
            print(redact(f"[{args.name} run {run}/{last}] call={row.get('tool_call_at_ms')} "
                         f"stop={row.get('stop_sent_at_ms')} decision={row.get('decision')} "
                         f"service={row.get('service')} unwanted={row.get('unwanted_commit')} "
                         f"double={row.get('double_booking')} claim={row.get('claim')} "
                         f"consistent={row.get('model_statement_consistent')} "
                         f"said=\"{row.get('model_after_stop')}\" errors={row.get('errors')}"))
            if quota:
                out.write(run, -1, "stopped_on_quota_or_billing_error", detail=quota)
                print(redact(f"[{args.name}] stopping: quota/billing error: {quota}"))
                break
            if run < last:
                await asyncio.sleep(args.between_runs)
    finally:
        out.close()
    print()
    print(redact(scenario_table(args.name, meta, rows)))
    print("(run `uv run aggregate.py` to rebuild results/summary.md)")
    return EXIT_QUOTA if quota else 0


def main() -> None:
    args = parse_args()
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
