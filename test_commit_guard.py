"""Offline tests for commit_guard.py (no network, no key): `uv run python -m unittest -v`.

The scenario tests replay server-event timelines taken from the stop-test harness
JSONL (gemini-3.8-live, speech input) through the guard, with every duration scaled
by SCALE so the suite runs in a few seconds. They check the commit decisions only;
what the model says is measured by guard_test.py against the live API.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from google.genai import types

from commit_guard import CommitGuard, GuardConfig, GuardedTool, IntentCheck, TruthCheck
from guard_test import FakeBookingService, canonical_slot, describe_booking

SCALE = 0.05  # 1 s of the recorded timeline = 50 ms here


def vad(kind: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(voice_activity=types.VoiceActivity(voice_activity_type=kind))


def transcript(text: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(server_content=types.LiveServerContent(
        input_transcription=types.Transcription(text=text)))


def interrupted() -> types.LiveServerMessage:
    return types.LiveServerMessage(server_content=types.LiveServerContent(interrupted=True))


def call(call_id: str, slot: str = "tomorrow 3pm") -> types.LiveServerMessage:
    return types.LiveServerMessage(tool_call=types.LiveServerToolCall(function_calls=[
        types.FunctionCall(id=call_id, name="book_slot", args={"slot": slot})]))


BOOK = "Book me the 3:00 p.m. slot tomorrow, please."
STOP = "Actually, stop. Don't book it."

# (seconds since session start, message), from the stop-test JSONL (run 1 of each).
REQUEST = [(0.53, vad("ACTIVITY_START")), (4.179, transcript(BOOK)), (4.18, vad("ACTIVITY_END")),
           (4.18, call("c1"))]
STOP_AFTER_CALL = [(5.33, vad("ACTIVITY_START")), (8.655, transcript(STOP)),
                   (8.655, vad("ACTIVITY_END"))]
A = REQUEST + STOP_AFTER_CALL                                     # audio_A, latency 4 s
C = REQUEST + [(5.329, vad("ACTIVITY_START")), (5.33, interrupted()),   # audio_C rerun run 2
               (8.655, transcript(STOP)), (8.655, vad("ACTIVITY_END")),
               (9.009, call("c2", "tomorrow at 3pm"))]
F = [(0.43, vad("ACTIVITY_START")),                                # audio_F: one merged turn
     (6.612, transcript("Book me the 3:00 p.m. slot tomorrow, please. Actually, stop, don't book it.")),
     (6.613, vad("ACTIVITY_END")), (6.613, call("c1"))]
FOLLOWUP = [(4.829, vad("ACTIVITY_START")),                        # audio_G2, latency 7 s
            (8.893, transcript("While you do that. What should I bring to the appointment?")),
            (8.894, vad("ACTIVITY_END"))]
G2 = REQUEST + FOLLOWUP + [(10.255, vad("ACTIVITY_START")), (10.255, interrupted()),
                           (13.573, transcript(STOP)), (13.574, vad("ACTIVITY_END"))]


class FakeSession:
    def __init__(self) -> None:
        self.responses: list[types.FunctionResponse] = []
        self.notes: list[str] = []

    async def send_tool_response(self, *, function_responses: list[types.FunctionResponse]) -> None:
        self.responses += function_responses

    async def send_client_content(self, *, turns: types.Content, turn_complete: bool) -> None:
        assert turn_complete and turns.role == "user"
        self.notes.append(turns.parts[0].text or "")


def scaled(cfg: GuardConfig) -> GuardConfig:
    cfg.grace_s *= SCALE
    cfg.transcript_timeout_s *= SCALE
    cfg.trigger_lookback_s *= SCALE
    cfg.own_interrupt_s *= SCALE
    return cfg


async def replay(events: list[tuple[float, Any]], cfg: GuardConfig, latency_s: float = 4.0,
                 blocking: bool = False) -> tuple[CommitGuard, FakeBookingService, FakeSession]:
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    service = FakeBookingService(latency_s * SCALE, lambda: int((loop.time() - t0) / SCALE * 1000))
    session = FakeSession()
    tool = GuardedTool("book_slot", blocking=blocking, canonical=canonical_slot,
                       describe=describe_booking)
    guard = CommitGuard(session, service, [tool], scaled(cfg), now=lambda: loop.time() - t0)
    for t, msg in sorted(events, key=lambda e: e[0]):
        await asyncio.sleep(max(0.0, t * SCALE - (loop.time() - t0)))
        guard.observe(msg)
    for _ in range(400):  # let pending jobs finish (at most 20 s of recorded time)
        if not guard.pending():
            break
        await asyncio.sleep(0.05 * SCALE * 20)
    await guard.aclose()
    return guard, service, session


def status(session: FakeSession) -> list[tuple[str, str]]:
    return [(r.id or "", (r.response or {}).get("status", "")) for r in session.responses]


class ScenarioTests(unittest.IsolatedAsyncioTestCase):
    async def test_A_off_commits_after_stop(self) -> None:
        g, svc, s = await replay(A, GuardConfig.off())
        self.assertEqual(len(svc.committed), 1)
        self.assertEqual(status(s), [("c1", "booked")])
        self.assertEqual(s.notes, [])

    async def test_A_on_cancels_before_commit_and_injects_note(self) -> None:
        g, svc, s = await replay(A, GuardConfig())
        self.assertEqual(svc.committed, [])
        self.assertEqual(status(s), [("c1", "cancelled")])
        self.assertEqual(len(s.notes), 1)
        self.assertIn("was cancelled before it was committed", s.notes[0])

    async def test_A_hold_off_inject_on_says_committed_after_stop(self) -> None:
        g, svc, s = await replay(A, GuardConfig(hold=False))
        self.assertEqual(len(svc.committed), 1)
        self.assertIn("BK-1001", s.notes[0])
        self.assertIn("asked to stop after it was committed", s.notes[0])

    async def test_C_off_double_books(self) -> None:
        g, svc, s = await replay(C, GuardConfig.off(), blocking=True)
        self.assertEqual(len(svc.committed), 2)

    async def test_C_on_cancels_skips_dead_id_answers_reissue(self) -> None:
        g, svc, s = await replay(C, GuardConfig(), blocking=True)
        self.assertEqual(svc.committed, [])
        self.assertTrue(g.jobs[0].abandoned)
        self.assertEqual(status(s), [("c2", "cancelled")])   # c1 was dropped by the server
        self.assertEqual(s.responses[0].response["duplicate_of"], "c1")
        self.assertEqual(len(g.duplicates), 1)
        self.assertEqual(len(s.notes), 1)

    async def test_C_dedupe_alone_prevents_double_booking(self) -> None:
        g, svc, s = await replay(C, GuardConfig(hold=False, inject=False, abandon=False),
                                 blocking=True)
        self.assertEqual(len(svc.committed), 1)
        self.assertEqual(status(s), [("c1", "booked"), ("c2", "booked")])

    async def test_C_abandon_alone_holds_the_commit(self) -> None:
        g, svc, s = await replay(C, GuardConfig(hold=False, dedupe=False, inject=False),
                                 blocking=True)
        # c1 is held as abandoned and cancelled by the stop; c2 is a new, unheld job.
        self.assertEqual([j.state for j in g.jobs], ["cancelled", "committed"])

    async def test_F_on_cancels_from_the_triggering_utterance(self) -> None:
        g, svc, s = await replay(F, GuardConfig())
        self.assertEqual(svc.committed, [])
        self.assertIn("triggered the call", g.decisions[0]["reason"])
        self.assertEqual(status(s), [("c1", "cancelled")])

    async def test_F_off_commits(self) -> None:
        g, svc, s = await replay(F, GuardConfig.off())
        self.assertEqual(len(svc.committed), 1)

    async def test_G2_followup_question_does_not_cancel(self) -> None:
        g, svc, s = await replay(REQUEST + FOLLOWUP, GuardConfig(), latency_s=7.0)
        self.assertEqual(len(svc.committed), 1)
        self.assertIn("no stop intent", g.decisions[0]["reason"])
        self.assertEqual(s.notes, [])

    async def test_G2_on_holds_past_grace_until_the_stop_transcript(self) -> None:
        g, svc, s = await replay(G2, GuardConfig(), latency_s=7.0)
        self.assertEqual(svc.committed, [])
        self.assertEqual(g.jobs[0].state, "cancelled")

    async def test_G2_decision_comes_after_the_grace_window(self) -> None:
        # Prepared at 11.18 s, due at 12.68 s, stop transcript at 13.57 s: the grace
        # window alone would have committed; the speech hold is what waits.
        g, svc, s = await replay(G2, GuardConfig(), latency_s=7.0)
        job = g.jobs[0]
        assert job.prepared_at is not None and job.decided_at is not None
        self.assertGreater(job.decided_at, job.prepared_at + g.cfg.grace_s)

    async def test_timeout_without_transcript_fails_closed(self) -> None:
        g, svc, s = await replay(REQUEST + [(5.33, vad("ACTIVITY_START"))], GuardConfig())
        self.assertEqual(svc.committed, [])
        self.assertIn("could not be confirmed", s.notes[0])

    async def test_timeout_policy_commit(self) -> None:
        g, svc, s = await replay(REQUEST + [(5.33, vad("ACTIVITY_START"))],
                                 GuardConfig(on_timeout="commit"))
        self.assertEqual(len(svc.committed), 1)

    async def test_new_request_after_cancel_is_not_a_duplicate(self) -> None:
        again = [(10.0, vad("ACTIVITY_START")), (11.0, transcript("OK, book it after all.")),
                 (11.0, vad("ACTIVITY_END")), (11.5, call("c3"))]
        g, svc, s = await replay(A + again, GuardConfig())
        self.assertEqual(len(svc.committed), 1)
        self.assertEqual(g.duplicates, [])


class RuleTests(unittest.TestCase):
    def test_intent(self) -> None:
        ic = IntentCheck()
        self.assertEqual(ic.classify(STOP), "stop")
        self.assertEqual(ic.classify("Never mind."), "stop")
        self.assertEqual(ic.classify("Hold on a second"), "stop")
        self.assertEqual(ic.classify(BOOK), "other")
        self.assertEqual(ic.classify("While you do that. What should I bring to the appointment?"),
                         "other")
        self.assertEqual(ic.classify(""), "unknown")

    def test_slot_key(self) -> None:
        self.assertEqual(canonical_slot({"slot": "tomorrow 3pm"}),
                         canonical_slot({"slot": "Tomorrow at 3 P.M."}))
        self.assertNotEqual(canonical_slot({"slot": "tomorrow 3pm"}),
                            canonical_slot({"slot": "tomorrow 4pm"}))

    def test_truth_rules_on_recorded_statements(self) -> None:
        tc = TruthCheck()
        booked = [
            "I'm sorry, but the 3:00 PM slot for tomorrow was already booked. Please let me "
            "know if you would like me to cancel it.",
            "The booking for tomorrow at 3 PM was already made before you asked to cancel.",
            "It is too late to stop; the booking for tomorrow at 3 PM was already made.",
            "The booking was already made and cannot be canceled.",
            "I have already booked that slot for you.",
            "Booking BK-1001 for tomorrow 3pm is confirmed. Would you like me to cancel it?",
            "I'm sorry, the booking was already confirmed before you cancelled.",
        ]
        not_booked = [
            "I have not booked the slot. It has been canceled as requested.",
            "I've stopped the process, and the booking booking was not made.",
            "The booking was not made, so nothing has been scheduled.",
            "I haven't made that booking for you.",
            "Okay, I've cancelled it. Nothing is scheduled.",
            "It was booked, but I cancelled it.",
            "The booking was cancelled before it was made, so nothing is scheduled.",
        ]
        neither = ["I am booking the 3 p.m. slot for you now.", "Okay.",
                   "Would you like me to book it?"]
        for s in booked:
            self.assertEqual(tc.classify(s), "claims_booked", s)
        for s in not_booked:
            self.assertEqual(tc.classify(s), "claims_not_booked", s)
        for s in neither:
            self.assertEqual(tc.classify(s), "neither", s)
        r = tc.check(["I have not ", "booked that slot for you."], committed=False)
        self.assertEqual((r.claim, r.consistent), ("claims_not_booked", True))
        r = tc.check(["you should bring?", "The booking has already been made."], committed=False)
        self.assertEqual((r.claim, r.consistent), ("claims_booked", False))


if __name__ == "__main__":
    unittest.main()
