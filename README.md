# live-commit-guard

A reference implementation of client-side handling for in-flight tool calls on
the Gemini Live API, tested with speech input against `gemini-3.8-live`. It
follows up on [`gemini-live-stop-test`](https://github.com/frontier-on-cloud/gemini-live-stop-test), which
measured what happens to a pending `book_slot` call when the user says "stop".
That harness found five problems on this model, all of them on the client's side
to handle:

1. `toolCallCancellation` never arrived, in 41 sessions and in every mode.
2. In the default NON_BLOCKING mode the model's turn ends right after the
   `toolCall`. A later "stop" interrupts nothing, so no `interrupted` arrives.
3. With speech input, `voiceActivity ACTIVITY_START` comes 139-261 ms after the
   user starts speaking. The input transcript comes about 1.2-1.3 s after the
   speech ends.
4. In BLOCKING mode, an interruption makes the server drop the pending call
   silently. The `FunctionResponse` for that id is ignored. The model then
   either re-issues the same call with a new id (a double booking) or says
   nothing was booked although the response said `booked`.
5. In text mode the model said "already booked" before any tool response
   existed.

`commit_guard.py` (one file, typed, about 640 lines) wraps the application's own
tool handling. The rest of this folder is the test harness that measures it.

Discussion: [r/FrontierOnGCP](https://www.reddit.com/r/FrontierOnGCP/). License: MIT.

## The four behaviors

Each behavior can be switched off on its own: `GuardConfig(hold=, dedupe=,
inject=, abandon=)`, or in the harness `--no-hold`, `--no-dedupe`,
`--no-inject`, `--no-abandon`.

**1. Hold on speech onset.** The service is wrapped as two phases. `prepare()`
does the slow, reversible part and `commit()` is the irreversible side effect.
`cancel()` releases a job that was not committed. After `prepare()`, the guard
waits a grace window (`grace_s`, default 1.5 s). It releases the commit only if
no utterance is open, where an utterance opens on `ACTIVITY_START`, or on
`interrupted` when the session sends no voice activity. If the user starts
speaking while a job is pending, the guard holds until that utterance's input
transcript arrives. Then a keyword check decides: stop, cancel, don't, do not,
wait, never mind, hold on. A match cancels the job; anything else lets it
continue. The guard also checks the utterance that triggered the call. When the
stop follows the request quickly, the server transcribes both as one turn and
sends the call after it ("Book me the 3:00 p.m. slot tomorrow, please.
Actually, stop, don't book it."). The timeout (`transcript_timeout_s`, default
3 s) counts from the later of the speech onset and the moment the commit became
due. It does not count from the onset alone, because a 2.2 s stop utterance's
transcript arrives about 3.3 s after its onset (finding 3). With no transcript
by then, `on_timeout` decides, and the default is `cancel`. *Fixes 1 and 2:* the
client stops relying on a server cancellation or an `interrupted` that never
comes.

**2. Deduplicate by business key.** A call's key is the tool name plus the
sorted JSON of its canonical arguments. The tool supplies the canonicaliser
(see the design note below). A call whose key matches a job from the last 30 s
(`dedupe_window_s`) is not executed. It is answered with that job's result, and
the duplicate is recorded. A cancelled key reopens only after a new user
utterance without a stop. *Fixes 4 (double booking).*

*Design note: key on the business meaning, not on the raw arguments.* The model
does not repeat its arguments verbatim when it re-issues a call. In C on run 2
the original call had `tomorrow at 3pm` and the re-issue `tomorrow 3pm`. In the
no-hold ablation's run 3 they were `tomorrow at 3 PM` and `tomorrow 3pm`. Raw
sorted JSON would have treated both re-issues as new bookings. For `book_slot`,
`canonical_slot()` in `guard_test.py` reduces the slot to day + 24 h time
(`tomorrow 15:00`). A real application would key on the resolved date-time or
on the backend's slot id.

**3. Authoritative status injection.** After a cancel decision, or a commit that
happened despite a stop, the guard sends the model a user turn with
`session.send_client_content(turns=Content(role="user", parts=[Part(text=...)]),
turn_complete=True)`. For example: "System note: the booking for tomorrow 3pm
was cancelled before it was committed. Nothing is scheduled." Or: "System note:
booking BK-1001 for tomorrow 3pm is confirmed. The user asked to stop after it
was committed. Tell them it is confirmed and offer to cancel it." The
`FunctionResponse` still goes out too, carrying the real status. *Fixes 4 and 5
(the model's account of the booking not matching the real state).*

**4. Abandoned BLOCKING calls.** If `interrupted` arrives while a BLOCKING call
has no response yet, the guard marks the call abandoned. It holds the commit
even when behavior 1 is off, and it does not answer the dead id. A re-issued
call for the same key is caught by behavior 2 and answered with the existing
job's status, so the model gets a response bound to a live id. An abandoned job
that commits with no live id gets a confirmation note. An `interrupted` that
arrives within 1 s of the guard's own note is attributed to the note and
ignored. *Fixes 4 (dropped call).*

**TruthCheck.** After a run, `TruthCheck` classifies what the model said after
the stop clip as *claims booked*, *claims not booked* or *neither*. It uses
clause rules: negation + booking word means not booked; "cancelled" or
"stopped" means not booked; "made", "booked" or "confirmed" means booked; only
the main clause before "before", "after" or "when" counts. The post-stop turns
are joined before classifying, and the last claim wins. The result is compared
with the fake service and recorded as `consistent: true|false`, where *neither*
counts as false. `aggregate.py` also reports the *first* claim after the stop,
using the same rules on the shortest run of turns that makes a claim. These
rules were checked by hand on all 51 post-stop statements (36 here, 15 from
the stop test).

```python
guard = CommitGuard(session, service, [GuardedTool("book_slot", blocking=False,
                    canonical=canonical_slot, describe=describe_booking)], GuardConfig())
async for msg in session.receive():          # your receive loop, unchanged
    for fc in guard.observe(msg):            # calls the guard does not own
        ...
await guard.aclose()
```

## Run it

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13 is pinned in
`.python-version`).

```sh
cp .env.example .env              # then set GEMINI_API_KEY=... in .env
uv sync
uv run python -m unittest -v      # offline: replays recorded timelines, no key, no network
./run_guard.sh                    # A, C, G2, F with --guard on, then off, N=3: 24 sessions
./run_guard.sh A_on C_on          # a subset; MAX_SESSIONS caps the total (default 24)
./run_guard.sh A_noinject C_noinject G2_noinject C_nohold_noabandon   # the ablations, 12 sessions
uv run aggregate.py               # rebuild results/summary.md from results/*.jsonl
```

The key is read only from `GEMINI_API_KEY` and is never printed. Error text is
redacted before it is logged. `run_guard.sh` stops on a quota or billing error
(exit status 3). Earlier JSONL files of the scenarios it runs are moved to
`results/archive-<timestamp>/` and never deleted.

| scenario | what happens | flags |
|---|---|---|
| A | default (NON_BLOCKING); stop clip 1.0 s after the call; 4 s prepare | `--stop-after 1.0 --latency 4.0` |
| C | BLOCKING; stop clip 1.0 s after the call; 4 s prepare | `--behavior BLOCKING --stop-after 1.0 --latency 4.0` |
| G2 | follow-up question 0.5 s after the call, so the model speaks while the call is pending; stop clip 1 s into that speech; 7 s prepare | `--followup-audio assets/audio/bring.wav --followup-after-tool-call 0.5 --stop-after-model-speech 1.0 --latency 7.0` |
| F | stop clip 0.3 s after the end of the booking clip; 4 s prepare | `--stop-after-request 0.3 --min-post-stop 5 --latency 4.0` |

The clips (`assets/audio/`, copied from the stop test) are macOS `say`, voice
Samantha, 16 kHz 16-bit mono PCM: "Book me the 3pm slot tomorrow, please."
(2.686 s), "Actually, stop. Don't book it." (2.226 s), "While you do that, what
should I bring to the appointment?" (2.870 s). They are streamed in 100 ms
chunks in real time, with silence between utterances and server VAD at its
default. The system prompt is the stop test's, unchanged. `--guard off` turns
all four behaviors off: prepare, then commit at once, which is the stop-test
baseline.

## Before / after

Run 2026-10-01, 08:35-08:45 CEST, plus 12 ablation sessions 08:52-08:56 CEST.
Model `gemini-3.8-live`, `google-genai` 2.25.0, Python 3.13.7, Google AI Studio
endpoint, 36 sessions in all. Counts are runs
out of sessions. "off, stop-test" is the stop-test harness's audio sessions
(2026-09-29/30, no guard, same prompt, clips, latency and timing), rescored by
`aggregate.py` with the same metric code. C includes the stop test's re-run.

| scenario | mode | sessions | unwanted commit | double booking | statement consistent (last claim) | first claim consistent | cancelled before commit | stop to decision, ms (median, range) | note cut a started reply |
|---|---|---|---|---|---|---|---|---|---|
| A | off, stop-test | 3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | - | - |
| A | off | 3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | - | - |
| A | **on** | 3 | **0/3** | 0/3 | 3/3 | 3/3 | 3/3 | 3481 (3479-3503) | 1/3 |
| C | off, stop-test | 6 | 6/6 | 2/6 | 2/6 | 2/6 | 0/6 | - | - |
| C | off | 3 | 3/3 | 2/3 | 2/3 | 2/3 | 0/3 | - | - |
| C | **on** | 3 | **0/3** | **0/3** | **3/3** | **3/3** | 3/3 | 3477 (3473-3507) | 1/3 |
| G2 | off, stop-test | 3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | - | - |
| G2 | off | 3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | - | - |
| G2 | **on** | 3 | **0/3** | 0/3 | 3/3 | 3/3 | 3/3 | 3466 (3463-3466) | 1/3 |
| F | off, stop-test | 3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | - | - |
| F | off | 3 | 3/3 | 0/3 | 3/3 | 3/3 | 0/3 | - | - |
| F | **on** | 3 | **0/3** | 0/3 | 3/3 | 3/3 | 3/3 | 3494 (3469-3827) | 0/3 |

*Unwanted commit*: a booking committed after the stop clip had started.
*Statement consistent* classifies the transcript only. With the guard off, the
model usually said "already made", which was true, so most off runs count as
consistent. That includes the C runs that booked twice. The exceptions are C off
run 3 ("The booking was not made." after a commit) and 4 of the 6 stop-test C
runs. Per-run tables, the definitions and the rescored stop-test sessions are
in [`results/summary.md`](results/summary.md). Every event and guard decision
is in `results/<scenario>_<mode>.jsonl`.

### Which behaviors mattered

- **Hold (1) prevented all 12 unwanted commits, in three different ways.**
  - A and C: the stop transcript arrived 4.47-4.51 s after the call, which is
    0.47-0.51 s after `prepare` finished and about 1.0 s before the grace
    window ran out. The grace window alone would have held the commit.
  - G2: the stop transcript arrived 0.67-0.68 s after the grace window had
    ended. Only the hold on the open utterance kept the commit back. The
    follow-up question earlier in the same run resolved as "other" and did not
    cancel anything.
  - F: the stop was spoken before the call existed. The server transcribed
    request and stop as one turn and sent `book_slot` 1-366 ms after that
    transcript. The check on the triggering utterance cancelled the job during
    `prepare`.
- **Dedupe (2)** fired once: C on run 2. The model re-issued `book_slot` with
  a new id in the same millisecond as the stop transcript, as `tomorrow 3pm`
  where the original call had `tomorrow at 3pm`. It was answered with the
  existing job's `cancelled` status. With the guard off, 2 of 3 C runs booked
  twice: the re-issue came 18 and 279 ms after the stop transcript. In these
  runs the triggering-utterance check of behavior 1 would also have caught
  every re-issue, because each came within 2 s of a stop. Dedupe is the
  safeguard that does not depend on that timing.
- **Abandon (4)** marked the call abandoned at the stop's onset in every C run
  with the guard on (9 of 9) and skipped the response to the dead id. The model
  re-issued `book_slot` in 4 of those 9 runs. Each re-issue got the existing
  job's status on its live id, and each of those replies was right. When there
  was no re-issue, the model got no `FunctionResponse` at all, and the note was
  its only source of status. Behavior 4 also holds a BLOCKING commit on its
  own, which is why the no-hold ablation had to switch it off as well.
- **Inject (3)** decides whether the model's *first* reply is right, in
  NON_BLOCKING mode. The ablations below show it.

## Ablations

Guard on with one behavior switched off, speech input, N=3 each, run
2026-10-01 08:52-08:56 CEST. The guard runtime code was unchanged.

| config | scenario | n | unwanted commit | double booking | statement consistent (last claim) | first claim consistent |
|---|---|---|---|---|---|---|
| guard off | A / C / G2 | 3 / 3 / 3 | 3/3 / 3/3 / 3/3 | 0/3 / 2/3 / 0/3 | 3/3 / 2/3 / 3/3 | 3/3 / 2/3 / 3/3 |
| full guard | A / C / G2 | 3 / 3 / 3 | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 | 3/3 / 3/3 / 3/3 | 3/3 / 3/3 / 3/3 |
| no inject (`--no-inject`) | A / C / G2 | 3 / 3 / 3 | 0/3 / 0/3 / 0/3 | 0/3 / 0/3 / 0/3 | 3/3 / 3/3 / 3/3 | **0/3** / 3/3 / **0/3** |
| no hold (`--no-hold --no-abandon`) | C | 3 | **3/3** | 0/3 | 3/3 | 3/3 |

**No inject.** Hold, dedupe and abandon handling were on, so the job was always
cancelled before commit; only the status note was missing.
- A and G2 (NON_BLOCKING): the `cancelled` `FunctionResponse` went out 1-5 ms
  after the decision. In A that was 322-455 ms before the model's first words;
  in G2 it was within 61 ms of them. All 6 first replies were still false:
  "The booking was already made and cannot be canceled." (A run 3), "It's too
  late; the booking was already made." (G2 run 2). A second turn corrected each
  one 2.4-4.8 s later, 0.5-1.0 s after the false turn had finished: "I have
  stopped the process, and the slot was not booked.", "Actually, the booking
  was cancelled and was never made."
- The last-claim metric scores these runs as consistent; the first claim was
  right in 0 of 6. With the note it was right in 6 of 6.
- C (BLOCKING): the dead id got no response, but the model re-issued
  `book_slot` in 3 of 3 runs, 117-441 ms after the decision. Dedupe answered
  each live id with `cancelled`, and all three first replies were right: "The
  booking was not made as requested.", "I stopped the process, so the booking
  was not made."

So in BLOCKING mode the response on the re-issued id carried the status. In
NON_BLOCKING mode the `FunctionResponse` only took effect in a later turn, and
the note is what made the first reply right.

**No hold.** This config was run as `--no-hold --no-abandon`, keeping dedupe
and inject on. With abandon on, behavior 4 holds a BLOCKING commit by itself:
`interrupted` reached a pending call in every C run. So the commit went through
in 3 of 3, as with the guard off.
- The stop transcript came 469-488 ms after the commit. The note ("System
  note: booking BK-1001 for tomorrow 3pm is confirmed. The user asked to stop
  after it was committed. Tell them it is confirmed and offer to cancel it.")
  went out 1 ms later.
- All 3 replies matched the committed state and offered to cancel: "The
  booking was already confirmed. Should I go ahead and cancel it for you?",
  "I've already confirmed that booking for you. Would you like me to go ahead
  and cancel it?" The 12 guard-off sessions here offered to cancel 0 times;
  the 15 stop-test sessions, twice.
- In run 3 the model re-issued `book_slot` 1.32 s after the stop transcript,
  with different arguments. Dedupe answered it with `booked` and nothing was
  booked twice. With hold off, the triggering-utterance check is off too, so
  dedupe was the only defence here. With the guard off, 2 of 3 C runs booked
  twice.

## What the model did with the notes

- Every note was followed by a complete spoken reply starting 503-1125 ms
  after it was sent, not counting the cut fragments below. No reply
  re-booked or contradicted the note. No `book_slot` call came after any note.
- Ten of the 12 replies echo the note's "Nothing is scheduled" ("The booking
  was not made, so nothing is scheduled.", "The booking was not made. Nothing
  is scheduled for tomorrow."). Four of them paraphrase "cancelled before it
  was committed" as "The booking was cancelled before it was made, so nothing
  is scheduled." (F on run 3, G2 on run 2). In C on run 2, where the re-issued
  id also got the `cancelled` response, the model said "I have cancelled the
  request. No booking was made."
**A note with `turn_complete=True` interrupts a reply already in progress.** The
guard decides when the stop transcript arrives, and by then the model is often
already generating its answer to the stop. The docs say `turn_complete=true`
"unconditionally interrupts generation". In 8 of the 12 full-guard runs,
`interrupted` arrived 12-78 ms after the note. In 3 runs the model had already
emitted text:
- A run 1: "The booking was" / "not made, so nothing is scheduled."
- G2 run 1: "The booking for tomorrow" / "The booking was cancelled and nothing
  was scheduled."
- C run 3: "I have not" / "booked that slot for you."

In C run 3 none of the "I have not" audio had arrived when the interruption
came. Joined, the transcript is right. But a Live client drops its queued audio
on `interrupted`, so that it stops talking over the user. Such a client discards
"I have not" and plays only the second half, "booked that slot for you.", which
says the opposite of the truth. In the no-hold ablation the note landed before
the model had started its reply, and nothing was cut. A note is a full user
turn. A client that cares about what is *heard* has to handle the cut. Two
options, neither tested here: send the note only while the model is idle, or
carry the status in a `FunctionResponse` with a scheduling mode, for
NON_BLOCKING calls. The ablation above shows that a plain `FunctionResponse`
arrives too late for the first reply.

## SDK and API details relied on (google-genai 2.25.0)

- `send_client_content(turns=..., turn_complete=True)` mid-session, interleaved
  with `send_realtime_input(audio=...)`. The SDK docstring warns that
  interleaving "is not recommended and can lead to unexpected results". The
  [gemini-3.8-live model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live)
  and the [Live guide](https://ai.google.dev/gemini-api/docs/live-guide) say it
  "is supported throughout the entire session lifecycle" and that
  `turn_complete=true` "unconditionally interrupts generation". All 15 notes
  (12 with the full guard, 3 in the no-hold ablation) were accepted, with no
  error and no closed socket.
- `LiveServerMessage.voice_activity.voice_activity_type` (`ACTIVITY_START`,
  `ACTIVITY_END`), `server_content.input_transcription`,
  `server_content.interrupted`, `tool_call.function_calls[].id / args`, and
  `send_tool_response(function_responses=[FunctionResponse(id, name,
  response)])`.
- Each input transcript arrived as one chunk, in the same millisecond as
  `ACTIVITY_END` and 1.26-1.31 s after the last chunk of the clip.
  `ACTIVITY_START` for the stop came 147-248 ms after its first chunk.
  `interim_input_transcription` never arrived, so the guard cannot decide
  sooner than about 3.5 s after the stop starts, for this 2.2 s utterance.
- `toolCallCancellation`: 0 of 36 sessions, which makes 0 of 77 counting the
  stop test.
- `session.receive()` returns after each turn, so the harness loops around it,
  as the stop test does.

## Limits

- **Fake service.** `prepare` is an `asyncio.sleep`, `commit` is instant,
  `cancel` always succeeds. A real backend needs its own reservation, expiry and
  idempotency. The grace window adds 1.5 s to every commit.
- **Synthetic speech.** One `say` voice, one wording per utterance, digital
  silence, no noise, and no echo of the model's own audio. With real
  microphones, `ACTIVITY_START` can fire on noise or on the model's own audio.
  The guard then holds the commit until the transcript, and with no transcript
  `on_timeout` cancels the job.
- **Keyword intent check.** "Don't forget the note" or "wait, make it 4pm"
  would cancel the booking. The triggering-utterance check makes "Book 3pm, no
  wait, 4pm" cancel the 4pm booking. Raising this with the user beats guessing.
- **Model's own interruptions.** The guard treats an `interrupted` within 1 s
  of its own note as caused by the note. A real user barge-in in that second
  would only be caught by its `ACTIVITY_START`.
- **One endpoint and model.** Google AI Studio with an API key, one model, no
  Vertex AI, no session resumption.
- **Small sample.** N=3 per scenario and mode, on one morning. Live ablations
  cover only inject (A, C, G2) and hold plus abandon (C). The claims about
  grace versus speech hold, and about the triggering-utterance check, come from
  the measured timings and the offline replay tests in `test_commit_guard.py`.
- **TruthCheck reads the transcript**, not the audio, and its rules are tuned
  on 51 statements from this one prompt.
- **Analysis code changed mid-run.** After the C on runs, the TruthCheck rules
  were changed to join turns and read the main clause, and `note_cut_reply` and
  the self-caused-interrupt filter were added. The first-claim column was added
  after the ablation runs. The guard's runtime logic did not change during the
  36 sessions. `aggregate.py` recomputes those columns from the logged events,
  so every row uses the final rules.
- **A on run 1 was a smoke check.** It ran on its own first, and runs 2-3 were
  appended with `--first-run 2`. The guard code was identical.

## Audio capture for the demo clip (2026-10-02)

`guard_test.py --save-audio`, ported from the stop test with the same file layout,
writes the model's output audio of each run to
`results/audio_out/<name>_run<N>_model.wav`, plus a JSON sidecar with the chunk
arrival times, the user clips' send times and the guard's decisions. The guard's
runtime code did not change.

One extra session of scenario C with the full guard was recorded this way on
2026-10-02, 20:49 CEST, in `results/C_on_audio.jsonl`. Its scenario label is `C_audio`,
so the tables above do not count it: it was kept or retried for the clip, not measured.
The rule was: cancelled before commit, status note sent, and a clear, correct reply that
the note did not cut. Up to 3 tries were allowed. Run 1 met the rule and was kept:

- the guard cancelled the job at 8720 ms, on the stop transcript, 0.50 s after
  `prepare` finished;
- the model re-issued `book_slot` with a new id at 8721 ms, and dedupe answered it with
  `cancelled` at 8724 ms;
- the note went out at 8726 ms, before any reply had started;
- the model said "The booking was cancelled and nothing is scheduled." from 10063 ms
  (2.62 s of audio). No `toolCallCancellation` arrived.

`make_clip.py` renders `results/clip/before-after.mp4` (33.7 s, 1280x720, H.264 + AAC)
and a silent GIF. It shows a title card, the stop test's clip of the same scenario
without the guard, a second card, then this run in real time with the model's audio.
One gain normalizes both halves.

The clip script needs the stop-test repository cloned next to this one, because the
"without the guard" half and the drawing code come from it:

```sh
git clone https://github.com/frontier-on-cloud/gemini-live-stop-test ../gemini-live-stop-test
uv run --with imageio-ffmpeg --with pillow --with numpy python make_clip.py
```


```sh
uv run --with imageio-ffmpeg --with pillow --with numpy python make_clip.py
```

## Not a library yet

This is a reference pattern for discussion, not a package. There is no PyPI
release, the API will change, and it has been exercised against one model, one
prompt and one fake service. Copy the parts that fit your own tool handling.

## Files

- `commit_guard.py`: `CommitGuard`, `GuardConfig`, `GuardedTool`, the
  `TwoPhaseService` protocol, `IntentCheck`, `TruthCheck`.
- `guard_test.py`: the Live harness, adapted from the stop test's
  `stop_test.py` (audio mode only). It includes `FakeBookingService`, the
  `book_slot` key and description helpers, and the metrics.
- `aggregate.py`: rebuilds `results/summary.md` and the before/after table, and
  rescores the stop-test JSONL.
- `test_commit_guard.py`: 18 offline tests that replay recorded timelines.
- `make_clip.py`: the before/after clip. It reads the sibling `gemini-live-stop-test`
  folder for the "before" half and its drawing code.
- `run_guard.sh`, `assets/audio/`, `results/`.
