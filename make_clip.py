#!/usr/bin/env python3
"""Render the before/after demo clip: the same BLOCKING call and the same spoken "stop",
first without the guard (the stop test's clip), then with it (a measured run here).

  2 s title card  "Without the guard"
  15 s            ../gemini-live-stop-test/results/clip/stop-test-C1.mp4 (frames unchanged)
  2 s title card  "With the guard"
  after half      results/C_on_audio.jsonl run 1, rendered here in real time

The after half uses the stop-test clip's own drawing code, fonts, colors and layout
(imported from ../gemini-live-stop-test/make_clip.py), so both halves read as one
piece. Its data are results/C_on_audio.jsonl (run 1) and the files guard_test.py
--save-audio wrote for that run in results/audio_out/ (model WAV + JSON sidecar). Every
time shown is read from those files (ms since session start); nothing is sped up. The
model's audio is placed as a Live client would play it: each chunk at its arrival time
or right after the previous one, and anything still queued when `interrupted` arrives
is dropped.

Audio: the before half's track is rebuilt from the same sources and offsets as the
stop-test clip (checked against that clip's AAC track), the after half's from the
clips and the saved model WAV, and one gain (-1 dBFS peak over the whole track)
normalizes both. Title cards are silent.

Output:
  results/clip/before-after.mp4  1280x720, 30 fps, H.264 + AAC (mono, 24 kHz), faststart
  results/clip/before-after.gif  800 px wide, 10 fps, no audio

Run (Homebrew ffmpeg is not needed; imageio-ffmpeg ships a static ffmpeg):
  uv run --with imageio-ffmpeg --with pillow --with numpy python make_clip.py
  ... make_clip.py --preview DIR     # PNG stills of the after half and the cards only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
STOP_TEST = ROOT.parent / "gemini-live-stop-test"


def _load_before():
    spec = importlib.util.spec_from_file_location("stop_test_make_clip", STOP_TEST / "make_clip.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {STOP_TEST / 'make_clip.py'}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load_before()  # the stop-test clip: drawing code, layout, colors, data loader

SCENARIO = "C_on_audio"
RUN = 1
JSONL = ROOT / "results" / f"{SCENARIO}.jsonl"
AUDIO_DIR = ROOT / "results" / "audio_out"
SIDECAR = AUDIO_DIR / f"{SCENARIO}_run{RUN}_audio.json"
MODEL_WAV = AUDIO_DIR / f"{SCENARIO}_run{RUN}_model.wav"
USER_CLIPS = {"book_request": ROOT / "assets" / "audio" / "book.wav",
              "stop": ROOT / "assets" / "audio" / "stop.wav"}
BEFORE_MP4 = B.OUT_MP4
OUT_DIR = ROOT / "results" / "clip"
OUT_MP4 = OUT_DIR / "before-after.mp4"
OUT_GIF = OUT_DIR / "before-after.gif"

W, H, FPS, SR, SS = B.W, B.H, B.FPS, B.SR, B.SS
SPF = SR // FPS  # audio samples per video frame (800)
CARD_S = 2.0
CARDS = ("Without the guard", "With the guard")
TAIL_MS = 1000  # after-half timeline runs this long past the end of the model's audio
HOLD_S = 1.0    # then the last frame holds (clock stopped)
GIF_FPS = 10
GIF_W = 800
PEAK = 0.89     # -1 dBFS, as in the stop-test clip

# colors: the stop-test clip's, plus the reference palette's status "critical" for the
# cancel (white bold text on it: 4.7:1)
BG, PANEL, BORDER = B.BG, B.PANEL, B.BORDER
INK, INK_2, INK_3, TRACK = B.INK, B.INK_2, B.INK_3, B.TRACK
BLUE, WHITE = B.BLUE, B.WHITE
RED = (208, 59, 59)
HATCH_BG, HATCH = (226, 224, 218), (150, 148, 142)

font, tw, wrap, secs = B.font, B.tw, B.wrap, B.secs


# ---------------------------------------------------------------- data
def load_run() -> dict:
    events = [json.loads(line) for line in JSONL.read_text().splitlines() if line.strip()]
    events = [e for e in events if e.get("run") == RUN]
    if not events:
        raise SystemExit(f"run {RUN} not found in {JSONL}")
    timed = [e for e in events if e["t_ms"] >= 0]

    def all_of(event: str, **match) -> list[dict]:
        return [e for e in timed if e["event"] == event and all(e.get(k) == v for k, v in match.items())]

    def one(event: str, **match) -> dict:
        hits = all_of(event, **match)
        if not hits:
            raise SystemExit(f"no {event} {match} in run {RUN}")
        return hits[0]

    side = json.loads(SIDECAR.read_text())
    start = one("run_start")
    summary = next(e["summary"] for e in events if e["event"] == "run_end")
    stop_ms = one("user_audio_start", label="stop")["t_ms"]
    calls = all_of("tool_call_received", name="book_slot")
    decision = one("guard_decision")
    note = one("guard_status_note_sent")
    responses = all_of("guard_tool_response_sent")
    transcripts = [e for e in all_of("model_transcript") if e["t_ms"] >= note["t_ms"]]
    stop_transcript = next(e for e in all_of("input_transcript") if e["t_ms"] >= stop_ms)
    d = {
        "book_ms": one("user_audio_start", label="book_request")["t_ms"],
        "stop_ms": stop_ms,
        "tool_call_ms": calls[0]["t_ms"],
        "call_id": calls[0]["call_id"],
        "reissue": ({"ms": calls[1]["t_ms"], "call_id": calls[1]["call_id"]}
                    if len(calls) > 1 else None),
        "latency_ms": int(round(start["latency_s"] * 1000)),
        "grace_ms": int(round(start["grace_s"] * 1000)),
        "job_start_ms": one("guard_job_started")["t_ms"],
        "onset_ms": next(e["t_ms"] for e in all_of("voice_activity", voice_activity_type="ACTIVITY_START")
                         if e["t_ms"] >= stop_ms),
        "prepared_ms": one("guard_prepared")["t_ms"],
        "transcript_ms": stop_transcript["t_ms"],
        "transcript": stop_transcript["text"],
        "matched": one("guard_utterance_resolved", intent="stop")["matched"],
        "decision_ms": decision["t_ms"],
        "decision": decision["decision"],
        "cancelled_ms": one("guard_cancelled")["t_ms"],
        "response": ({"ms": responses[0]["t_ms"], "status": responses[0]["response"].get("status"),
                      "call_id": responses[0]["call_id"]} if responses else None),
        "response_skipped_ms": (all_of("guard_response_skipped") or [{"t_ms": None}])[0]["t_ms"],
        "note_ms": note["t_ms"],
        "note_text": note["text"],
        "model_text_ms": transcripts[0]["t_ms"],
        "model_text": "".join(e["text"] for e in transcripts).strip(),
        "session_closed_ms": one("session_closed")["t_ms"],
        "n_cancellations": len(all_of("tool_call_cancellation")) + len(side["events"].get("cancellations") or []),
        "commits": summary.get("commits") or [],
        "note_cuts": summary.get("note_cuts") or [],
        "side": side,
    }
    # The clip claims these; refuse to render a run where they do not hold.
    assert d["decision"] == "cancel", "the guard did not cancel"
    assert not d["commits"], "something was committed"
    assert not d["note_cuts"], "the note cut a reply that had started"
    clips = {c["label"]: c for c in side["user_clips"]}
    assert clips["book_request"]["sent_start_ms"] == d["book_ms"], "book clip offset mismatch"
    assert clips["stop"]["sent_start_ms"] == d["stop_ms"], "stop clip offset mismatch"
    return d


# ---------------------------------------------------------------- audio
def model_playback(side: dict, wav: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(start_ms, samples) per model chunk as a Live client plays them: a chunk starts
    at its arrival or when the previous one ends; `interrupted` drops what is queued."""
    rate = side["model_audio"]["sample_rate"]
    assert rate == SR, f"model audio at {rate} Hz, expected {SR}"
    interrupts = sorted(side["events"]["interrupted_ms"])
    out, cursor = [], 0.0
    for c in side["chunks"]:
        i0 = int(round(c["wav_offset_ms"] * SR / 1000))
        seg = wav[i0: i0 + int(round(c["duration_ms"] * SR / 1000))]
        start = max(float(c["t_ms"]), cursor)
        cut = next((t for t in interrupts if c["t_ms"] <= t < start + len(seg) * 1000 / SR), None)
        if cut is not None:
            seg = seg[: max(0, int((cut - start) * SR / 1000))]
        if len(seg):
            out.append((start, seg))
        cursor = start + len(seg) * 1000 / SR
    return out


def place(mix: np.ndarray, x: np.ndarray, at_ms: float) -> None:
    i = int(round(at_ms * SR / 1000))
    seg = x[: max(0, len(mix) - i)]
    mix[i: i + len(seg)] += seg


def before_audio(n: int) -> np.ndarray:
    """The stop-test clip's mix before its normalization: same sources, same offsets
    (stop-test make_clip.build_mix)."""
    d = B.load_run()
    mix = np.zeros(n, dtype=np.float32)
    place(mix, B.decode(B.USER_CLIPS["book_request"]), d["book_ms"])
    place(mix, B.decode(B.USER_CLIPS["stop"]), d["stop_ms"])
    place(mix, B.decode(B.MODEL_WAV), d["model_first_chunk_ms"])
    return mix


def after_audio(d: dict, segments: list, from_ms: float, n: int) -> np.ndarray:
    full = np.zeros(int(round(from_ms * SR / 1000)) + n, dtype=np.float32)
    place(full, B.decode(USER_CLIPS["book_request"]), d["book_ms"])
    place(full, B.decode(USER_CLIPS["stop"]), d["stop_ms"])
    for start, seg in segments:
        place(full, seg, start)
    return full[len(full) - n:]


def check_before_audio(raw: np.ndarray) -> str:
    """Compare the rebuilt before-half mix with the AAC track of the existing clip."""
    ref = B.decode(BEFORE_MP4)
    x = raw * (PEAK / float(np.abs(raw).max()))
    n = min(len(x), len(ref))
    best = max(range(-240, 241, 8), key=lambda k: float(np.dot(x[max(0, k):n + min(0, k)],
                                                               ref[max(0, -k):n - max(0, k)])))
    a, b = x[:n], ref[:n]
    corr = float(np.dot(a, b) / math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b))))
    if corr < 0.9 or best != 0:
        raise SystemExit(f"rebuilt before-half audio does not match {BEFORE_MP4.name}: "
                         f"corr {corr:.3f}, best lag {best} samples")
    return f"corr {corr:.3f} at lag 0 with the clip's AAC track"


def voiced_rms_db(x: np.ndarray) -> float:
    fr = x[: len(x) // 480 * 480].reshape(-1, 480)
    r = np.sqrt((fr ** 2).mean(axis=1))
    v = r[r > 10 ** (-50 / 20)]
    return 20 * math.log10(float(np.sqrt((v ** 2).mean())))


# ---------------------------------------------------------------- drawing
M, LEFT, RIGHT, STRIP = B.M, B.LEFT, B.RIGHT, B.STRIP
AX_X0, AX_X1, AX_Y = B.AX_X0, B.AX_X1, B.AX_Y
# label rows 2-6 px higher than the stop-test clip's: here a label sits right of its
# tick on the lowest row, and the playhead passes under it afterwards
ROW_Y = {1: 550, 2: 578, 3: 606}
TITLE = B.TITLE


def draw_header(c, t_ms: float) -> None:
    """Same geometry as the stop-test clip's header; the subtitle names this run."""
    c.text(M + 4, 16, TITLE, font("bold", 30), INK)
    c.text(M + 4, 52, f"Measured run: scenario C with the commit guard on ({SCENARIO}), run {RUN}. "
           "Real time, session clock.", font("regular", 18), INK_2)
    clock = f"{t_ms / 1000:.2f} s"
    fc = font("mono_bold", 34)
    c.text(W - M - 4, 14, clock, fc, INK, anchor="ra")
    c.text(W - M - 4 - tw(clock, fc) - 10, 26, "t =", font("regular", 22), INK_2, anchor="ra")


def fit(s: str, kind: str, size: int, max_w: float, min_size: int = 15):
    while size > min_size and tw(s, font(kind, size)) > max_w:
        size -= 1
    return font(kind, size)


def hatched(c, box, r: int, phase: float) -> None:
    """Rounded bar with 45 degree stripes; `phase` (px) moves them while the hold lasts."""
    x0, y0, x1, y1 = (round(v * SS) for v in box)
    w, h = x1 - x0, y1 - y0
    tile = Image.new("RGB", (w, h), HATCH_BG)
    td = ImageDraw.Draw(tile)
    step = 14 * SS
    off = int(phase * SS) % step
    for k in range(-h - step + off, w + step, step):
        td.line([(k, h), (k + h, 0)], fill=HATCH, width=5 * SS)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=r * SS, fill=255)
    c.im.paste(tile, (x0, y0), mask)


def draw_service(c, d: dict, t_ms: float) -> None:
    x0, y0, x1, _ = RIGHT
    ix0, ix1 = x0 + 20, x1 - 20
    title = "Booking service (two-phase)"
    note = f"fake service, {d['latency_ms'] / 1000:.1f} s prepare"
    room = ix1 - ix0 - tw(title, font("bold", 24)) - 16
    c.rrect(RIGHT, 12, fill=PANEL, outline=BORDER, width=1)
    c.text(ix0, y0 + 16, title, font("bold", 24), INK)
    c.text(ix1, y0 + 20, note, fit(note, "regular", 18, room), INK_2, anchor="ra")
    f_time, f_row = font("regular", 19), font("regular", 20)
    if t_ms < d["job_start_ms"]:
        c.text(ix0, y0 + 70, "idle, no tool call yet", font("regular", 24), INK_3)
        return
    # job started
    y = y0 + 62
    c.text(ix0, y, "job started", font("bold", 26), INK)
    c.text(ix1, y + 4, secs(d["job_start_ms"]), f_time, INK_2, anchor="ra")
    cid_x = ix0 + tw("job started", font("bold", 26)) + 14
    cid = f"call {d['call_id']}"
    c.text(cid_x, y + 6, cid, fit(cid, "mono", 18, ix1 - 70 - cid_x), INK_2)
    # prepare bar over the configured latency (reversible part)
    by = y + 40
    frac = min(1.0, max(0.0, (t_ms - d["job_start_ms"]) / d["latency_ms"]))
    c.rrect((ix0, by, ix1, by + 20), 10, fill=TRACK)
    if frac > 0:
        c.rrect((ix0, by, ix0 + max(20, (ix1 - ix0) * frac), by + 20), 10, fill=INK_2)
    if t_ms < d["prepared_ms"]:
        s = f"prepare {d['latency_ms'] / 1000:.1f} s, elapsed {(t_ms - d['job_start_ms']) / 1000:.2f} s"
    else:
        s = f"prepare {d['latency_ms'] / 1000:.1f} s, done at {secs(d['prepared_ms'])}"
    c.text(ix0, by + 28, s, f_row, INK_2)
    # commit: held while the user speaks
    hy = by + 62
    held_until = min(t_ms, d["decision_ms"])
    if t_ms < d["prepared_ms"]:
        c.rrect((ix0, hy, ix1, hy + 20), 10, fill=TRACK)
        if t_ms < d["onset_ms"]:
            s, col = f"commit: after prepare plus {d['grace_ms'] / 1000:.1f} s grace", INK_3
        else:
            s, col = f"user speaking while pending ({secs(d['onset_ms'])}): commit will be held", INK_2
    else:
        hatched(c, (ix0, hy, ix1, hy + 20), 10, phase=(held_until - d["prepared_ms"]) / 1000 * 40)
        held = (held_until - d["prepared_ms"]) / 1000
        if t_ms < d["decision_ms"]:
            s, col = f"prepared, commit held {held:.2f} s: waiting for the transcript", INK
        else:
            words = ", ".join(f"“{m}”" for m in d["matched"])
            s, col = f"commit held {held:.2f} s; transcript matched {words}", INK
    c.text(ix0, hy + 28, s, fit(s, "regular", 20, ix1 - ix0), col)
    # cancelled
    cy = hy + 62
    if t_ms >= d["decision_ms"]:
        c.rrect((ix0, cy, ix1, cy + 54), 10, fill=RED)
        cx, cm = ix0 + 32, cy + 27  # an X, drawn (not a glyph)
        c.line([(cx - 10, cm - 10), (cx + 10, cm + 10)], WHITE, width=4)
        c.line([(cx - 10, cm + 10), (cx + 10, cm - 10)], WHITE, width=4)
        label = f"CANCELLED before commit, {secs(d['cancelled_ms'])}"
        c.text(ix0 + 60, cm, label, fit(label, "bold", 28, ix1 - ix0 - 76, 18), WHITE, anchor="lm")
    # what the model was told
    ry = cy + 68
    if d["response"] and t_ms >= d["response"]["ms"]:
        s = f"re-issued call (new id) answered: {d['response']['status']}"
        c.text(ix0, ry, s, fit(s, "regular", 21, ix1 - ix0 - 80), INK)
        c.text(ix1, ry + 2, secs(d["response"]["ms"]), f_time, INK_2, anchor="ra")
    elif not d["response"] and d["response_skipped_ms"] and t_ms >= d["response_skipped_ms"]:
        s = "tool response: none, the server dropped the call"
        c.text(ix0, ry, s, fit(s, "regular", 21, ix1 - ix0 - 80), INK)
    ny = ry + 29
    if t_ms >= d["note_ms"]:
        c.text(ix0, ny, "status note sent to the model", font("regular", 21), INK)
        c.text(ix1, ny + 2, secs(d["note_ms"]), f_time, INK_2, anchor="ra")
        f_note = font("regular", 18)
        for i, line in enumerate(wrap(f"“{d['note_text']}”", f_note, ix1 - ix0 - 12)[:2]):
            c.text(ix0 + 12, ny + 28 + i * 22, line, f_note, INK_2)


def strip_ticks(d: dict) -> list[tuple]:
    """label, time, color, row, side, marker. Ordered for drawing: the note's ring goes
    under the decision's dot, 6 ms apart."""
    t = [
        (f"toolCall {secs(d['tool_call_ms'])}", d["tool_call_ms"], BLUE, 3, "left", "dot"),
        (f"stop (speech onset) {secs(d['stop_ms'])}", d["stop_ms"], INK, 2, "left", "dot"),
        (f"voiceActivity {secs(d['onset_ms'])} (+{d['onset_ms'] - d['stop_ms']} ms)",
         d["onset_ms"], BLUE, 1, "left", "dot"),
        (f"prepared {secs(d['prepared_ms'])}", d["prepared_ms"], INK_3, 3, "left", "dot"),
        (f"status note {secs(d['note_ms'])}", d["note_ms"], INK, 2, "right", "ring"),
        (f"transcript, decision: {d['decision']} {secs(d['decision_ms'])}",
         d["decision_ms"], RED, 1, "right", "dot"),
        (f"model reply {secs(d['model_text_ms'])}", d["model_text_ms"], BLUE, 3, "right", "dot"),
    ]
    assert abs(d["transcript_ms"] - d["decision_ms"]) <= 5, "transcript and decision not together"
    return t


FINAL_LINE = "toolCallCancellation: never received, the guard decided"


def strip_boxes(d: dict, end_ms: float) -> tuple[list, list]:
    """Label boxes and leader segments of the full strip, for the overlap check."""
    f_lab = font("regular", 20)
    boxes, leaders = [], []
    for label, at, _, row, side, _ in strip_ticks(d):
        x, ly = B.tx(at, end_ms), ROW_Y[row]
        w = tw(label, f_lab)
        boxes.append((label, (x + 8, ly, x + 8 + w, ly + 22) if side == "right" else (x - 8 - w, ly, x - 8, ly + 22)))
        leaders.append((label, (x - 1, ly + 2, x + 1, AX_Y)))
    fb = font("bold", 20)
    boxes.append(("final", (STRIP[2] - 20 - tw(FINAL_LINE, fb) - 26, STRIP[1] + 10, STRIP[2] - 20, STRIP[1] + 34)))
    fe = font("bold", 22)
    head = "Events" + " " * 2 + "seconds since session start"
    boxes.append(("header", (STRIP[0] + 20, STRIP[1] + 12, STRIP[0] + 30 + tw(head, fe), STRIP[1] + 36)))
    return boxes, leaders


def overlaps(d: dict, end_ms: float) -> list[str]:
    def hit(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    boxes, leaders = strip_boxes(d, end_ms)
    bad = [f"{n1!r} x {n2!r}" for i, (n1, b1) in enumerate(boxes) for n2, b2 in boxes[i + 1:] if hit(b1, b2)]
    bad += [f"{n1!r} x leader of {n2!r}" for n1, b1 in boxes for n2, l2 in leaders if n1 != n2 and hit(b1, l2)]
    bad += [f"{n!r} outside the strip" for n, b in boxes if b[0] < STRIP[0] + 8 or b[2] > STRIP[2] - 8]
    playhead = (AX_X0 - 7, AX_Y - 22, AX_X1 + 7, AX_Y - 10)  # the path of its triangle, 2 px margin
    bad += [f"{n!r} x playhead path" for n, b in boxes if hit(b, playhead)]
    return bad


def draw_strip(c, d: dict, t_ms: float, end_ms: float, final_ms: float) -> None:
    c.rrect(STRIP, 12, fill=PANEL, outline=BORDER, width=1)
    c.text(STRIP[0] + 20, STRIP[1] + 12, "Events", font("bold", 22), INK)
    c.text(STRIP[0] + 20 + tw("Events", font("bold", 22)) + 10, STRIP[1] + 16,
           "seconds since session start", font("regular", 18), INK_2)
    f_lab = font("regular", 20)
    now_x = B.tx(min(t_ms, end_ms), end_ms)
    c.line([(AX_X0, AX_Y), (AX_X1, AX_Y)], TRACK, width=3)
    c.line([(AX_X0, AX_Y), (now_x, AX_Y)], INK_3, width=3)
    for s in range(0, int(end_ms // 1000) + 1):
        x = B.tx(s * 1000, end_ms)
        c.line([(x, AX_Y + 4), (x, AX_Y + 9)], INK_3, width=1)
        if s % 2 == 0:
            c.text(x, AX_Y + 12, f"{s} s", font("regular", 17), INK_3, anchor="ma")
    ticks = [tk for tk in strip_ticks(d) if t_ms >= tk[1]]
    for label, at, color, row, side, _ in ticks:
        x, ly = B.tx(at, end_ms), ROW_Y[row]
        c.line([(x, AX_Y), (x, ly + 2)], color, width=2)
        if side == "right":
            c.text(x + 8, ly, label, f_lab, INK)
        else:
            c.text(x - 8, ly, label, f_lab, INK, anchor="ra")
    for label, at, color, row, side, marker in ticks:
        x = B.tx(at, end_ms)
        if marker == "ring":
            c.dot(x, AX_Y, 10, PANEL)
            c.ring(x, AX_Y, 10, color, width=2)
        else:
            c.dot(x, AX_Y, 6, color, ring=PANEL)
    if t_ms >= final_ms and d["n_cancellations"] == 0:
        fb = font("bold", 20)
        c.text(STRIP[2] - 20, STRIP[1] + 10, FINAL_LINE, fb, INK, anchor="ra")
        c.ring(STRIP[2] - 20 - tw(FINAL_LINE, fb) - 16, STRIP[1] + 21, 7, INK, width=2)
    c.poly([(now_x - 7, AX_Y - 20), (now_x + 7, AX_Y - 20), (now_x, AX_Y - 10)], INK)


def render_after(d: dict, t_ms: float, end_ms: float, final_ms: float) -> Image.Image:
    c = B.Canvas()
    draw_header(c, t_ms)
    B.draw_conversation(c, d, t_ms)
    draw_service(c, d, t_ms)
    draw_strip(c, d, t_ms, end_ms, final_ms)
    return c.frame()


def render_card(text: str) -> Image.Image:
    c = B.Canvas()
    c.text(W / 2, H / 2, text, font("bold", 46), INK, anchor="mm")
    return c.frame()


# ---------------------------------------------------------------- main
def mp4_frames(path: Path):
    """Decoded RGB frames of an existing clip, in order."""
    proc = subprocess.Popen([B.FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = W * H * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield buf
    proc.wait()


def count_frames(path: Path) -> int:
    return sum(1 for _ in mp4_frames(path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preview", type=Path, help="write PNG stills to this folder and stop")
    ap.add_argument("--after-from", type=float, default=0.0,
                    help="start the after half this many seconds into the session (default: 0, "
                         "the whole session from its start)")
    a = ap.parse_args()

    d = load_run()
    segments = model_playback(d["side"], B.decode(MODEL_WAV))
    model_end_ms = max(s + len(x) * 1000 / SR for s, x in segments)
    end_ms = model_end_ms + TAIL_MS
    final_ms = model_end_ms
    from_ms = a.after_from * 1000
    n_run = math.ceil((end_ms - from_ms) / 1000 * FPS)
    n_hold = int(round(HOLD_S * FPS))
    n_after = n_run + n_hold
    n_card = int(round(CARD_S * FPS))

    print(f"after half, run {RUN}: book {d['book_ms']}, toolCall {d['tool_call_ms']}, stop {d['stop_ms']}, "
          f"onset {d['onset_ms']}, prepared {d['prepared_ms']}, transcript {d['transcript_ms']}, "
          f"decision {d['decision']} {d['decision_ms']}, cancelled {d['cancelled_ms']}, "
          f"re-issue {d['reissue']}, response {d['response']}, note {d['note_ms']}, "
          f"model {d['model_text_ms']} ms, model audio "
          + ", ".join(f"{s:.0f}-{s + len(x) * 1000 / SR:.0f}" for s, x in segments)
          + f" ms, cancellations {d['n_cancellations']}, session closed {d['session_closed_ms']}")
    print(f"model said: {d['model_text']!r}")
    bad = overlaps(d, end_ms)
    for b in bad:
        print(f"OVERLAP in the events strip: {b}")
    if bad:
        raise SystemExit("fix the strip layout first")

    if a.preview:
        a.preview.mkdir(parents=True, exist_ok=True)
        render_card(CARDS[0]).save(a.preview / "card-1.png")
        render_card(CARDS[1]).save(a.preview / "card-2.png")
        for t in (d["job_start_ms"] + 1500, d["onset_ms"] + 300, d["prepared_ms"] + 250,
                  d["decision_ms"] + 300, d["model_text_ms"] + 1500, end_ms):
            render_after(d, t, end_ms, final_ms).save(a.preview / f"after-{int(t):05d}.png")
        print(f"stills in {a.preview}")
        return

    n_before = count_frames(BEFORE_MP4)
    total_frames = 2 * n_card + n_before + n_after
    print(f"frames: card {n_card} + before {n_before} + card {n_card} + after {n_run} + hold {n_hold} "
          f"= {total_frames} ({total_frames / FPS:.3f} s)")

    # audio: silence, before mix, silence, after mix; one gain for everything
    pre = before_audio(n_before * SPF)
    print(f"before-half audio rebuilt: {check_before_audio(pre)}")
    post = after_audio(d, segments, from_ms, n_after * SPF)
    gap = np.zeros(n_card * SPF, dtype=np.float32)
    track = np.concatenate([gap, pre, gap, post])
    peak = float(np.abs(track).max())
    gain = PEAK / peak
    track *= gain
    for name, x in (("before", pre * gain), ("after", post * gain)):
        print(f"{name} half: peak {20 * math.log10(float(np.abs(x).max())):.1f} dBFS, "
              f"voiced rms {voiced_rms_db(x):.1f} dBFS")
    print(f"one gain for both halves: {20 * math.log10(gain):+.2f} dB (raw peak {peak:.3f} -> -1 dBFS)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mix_path = Path(tmp) / "mix.wav"
        pcm = np.clip(np.round(track * 32767), -32768, 32767).astype("<i2")
        with wave.open(str(mix_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        cmd = [B.FFMPEG, "-y", "-v", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", str(mix_path),
               "-map", "0:v", "-map", "1:a",
               "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "128k", "-ar", str(SR), "-ac", "1",
               "-movflags", "+faststart", str(OUT_MP4)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        card = [render_card(s).tobytes() for s in CARDS]
        for _ in range(n_card):
            proc.stdin.write(card[0])
        n = 0
        for buf in mp4_frames(BEFORE_MP4):
            proc.stdin.write(buf)
            n += 1
        assert n == n_before, f"decoded {n} frames, expected {n_before}"
        for _ in range(n_card):
            proc.stdin.write(card[1])
        for i in range(n_after):
            t = min(from_ms + i * 1000 / FPS, end_ms)
            proc.stdin.write(render_after(d, t, end_ms, final_ms).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg (mp4) failed")

    vf = (f"fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos,split[a][b];"
          "[a]palettegen=max_colors=256:stats_mode=full:reserve_transparent=0[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    subprocess.run([B.FFMPEG, "-y", "-v", "error", "-i", str(OUT_MP4), "-an", "-vf", vf, str(OUT_GIF)],
                   check=True)
    starts = {"card 1": 0.0, "before": n_card / FPS, "card 2": (n_card + n_before) / FPS,
              "after": (2 * n_card + n_before) / FPS}
    print("segment starts (s): " + ", ".join(f"{k} {v:.3f}" for k, v in starts.items()))
    for p in (OUT_MP4, OUT_GIF):
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
