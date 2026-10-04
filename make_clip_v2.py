#!/usr/bin/env python3
"""Render the before/after demo clip (v2) in two formats: the same BLOCKING booking call
and the same spoken "stop", first with the commit guard off, then on. Two people share one
microphone: Person 1 asks for the booking (Kokoro-82M af_heart), Person 2 says stop
(Kokoro-82M am_michael). Both halves are sessions recorded for the clip on 2026-10-04 with
--save-audio; the voices were sent to the model as one live audio stream. They are not
among the 36 sessions of the measured tables, which used a single voice.

  4 s cold open  "Did the booking happen?": the outcome of each session side by side,
                 labelled as an excerpt, with the guard-on session's spoken reply (the
                 same samples as in its full run below)
  2 s card       "Without the guard"
  real time      results/clip/C_off_two_voices.jsonl run 1, from the session start to
                 session_closed, then a 1 s hold (clock stopped)
  2 s card       "With the guard"
  real time      results/clip/C_on_two_voices.jsonl run 1, same span, then a 1 s hold
  5 s results    what happened in each session, in sentences built from the two runs'
                 events, with the verbatim replies

Nothing is time-compressed: neither session has a stretch where nothing happens. The waits
on screen are the server's transcription latency, the 4 s prepare of each booking job and
the model's own delay before it speaks; the 1.5 s after the reply is the harness's quiet
window before it closes the session. If a re-recorded session had dead time, compress it
here and mark it on screen, as ../gemini-live-resume-test/make_clip.py does.

Formats:
  results/clip/before-after-v2.mp4       1280x720, the full layout: conversation, booking
                                         service, events strip (as before-after.mp4)
  results/clip/before-after-v2.gif       800 px wide, 10 fps, no audio
  results/clip/before-after-v2-feed.mp4  1080x1350 (4:5), for phone feeds: the
                                         conversation and one booking status block, large
                                         text, no events strip
Both are 30 fps, H.264 + AAC (mono, 24 kHz), faststart, with the same timing and audio.

Every time and text shown is read from those files (ms since session start). Captions are
verbatim: the people's lines from the server's input transcription, the model's from its
output transcription, the status note from the guard's log. Audio: the clips the harness
sent (the --book-audio / --stop-audio paths in each JSONL's scenario_meta, checked byte for
byte against the copies --save-audio wrote) at their send times, and the model's audio
(results/clip/audio_out/<name>_run1_model.wav) placed as a Live client plays it: each chunk
at its arrival or right after the previous one, and whatever is still queued when
`interrupted` arrives is dropped. One gain (-1 dBFS peak) for the whole track. Cards and
the results slide are silent.

The 16:9 drawing code follows ../gemini-live-stop-test/make_clip.py and
../gemini-live-resume-test/make_clip.py, copied here so this script runs on its own. The
feed version uses the light palette of frontier-on-cloud.github.io. The first clip,
results/clip/before-after.mp4 (macOS `say` voice), is make_clip.py's and stays as it was.

Run (Homebrew ffmpeg is not needed; imageio-ffmpeg ships a static ffmpeg):
  uv run --with imageio-ffmpeg --with pillow --with numpy python make_clip_v2.py
  ... make_clip_v2.py --format feed       # only one format (16x9, feed, both)
  ... make_clip_v2.py --preview DIR       # PNG stills only
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
CLIP_DIR = ROOT / "results" / "clip"
AUDIO_OUT = CLIP_DIR / "audio_out"
OFF, ON = "C_off_two_voices", "C_on_two_voices"
RUN = 1
PEOPLE = {"book_request": "Person 1", "stop": "Person 2"}
VOICE_DIRS = {"book_request": "assets/audio/af_heart/", "stop": "assets/audio/am_michael/"}
VOICES = ("Person 1: Kokoro-82M af_heart. Person 2: Kokoro-82M am_michael. "
          "Synthetic voices, sent as one live audio stream.")
OUT_MP4 = CLIP_DIR / "before-after-v2.mp4"
OUT_GIF = CLIP_DIR / "before-after-v2.gif"
OUT_FEED = CLIP_DIR / "before-after-v2-feed.mp4"
REPO_URL = "github.com/frontier-on-cloud/gemini-live-commit-guard"
TITLE = "Gemini 3.8 Live, BLOCKING tool call: one person books, the other says stop"
COLD_TITLE = ["Two people, one assistant. One books, the other says stop.", "Did the booking happen?"]
CLOSING = f"Client-side guard for the Gemini Live API. Code and data: {REPO_URL}"
FPS, SS, SR = 30, 2, 24000
SPF = SR // FPS
COLD_S, CARD_S, HOLD_S, RESULT_S = 4.0, 2.0, 1.0, 5.0
COLD_REPLY = "on"        # whose spoken reply plays under the cold open (it must fit in COLD_S)
COLD_AUDIO_AT_MS = 400   # where that reply starts in the cold open
GIF_FPS, GIF_W = 10, 800
PEAK = 0.89              # -1 dBFS
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# 16:9 colors: the stop-test clip's (light surface, blue for server and model events, green
# for a committed booking) plus the reference palette's status red for the cancel
BG, PANEL, BORDER = (246, 245, 242), (255, 255, 255), (220, 218, 212)
INK, INK_2, INK_3 = (26, 26, 25), (82, 81, 78), (130, 129, 124)
TRACK, USER_FILL = (232, 230, 225), (236, 234, 230)
BLUE, BLUE_TINT = (42, 120, 214), (226, 237, 251)
GREEN, WHITE = (0, 131, 0), (255, 255, 255)
RED = (208, 59, 59)
LABEL_BG = (26, 26, 25)
# feed colors: frontier-on-cloud.github.io light theme (--bg, --fg, --muted, --line,
# --surface, --accent), plus the same green and red for booked and cancelled
F_BG, F_FG, F_MUTED, F_LINE = (250, 249, 246), (36, 41, 47), (87, 96, 106), (216, 222, 228)
F_SURF, F_ACC, F_ACC_TINT = (238, 240, 242), (8, 145, 178), (224, 242, 247)


# ---------------------------------------------------------------- data
def load_session(name: str) -> dict:
    path = CLIP_DIR / f"{name}.jsonl"
    events = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    meta = next(e for e in events if e["event"] == "scenario_meta")
    events = [e for e in events if e.get("run") == RUN]
    if not events:
        raise SystemExit(f"run {RUN} not found in {path}")
    timed = [e for e in events if e["t_ms"] >= 0]

    def all_of(event: str, **match) -> list[dict]:
        return [e for e in timed if e["event"] == event and all(e.get(k) == v for k, v in match.items())]

    def one(event: str, **match) -> dict:
        hits = all_of(event, **match)
        if not hits:
            raise SystemExit(f"no {event} {match} in {name} run {RUN}")
        return hits[0]

    def first_or_none(event: str, **match) -> dict | None:
        hits = all_of(event, **match)
        return hits[0] if hits else None

    side = json.loads((AUDIO_OUT / f"{name}_run{RUN}_audio.json").read_text(encoding="utf-8"))
    summary = next(e["summary"] for e in events if e["event"] == "run_end")
    start = one("run_start")
    argv = meta["argv"]

    def arg(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    # the clips: the files the harness was told to send, checked against its copies
    clip_paths = {"book_request": arg("--book-audio"), "stop": arg("--stop-audio")}
    clips = {c["label"]: c for c in side["user_clips"]}
    for label, rel in clip_paths.items():
        assert rel.startswith(VOICE_DIRS[label]), f"{name}: {label} clip is {rel}, not in {VOICE_DIRS[label]}"
        assert (ROOT / rel).read_bytes() == (AUDIO_OUT / clips[label]["file"]).read_bytes(), \
            f"{name}: {rel} differs from the copy the harness saved"
        assert clips[label]["sent_start_ms"] == one("user_audio_start", label=label)["t_ms"], label
    assert "--save-audio" in argv and arg("--behavior") == "BLOCKING", argv

    book_ms = one("user_audio_start", label="book_request")["t_ms"]
    stop_ms = one("user_audio_start", label="stop")["t_ms"]
    inputs = all_of("input_transcript")
    book_tr = next(e for e in inputs if book_ms <= e["t_ms"] < stop_ms)
    stop_tr = next(e for e in inputs if e["t_ms"] >= stop_ms)
    calls = all_of("tool_call_received", name="book_slot")
    texts = [e for e in all_of("model_transcript") if e["t_ms"] >= stop_ms]
    note = first_or_none("guard_status_note_sent")
    skipped = first_or_none("guard_response_skipped")
    commits = summary.get("commits") or []
    jobs = []
    for js in all_of("guard_job_started"):
        jid = js["job_id"]
        prep = first_or_none("guard_prepared", job_id=jid)
        com = first_or_none("guard_committed", job_id=jid)
        can = first_or_none("guard_cancelled", job_id=jid)
        resp = first_or_none("guard_tool_response_sent", job_id=jid)
        jobs.append({"id": jid, "call_id": js["call_id"], "start": js["t_ms"],
                     "prepared": prep["t_ms"] if prep else None,
                     "committed": com["t_ms"] if com else None,
                     "confirmation": com["result"]["confirmation_id"] if com else None,
                     "cancelled": can["t_ms"] if can else None,
                     "response": ({"ms": resp["t_ms"], "status": resp["response"].get("status")}
                                  if resp else None)})
    decision = one("guard_decision")
    d = {
        "name": name, "guard": start["guard"], "wall": start["wall"],
        "latency_ms": int(round(start["latency_s"] * 1000)),
        "grace_ms": int(round(start["grace_s"] * 1000)),
        "book_ms": book_ms, "book_text": " ".join(book_tr["text"].split()), "book_tr_ms": book_tr["t_ms"],
        "stop_ms": stop_ms, "stop_text": " ".join(stop_tr["text"].split()), "stop_tr_ms": stop_tr["t_ms"],
        "calls": [{"ms": c["t_ms"], "call_id": c["call_id"], "first": c.get("first")} for c in calls],
        "jobs": jobs,
        "onset_ms": next(e["t_ms"] for e in all_of("voice_activity", voice_activity_type="ACTIVITY_START")
                         if e["t_ms"] >= stop_ms),
        "interrupted_ms": next(e["t_ms"] for e in all_of("interrupted") if e["t_ms"] >= stop_ms),
        "abandoned_ms": (first_or_none("guard_call_abandoned") or {}).get("t_ms"),
        "decision_ms": decision["t_ms"], "decision": decision["decision"],
        "matched": (first_or_none("guard_utterance_resolved", intent="stop") or {}).get("matched") or [],
        "cancelled_ms": jobs[0]["cancelled"],
        "commits": commits,
        "skipped_ms": skipped["t_ms"] if skipped else None,
        "note_ms": note["t_ms"] if note else None, "note_text": note["text"] if note else None,
        "reply_ms": texts[0]["t_ms"], "reply": "".join(e["text"] for e in texts).strip(),
        "closed_ms": one("session_closed")["t_ms"],
        "n_cancellations": len(all_of("tool_call_cancellation")) + len(side["events"].get("cancellations") or []),
        "claim": summary["claim"], "consistent": summary["model_statement_consistent"] == "true",
        "unwanted": summary["unwanted_commit"] == "yes", "note_cuts": summary.get("note_cuts") or [],
        "clips": {label: {"path": ROOT / rel, "sent_start_ms": clips[label]["sent_start_ms"]}
                  for label, rel in clip_paths.items()},
        "side": side, "wav": AUDIO_OUT / f"{name}_run{RUN}_model.wav",
    }
    assert [k["at_ms"] for k in commits] == [j["committed"] for j in jobs if j["committed"]]
    # The clip claims these; refuse to render sessions where they do not hold.
    if d["guard"] == "off":
        assert d["unwanted"] and all(k["at_ms"] > stop_ms for k in commits), f"{name}: no commit after the stop"
        assert len(jobs) <= 2 and len(jobs) == len(calls), f"{name}: the layout shows one or two jobs"
        assert all(j["committed"] for j in jobs), f"{name}: a job was not committed"
    else:
        assert d["decision"] == "cancel" and not commits, f"{name}: the guard did not cancel"
        assert len(jobs) == 1, f"{name}: the layout shows one job"
        assert d["consistent"] and d["claim"] == "claims_not_booked", f"{name}: the reply was wrong"
        assert not d["note_cuts"], f"{name}: the note cut a reply (the clip does not show that case)"
        assert d["note_ms"] is not None
    return d


# ---------------------------------------------------------------- audio
def decode(path: Path) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
           "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def model_playback(side: dict, wav: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(start_ms, samples) per chunk as a Live client plays them: a chunk starts at its
    arrival or when the previous one ends; `interrupted` drops what is still queued."""
    assert side["model_audio"]["sample_rate"] == SR
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
    if i < 0:
        x, i = x[-i:], 0
    seg = x[: max(0, len(mix) - i)]
    mix[i: i + len(seg)] += seg


def voiced_rms_db(x: np.ndarray) -> float:
    fr = x[: len(x) // 480 * 480].reshape(-1, 480)
    r = np.sqrt((fr ** 2).mean(axis=1))
    v = r[r > 10 ** (-50 / 20)]
    return 20 * math.log10(float(np.sqrt((v ** 2).mean()))) if len(v) else float("-inf")


# ---------------------------------------------------------------- drawing primitives
FONT_CANDIDATES = {
    "regular": [("/System/Library/Fonts/HelveticaNeue.ttc", 0), ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
                ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)],
    "bold": [("/System/Library/Fonts/HelveticaNeue.ttc", 1), ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)],
    "mono": [("/System/Library/Fonts/Menlo.ttc", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0)],
    "mono_bold": [("/System/Library/Fonts/Menlo.ttc", 1), ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 0)],
}
_fonts: dict = {}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    key = (kind, size)
    if key not in _fonts:
        for path, index in FONT_CANDIDATES[kind]:
            if Path(path).exists():
                _fonts[key] = ImageFont.truetype(path, size * SS, index=index)
                break
        else:
            raise SystemExit(f"no {kind} font found")
    return _fonts[key]


def tw(s: str, f) -> float:
    return f.getlength(s) / SS


def wrap(s: str, f, max_w: float) -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if cur and tw(trial, f) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


def fit(s: str, kind: str, size: int, max_w: float, min_size: int = 14):
    while size > min_size and tw(s, font(kind, size)) > max_w:
        size -= 1
    return font(kind, size)


def secs(ms: float) -> str:
    """ms as seconds with two decimals, rounded half up (7175 ms -> 7.18 s)."""
    return f"{math.floor(ms / 10 + 0.5) / 100:.2f} s"


class Canvas:
    """Supersampled drawing surface; coordinates are always frame pixels."""

    def __init__(self, w: int, h: int, bg) -> None:
        self.w, self.h = w, h
        self.im = Image.new("RGB", (int(w * SS), int(h * SS)), bg)
        self.d = ImageDraw.Draw(self.im)

    @staticmethod
    def _s(v):
        return [round(x * SS) for x in v]

    def rrect(self, box, r, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(self._s(box), radius=r * SS, fill=fill, outline=outline,
                                 width=round(width * SS) if outline else 0)

    def line(self, pts, fill, width=1):
        self.d.line(self._s([c for p in pts for c in p]), fill=fill, width=round(width * SS))

    def dot(self, x, y, r, fill, ring=None):
        if ring:
            self.d.ellipse(self._s((x - r - 2, y - r - 2, x + r + 2, y + r + 2)), fill=ring)
        self.d.ellipse(self._s((x - r, y - r, x + r, y + r)), fill=fill)

    def ring(self, x, y, r, color, width=2):
        self.d.ellipse(self._s((x - r, y - r, x + r, y + r)), outline=color, width=round(width * SS))

    def poly(self, pts, fill):
        self.d.polygon(self._s([c for p in pts for c in p]), fill=fill)

    def text(self, x, y, s, f, fill, anchor="la"):
        self.d.text((x * SS, y * SS), s, font=f, fill=fill, anchor=anchor)

    def frame(self) -> Image.Image:
        return self.im.resize((self.w, self.h), Image.LANCZOS)


def check_mark(c: Canvas, x: float, ym: float, k: float = 1.0) -> None:
    c.line([(x - 12 * k, ym + 1 * k), (x - 4 * k, ym + 9 * k), (x + 12 * k, ym - 9 * k)], WHITE, width=4 * k)


def cross_mark(c: Canvas, x: float, ym: float, k: float = 1.0) -> None:
    c.line([(x - 10 * k, ym - 10 * k), (x + 10 * k, ym + 10 * k)], WHITE, width=4 * k)
    c.line([(x - 10 * k, ym + 10 * k), (x + 10 * k, ym - 10 * k)], WHITE, width=4 * k)


# ---------------------------------------------------------------- shared wording
def since_stop(d: dict, at_ms: float) -> str:
    return f"{(at_ms - d['stop_ms']) / 1000:.2f} s after Person 2 began to say stop"


def verdict(d: dict) -> str:
    n = len(d["commits"])
    return {0: "Nothing booked.", 1: "Booked after the stop.", 2: "Booked twice after the stop."}[n]


def result_lines(off: dict, on: dict) -> list[tuple[str, str]]:
    """The results slide: what each session did, from its events, with the verbatim reply."""
    k = off["commits"]
    s = (f"Person 2 said stop at {secs(off['stop_ms'])}. The booking was committed at "
         f"{secs(k[0]['at_ms'])} anyway.")
    if len(k) == 2:
        assert off["calls"][1]["first"] is False
        s += (f" The assistant then re-issued the call, and a second booking was committed at "
              f"{secs(k[1]['at_ms'])}.")
    s += f" The assistant said: “{off['reply']}”"
    j = on["jobs"][0]
    assert j["cancelled"] is not None and j["committed"] is None and on["onset_ms"] < j["cancelled"]
    t = (f"Person 2 said stop at {secs(on['stop_ms'])}. The guard held the commit and cancelled the "
         f"booking at {secs(j['cancelled'])}, before it was committed. Nothing was booked. The "
         f"assistant said: “{on['reply']}”")
    return [("Without the guard", s), ("With the guard", t)]


# ================================================================ 16:9
W, H = 1280, 720
M = 24
LEFT = (M, 86, 640, 500)
RIGHT = (656, 86, W - M, 500)
STRIP = (M, 512, W - M, H - 10)
AX_X0, AX_X1, AX_Y = 64, 1216, 666
ROW_Y = {1: 548, 2: 571, 3: 594, 4: 617}   # label rows (text top), 4 nearest the axis
F_LAB = ("regular", 19)
CONV_TOP, CONV_BOT = LEFT[1] + 54, LEFT[3] - 10
PILL_Y = 50
F_PILL = ("bold", 16)
SPEED = "real time"


def pill_box(label: str) -> tuple[float, float, float, float]:
    w = tw(label, font(*F_PILL)) + 24
    return (W - M - 4 - w, PILL_Y, W - M - 4, PILL_Y + 24)


def draw_header(c: Canvas, t_ms: float, subtitle: str) -> None:
    c.text(M + 4, 16, TITLE, fit(TITLE, "bold", 30, 980), INK)
    box = pill_box(SPEED)
    assert M + 4 + tw(subtitle, font("regular", 18)) < box[0] - 16, f"subtitle too long: {subtitle}"
    c.text(M + 4, 52, subtitle, font("regular", 18), INK_2)
    clock = f"{t_ms / 1000:.2f} s"
    fc = font("mono_bold", 34)
    c.text(W - M - 4, 14, clock, fc, INK, anchor="ra")
    c.text(W - M - 4 - tw(clock, fc) - 10, 26, "t =", font("regular", 22), INK_2, anchor="ra")
    c.rrect(box, 12, fill=PANEL, outline=BORDER, width=1)
    c.text((box[0] + box[2]) / 2, box[1] + 12, SPEED, font(*F_PILL), INK_2, anchor="mm")


def draw_panel(c: Canvas, box, title: str, note: str = "") -> None:
    c.rrect(box, 12, fill=PANEL, outline=BORDER, width=1)
    c.text(box[0] + 20, box[1] + 16, title, font("bold", 24), INK)
    if note:
        room = box[2] - box[0] - 60 - tw(title, font("bold", 24))
        c.text(box[2] - 20, box[1] + 20, note, fit(note, "regular", 18, room), INK_2, anchor="ra")


# -- conversation
F_BODY, F_ROLE, F_NOTE = ("regular", 24), ("regular", 17), ("regular", 16)
BODY_LH, PAD_X, PAD_Y = 30, 16, 10
MAX_TEXT = 570


def conversation(d: dict) -> list[dict]:
    return [
        {"kind": "user", "at": d["book_ms"], "role": f"{PEOPLE['book_request']} (voice), {secs(d['book_ms'])}",
         "text": d["book_text"], "foot": f"server transcript at {secs(d['book_tr_ms'])}",
         "foot_at": d["book_tr_ms"]},
        {"kind": "user", "at": d["stop_ms"], "role": f"{PEOPLE['stop']} (voice), {secs(d['stop_ms'])}",
         "text": d["stop_text"], "foot": f"server transcript at {secs(d['stop_tr_ms'])}",
         "foot_at": d["stop_tr_ms"]},
        {"kind": "model", "at": d["reply_ms"], "role": f"Model (spoken reply), {secs(d['reply_ms'])}",
         "text": d["reply"]},
    ]


def bubble_h(text: str, foot: bool) -> int:
    return 24 + len(wrap(text, font(*F_BODY), MAX_TEXT)) * BODY_LH + 2 * PAD_Y + (22 if foot else 0) + 12


def draw_conversation(c: Canvas, d: dict, t_ms: float) -> None:
    draw_panel(c, LEFT, "Conversation", "captions: the server's transcriptions")
    msgs = conversation(d)
    total = sum(bubble_h(m["text"], bool(m.get("foot"))) for m in msgs)
    assert total <= CONV_BOT - CONV_TOP, f"conversation overflows by {total - (CONV_BOT - CONV_TOP)} px"
    x0, x1 = LEFT[0] + 20, LEFT[2] - 20
    y = CONV_TOP
    f = font(*F_BODY)
    for m in msgs:
        lines = wrap(m["text"], f, MAX_TEXT)
        h = bubble_h(m["text"], bool(m.get("foot")))
        if t_ms >= m["at"]:
            bw = max(tw(s, f) for s in lines) + 2 * PAD_X
            bh = len(lines) * BODY_LH + 2 * PAD_Y
            if m["kind"] == "user":
                bx1 = x1
                bx0 = bx1 - bw
                c.text(bx1, y, m["role"], font(*F_ROLE), INK_2, anchor="ra")
                c.rrect((bx0, y + 24, bx1, y + 24 + bh), 16, fill=USER_FILL)
            else:
                bx0, bx1 = x0, x0 + bw
                c.dot(bx0 + 6, y + 10, 5, BLUE)
                c.text(bx0 + 18, y, m["role"], font(*F_ROLE), INK_2)
                c.rrect((bx0, y + 24, bx1, y + 24 + bh), 16, fill=BLUE_TINT, outline=BLUE, width=2)
            for i, s in enumerate(lines):
                c.text(bx0 + PAD_X, y + 24 + PAD_Y + i * BODY_LH - 1, s, f, INK)
            if m.get("foot") and t_ms >= m["foot_at"]:
                fx = bx1 if m["kind"] == "user" else bx0 + 4
                c.text(fx, y + 24 + bh + 4, m["foot"], font(*F_NOTE), INK_3,
                       anchor="ra" if m["kind"] == "user" else "la")
        y += h


# -- booking service
def prepare_text(d: dict, j: dict, t_ms: float) -> str:
    lat = d["latency_ms"] / 1000
    if j["cancelled"] is not None and t_ms >= j["cancelled"] and (j["prepared"] is None or j["cancelled"] < j["prepared"]):
        return f"prepare {lat:.1f} s, stopped at {(j['cancelled'] - j['start']) / 1000:.2f} s: cancelled"
    if j["prepared"] is not None and t_ms >= j["prepared"]:
        return f"prepare {lat:.1f} s, done at {secs(j['prepared'])}"
    return f"prepare {lat:.1f} s, elapsed {(t_ms - j['start']) / 1000:.2f} s"


def prepare_bar(c: Canvas, d: dict, j: dict, t_ms: float, box, r: float) -> None:
    x0, y0, x1, y1 = box
    end = min(t_ms, j["cancelled"] if j["cancelled"] is not None else math.inf)
    frac = min(1.0, max(0.0, (end - j["start"]) / d["latency_ms"]))
    c.rrect(box, r, fill=TRACK)
    if frac > 0:
        done = j["committed"] is not None and t_ms >= j["committed"]
        c.rrect((x0, y0, x0 + max(y1 - y0, (x1 - x0) * frac), y1), r, fill=GREEN if done else INK_2)


def commit_badge(c: Canvas, box, j: dict, t_ms: float) -> None:
    x0, y0, x1, y1 = box
    ym = (y0 + y1) / 2
    c.rrect(box, 10, fill=GREEN)
    check_mark(c, x0 + 26, ym, 0.9)
    label = f"COMMITTED {j['confirmation']}, {secs(j['committed'])}"
    c.text(x0 + 52, ym, label, font("bold", 22), WHITE, anchor="lm")


def draw_service_off(c: Canvas, d: dict, t_ms: float) -> None:
    draw_panel(c, RIGHT, "Booking service (two-phase)", f"fake service, {d['latency_ms'] / 1000:.1f} s prepare")
    x0, y0, x1, _ = RIGHT
    ix0, ix1 = x0 + 20, x1 - 20
    if t_ms < d["jobs"][0]["start"]:
        c.text(ix0, y0 + 70, "idle, no tool call yet", font("regular", 24), INK_3)
        return
    y = y0 + 58
    for k, j in enumerate(d["jobs"]):
        if t_ms < j["start"]:
            break
        head = "job 1 started" if k == 0 else f"job {k + 1} started: re-issued call, new id"
        fh = font("bold", 22)
        c.text(ix0, y, head, fh, INK)
        c.text(ix1, y + 3, secs(j["start"]), font("regular", 19), INK_2, anchor="ra")
        if k == 0:
            cid_x = ix0 + tw(head, fh) + 12
            cid = f"call {j['call_id']}"
            c.text(cid_x, y + 4, cid, fit(cid, "mono", 16, ix1 - 80 - cid_x), INK_2)
        by = y + 32
        prepare_bar(c, d, j, t_ms, (ix0, by, ix1, by + 16), 8)
        c.text(ix0, by + 22, prepare_text(d, j, t_ms), font("regular", 19), INK_2)
        if j["response"] and t_ms >= j["response"]["ms"]:
            s = f"tool response: {j['response']['status']}"
            c.text(ix1, by + 22, s, font("regular", 19), INK, anchor="ra")
        yy = by + 48
        if k == 0:
            if t_ms < d["onset_ms"]:
                s, col = "commit: at once when prepare is done (guard off)", INK_3
            elif t_ms < d["stop_tr_ms"]:
                s, col = f"Person 2 speaking while pending ({secs(d['onset_ms'])}): guard off, not held", INK_2
            elif t_ms < j["committed"]:
                s, col = f"stop transcript at {secs(d['stop_tr_ms'])}: guard off, not checked", INK
            else:
                s, col = f"committed {since_stop(d, j['committed'])}", INK
            c.text(ix0, yy, s, fit(s, "regular", 20, ix1 - ix0), col)
            yy += 28
        if t_ms >= j["committed"]:
            commit_badge(c, (ix0, yy, ix1, yy + 44), j, t_ms)
        y = yy + 54 + 6


def draw_service_on(c: Canvas, d: dict, t_ms: float) -> None:
    draw_panel(c, RIGHT, "Booking service (two-phase)", f"fake service, {d['latency_ms'] / 1000:.1f} s prepare")
    x0, y0, x1, _ = RIGHT
    ix0, ix1 = x0 + 20, x1 - 20
    j = d["jobs"][0]
    f_time, f_row = font("regular", 19), font("regular", 20)
    if t_ms < j["start"]:
        c.text(ix0, y0 + 70, "idle, no tool call yet", font("regular", 24), INK_3)
        return
    y = y0 + 62
    c.text(ix0, y, "job started", font("bold", 26), INK)
    c.text(ix1, y + 4, secs(j["start"]), f_time, INK_2, anchor="ra")
    cid_x = ix0 + tw("job started", font("bold", 26)) + 14
    cid = f"call {j['call_id']}"
    c.text(cid_x, y + 6, cid, fit(cid, "mono", 18, ix1 - 70 - cid_x), INK_2)
    by = y + 40
    prepare_bar(c, d, j, t_ms, (ix0, by, ix1, by + 20), 10)
    c.text(ix0, by + 28, prepare_text(d, j, t_ms), f_row, INK_2)
    hy = by + 62
    if t_ms < d["decision_ms"]:
        if t_ms < d["onset_ms"]:
            s, col = f"commit: after prepare plus {d['grace_ms'] / 1000:.1f} s grace", INK_3
        else:
            s, col = f"Person 2 speaking while pending ({secs(d['onset_ms'])}): commit held", INK_2
        c.rrect((ix0, hy, ix1, hy + 20), 10, fill=TRACK)
        c.text(ix0, hy + 28, s, fit(s, "regular", 20, ix1 - ix0), col)
    else:
        words = ", ".join(f"“{m}”" for m in d["matched"])
        s = f"stop transcript at {secs(d['stop_tr_ms'])} matched {words}"
        c.text(ix0, hy + 4, s, fit(s, "regular", 20, ix1 - ix0), INK)
    cy = hy + 62
    if t_ms >= d["decision_ms"]:
        c.rrect((ix0, cy, ix1, cy + 54), 10, fill=RED)
        cross_mark(c, ix0 + 32, cy + 27)
        label = f"CANCELLED before commit, {secs(j['cancelled'])}"
        c.text(ix0 + 60, cy + 27, label, fit(label, "bold", 28, ix1 - ix0 - 76, 18), WHITE, anchor="lm")
    ry = cy + 68
    if d["skipped_ms"] is not None and t_ms >= d["skipped_ms"]:
        s = "tool response: none, the server dropped the call"
        c.text(ix0, ry, s, fit(s, "regular", 21, ix1 - ix0 - 80), INK)
    ny = ry + 29
    if t_ms >= d["note_ms"]:
        c.text(ix0, ny, "status note sent to the model", font("regular", 21), INK)
        c.text(ix1, ny + 2, secs(d["note_ms"]), f_time, INK_2, anchor="ra")
        f_note = font("regular", 18)
        lines = wrap(f"“{d['note_text']}”", f_note, ix1 - ix0 - 12)
        assert len(lines) <= 2, "status note needs more than two lines"
        for i, line in enumerate(lines):
            c.text(ix0 + 12, ny + 28 + i * 22, line, f_note, INK_2)


# -- events strip
class Axis:
    def __init__(self, t1: float) -> None:
        self.t1 = t1

    def x(self, t_ms: float) -> float:
        return AX_X0 + (AX_X1 - AX_X0) * min(max(t_ms, 0.0), self.t1) / self.t1


def strip_ticks(d: dict) -> list[tuple]:
    """label, time, color, row, side, marker; rows and sides chosen so nothing overlaps
    (checked by overlaps())."""
    j = d["jobs"]
    if d["guard"] == "off":
        t = [
            (f"toolCall {secs(j[0]['start'])}", d["calls"][0]["ms"], BLUE, 4, "left", "dot"),
            (f"stop clip starts {secs(d['stop_ms'])}", d["stop_ms"], INK, 3, "left", "dot"),
            (f"interrupted {secs(d['interrupted_ms'])}", d["interrupted_ms"], BLUE, 2, "left", "dot"),
            (f"stop transcript {secs(d['stop_tr_ms'])}", d["stop_tr_ms"], INK, 1, "left", "dot"),
            (f"committed {j[0]['confirmation']} {secs(j[0]['committed'])}", j[0]["committed"], GREEN, 2,
             "right", "dot"),
        ]
        if len(j) == 2:
            t += [(f"re-issued toolCall {secs(d['calls'][1]['ms'])}", d["calls"][1]["ms"], BLUE, 3, "right", "dot"),
                  (f"committed {j[1]['confirmation']} {secs(j[1]['committed'])}", j[1]["committed"], GREEN, 1,
                   "right", "dot")]
        return t + [(f"model reply {secs(d['reply_ms'])}", d["reply_ms"], BLUE, 4, "right", "dot")]
    assert d["abandoned_ms"] == d["interrupted_ms"], "abandon and interrupted not together"
    assert d["note_ms"] - d["decision_ms"] <= 5 and abs(d["stop_tr_ms"] - d["decision_ms"]) <= 5
    return [
        (f"toolCall {secs(d['calls'][0]['ms'])}", d["calls"][0]["ms"], BLUE, 4, "left", "dot"),
        (f"stop clip starts {secs(d['stop_ms'])}", d["stop_ms"], INK, 3, "left", "dot"),
        (f"interrupted, call abandoned {secs(d['interrupted_ms'])}", d["interrupted_ms"], BLUE, 1,
         "right", "dot"),
        (f"status note {secs(d['note_ms'])}", d["note_ms"], INK, 3, "right", "ring"),
        (f"stop transcript, decision: {d['decision']} {secs(d['decision_ms'])}", d["decision_ms"], RED, 2,
         "right", "dot"),
        (f"model reply {secs(d['reply_ms'])}", d["reply_ms"], BLUE, 4, "right", "dot"),
    ]


def final_line(d: dict) -> str:
    s = "toolCallCancellation: never received"
    return s if d["guard"] == "off" else s + ", the guard decided"


def strip_boxes(d: dict, axis: Axis) -> tuple[list, list]:
    f_lab = font(*F_LAB)
    boxes, leaders = [], []
    for label, at, _, row, side, _ in strip_ticks(d):
        x, ly = axis.x(at), ROW_Y[row]
        w = tw(label, f_lab)
        boxes.append((label, (x + 8, ly, x + 8 + w, ly + 21) if side == "right" else (x - 8 - w, ly, x - 8, ly + 21)))
        leaders.append((label, (x - 1, ly + 2, x + 1, AX_Y)))
    fb = font("bold", 19)
    fl = final_line(d)
    boxes.append(("final", (STRIP[2] - 20 - tw(fl, fb) - 26, STRIP[1] + 10, STRIP[2] - 20, STRIP[1] + 33)))
    head = tw("Events", font("bold", 22)) + 10 + tw("seconds since session start", font("regular", 18))
    boxes.append(("header", (STRIP[0] + 20, STRIP[1] + 12, STRIP[0] + 20 + head, STRIP[1] + 36)))
    return boxes, leaders


def overlaps(d: dict, axis: Axis) -> list[str]:
    def hit(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    boxes, leaders = strip_boxes(d, axis)
    bad = [f"{a!r} x {b!r}" for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:] if hit(ba, bb)]
    bad += [f"{a!r} x leader of {b!r}" for a, ba in boxes for b, lb in leaders if a != b and hit(ba, lb)]
    bad += [f"{a!r} outside the strip" for a, ba in boxes if ba[0] < STRIP[0] + 8 or ba[2] > STRIP[2] - 8]
    play = (AX_X0 - 7, AX_Y - 22, AX_X1 + 7, AX_Y - 10)
    bad += [f"{a!r} x playhead path" for a, ba in boxes if hit(ba, play)]
    return [f"{d['name']}: {b}" for b in bad]


def draw_strip(c: Canvas, d: dict, t_ms: float, axis: Axis) -> None:
    c.rrect(STRIP, 12, fill=PANEL, outline=BORDER, width=1)
    c.text(STRIP[0] + 20, STRIP[1] + 12, "Events", font("bold", 22), INK)
    c.text(STRIP[0] + 20 + tw("Events", font("bold", 22)) + 10, STRIP[1] + 16,
           "seconds since session start", font("regular", 18), INK_2)
    now_x = axis.x(t_ms)
    c.line([(AX_X0, AX_Y), (AX_X1, AX_Y)], TRACK, width=3)
    c.line([(AX_X0, AX_Y), (now_x, AX_Y)], INK_3, width=3)
    for s in range(0, int(axis.t1 // 1000) + 1):
        x = axis.x(s * 1000)
        c.line([(x, AX_Y + 4), (x, AX_Y + 9)], INK_3, width=1)
        if s % 2 == 0:
            c.text(x, AX_Y + 12, f"{s} s", font("regular", 17), INK_3, anchor="ma")
    ticks = [tk for tk in strip_ticks(d) if t_ms >= tk[1]]
    for label, at, color, row, side, _ in ticks:
        x, ly = axis.x(at), ROW_Y[row]
        c.line([(x, AX_Y), (x, ly + 2)], color, width=2)
        if side == "right":
            c.text(x + 8, ly, label, font(*F_LAB), INK)
        else:
            c.text(x - 8, ly, label, font(*F_LAB), INK, anchor="ra")
    for label, at, color, row, side, marker in ticks:
        x = axis.x(at)
        if marker == "ring":
            c.dot(x, AX_Y, 10, PANEL)
            c.ring(x, AX_Y, 10, color, width=2)
        else:
            c.dot(x, AX_Y, 6, color, ring=PANEL)
    if t_ms >= d["closed_ms"] and d["n_cancellations"] == 0:
        fb = font("bold", 19)
        fl = final_line(d)
        c.text(STRIP[2] - 20, STRIP[1] + 10, fl, fb, INK, anchor="ra")
        c.ring(STRIP[2] - 20 - tw(fl, fb) - 16, STRIP[1] + 21, 7, INK, width=2)
    c.poly([(now_x - 7, AX_Y - 20), (now_x + 7, AX_Y - 20), (now_x, AX_Y - 10)], INK)


def subtitle(d: dict) -> str:
    mode = "Without the guard (--guard off)" if d["guard"] == "off" else "With the guard (--guard on)"
    return f"{mode}. Session recorded for this clip, not in the measured tables."


def render_half(d: dict, t_ms: float, axis: Axis) -> Image.Image:
    c = Canvas(W, H, BG)
    draw_header(c, t_ms, subtitle(d))
    draw_conversation(c, d, t_ms)
    (draw_service_off if d["guard"] == "off" else draw_service_on)(c, d, t_ms)
    draw_strip(c, d, t_ms, axis)
    return c.frame()


# -- cold open, cards, results
COLD_LABEL = "Excerpt: the outcome of each session. Full sessions follow."


def cold_column(c: Canvas, box, heading: str, d: dict) -> None:
    x0, y0, x1, y1 = box
    ix0, ix1 = x0 + 24, x1 - 24
    c.rrect(box, 12, fill=PANEL, outline=BORDER, width=1)
    c.text(ix0, y0 + 18, heading, font("bold", 30), INK)
    c.text(ix1, y0 + 26, f"--guard {d['guard']}", font("mono", 18), INK_3, anchor="ra")
    f = font("regular", 23)
    y = y0 + 72
    c.text(ix1, y, f"{PEOPLE['stop']} (voice), {secs(d['stop_ms'])}", font("regular", 17), INK_2, anchor="ra")
    bw = tw(d["stop_text"], f) + 2 * PAD_X
    c.rrect((ix1 - bw, y + 24, ix1, y + 24 + 28 + 2 * PAD_Y), 16, fill=USER_FILL)
    c.text(ix1 - bw + PAD_X, y + 24 + PAD_Y - 1, d["stop_text"], f, INK)
    y += 24 + 28 + 2 * PAD_Y + 18
    c.text(ix0, y, "Booking service", font("regular", 17), INK_2)
    y += 24
    if d["commits"]:
        for j in d["jobs"]:
            c.rrect((ix0, y, ix1, y + 44), 10, fill=GREEN)
            check_mark(c, ix0 + 26, y + 22, 0.9)
            c.text(ix0 + 52, y + 22, f"COMMITTED {j['confirmation']}, {secs(j['committed'])}",
                   font("bold", 23), WHITE, anchor="lm")
            y += 52
        line = f"first commit {since_stop(d, d['jobs'][0]['committed'])}"
    else:
        c.rrect((ix0, y, ix1, y + 44), 10, fill=RED)
        cross_mark(c, ix0 + 26, y + 22, 0.9)
        c.text(ix0 + 52, y + 22, f"CANCELLED before commit, {secs(d['cancelled_ms'])}",
               font("bold", 23), WHITE, anchor="lm")
        y += 52
        line = f"cancelled {since_stop(d, d['cancelled_ms'])}"
    c.text(ix0, y, line, fit(line, "regular", 19, ix1 - ix0), INK_2)
    y += 40
    c.dot(ix0 + 6, y + 10, 5, BLUE)
    c.text(ix0 + 18, y, f"Model (spoken reply), {secs(d['reply_ms'])}", font("regular", 17), INK_2)
    lines = wrap(d["reply"], f, ix1 - ix0 - 2 * PAD_X)
    bw = max(tw(s, f) for s in lines) + 2 * PAD_X
    bh = len(lines) * 29 + 2 * PAD_Y
    c.rrect((ix0, y + 24, ix0 + bw, y + 24 + bh), 16, fill=BLUE_TINT, outline=BLUE, width=2)
    for i, s in enumerate(lines):
        c.text(ix0 + PAD_X, y + 24 + PAD_Y + i * 29 - 1, s, f, INK)
    y += 24 + bh + 20
    c.text(ix0, y, verdict(d), font("bold", 26), INK)
    assert y + 32 < y1 - 6, f"cold-open column overflows ({heading})"


def render_cold(off: dict, on: dict) -> Image.Image:
    c = Canvas(W, H, BG)
    c.text(W / 2, 14, COLD_TITLE[0], fit(COLD_TITLE[0], "bold", 34, W - 2 * M), INK, anchor="ma")
    c.text(W / 2, 56, COLD_TITLE[1], font("bold", 34), INK, anchor="ma")
    top, bot = 106, 652
    cold_column(c, (M, top, W / 2 - 8, bot), "Without the guard", off)
    cold_column(c, (W / 2 + 8, top, W - M, bot), "With the guard", on)
    f = font("regular", 19)
    w = tw(COLD_LABEL, f)
    bx1, by1 = W - M, H - 12
    c.rrect((bx1 - w - 28, by1 - 40, bx1, by1), 8, fill=LABEL_BG)
    c.text(bx1 - 14, by1 - 20, COLD_LABEL, f, WHITE, anchor="rm")
    return c.frame()


def render_card(lines: list[tuple[str, int, tuple]]) -> Image.Image:
    c = Canvas(W, H, BG)
    total = sum(size + 18 for _, size, _ in lines) - 18
    y = H / 2 - total / 2
    for s, size, col in lines:
        c.text(W / 2, y + size / 2, s, fit(s, "bold" if size >= 40 else "regular", size, W - 80), col, anchor="mm")
        y += size + 18
    return c.frame()


def cards(off: dict, on: dict) -> dict:
    return {
        "card_off": render_card([("Without the guard", 46, INK),
                                 ("BLOCKING book_slot call, 4 s prepare, then commit at once.", 24, INK_2),
                                 (VOICES, 20, INK_3)]),
        "card_on": render_card([("With the guard", 46, INK),
                                ("Same request, same stop, commit_guard.py on.", 24, INK_2)]),
    }


def render_results(off: dict, on: dict) -> Image.Image:
    c = Canvas(W, H, BG)
    x0, width = 80, W - 160
    fb, fp = font("bold", 34), font("regular", 29)
    blocks = [(h, wrap(s, fp, width)) for h, s in result_lines(off, on)]
    foot = wrap(CLOSING, font("regular", 24), width)
    src = (f"Two sessions recorded for this clip on {on['wall'][:10]}, not part of the measured tables. "
           "Two synthetic voices, one audio stream.")
    total = sum(46 + len(ls) * 39 + 30 for _, ls in blocks) + 10 + len(foot) * 32 + 12 + 26
    assert total < H - 40, f"results slide overflows ({total} px)"
    y = (H - total) / 2
    for head, ls in blocks:
        c.text(x0, y, head, fb, INK)
        y += 46
        for s in ls:
            c.text(x0, y, s, fp, INK)
            y += 39
        y += 30
    c.line([(x0, y - 8), (x0 + width, y - 8)], BORDER, width=1)
    y += 10
    for s in foot:
        c.text(x0, y, s, font("regular", 24), INK_2)
        y += 32
    y += 12
    c.text(x0, y, src, fit(src, "regular", 18, width), INK_3)
    return c.frame()


# ================================================================ 4:5 feed
FW, FH = 1080, 1350
FM = 40
F_CONV = (FM, 176, FW - FM, 820)
F_STAT = (FM, 846, FW - FM, 1262)
F_TEXT, F_TEXT_LH = ("regular", 36), 46
F_ROLE_ = ("regular", 28)


def feed_source(c: Canvas, d: dict) -> None:
    s = (f"Gemini 3.8 Live, {d['wall'][:10]}. Recorded for this clip, not in the measured tables. "
         "Two synthetic voices.")
    c.text(FW / 2, FH - 52, s, fit(s, "regular", 24, FW - 2 * FM), F_MUTED, anchor="ma")


def feed_bubble(c: Canvas, x_left: float, x_right: float, y: float, role: str, text: str, kind: str,
                draw: bool) -> float:
    """A caption bubble; returns its height. Person lines are right-aligned on the surface
    color, the assistant's left-aligned on the accent tint."""
    f = font(*F_TEXT)
    lines = wrap(text, f, x_right - x_left - 140 - 48)
    bw = max(tw(s, f) for s in lines) + 48
    bh = len(lines) * F_TEXT_LH + 28
    if draw:
        if kind == "person":
            c.text(x_right, y, role, font(*F_ROLE_), F_MUTED, anchor="ra")
            c.rrect((x_right - bw, y + 38, x_right, y + 38 + bh), 22, fill=F_SURF)
            bx0 = x_right - bw
        else:
            c.text(x_left, y, role, font(*F_ROLE_), F_MUTED)
            c.rrect((x_left, y + 38, x_left + bw, y + 38 + bh), 22, fill=F_ACC_TINT, outline=F_ACC, width=3)
            bx0 = x_left
        for i, s in enumerate(lines):
            c.text(bx0 + 24, y + 38 + 14 + i * F_TEXT_LH, s, f, F_FG)
    return 38 + bh + 26


def feed_status(d: dict, t_ms: float) -> tuple[str, tuple, list[str], float | None]:
    """Badge text, badge color, text lines and the prepare fraction (or None) of the one
    booking status block."""
    jobs, lat = d["jobs"], d["latency_ms"]
    stop_line = f"Person 2 said stop at {secs(d['stop_ms'])}."

    def frac(j: dict) -> float:
        return min(1.0, max(0.0, (t_ms - j["start"]) / lat))

    if t_ms < jobs[0]["start"]:
        return "No booking yet", F_SURF, [], None
    if d["guard"] == "off":
        j1 = jobs[0]
        j2 = jobs[1] if len(jobs) > 1 else None
        first = f"{j1['confirmation']} committed at {secs(j1['committed'])}, {(j1['committed'] - d['stop_ms']) / 1000:.2f} s after the stop."
        if t_ms < j1["committed"]:
            lines = [f"Preparing: {(t_ms - j1['start']) / 1000:.1f} of {lat / 1000:.1f} s."]
            if t_ms >= d["stop_ms"]:
                lines.append(stop_line)
            if t_ms >= d["stop_tr_ms"]:
                lines.append("Guard off: the stop is not checked.")
            return "Booking in progress", F_SURF, lines, frac(j1)
        if j2 is None or t_ms < j2["start"]:
            return f"BOOKED  {j1['confirmation']}", GREEN, [stop_line, first], None
        if t_ms < j2["committed"]:
            return (f"BOOKED  {j1['confirmation']}", GREEN,
                    [first, f"Re-issued call at {secs(j2['start'])}: second booking in progress."], frac(j2))
        return ("BOOKED TWICE", GREEN,
                [first, f"{j2['confirmation']} committed at {secs(j2['committed'])}, from the re-issued call."],
                None)
    j = jobs[0]
    if t_ms < d["decision_ms"]:
        lines = [f"Preparing: {(t_ms - j['start']) / 1000:.1f} of {lat / 1000:.1f} s."]
        if t_ms >= d["stop_ms"]:
            lines.append(stop_line)
        if t_ms >= d["onset_ms"]:
            lines.append("Guard: commit on hold while Person 2 speaks.")
        return "Booking in progress", F_SURF, lines, frac(j)
    lines = [stop_line, f"Cancelled at {secs(j['cancelled'])}, before any commit."]
    if t_ms >= d["note_ms"]:
        lines.append("Status note sent to the model.")
    return "NOT BOOKED", RED, lines, None


def feed_badge(c: Canvas, box, text: str, fill: tuple, size: int = 46) -> None:
    x0, y0, x1, y1 = box
    ym = (y0 + y1) / 2
    c.rrect(box, 18, fill=fill)
    if fill == F_SURF:
        c.text(x0 + 32, ym, text, font("bold", size), F_FG, anchor="lm")
        return
    (check_mark if fill == GREEN else cross_mark)(c, x0 + 46, ym, 1.6)
    c.text(x0 + 90, ym, text, fit(text, "bold", size, x1 - x0 - 110), WHITE, anchor="lm")


def render_feed_half(d: dict, t_ms: float) -> Image.Image:
    c = Canvas(FW, FH, F_BG)
    head = "Without the guard" if d["guard"] == "off" else "With the guard"
    c.text(FM + 10, 40, head, font("bold", 62), F_FG)
    clock = f"{t_ms / 1000:.2f} s"
    fc = font("mono_bold", 50)
    c.text(FW - FM - 10, 44, clock, fc, F_FG, anchor="ra")
    c.text(FW - FM - 10 - tw(clock, fc) - 12, 58, "t =", font("regular", 32), F_MUTED, anchor="ra")
    c.text(FM + 10, 118, "Real time, seconds since the session started.", font("regular", 30), F_MUTED)
    # conversation
    c.rrect(F_CONV, 24, fill=PANEL, outline=F_LINE, width=2)
    x_l, x_r = F_CONV[0] + 30, F_CONV[2] - 30
    y = F_CONV[1] + 28
    msgs = [("person", d["book_ms"], f"{PEOPLE['book_request']}, {secs(d['book_ms'])}", d["book_text"]),
            ("person", d["stop_ms"], f"{PEOPLE['stop']}, {secs(d['stop_ms'])}", d["stop_text"]),
            ("assistant", d["reply_ms"], f"Assistant (spoken reply), {secs(d['reply_ms'])}", d["reply"])]
    for kind, at, role, text in msgs:
        y += feed_bubble(c, x_l, x_r, y, role, text, kind, t_ms >= at)
    assert y <= F_CONV[3] + 10, f"feed conversation overflows ({y:.0f} > {F_CONV[3]})"
    # booking status
    c.rrect(F_STAT, 24, fill=PANEL, outline=F_LINE, width=2)
    sx0, sx1 = F_STAT[0] + 30, F_STAT[2] - 30
    c.text(sx0, F_STAT[1] + 24, f"Booking (fake backend, {d['latency_ms'] / 1000:.0f} s to prepare)",
           font("regular", 30), F_MUTED)
    text, fill, lines, frac = feed_status(d, t_ms)
    by = F_STAT[1] + 74
    feed_badge(c, (sx0, by, sx1, by + 100), text, fill)
    y = by + 124
    if frac is not None:
        c.rrect((sx0, y, sx1, y + 20), 10, fill=F_SURF)
        if frac > 0:
            c.rrect((sx0, y, sx0 + max(20, (sx1 - sx0) * frac), y + 20), 10, fill=F_MUTED)
        y += 40
    f = font(*F_TEXT)
    for s in lines:
        for part in wrap(s, f, sx1 - sx0):
            c.text(sx0, y, part, f, F_FG)
            y += F_TEXT_LH
    assert y <= F_STAT[3] - 10, f"feed status block overflows ({y:.0f} > {F_STAT[3] - 10}) at {t_ms:.0f} ms"
    feed_source(c, d)
    return c.frame()


def feed_outcome_h(d: dict, width: float) -> int:
    """Height feed_outcome needs for this session at this box width."""
    n_line = len(wrap(feed_outcome_line(d), font("regular", 34), width - 60))
    n_reply = len(wrap(f"“{d['reply']}”", font("regular", 36), width - 60 - 48))
    return 90 + 104 + n_line * 44 + 14 + 42 + n_reply * F_TEXT_LH + 28 + 30


def feed_outcome_line(d: dict) -> str:
    if d["commits"]:
        return f"First commit {(d['commits'][0]['at_ms'] - d['stop_ms']) / 1000:.2f} s after Person 2 said stop."
    return f"Cancelled {(d['cancelled_ms'] - d['stop_ms']) / 1000:.2f} s after Person 2 said stop."


def feed_outcome(c: Canvas, box, heading: str, d: dict) -> None:
    x0, y0, x1, y1 = box
    ix0, ix1 = x0 + 30, x1 - 30
    c.rrect(box, 24, fill=PANEL, outline=F_LINE, width=2)
    c.text(ix0, y0 + 24, heading, font("bold", 46), F_FG)
    y = y0 + 90
    line = feed_outcome_line(d)
    if d["commits"]:
        text = "BOOKED TWICE" if len(d["commits"]) == 2 else "BOOKED"
        feed_badge(c, (ix0, y, ix1, y + 88), text, GREEN, 44)
    else:
        feed_badge(c, (ix0, y, ix1, y + 88), "NOT BOOKED", RED, 44)
    y += 104
    f = font("regular", 34)
    for part in wrap(line, f, ix1 - ix0):
        c.text(ix0, y, part, f, F_FG)
        y += 44
    y += 14
    c.text(ix0, y, "The assistant said:", font("regular", 30), F_MUTED)
    y += 42
    fq = font("regular", 36)
    lines = wrap(f"“{d['reply']}”", fq, ix1 - ix0 - 48)
    bh = len(lines) * F_TEXT_LH + 28
    bw = max(tw(s, fq) for s in lines) + 48
    c.rrect((ix0, y, ix0 + bw, y + bh), 22, fill=F_ACC_TINT, outline=F_ACC, width=3)
    for i, s in enumerate(lines):
        c.text(ix0 + 24, y + 14 + i * F_TEXT_LH, s, fq, F_FG)
    assert y + bh <= y1 - 16, f"feed cold-open block overflows ({heading}: {y + bh:.0f} > {y1 - 16})"


def render_feed_cold(off: dict, on: dict) -> Image.Image:
    c = Canvas(FW, FH, F_BG)
    ft = font("bold", 52)
    y = 40
    for s in wrap(COLD_TITLE[0], ft, FW - 2 * FM - 20) + [COLD_TITLE[1]]:
        c.text(FW / 2, y, s, ft, F_FG, anchor="ma")
        y += 64
    hs = [feed_outcome_h(d, FW - 2 * FM) for d in (off, on)]
    y += 16 + (FH - 60 - 52 - 24 - (y + 16) - sum(hs) - 24) / 3   # spread the spare room
    for (head, d), h in zip((("Without the guard", off), ("With the guard", on)), hs):
        feed_outcome(c, (FM, y, FW - FM, y + h), head, d)
        y += h + 24
    f = font("regular", 30)
    label = "Excerpt. Full sessions follow."
    w = tw(label, f)
    c.rrect((FW / 2 - w / 2 - 22, y, FW / 2 + w / 2 + 22, y + 52), 10, fill=LABEL_BG)
    c.text(FW / 2, y + 26, label, f, WHITE, anchor="mm")
    assert y + 52 < FH - 60, f"feed cold open overflows ({y + 52:.0f})"
    feed_source(c, on)
    return c.frame()


def render_feed_card(title: str, sub: str, d: dict) -> Image.Image:
    c = Canvas(FW, FH, F_BG)
    c.text(FW / 2, FH / 2 - 60, title, font("bold", 76), F_FG, anchor="mm")
    f = font("regular", 38)
    for i, s in enumerate(wrap(sub, f, FW - 2 * FM - 40)):
        c.text(FW / 2, FH / 2 + 30 + i * 50, s, f, F_MUTED, anchor="ma")
    feed_source(c, d)
    return c.frame()


def render_feed_results(off: dict, on: dict) -> Image.Image:
    c = Canvas(FW, FH, F_BG)
    x0, width = FM + 20, FW - 2 * FM - 40
    fb, fp, ff = font("bold", 54), font("regular", 40), font("regular", 34)
    blocks = [(h, wrap(s, fp, width)) for h, s in result_lines(off, on)]
    foot = wrap(CLOSING, ff, width)
    total = sum(72 + len(ls) * 52 + 44 for _, ls in blocks) + 24 + len(foot) * 46
    assert total < FH - 140, f"feed results slide overflows ({total} px)"
    y = (FH - 80 - total) / 2
    for head, ls in blocks:
        c.text(x0, y, head, fb, F_FG)
        y += 72
        for s in ls:
            c.text(x0, y, s, fp, F_FG)
            y += 52
        y += 44
    c.line([(x0, y - 18), (x0 + width, y - 18)], F_LINE, width=2)
    y += 24
    for s in foot:
        c.text(x0, y, s, ff, F_MUTED)
        y += 46
    feed_source(c, on)
    return c.frame()


# ================================================================ main
def encode(path: Path, size: tuple[int, int], mix_path: Path, frames) -> None:
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-",
           "-i", str(mix_path), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-ar", str(SR), "-ac", "1",
           "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for buf in frames:
        proc.stdin.write(buf)
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg ({path.name}) failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--format", choices=["16x9", "feed", "both"], default="both")
    ap.add_argument("--preview", type=Path, help="write PNG stills to this folder and stop")
    a = ap.parse_args()

    off, on = load_session(OFF), load_session(ON)
    seg = {"off": model_playback(off["side"], decode(off["wav"])),
           "on": model_playback(on["side"], decode(on["wav"]))}
    axis = Axis(max(off["closed_ms"], on["closed_ms"]))
    for k, d in (("off", off), ("on", on)):
        print(f"{d['name']} ({d['wall']}): book {d['book_ms']} ({d['book_text']!r}, transcript {d['book_tr_ms']}), "
              f"calls {[c['ms'] for c in d['calls']]}, stop {d['stop_ms']} ({d['stop_text']!r}, transcript "
              f"{d['stop_tr_ms']}), onset {d['onset_ms']}, interrupted {d['interrupted_ms']}, decision "
              f"{d['decision']} {d['decision_ms']}, jobs "
              + "; ".join(f"{j['id']} {j['start']}-{j['prepared']} committed {j['committed']} {j['confirmation']} "
                          f"cancelled {j['cancelled']}" for j in d["jobs"])
              + f", note {d['note_ms']}, reply {d['reply_ms']} ({d['reply']!r}), claim {d['claim']}, "
              f"cancellations {d['n_cancellations']}, closed {d['closed_ms']}; model audio "
              + ", ".join(f"{t:.0f}-{t + len(x) * 1000 / SR:.0f}" for t, x in seg[k]) + " ms")
    for head, s in result_lines(off, on):
        print(f"results slide, {head}: {s}")
    bad = overlaps(off, axis) + overlaps(on, axis)
    for b in bad:
        print(f"OVERLAP in the events strip: {b}")
    if bad:
        raise SystemExit("fix the strip layout first")

    wide = {"cold": render_cold(off, on), "results": render_results(off, on)} | cards(off, on)
    feed = {"cold": render_feed_cold(off, on), "results": render_feed_results(off, on),
            "card_off": render_feed_card("Without the guard", "Person 1 asks for a booking. "
                                         "Person 2 says stop while it is being made.", off),
            "card_on": render_feed_card("With the guard", "Same request, same stop. "
                                        "The guard holds the commit when someone speaks.", on)}
    if a.preview:
        a.preview.mkdir(parents=True, exist_ok=True)
        for k, im in wide.items():
            im.save(a.preview / f"wide-{k}.png")
        for k, im in feed.items():
            im.save(a.preview / f"feed-{k}.png")
        j = off["jobs"]
        t_off = [j[0]["start"] + 600, off["onset_ms"] + 500, off["stop_tr_ms"] + 5, j[0]["committed"] + 300]
        t_off += [j[1]["start"] + 1500, j[1]["committed"] + 300] if len(j) > 1 else []
        t_off += [off["reply_ms"] + 2000, off["closed_ms"]]
        t_on = [on["jobs"][0]["start"] + 600, on["onset_ms"] + 500, on["decision_ms"] + 300,
                on["reply_ms"] + 1500, on["closed_ms"]]
        for d, ts in ((off, t_off), (on, t_on)):
            for t in ts:
                render_half(d, t, axis).save(a.preview / f"wide-{d['guard']}-{int(t):05d}.png")
                render_feed_half(d, t).save(a.preview / f"feed-{d['guard']}-{int(t):05d}.png")
        print(f"stills in {a.preview}")
        return

    n_cold, n_card, n_hold, n_res = (int(round(s * FPS)) for s in (COLD_S, CARD_S, HOLD_S, RESULT_S))
    n_off = math.ceil(off["closed_ms"] / 1000 * FPS)
    n_on = math.ceil(on["closed_ms"] / 1000 * FPS)
    parts = [("cold", n_cold), ("card_off", n_card), ("off", n_off + n_hold), ("card_on", n_card),
             ("on", n_on + n_hold), ("results", n_res)]
    total = sum(n for _, n in parts)
    starts, acc = {}, 0
    for kind, n in parts:
        starts[kind] = acc
        acc += n
    assert total / FPS <= 60, "over 60 s"
    print("frames: " + " + ".join(f"{k} {n}" for k, n in parts) + f" = {total} ({total / FPS:.2f} s)")

    # audio: one track, one gain, the same for both formats
    track = np.zeros(total * SPF, dtype=np.float32)
    cold_seg = seg[COLD_REPLY]
    t0 = min(t for t, _ in cold_seg)
    cold_end = max(t + len(x) * 1000 / SR for t, x in cold_seg) - t0 + COLD_AUDIO_AT_MS
    assert cold_end <= COLD_S * 1000, f"the cold-open reply runs past the cold open ({cold_end:.0f} ms)"
    cold_mix = np.zeros(n_cold * SPF, dtype=np.float32)
    for t, x in cold_seg:
        place(cold_mix, x, t - t0 + COLD_AUDIO_AT_MS)
    track[: len(cold_mix)] += cold_mix
    placements = [(f"cold open: guard-{COLD_REPLY} model reply", COLD_AUDIO_AT_MS / 1000, cold_end / 1000)]
    for kind, d in (("off", off), ("on", on)):
        n = (n_off if kind == "off" else n_on) + n_hold
        mix = np.zeros(n * SPF, dtype=np.float32)
        for label, cl in d["clips"].items():
            x = decode(cl["path"])
            place(mix, x, cl["sent_start_ms"])
            at = starts[kind] / FPS + cl["sent_start_ms"] / 1000
            placements.append((f"{kind}: {PEOPLE[label]} ({cl['path'].relative_to(ROOT)})", at, at + len(x) / SR))
        for t, x in seg[kind]:
            place(mix, x, t)
        m0 = min(t for t, _ in seg[kind])
        m1 = max(t + len(x) * 1000 / SR for t, x in seg[kind])
        placements.append((f"{kind}: model audio", starts[kind] / FPS + m0 / 1000, starts[kind] / FPS + m1 / 1000))
        track[starts[kind] * SPF: starts[kind] * SPF + len(mix)] += mix
    peak = float(np.abs(track).max())
    gain = PEAK / peak
    track *= gain
    print(f"audio: one gain {20 * math.log10(gain):+.2f} dB (raw peak {peak:.3f} -> -1 dBFS); "
          f"voiced rms {voiced_rms_db(track):.1f} dBFS")
    print("audio placements (s in the clip): " + "; ".join(f"{k} {a0:.2f}-{a1:.2f}" for k, a0, a1 in placements))

    def frames(stills: dict, half):
        for kind, n in parts:
            if kind in stills:
                buf = stills[kind].tobytes()
                for _ in range(n):
                    yield buf
            else:
                d = off if kind == "off" else on
                for i in range(n):
                    yield half(d, min(i * 1000 / FPS, d["closed_ms"])).tobytes()

    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mix_path = Path(tmp) / "mix.wav"
        pcm = np.clip(np.round(track * 32767), -32768, 32767).astype("<i2")
        with wave.open(str(mix_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        if a.format in ("16x9", "both"):
            encode(OUT_MP4, (W, H), mix_path, frames(wide, lambda d, t: render_half(d, t, axis)))
            vf = (f"fps={GIF_FPS},scale={GIF_W}:-1:flags=lanczos,split[a][b];"
                  "[a]palettegen=max_colors=256:stats_mode=full:reserve_transparent=0[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
            subprocess.run([FFMPEG, "-y", "-v", "error", "-i", str(OUT_MP4), "-an", "-vf", vf, str(OUT_GIF)],
                           check=True)
        if a.format in ("feed", "both"):
            encode(OUT_FEED, (FW, FH), mix_path, frames(feed, render_feed_half))

    print("part starts (s): " + ", ".join(f"{k} {v / FPS:.2f}" for k, v in starts.items()))
    j = off["jobs"]
    print("key moments (s in the clip): "
          + ", ".join(f"without: commit {x['confirmation']} {starts['off'] / FPS + x['committed'] / 1000:.2f}"
                      for x in j)
          + f", with: cancel {starts['on'] / FPS + on['decision_ms'] / 1000:.2f}")
    for p in (OUT_MP4, OUT_GIF, OUT_FEED):
        if p.exists():
            print(f"{p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
