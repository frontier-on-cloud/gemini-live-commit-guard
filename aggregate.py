"""Rebuild results/summary.md from results/*.jsonl and add the before/after table.

    uv run aggregate.py [--harness-results ../gemini-live-stop-test/results]

The "off, stop-test" rows rescore the audio sessions of the stop-test harness
(2026-09-29/30, no guard) with the same metric code and TruthCheck rules, from the
`run_end` summaries in its JSONL files. No API calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from commit_guard import TruthCheck
from guard_test import (HERE, first_claim, fmt_cuts, md, note_cuts, post_stop_truth,
                        scenario_table, user_interrupt_ms)

ORDER = ["A", "C", "G2", "F"]
LABELS = {
    "A": "A: default (NON_BLOCKING), stop 1.0 s after the call, 4 s service",
    "C": "C: BLOCKING, stop 1.0 s after the call, 4 s service",
    "G2": "G2: model speaking during a pending NON_BLOCKING call (7 s), stop 1 s into its speech",
    "F": "F: stop 0.3 s after the end of the booking clip, 4 s service",
}
HARNESS_FILES = {
    "A": ["audio_A_default_stop1.0_honor.jsonl"],
    "C": ["audio_C_blocking_stop1.0_honor.jsonl", "audio_C_rerun_blocking_stop1.0_honor.jsonl"],
    "G2": ["audio_G2_default_speechstop1.0_lat7.0_honor.jsonl"],
    "F": ["audio_F_default_stopend0.3_honor.jsonl"],
}
METRICS = ["unwanted_commit", "double_booking", "model_statement_consistent",
           "first_statement_consistent", "cancellation_before_commit"]
# `behaviors` column (hold/dedupe/inject/abandon) -> ablation label
CONFIGS = [("----", "guard off"), ("HDIA", "full guard"), ("HD-A", "no inject"),
           ("-DI-", "no hold, no abandon")]


def read_jsonl(path: Path) -> tuple[dict[str, Any] | None, list[dict], str]:
    """Scenario meta, one summary row per run, first wall time. The TruthCheck claim
    and the note-cut column are recomputed from the logged turns and raw events, so
    every row uses the current rules."""
    meta, rows, wall = None, [], ""
    events: dict[int, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if obj["event"] == "scenario_meta":
            meta = obj
        elif obj["event"] == "run_start" and not wall:
            wall = obj.get("wall", "")
        elif obj["event"] == "run_end":
            rows.append(obj["summary"])
        elif obj["t_ms"] >= 0:
            events.setdefault(obj["run"], []).append(obj)
    for row in rows:
        if "model_turns" in row:
            truth = post_stop_truth(row["model_turns"], row.get("stop_audio_end_ms"),
                                    bool(row.get("commits")))
            row.update(claim=truth.claim, statement_used=truth.statement,
                       model_statement_consistent=str(truth.consistent).lower())
            fc = first_claim(row["model_turns"], row.get("stop_audio_end_ms"),
                             bool(row.get("commits")))
            row.update(first_claim=fc[0], first_statement_consistent=str(fc[1]).lower(),
                       first_statement=fc[2])
            evs = events.get(row["run"], [])
            row["note_cuts"] = note_cuts(evs)
            row["note_cut_reply"] = fmt_cuts(row["note_cuts"])
            intr = user_interrupt_ms(evs, row.get("stop_sent_at_ms"))
            row["interrupted_ms"] = intr if intr is not None else "no"
    return meta, rows, wall


def rescore_harness(path: Path) -> list[dict]:
    """Stop-test raw events -> the same metrics as guard_test.summarize(). Model turns
    are rebuilt like the harness does it: a new turn after interrupted/turn_complete."""
    runs: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        r = runs.setdefault(e["run"], {"commits": [], "turns": {}, "turn": 0,
                                       "stop": None, "end": None})
        ev, t = e["event"], e["t_ms"]
        if ev == "user_audio_start" and e.get("label") == "stop":
            r["stop"] = t
        elif ev == "user_audio_end" and e.get("label") == "stop":
            r["end"] = t
        elif ev == "service_committed":
            r["commits"].append(e.get("committed_at", t))
        elif ev in ("model_transcript", "model_text"):
            turn = r["turns"].setdefault(r["turn"], {"start_ms": t, "text": ""})
            turn["text"] += e["text"]
        elif ev in ("interrupted", "turn_complete"):
            r["turn"] += 1
    out = []
    for run, r in sorted(runs.items()):
        if r["stop"] is None or r["end"] is None:
            continue
        after = [x["text"].strip() for x in r["turns"].values() if x["start_ms"] >= r["end"]]
        truth = TruthCheck().check(after, committed=bool(r["commits"]))
        fc = first_claim(list(r["turns"].values()), r["end"], bool(r["commits"]))
        out.append({
            "file": path.name, "run": run, "commits": len(r["commits"]),
            "unwanted_commit": "yes" if any(c > r["stop"] for c in r["commits"]) else "no",
            "double_booking": "yes" if len(r["commits"]) > 1 else "no",
            "model_statement_consistent": str(truth.consistent).lower(),
            "first_statement_consistent": str(fc[1]).lower(),
            "cancellation_before_commit": "no",
            "claim": truth.claim, "statement": truth.statement,
        })
    return out


def count(rows: list[dict], metric: str) -> str:
    yes = sum(1 for r in rows if r.get(metric) in ("yes", "true"))
    return f"{yes}/{len(rows)}"


def decision_ms(rows: list[dict]) -> str:
    vals = [r["time_from_stop_to_decision_ms"] for r in rows
            if isinstance(r.get("time_from_stop_to_decision_ms"), int)]
    if not vals:
        return "-"
    return f"{int(statistics.median(vals))} ({min(vals)}-{max(vals)})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument("--harness-results",
                   default=str(HERE.parent / "gemini-live-stop-test" / "results"))
    a = p.parse_args()
    results = Path(a.results_dir)
    files = []
    for path in sorted(results.glob("*.jsonl")):
        meta, rows, wall = read_jsonl(path)
        files.append((wall, path, meta, rows))
    files.sort(key=lambda f: f[0])
    sessions = sum(len(rows) for *_, rows in files)

    lines = ["# live-commit-guard results", "",
             f"Rebuilt by `aggregate.py` on {datetime.now().strftime('%Y-%m-%d %H:%M')} from "
             f"`results/*.jsonl`: {sessions} Live API sessions in {len(files)} files. "
             "Speech input (macOS `say` clips, 16 kHz PCM, 100 ms chunks in real time, server "
             "VAD default), AUDIO output with transcription, Google AI Studio endpoint.", "",
             "Metric definitions: `unwanted_commit` = a booking committed after the stop clip "
             "had started. `double_booking` = more than one commit for the same business key. "
             "`cancellation_before_commit` = the guard cancelled the job and nothing was "
             "committed. `model_statement_consistent` = TruthCheck on the model turns that "
             "start after the last chunk of the stop clip, joined (a turn cut by an "
             "interruption continues in the next one), last claim wins, compared with the fake "
             "service; no claim counts as not consistent. It classifies the transcript, not "
             "the number of bookings: C off runs that double-booked and said \"the booking "
             "was already made\" count as consistent. `first_statement_consistent` = the same "
             "check on the FIRST post-stop claim (shortest run of joined turns that makes a "
             "claim): without a status note the model's first reply can be wrong and a later "
             "turn correct it. `time_from_stop_to_decision_ms` = first "
             "guard decision minus the first chunk of the stop clip (with the guard off, the "
             "decision is the automatic commit). `interrupted_ms` = first `interrupted` after "
             "the stop that the guard's own note did not cause. `note_cut_reply` = a model "
             "turn that had already produced text when the `interrupted` caused by the "
             "guard's note arrived, with how much of its audio had been received; a client "
             "that drops queued audio on `interrupted` plays at most that much of it. "
             "`aggregate.py` recomputes `claim`, `model_statement_consistent`, `first_claim`, "
             "`first_statement_consistent`, `note_cut_reply` and `interrupted_ms` from the "
             "logged events, so every row uses "
             "the current rules.", "",
             "## Per-run tables", ""]
    for wall, path, meta, rows in files:
        text = meta["meta"] if meta else "(no scenario_meta event)"
        lines.append(scenario_table(path.stem, f"{text} First run started {wall}.", rows))

    # ---------------------------------------------------------- before/after --
    harness = Path(a.harness_results)
    rescored: dict[str, list[dict]] = {}
    for sc, names in HARNESS_FILES.items():
        rescored[sc] = [r for n in names if (harness / n).exists()
                        for r in rescore_harness(harness / n)]
    lines += ["## Before / after", "",
              "Counts are runs out of sessions. \"off, stop-test\" rescores the existing "
              "stop-test harness sessions (no guard, same prompt, clips, latency and stop "
              "timing; C includes its re-run). \"off\" and \"on\" are this harness with "
              "`--guard off` / `--guard on` (all four behaviors; ablation runs are in the "
              "next section).", "",
              "| scenario | mode | sessions | unwanted commit | double booking | statement "
              "consistent (last claim) | first claim consistent | cancelled before commit | "
              "stop to decision, ms (median, range) | note cut a started reply |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    mine: dict[tuple[str, str], list[dict]] = {}
    for *_, rows in files:
        for r in rows:
            mine.setdefault((str(r.get("scenario")), str(r.get("behaviors"))), []).append(r)
    for sc in ORDER:
        for mode, rows in (("off, stop-test", rescored.get(sc, [])),
                           ("off", mine.get((sc, "----"), [])),
                           ("on", mine.get((sc, "HDIA"), []))):
            if not rows:
                lines.append(f"| {md(LABELS[sc])} | {mode} | 0 | not run | | | | | | |")
                continue
            lines.append(f"| {md(LABELS[sc])} | {mode} | {len(rows)} | "
                         + " | ".join(count(rows, m) for m in METRICS)
                         + f" | {decision_ms(rows) if mode == 'on' else '-'} | "
                         + (f"{sum(1 for r in rows if r.get('note_cuts'))}/{len(rows)}"
                            if mode == "on" else "-") + " |")
    # ------------------------------------------------------------- ablations --
    lines += ["", "## Ablations", "",
              "Guard on with single behaviors switched off (`behaviors` = H/D/I/A for "
              "hold/dedupe/inject/abandon). \"no inject\": `--no-inject`. \"no hold, no "
              "abandon\": `--no-hold --no-abandon`; abandon has to go too, because in BLOCKING "
              "mode `interrupted` always arrived while the call was pending and behavior 4 "
              "would hold the commit on its own. Guard off and full guard rows repeat the "
              "runs above for comparison. Claims count the TruthCheck classes booked / not "
              "booked / neither.", "",
              "| config | behaviors | scenario | n | unwanted commit | double booking | "
              "statement consistent (last claim) | first claim consistent | claims, last "
              "(booked / not booked / neither) | duplicates suppressed | status notes sent |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    for code, label in CONFIGS:
        for sc in ORDER:
            rows = mine.get((sc, code), [])
            if not rows or not any(mine.get((sc, c)) for c in ("HD-A", "-DI-")):
                continue
            claims = [sum(1 for r in rows if r.get("claim") == c)
                      for c in ("claims_booked", "claims_not_booked", "neither")]
            lines.append(f"| {label} | {code} | {sc} | {len(rows)} | "
                         + " | ".join(count(rows, m) for m in METRICS[:4])
                         + f" | {' / '.join(map(str, claims))} | "
                         f"{sum(int(r.get('duplicates_suppressed') or 0) for r in rows)} | "
                         f"{sum(len(r.get('notes') or []) for r in rows)} |")
            if code in ("HD-A", "-DI-"):
                detail += rows
    lines += ["", "### Ablation runs", "",
              "| scenario | behaviors | run | n_tool_calls | service | tool_responses | notes | "
              "note_cut_reply | first claim | last claim | last consistent | model after the "
              "stop clip (joined) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in detail:
        notes = "; ".join(f"{n['at_ms']}: {n['text']}" for n in r.get("notes") or []) or "none"
        lines.append(f"| {r['scenario']} | {r['behaviors']} | {r['run']} | {r['n_tool_calls']} | "
                     f"{md(r['service'])} | {md(r['tool_responses'])} | {md(notes)} | "
                     f"{md(r['note_cut_reply'])} | {r['first_claim']} | {r['claim']} | "
                     f"{r['model_statement_consistent']} | {md(r['statement_used']) or '-'} |")

    lines += ["", "### Stop-test sessions rescored", "",
              "| scenario | file | run | commits | unwanted commit | double booking | claim | "
              "consistent | statement used |", "|---|---|---|---|---|---|---|---|---|"]
    for sc in ORDER:
        for r in rescored.get(sc, []):
            lines.append(f"| {sc} | {r['file']} | {r['run']} | {r['commits']} | "
                         f"{r['unwanted_commit']} | {r['double_booking']} | {r['claim']} | "
                         f"{r['model_statement_consistent']} | {md(r['statement'][:160]) or '-'} |")
    (results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[lines.index("## Before / after"):]))


if __name__ == "__main__":
    main()
