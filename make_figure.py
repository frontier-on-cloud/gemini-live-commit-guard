"""Before/after figure: the commit guard off and on, plus the "no status note" ablation.

    uv run --with matplotlib python make_figure.py

Reads the per-run tables in results/summary.md (rebuilt by aggregate.py) and writes
results/figures/before-after.png (1600x900). Each dot is one run, in run order.
Nothing is typed in by hand: the script also checks the per-run values against the
summary tables and against the run_end records in results/*.jsonl, and exits with
status 1 if anything disagrees or if two text labels overlap.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary.md"
OUT = RESULTS / "figures" / "before-after.png"

# Palette: two categorical slots (validated light, all pairs) + grey chrome.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
RULE, EMPTY = "#e1e0d9", "#d4d3cc"
HARM, GOOD = "#eb6834", "#2a78d6"  # orange: unwanted side effect; blue: reply matched state

SCENARIOS = ["A", "C", "G2", "F"]  # described in the README and the article
FOOTER = ("gemini-3.8-live, speech input, N=3 per cell, 2026-10-01. "
          "github.com/frontier-on-cloud/gemini-live-commit-guard")

W, H = 1600, 900
R, PITCH = 20, 54            # dot radius and spacing, px
COL = {"off": 320, "on": 600, "nonote": 930, "full": 1230}   # centre of each dot triple
ROW_Y = {"A": 312, "C": 398, "G2": 484, "F": 570}
DOUBLE_Y = 742
LABEL_X = 64

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK,
})


# ------------------------------------------------------------------ data --


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def per_run_tables():
    """{'A_on': [{'run': '1', 'unwanted_commit': 'no', ...}, ...], ...} from summary.md."""
    tables, name, header = {}, None, None
    for line in SUMMARY.read_text().splitlines():
        m = re.match(r"^### (\w+)$", line)
        if m:
            name, header = m.group(1), None
            continue
        if line.startswith("## "):
            name = None
        if name is None or not line.startswith("|"):
            continue
        cells = split_row(line)
        if header is None:
            header = cells
        elif not set(cells[0]) <= {"-"}:
            tables.setdefault(name, []).append(dict(zip(header, cells)))
    return tables


def section_table(title):
    """First table under '## <title>' as a list of dicts."""
    text = SUMMARY.read_text().split(f"\n## {title}\n", 1)[1]
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    header = split_row(lines[0])
    out = []
    for ln in lines[2:]:
        if not ln.startswith("|"):
            break
        out.append(dict(zip(header, split_row(ln))))
    return out


def jsonl_summaries(name):
    rows = [json.loads(ln) for ln in (RESULTS / f"{name}.jsonl").read_text().splitlines() if ln.strip()]
    return {r["summary"]["run"]: r["summary"] for r in rows if r["event"] == "run_end"}


def load():
    tables = per_run_tables()
    problems = []

    def runs(name, col, true_value):
        rows = sorted(tables[name], key=lambda r: int(r["run"]))
        if len(rows) != 3:
            problems.append(f"{name}: {len(rows)} runs, expected 3")
        return [r[col] == true_value for r in rows]

    data = {"unwanted": {}, "double": {}, "first": {}}
    for sc in SCENARIOS:
        for mode in ("off", "on"):
            data["unwanted"][(sc, mode)] = runs(f"{sc}_{mode}", "unwanted_commit", "yes")
            data["double"][(sc, mode)] = runs(f"{sc}_{mode}", "double_booking", "yes")
    for sc in ("A", "C", "G2"):
        data["first"][(sc, "nonote")] = runs(f"{sc}_noinject", "first_statement_consistent", "true")
        data["first"][(sc, "full")] = runs(f"{sc}_on", "first_statement_consistent", "true")

    # Cross-check 1: totals in the summary's own Before / after and Ablations tables.
    for row in section_table("Before / after"):
        sc, mode = row["scenario"].split(":")[0], row["mode"]
        if mode not in ("off", "on"):
            continue
        for metric, col in (("unwanted", "unwanted commit"), ("double", "double booking")):
            got = f"{sum(data[metric][(sc, mode)])}/3"
            if row[col] != got:
                problems.append(f"Before/after {sc} {mode} {col}: table {row[col]}, runs {got}")
    for row in section_table("Ablations"):
        mode = {"no inject": "nonote", "full guard": "full"}.get(row["config"])
        if mode is None or (row["scenario"], mode) not in data["first"]:
            continue
        got = f"{sum(data['first'][(row['scenario'], mode)])}/3"
        if row["first claim consistent"] != got:
            problems.append(f"Ablations {row['scenario']} {row['config']}: table "
                            f"{row['first claim consistent']}, runs {got}")

    # Cross-check 2: the raw run_end records in the JSONL files.
    for (sc, mode), values in data["unwanted"].items():
        raw = jsonl_summaries(f"{sc}_{mode}")
        for metric, field in (("unwanted", "unwanted_commit"), ("double", "double_booking")):
            from_jsonl = [raw[i + 1][field] == "yes" for i in range(3)]
            if from_jsonl != data[metric][(sc, mode)]:
                problems.append(f"{sc}_{mode} {field}: summary.md and JSONL disagree")
    return data, problems


# ---------------------------------------------------------------- drawing --


def dots(ax, cx, y, values, color):
    for i, hit in enumerate(values):
        ax.add_patch(Circle((cx + (i - 1) * PITCH, y), R, facecolor=color if hit else EMPTY,
                            edgecolor="none", zorder=3))
    return ax.text(cx + PITCH + R + 16, y, f"{sum(values)}/{len(values)}", ha="left",
                   va="center", fontsize=26, color=INK2)


def main():
    data, problems = load()

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    labels = []

    def text(x, y, s, **kw):
        t = ax.text(x, y, s, **kw)
        labels.append((s, t))
        return t

    total_off = sum(sum(v) for (sc, m), v in data["unwanted"].items() if m == "off")
    total_on = sum(sum(v) for (sc, m), v in data["unwanted"].items() if m == "on")
    n_cells = len(SCENARIOS) * 3
    text(LABEL_X, 60, f"Unwanted bookings: {total_off} of {n_cells} off, {total_on} of {n_cells} on",
         fontsize=34, fontweight="bold", va="center")
    text(LABEL_X, 116, "Each dot is one run (filled: it happened). Right: guard on, status note off.",
         fontsize=21, color=INK2, va="center")

    left = {k: COL[k] - PITCH - R for k in COL}
    text(left["off"], 186, "Booked after the stop", fontsize=26, fontweight="bold", va="center")
    text(left["nonote"], 186, "First reply matched the booking state", fontsize=26, fontweight="bold",
         va="center")
    for key, label in (("off", "guard off"), ("on", "guard on"),
                       ("nonote", "no status note"), ("full", "full guard")):
        text(left[key], 238, label, fontsize=24, color=INK2, va="center")
    divider_x = (left["on"] + 2 * PITCH + 2 * R + 100 + left["nonote"]) / 2
    ax.plot([divider_x, divider_x], [160, 610], color=RULE, lw=2, zorder=1)
    ax.plot([LABEL_X, W - LABEL_X], [268, 268], color=RULE, lw=2, zorder=1)

    for sc in SCENARIOS:
        y = ROW_Y[sc]
        text(LABEL_X, y, sc, fontsize=32, fontweight="bold", va="center")
        for mode in ("off", "on"):
            labels.append((f"{sc} {mode}", dots(ax, COL[mode], y, data["unwanted"][(sc, mode)], HARM)))
        for mode in ("nonote", "full"):
            if (sc, mode) in data["first"]:
                labels.append((f"{sc} {mode}", dots(ax, COL[mode], y, data["first"][(sc, mode)], GOOD)))
            else:
                text(left[mode], y, "not run", fontsize=22, color=MUTED, va="center")

    # Double booking, C only.
    ax.plot([LABEL_X, W - LABEL_X], [628, 628], color=RULE, lw=2, zorder=1)
    text(left["off"], 680, "Double booking", fontsize=26, fontweight="bold", va="center")
    text(LABEL_X, DOUBLE_Y, "C", fontsize=32, fontweight="bold", va="center")
    for mode in ("off", "on"):
        labels.append((f"C double {mode}", dots(ax, COL[mode], DOUBLE_Y, data["double"][("C", mode)], HARM)))

    text(LABEL_X, 852, FOOTER, fontsize=18, color=INK2, va="center")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(name, t.get_window_extent(renderer)) for name, t in labels]
    for i, (na, a) in enumerate(boxes):
        if a.x0 < 0 or a.x1 > W or a.y0 < 0 or a.y1 > H:
            problems.append(f"label runs off the figure: {na!r}")
        for nb, b in boxes[i + 1:]:
            if a.overlaps(b):
                problems.append(f"labels overlap: {na!r} / {nb!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(ROOT)}")
    for p in problems:
        print("problem:", p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
