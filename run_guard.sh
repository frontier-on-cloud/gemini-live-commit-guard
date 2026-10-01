#!/usr/bin/env bash
# Run the guard matrix with speech input, N=3 sessions per scenario, in priority order:
# A, C, G2, F with --guard on, then the same four with --guard off. Then rebuild
# results/summary.md (aggregate.py). At most MAX_SESSIONS sessions (default 24).
#
#   ./run_guard.sh              # all eight scenarios
#   ./run_guard.sh A_on C_off   # a subset; names are <scenario>_<mode>
#
# Modes: on, off, and two ablations (guard on with behaviors switched off):
#   noinject           --no-inject: hold, dedupe, abandon on; no status note
#   nohold_noabandon   --no-hold --no-abandon: dedupe and inject on, the commit goes
#                      through (in BLOCKING mode abandon would otherwise hold it)
#   ./run_guard.sh A_noinject C_noinject G2_noinject C_nohold_noabandon
#
# Earlier results/<name>.jsonl files of the scenarios being run are moved to
# results/archive-<timestamp>/ (never deleted) unless APPEND=1. Stops on a quota or
# billing error (guard_test.py exit status 3).
set -uo pipefail
cd "$(dirname "$0")"
N="${N:-3}"
MAX_SESSIONS="${MAX_SESSIONS:-24}"

if [ -z "${GEMINI_API_KEY:-}" ] && ! grep -qs '^GEMINI_API_KEY=.' .env; then
  echo "GEMINI_API_KEY is not set (put it in .env next to this script). Nothing was run." >&2
  exit 2
fi

args_for() {
  case "$1" in
    A)  echo "--stop-after 1.0 --latency 4.0" ;;
    C)  echo "--behavior BLOCKING --stop-after 1.0 --latency 4.0" ;;
    G2) echo "--followup-audio assets/audio/bring.wav --followup-after-tool-call 0.5
              --stop-after-model-speech 1.0 --latency 7.0" ;;
    F)  echo "--stop-after-request 0.3 --min-post-stop 5 --latency 4.0" ;;
    *)  return 1 ;;
  esac
}

if [ $# -eq 0 ]; then
  set -- A_on C_on G2_on F_on A_off C_off G2_off F_off
fi

if [ "${APPEND:-0}" != "1" ]; then
  archive="results/archive-$(date +%Y%m%d-%H%M%S)"
  for name in "$@"; do
    if [ -f "results/$name.jsonl" ]; then
      mkdir -p "$archive" && mv "results/$name.jsonl" "$archive"/
    fi
  done
fi

used=0
for name in "$@"; do
  sc="${name%%_*}"; mode="${name#*_}"
  case "$mode" in
    on|off)           guard="$mode"; flags="" ;;
    noinject)         guard=on; flags="--no-inject" ;;
    nohold_noabandon) guard=on; flags="--no-hold --no-abandon" ;;
    *)                guard="" ;;
  esac
  if ! extra=$(args_for "$sc") || [ -z "$guard" ]; then
    echo "unknown scenario name: $name (expected A|C|G2|F, then _on, _off, _noinject" \
         "or _nohold_noabandon)" >&2
    continue
  fi
  left=$((MAX_SESSIONS - used))
  if (( left <= 0 )); then
    echo "session budget ($MAX_SESSIONS) used up: skipping $name"
    continue
  fi
  n=$(( N < left ? N : left ))
  echo
  echo "=== $name (N=$n) ==="
  # shellcheck disable=SC2086
  uv run guard_test.py --name "$name" --scenario "$sc" --guard "$guard" -n "$n" $flags $extra
  status=$?
  used=$((used + n))
  if (( status == 3 )); then
    echo "quota/billing error in $name: stopping"
    break
  elif (( status != 0 )); then
    echo "$name exited with status $status (continuing)"
  fi
done

echo
echo "sessions started by this script: at most $used"
uv run aggregate.py
