#!/usr/bin/env bash
# A trace leg: restart with TF_GLM_DFLASH2_TRACE=1 (plus any extra env), drive a decode workload,
# then harvest the per-window `dft ...` records from the container's own output and fit the cost line.
#
#   docs/traceleg.sh <label> [KEY=VALUE ...]
#
# The trace goes to rank 0's stdout, so it is read with `docker logs --since`, not from the snapshot
# that start.sh copied at launch. Fitting is docs/trace.py.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
LABEL=${1:?usage: traceleg.sh <label> [KEY=VALUE ...]}; shift
OUT=docs/speed; mkdir -p "$OUT"
LOG=$OUT/trace-$LABEL.log; : > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "### trace leg $LABEL  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
export TF_GLM_DFLASH2_TRACE=1
say "  set TF_GLM_DFLASH2_TRACE=1"
for kv in "$@"; do export "$kv"; say "  set $kv"; done

say "--- restart ---"
bash docs/safestart.sh >> "$LOG" 2>&1 || { say "FAILED: safestart (verified stop or health)"; exit 3; }
grep -hE "dflash2 windows:|CUDA sm_121 device 0" logs/current/rank0.log | tee -a "$LOG"

SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
say "--- workload (fixed prompts, greedy; widths are cycled by FORCE when set) ---"
# Enough windows for a fit at every width: four long replies at one stream, then a 2-stream pair so
# shared rounds (w=0) are represented as well as lone ones (w=1).
LOADS=1 TRIALS=2 MAXTOK=300 python3 -I docs/dec.py "$LABEL-trace" 2>&1 | tee -a "$LOG"
LOADS=2 TRIALS=1 MAXTOK=300 python3 -I docs/dec.py "$LABEL-trace2" 2>&1 | tee -a "$LOG"

say "--- harvest ---"
docker logs --since "$SINCE" glm53-spark > "$OUT/dft-$LABEL.log" 2>&1
say "  $(grep -c 'dft r=' "$OUT/dft-$LABEL.log" || true) dft records in $OUT/dft-$LABEL.log"
python3 -I docs/trace.py "$OUT/dft-$LABEL.log" "$LABEL" 2>&1 | tee -a "$LOG"
say "### trace leg $LABEL done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
