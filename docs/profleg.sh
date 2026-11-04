#!/usr/bin/env bash
# A profiling leg: restart with TF_GLM_PROF=1, drive decode and a cold prefill, harvest `glm prof`.
# Profiling perturbs timing, so this leg is for attribution only - never for a throughput gate.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
LABEL=${1:?usage: profleg.sh <label> [KEY=VALUE ...]}; shift
OUT=docs/speed; mkdir -p "$OUT"; LOG=$OUT/prof-$LABEL.log; : > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
say "### prof leg $LABEL  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
export TF_GLM_PROF=1; say "  set TF_GLM_PROF=1"
for kv in "$@"; do export "$kv"; say "  set $kv"; done
bash docs/safestart.sh >> "$LOG" 2>&1 || { say "FAILED: safestart (verified stop or health)"; exit 3; }
SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
say "--- decode, 1 stream then 4 ---"
LOADS=1 TRIALS=2 MAXTOK=300 python3 -I docs/dec.py "$LABEL-prof1" 2>&1 | tee -a "$LOG"
LOADS=4 TRIALS=1 MAXTOK=300 python3 -I docs/dec.py "$LABEL-prof4" 2>&1 | tee -a "$LOG"
say "--- cold prefill (fresh content) ---"
python3 -I docs/pf2.py 2>&1 | tee -a "$LOG"
say "--- harvest ---"
docker logs --since "$SINCE" glm53-spark > "$OUT/profraw-$LABEL.log" 2>&1
say "  $(grep -c 'glm prof' "$OUT/profraw-$LABEL.log" || true) prof reports"
python3 -I docs/prof.py "$OUT/profraw-$LABEL.log" 2>&1 | tee -a "$LOG"
say "### prof leg $LABEL done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
