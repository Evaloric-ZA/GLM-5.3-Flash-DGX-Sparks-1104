#!/usr/bin/env bash
# The coverage-bench bimodality probe. The 10-trial 4-stream coverage block rounds bimodally
# (modes ~110 and ~150-185 ms) while the leg sweep after the same restarts sits at 108-112 ms.
# Question posed by the user: instrument noise, or a real serving regime?
# Design: restart with TF_GLM_DFLASH2_TRACE=1, run the exact bimodal block, then the stable
# leg sweep, sampling free memory and GPU clocks throughout; the dft records say whether slow
# rounds are the same windows taking longer or a different window/rows mix.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
OUT=docs/speed; LOG=$OUT/bimodal-probe.log; : > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "### bimodal probe  $(date -u +%H:%M:%SZ)"
export TF_GLM_DFLASH2_TRACE=1
say "  set TF_GLM_DFLASH2_TRACE=1 (per-window dft records on rank 0's stdout)"
bash docs/safestart.sh >> "$LOG" 2>&1 || { say "FAILED: safestart"; exit 3; }
grep -hE 'CUDA sm_121 device 0' logs/current/rank0.log | sed 's/^/  /' | tee -a "$LOG"
docker ps --filter name=glm53-spark --format '  image={{.Image}}' | tee -a "$LOG"

say "--- samplers (2 s: free -m + nvidia-smi) ---"
SAMPLER=$OUT/bimodal-sampler.csv
echo 't,free_mb,avail_mb,used_gpu_mb,sm_clock,temp,gpu_util' > "$SAMPLER"
(
  while :; do
    f=$(free -m | awk '/^Mem:/{print $4","$7}')
    g=$(nvidia-smi --query-gpu=memory.used,clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sed 's/\[N\/A\]/0/g')
    echo "$(date +%s.%N),$f,$g"
    sleep 2
  done
) > "$SAMPLER" 2>/dev/null &
SAMPID=$!
trap 'kill $SAMPID 2>/dev/null' EXIT

SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
T0=$(date +%s.%N)
say "--- the bimodal block: 4 streams x 10 trials, exactly as coverage.sh runs it ---"
LOADS=4 TRIALS=10 python3 -I docs/dec.py bimodal-c4 2>&1 | tee -a "$LOG"
T1=$(date +%s.%N)
say "--- the stable instrument, same start: the leg-sweep 1/2/4 x3 ---"
LOADS=1,2,4 python3 -I docs/dec.py bimodal-sweep 2>&1 | tee -a "$LOG"
sleep 3
kill $SAMPID 2>/dev/null

say "--- harvest dft records since $SINCE ---"
docker logs --since "$SINCE" --timestamps glm53-spark > "$OUT/dft-bimodal.log" 2>&1
say "  $(grep -c 'dft r=' "$OUT/dft-bimodal.log") dft records; block ended $(date -u -d @$T1 '+%H:%M:%SZ' 2>/dev/null || echo '')"
say "  trial span T0=$T0 T1=$T1 (epoch s); sampler in $SAMPLER"
say "### probe done  $(date -u +%H:%M:%SZ)"
