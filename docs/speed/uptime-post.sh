#!/usr/bin/env bash
# The post-reboot battery for spark3. Run ONCE, right after the box comes back (the pair starts down).
#   bash docs/speed/uptime-post.sh
# Three legs, ~18 min:
#   1. the traced bimodal probe  - a fresh boot is the biggest unified-memory fill there is; if the
#      coverage wobble is fill churn, it shows up HERE, with the per-window tracer on;
#   2. the bf16 uptime leg       - the A-arm comparison against 37.42 / 52.92 / 68.89 (07:54-08:07Z,
#      settled box) and against the +1.5% post-power-cycle claim; settled by leg 1's churn;
#   3. the shipped q8 leg        - the ship state on a fresh pair; expect ~38.6 / 54.1 / 70.2.
# The pair is left serving the shipped configuration.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
LOG=docs/speed/uptime-post.log; : > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

say "### uptime post-reboot battery  $(date -u +%Y-%m-%dT%H:%M:%SZ)  uptime: $(uptime -s 2>/dev/null || echo '?')"
say "--- leg 1/3: traced bimodal probe (fill-churn capture) ---"
bash docs/speed/bimodal-probe.sh 2>&1 | tail -25 | tee -a "$LOG" || say "  probe FAILED"

say "--- leg 2/3: bf16 uptime leg (settled) ---"
bash docs/leg.sh upt-bf16 HEAD=bf16 2>&1 | tail -14 | tee -a "$LOG" || say "  bf16 leg FAILED"

say "--- leg 3/3: shipped q8 leg ---"
bash docs/leg.sh upt-q8 2>&1 | tail -14 | tee -a "$LOG" || say "  q8 leg FAILED"

say "--- state ---"
docker ps --filter name=glm53-spark --format 'serving {{.Image}} {{.Status}}' | tee -a "$LOG"
curl -s -m 5 http://127.0.0.1:8888/health | head -c 120; echo
say "### battery done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "compare: dec-upt-bf16 vs dec-head-a1/head-a2 (37.42/52.92/68.89) and dec-upt-q8 vs dec-head-b (38.59/54.14/70.20)"
