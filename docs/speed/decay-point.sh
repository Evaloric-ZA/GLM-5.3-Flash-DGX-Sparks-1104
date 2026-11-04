#!/usr/bin/env bash
# One within-boot decay point: mean (1/2/4 x3, the battery-comparable sweep) PLUS the 10-trial 4-stream
# incidence block, on the SAME boot as every other point - no restarts inside the series. Appends one
# summary block to docs/speed/decay-log.txt and writes dec-decay<tag>-*.json. Slow-mode trial = round-ms
# mean over 130 (midpoint of the observed 110 / 150-185 modes).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
TAG=$1   # e.g. t+1h
BOOT=$2  # epoch of boot, for hours-since
LOG=docs/speed/decay-log.txt
{
echo "=== decay $TAG  $(date -u +%Y-%m-%dT%H:%M:%SZ)  uptime_h=$(python3 -c "print(f'{($(date +%s)-$BOOT)/3600:.2f}')")  load=$(cut -d' ' -f1-3 /proc/loadavg)  freeMB=$(free -m | awk '/^Mem:/{print $7}')"
LOADS=4 TRIALS=10 python3 -I docs/dec.py "decay$TAG-c4" 2>&1 | grep -E '^  -> '
LOADS=1,2,4    python3 -I docs/dec.py "decay$TAG"    2>&1 | grep -E '^  -> '
python3 - "$TAG" <<'PY'
import json, sys
tag = sys.argv[1]
try: d = json.load(open(f"docs/speed/dec-decay{tag}-c4.json"))["4"]
except Exception as e: print(f"  incidence: unreadable ({e})"); raise SystemExit
tr = d["trials"]; rms = [t["round_ms"][0] if t.get("round_ms") else None for t in tr]
slow = sum(1 for x in rms if x and x > 130)
tps = [t["total_tps"] for t in tr]
p95 = d.get("p95_round_ms"); mx = d.get("max_round_ms")
print(f"  incidence: slow-trials {slow}/{len(rms)} (round>130ms) | tps med {sorted(tps)[len(tps)//2]:.1f} spread {(max(tps)-min(tps))/sorted(tps)[len(tps)//2]*100:.1f}% | round p95 {p95} max {mx}")
PY
} >> "$LOG" 2>&1
echo "decay point $TAG done" >> "$LOG"
