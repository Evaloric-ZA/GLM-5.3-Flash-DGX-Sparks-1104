#!/usr/bin/env bash
# One coverage pass against whatever is currently serving. Does NOT restart anything.
#   coverage.sh <label>
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
L=${1:?usage: coverage.sh <label>}
echo "=== coverage pass $L  $(date -u +%H:%M:%SZ) ==="
docker ps --filter name=glm53-spark --format '  serving image={{.Image}}'
echo "--- 4 streams, 10 trials (resolve the sub-1% question) ---"
LOADS=4 TRIALS=10 python3 -I docs/dec.py "$L-c4"
echo "--- long-context decode, ~45K in the window, 1 and 4 streams ---"
CTXWORDS=12000 LOADS=1,4 TRIALS=3 python3 -I docs/dec.py "$L-ctx"
echo "--- mixed prefill + decode ---"
python3 -I docs/mixed.py "$L"
echo "--- contract corpus ---"
python3 -I docs/dec.py "$L" --contract
echo "=== coverage pass $L done  $(date -u +%H:%M:%SZ) ==="
