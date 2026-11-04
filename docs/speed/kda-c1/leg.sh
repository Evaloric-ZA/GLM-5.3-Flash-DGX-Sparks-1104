#!/bin/bash
# One A/B leg for 0103 on pair A (run on spark3): restart the pair with the given env, wait for health, run acc.py from
# spark5 (the client host), harvest rank 0's TF_GLM_PROF reports since the restart, summarise the kda section by label.
# Usage: docs/speed/kda-c1/leg.sh <name> [VAR=value ...]      e.g. leg.sh off1 TF_GLM_PROF=1 TF_GLM_PDL_CHAIN=reduce,hc,p2p,moe
set -uo pipefail
name=$1; shift
REPO=/home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig
OUT=$REPO/docs/speed/kda-c1
cd "$REPO"
t0=$(date -u +%Y-%m-%dT%H:%M:%S)
echo "[$name] restart at $t0 with: $*"
env "$@" ./start.sh restart > "$OUT/restart-$name.log" 2>&1 || { echo "[$name] start.sh restart exit $?"; tail -20 "$OUT/restart-$name.log"; }
for i in $(seq 1 180); do curl -sf -m 5 http://localhost:8888/health 2>/dev/null | grep -q '"warming": false' && break; sleep 5; done
echo "[$name] healthy after ~$((i*5)) s; image: $(docker inspect glm53-spark --format '{{.Config.Image}}'); env: $(docker inspect glm53-spark --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E 'TF_GLM_PROF|TF_GLM_PDL' | tr '\n' ' ')"
sleep 15
ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc.py http://spark3:8888 GLM-5.3-Flash $name 2 && python3 acc4only.py http://spark3:8888 GLM-5.3-Flash $name" 2>&1 | tee "$OUT/acc-$name.log"
sleep 5
docker logs --since "$t0" glm53-spark 2>&1 | grep "glm prof" > "$OUT/prof-$name.log" || true
python3 - "$OUT/prof-$name.log" "$name" <<'PY'
import sys, re, collections
acc = collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.search(r'glm prof (.+?) \((\d+) calls\): wall ([\d.]+) ms, gpu ([\d.]+) ms a call;(.*)', line)
    if not m: continue
    secs = dict(re.findall(r' (\w+) ([\d.]+)', m.group(5)))
    acc[m.group(1)].append((float(m.group(3)), float(m.group(4)), float(secs.get('kda', 'nan'))))
for label, v in sorted(acc.items()):
    n = len(v)
    print(f"[{sys.argv[2]}] prof {label}: {n} reports, wall {sum(x[0] for x in v)/n:.2f} ms, gpu {sum(x[1] for x in v)/n:.2f} ms, kda {sum(x[2] for x in v)/n:.3f} ms")
PY
echo "[$name] done $(date -u +%H:%M:%SZ)"
