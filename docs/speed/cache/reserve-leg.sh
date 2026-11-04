#!/bin/bash
# Reserve 8 -> 9.5 on pair A: restart, read the startup plan, replay the agent loop (cache hits), decode band + hashes,
# prefill/mixed (bench2), 8 streams (acc8); memory low-water on both Sparks before and after.
cd /home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig
OUT=docs/speed/cache
mem() { echo "[$1] spark3 MemAvailable $(grep MemAvailable /proc/meminfo | awk '{printf "%.1f", $2/1048576}') GiB; spark4 $(ssh -o BatchMode=yes zurih@10.0.34.4 "grep MemAvailable /proc/meminfo" | awk '{printf "%.1f", $2/1048576}') GiB"; }
mem before-restart
t0=$(date -u +%Y-%m-%dT%H:%M:%S)
./start.sh restart > $OUT/restart-reserve95.log 2>&1 || { echo "[reserve95] start.sh restart exit $?"; tail -20 $OUT/restart-reserve95.log; }
for i in $(seq 1 180); do curl -sf -m 5 http://localhost:8888/health 2>/dev/null | grep -q '"warming": false' && break; sleep 5; done
echo "[reserve95] healthy; image $(docker inspect glm53-spark --format '{{.Config.Image}}'); reserve env $(docker inspect glm53-spark --format '{{range .Config.Env}}{{println .}}{{end}}' | grep RESERVE)"
docker logs glm53-spark 2>&1 | grep -i "pages of\|prompt cache\|cannot keep" | grep -v "GET /" | cut -c1-500
mem after-start
sleep 10
echo "--- replay"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && timeout 900 python3 cache-replay.py http://spark3:8888" 2>&1 | tee $OUT/replay-reserve95.log
echo "--- acc"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc.py http://spark3:8888 GLM-5.3-Flash reserve95 2" 2>&1 | tee $OUT/acc-reserve95.log
echo "--- acc8"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && timeout 900 python3 acc8.py http://spark3:8888 GLM-5.3-Flash reserve95" 2>&1 | tail -6 | tee $OUT/acc8-reserve95.log
echo "--- bench2"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && timeout 2400 python3 bench2.py http://spark3:8888 GLM-5.3-Flash reserve95" 2>&1 | tee $OUT/bench2-reserve95.log
mem after-workload
curl -s -m 5 http://localhost:8888/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('[reserve95] health memory', d.get('memory'))"
echo RESERVE_DONE
