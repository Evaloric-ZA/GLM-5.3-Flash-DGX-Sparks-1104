#!/bin/bash
# Pair B (spark1+2) brought up on the Python engine (ENGINE=python, TensorFold v0.6.0 image) at the user's request, then
# the same harness from spark5 against B and against A (spark3+4, image 1fdae27149cc) in the same hour.
OUT=/home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig/docs/speed/python-vs
echo "[B] stop zig engine $(date -u +%H:%M:%SZ)"
ssh -o BatchMode=yes spark1 "cd ~/newMod/GLM-5.3-Flash-EXL3-TensorFold && PATH=\$HOME/.local/bin:\$PATH ./stop.sh" > $OUT/stop-B.log 2>&1
echo "[B] start python engine $(date -u +%H:%M:%SZ)"
ssh -o BatchMode=yes spark1 "cd ~/newMod/GLM-5.3-Flash-EXL3-TensorFold && PATH=\$HOME/.local/bin:\$PATH ENGINE=python nohup setsid ./start.sh > ~/start-python.log 2>&1 < /dev/null & sleep 2; echo started"
for i in $(seq 1 240); do ssh -o BatchMode=yes spark1 "curl -sf -m 5 http://localhost:8888/health" 2>/dev/null | grep -q '"status": "ok"' && break; sleep 10; done
echo "[B] health after ~$((i*10)) s: $(ssh -o BatchMode=yes spark1 'curl -s -m 5 http://localhost:8888/health | head -c 160; echo; docker ps --format "{{.Names}} {{.Image}} {{.Status}}" | head -1; tail -3 ~/start-python.log' | cut -c1-200)"
sleep 30
echo "--- B greedy";  ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc.py http://spark1:8888 GLM-5.3-Flash-EXL3 B-python 2" 2>&1 | tee $OUT/acc-B-python.log
echo "--- B sampled"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc_sampled.py http://spark1:8888 GLM-5.3-Flash-EXL3 B-python 2 0.7" 2>&1 | tee $OUT/sampled-B-python.log
echo "--- A greedy";  ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc.py http://spark3:8888 GLM-5.3-Flash A-104 2" 2>&1 | tee $OUT/acc-A-104.log
echo "--- A sampled"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc_sampled.py http://spark3:8888 GLM-5.3-Flash A-104 2 0.7" 2>&1 | tee $OUT/sampled-A-104.log
echo "--- B bench2";  ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && timeout 3000 python3 bench2.py http://spark1:8888 GLM-5.3-Flash-EXL3 B-python" 2>&1 | tee $OUT/bench2-B-python.log
echo "--- B 6/8 streams"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && timeout 900 python3 acc8.py http://spark1:8888 GLM-5.3-Flash-EXL3 B-python" 2>&1 | tail -6 | tee $OUT/acc8-B-python.log
echo PYVS_DONE
