#!/bin/bash
# SHIP-OFF: pair A back to the 102-patch image, env-free, then the standing-band check from spark5.
cd /home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig
OUT=docs/speed/kda-c1
echo "[final] restart env-free at $(date -u +%H:%M:%SZ)"
./start.sh restart > $OUT/restart-final.log 2>&1 || { echo "[final] start.sh restart exit $?"; tail -20 $OUT/restart-final.log; }
for i in $(seq 1 180); do curl -sf -m 5 http://localhost:8888/health 2>/dev/null | grep -q '"warming": false' && break; sleep 5; done
echo "[final] healthy; image: $(docker inspect glm53-spark --format '{{.Config.Image}}'); TF_GLM env: $(docker inspect glm53-spark --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E 'TF_GLM_PROF|TF_GLM_PDL' | tr '\n' ' ')"
sleep 15
ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc.py http://spark3:8888 GLM-5.3-Flash final102 2" 2>&1 | tee $OUT/acc-final102.log
echo FINAL_DONE
