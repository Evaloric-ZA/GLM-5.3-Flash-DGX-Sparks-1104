#!/bin/bash
cd /home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig
OUT=docs/speed/cache
cp docs/speed/candidates/0106-lanes-release-a-stream-that-finishes-on-its-first-token-upstream-ed39acc5a.patch patches/engine/
L=$(scripts/prepare.sh --label); echo "[0106] patches: $(ls patches/engine | wc -l); label: $L"
docker image inspect tensorfold-glm53-spark:$L --format '{{.Id}}' > /dev/null || { echo "[0106] image for the label is not loaded here: abort"; exit 1; }
./start.sh restart > $OUT/restart-0106.log 2>&1 || { echo "[0106] start.sh restart exit $?"; tail -20 $OUT/restart-0106.log; }
for i in $(seq 1 180); do curl -sf -m 5 http://localhost:8888/health 2>/dev/null | grep -q '"warming": false' && break; sleep 5; done
echo "[0106] healthy; image $(docker inspect glm53-spark --format '{{.Config.Image}}')"
docker logs glm53-spark 2>&1 | grep -i "pages of\|prompt cache:\|cannot keep" | grep -v "GET /" | head -3 | cut -c1-200
sleep 10
echo "--- one-token replies, 20 in a row (the leaking path), then the health's live counters"
for i in $(seq 1 20); do curl -s -m 30 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"Reply with the single word yes."}],"max_tokens":1,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' > /dev/null; done
curl -s -m 5 http://localhost:8888/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('[0106] live', d.get('live'))"
echo "--- acc + replay"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc.py http://spark3:8888 GLM-5.3-Flash p0106 2 && timeout 900 python3 cache-replay.py http://spark3:8888" 2>&1 | tee $OUT/acc-0106.log
echo LEG0106_DONE
