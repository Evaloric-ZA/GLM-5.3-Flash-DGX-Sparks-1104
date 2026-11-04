#!/bin/bash
# The 104-patch image (0103 amended + 0104 warm turns) on pair A: promote the two candidates into patches/engine (the
# image is on both Sparks by label), restart env-free, then the pre-registered gate (RESULTS.md 2026-11-04 13:00).
cd /home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig
OUT=docs/speed/cache
cp docs/speed/candidates/0103-server-a-call-after-prose-that-quotes-its-opener-is-the-call.patch docs/speed/candidates/0104-glm-warm-turns-the-next-user-message-resumes-past-the-agent-loop.patch patches/engine/
echo "[port] patches: $(ls patches/engine | wc -l); label: $(scripts/prepare.sh --label)"
docker image inspect tensorfold-glm53-spark:$(scripts/prepare.sh --label) --format '{{.Id}}' > /dev/null || { echo "[port] image for the label is not loaded here: abort"; exit 1; }
t0=$(date -u +%Y-%m-%dT%H:%M:%S)
./start.sh restart > $OUT/restart-port.log 2>&1 || { echo "[port] start.sh restart exit $?"; tail -20 $OUT/restart-port.log; }
for i in $(seq 1 180); do curl -sf -m 5 http://localhost:8888/health 2>/dev/null | grep -q '"warming": false' && break; sleep 5; done
echo "[port] healthy; image $(docker inspect glm53-spark --format '{{.Config.Image}}')"
docker logs glm53-spark 2>&1 | grep -i "pages of\|prompt cache:\|cannot keep\|warm" | grep -v "GET /" | head -5 | cut -c1-300
sleep 10
echo "--- replay (quiet engine; the script waits 3 s after each answer)"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && timeout 900 python3 cache-replay.py http://spark3:8888" 2>&1 | tee $OUT/replay-port.log
echo "--- warm pass log lines since restart"; docker logs --since "$t0" glm53-spark 2>&1 | grep -i "next turn\|stepped aside\|warm" | grep -v "GET /" | tail -12 | cut -c1-220
echo "--- acc (back-to-back requests: each reply is followed at once by the next request)"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc.py http://spark3:8888 GLM-5.3-Flash port 2" 2>&1 | tee $OUT/acc-port.log
echo "--- stepped-aside count during acc"; docker logs --since "$t0" glm53-spark 2>&1 | grep -c "stepped aside"; docker logs --since "$t0" glm53-spark 2>&1 | grep -c "background prefill"
echo "--- cold vs warm greedy: the replay's final prompt answered cold (fresh wording) and after a completed warm pass"
ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && timeout 600 python3 - <<'PY'
import json, urllib.request, time
URL='http://spark3:8888/v1/chat/completions'
def ask(msgs, mt=48):
    body={'model':'GLM-5.3-Flash','messages':msgs,'temperature':0,'max_tokens':mt,'chat_template_kwargs':{'enable_thinking':False}}
    r=json.load(urllib.request.urlopen(urllib.request.Request(URL,json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=600))
    return r['choices'][0]['message']['content'], r.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens', r.get('usage',{}).get('cached_tokens'))
base=[{'role':'system','content':'You are terse. '+('Background fact %d is irrelevant. '%i)*300},{'role':'user','content':'Name three lighthouses in one line.'}]
a1,c1=ask(base); time.sleep(4)          # cold answer; the warm pass renders [base, a1, next user] after it
msgs=base+[{'role':'assistant','content':a1},{'role':'user','content':'Now name three more, one line.'}]
a2,c2=ask(msgs)                        # after the warm pass: should hit past the previous answer
time.sleep(4)
a3,c3=ask(msgs)                        # the same prompt again: the kept state
print('turn1 cached',c1); print('turn2 cached',c2,'answer',repr(a2[:80])); print('turn2 again cached',c3,'same answer:',a2==a3)
PY" 2>&1 | tee $OUT/coldwarm-port.log
curl -s -m 5 http://localhost:8888/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('[port] health memory', d.get('memory'))"
echo PORT_DONE
