#!/bin/bash
# 0105 on pair A: promote into patches/engine (the image is on both Sparks by label), restart env-free, probe the
# untyped-parameter tool call, the decode band and hashes, and the agent-loop replay.
cd /home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig
OUT=docs/speed/cache
cp docs/speed/candidates/0105-server-an-untyped-tool-parameter-takes-the-JSON-object-or-array-the-model-wrote.patch patches/engine/
L=$(scripts/prepare.sh --label); echo "[0105] patches: $(ls patches/engine | wc -l); label: $L"
docker image inspect tensorfold-glm53-spark:$L --format '{{.Id}}' > /dev/null || { echo "[0105] image for the label is not loaded here: abort"; exit 1; }
./start.sh restart > $OUT/restart-0105.log 2>&1 || { echo "[0105] start.sh restart exit $?"; tail -20 $OUT/restart-0105.log; }
for i in $(seq 1 180); do curl -sf -m 5 http://localhost:8888/health 2>/dev/null | grep -q '"warming": false' && break; sleep 5; done
echo "[0105] healthy; image $(docker inspect glm53-spark --format '{{.Config.Image}}')"
docker logs glm53-spark 2>&1 | grep -i "pages of\|prompt cache:\|cannot keep" | grep -v "GET /" | head -3 | cut -c1-200
sleep 10
echo "--- the probe that reproduced #180 (untyped payload, union note)"
curl -s -m 120 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"tools":[{"type":"function","function":{"name":"save_record","description":"Save a record","parameters":{"type":"object","properties":{"payload":{"description":"the record as an object with fields id (int) and tags (list of strings)"},"note":{"type":["string","object"]}},"required":["payload"]}}}],"tool_choice":"required","messages":[{"role":"user","content":"Save a record with id 7 and tags alpha and beta, note {\"k\":1}. Call save_record now."}]}' | python3 -c "
import sys,json; d=json.load(sys.stdin); m=d['choices'][0]['message']
for c in (m.get('tool_calls') or []): print('arguments:', c['function']['arguments'][:300])" | tee $OUT/probe-0105.log
echo "--- the same, streamed"
curl -s -N -m 120 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","stream":true,"max_tokens":200,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"tools":[{"type":"function","function":{"name":"save_record","description":"Save a record","parameters":{"type":"object","properties":{"payload":{"description":"the record as an object with fields id (int) and tags (list of strings)"}},"required":["payload"]}}}],"tool_choice":"required","messages":[{"role":"user","content":"Save a record with id 7 and tags alpha and beta. Call save_record now."}]}' | python3 -c "
import sys,json; args=''
for line in sys.stdin:
    if not line.startswith('data: ') or '[DONE]' in line: continue
    d=json.loads(line[6:])
    for c in d.get('choices',[]):
        for t in (c['delta'].get('tool_calls') or []): args+= (t.get('function') or {}).get('arguments') or ''
print('streamed arguments:', args[:300]); json.loads(args); print('valid JSON, payload type:', type(json.loads(args)['payload']).__name__)" | tee -a $OUT/probe-0105.log
echo "--- a plain text argument still streams (tool without a parameter schema)"
curl -s -N -m 120 http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash","stream":true,"max_tokens":120,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"tools":[{"type":"function","function":{"name":"shell","description":"Run a shell command given as the command parameter"}}],"tool_choice":"required","messages":[{"role":"user","content":"List the files in /tmp sorted by size. Call shell now."}]}' | python3 -c "
import sys,json; args=''; deltas=0
for line in sys.stdin:
    if not line.startswith('data: ') or '[DONE]' in line: continue
    d=json.loads(line[6:])
    for c in d.get('choices',[]):
        for t in (c['delta'].get('tool_calls') or []):
            a=(t.get('function') or {}).get('arguments') or ''
            if a: args+=a; deltas+=1
print('argument deltas:', deltas, '| arguments:', args[:200]); json.loads(args); print('valid JSON')" | tee -a $OUT/probe-0105.log
echo "--- acc + replay"; ssh -o BatchMode=yes spark5 "cd ~/c1bench/harness && python3 acc.py http://spark3:8888 GLM-5.3-Flash p0105 2 && timeout 900 python3 cache-replay.py http://spark3:8888" 2>&1 | tee $OUT/acc-0105.log
echo LEG0105_DONE
