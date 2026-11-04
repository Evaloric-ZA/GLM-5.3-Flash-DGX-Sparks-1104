import sys, urllib.request, time, json, random
base, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
P = "Write a long essay about the history of lighthouses and their keepers."
def one(prompt, max_tokens):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    ev=[]; usage={}
    for line in urllib.request.urlopen(req, timeout=600):
        if not line.startswith(b"data: ") or b"[DONE]" in line: continue
        d=json.loads(line[6:])
        for c in d.get("choices",[]):
            if c["delta"].get("content"): ev.append(time.time())
        usage.update(d.get("usage") or {})
    return (usage["completion_tokens"]-1)/max(ev[-1]-ev[0],1e-6)
one("Reply with exactly: ready",8)
for rep in range(2):
    print("[%s] 1stream rep%d %.1f" % (tag,rep,one(P+" Variant %d"%random.random(),400)),flush=True)
