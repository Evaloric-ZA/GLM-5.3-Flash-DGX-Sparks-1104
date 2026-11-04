import sys, urllib.request, threading, random, statistics as st, time, json
base, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
P = "Write a long essay about the history of lighthouses and their keepers."
def one(prompt, max_tokens, out, i):
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
    n=usage["completion_tokens"]; out[i]=(n-1)/max(ev[-1]-ev[0],1e-6)
import time, json
one("Reply with exactly: ready",8,{0:0},0)
for rep in range(2):
    out={}
    th=[threading.Thread(target=one,args=(P+" Focus %d: %r"%(i,random.random()),400,out,i)) for i in range(4)]
    t0=time.time();[t.start() for t in th];[t.join() for t in th]
    print("[%s] 4stream rep%d total %.1f" % (tag,rep,sum(out.values())),flush=True)
