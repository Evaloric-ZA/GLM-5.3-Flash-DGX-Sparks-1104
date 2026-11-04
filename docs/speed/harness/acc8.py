import json, sys, time, urllib.request, threading, random, statistics as st
base, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
P = "Write a long essay about the history of lighthouses and their keepers."

def one(prompt, max_tokens, out=None, i=0):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    ev = []; sp = {}; usage = {}
    for line in urllib.request.urlopen(req, timeout=600):
        if not line.startswith(b"data: ") or b"[DONE]" in line:
            continue
        d = json.loads(line[6:])
        for c in d.get("choices", []):
            if c["delta"].get("content"):
                ev.append(time.time())
        sp.update(d.get("speculative") or {})
        usage.update(d.get("usage") or {})
    n = usage["completion_tokens"]; dec = max(ev[-1] - ev[0], 1e-6); r = sp.get("rounds") or 0
    res = {"tps": (n - 1) / dec, "tpr": n / r if r else float("nan"),
           "ms": dec / r * 1000 if r else float("nan")}
    if out is not None:
        out[i] = res

for k in (6, 8):
    for rep in range(2):
        out = {}
        th = [threading.Thread(target=one, args=(P + " Focus %d: %r" % (i, random.random()), 300, out, i)) for i in range(k)]
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
        tot = sum(o["tps"] for o in out.values())
        per = " ".join("%.1f" % o["tps"] for o in out.values())
        mtp = st.mean(o["tpr"] for o in out.values()); mms = st.mean(o["ms"] for o in out.values())
        print("[%s] %d-stream rep%d: total %.1f tok/s | per stream %s | tok/round %.2f | ms/round %.0f | wall %.0fs"
              % (tag, k, rep, tot, per, mtp, mms, time.time() - t0))
