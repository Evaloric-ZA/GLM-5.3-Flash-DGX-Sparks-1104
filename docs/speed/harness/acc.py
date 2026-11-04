import json, sys, time, urllib.request, threading, random, statistics as st
base, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
P = "Write a long essay about the history of lighthouses and their keepers."
C = "Write a complete Python module implementing an LRU cache with TTL, type hints and docstrings, plus unit tests."
def one(prompt, max_tokens, out=None, i=0):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    ev = []; sp = {}; tf = {}; usage = {}
    for line in urllib.request.urlopen(req, timeout=600):
        if not line.startswith(b"data: ") or b"[DONE]" in line: continue
        d = json.loads(line[6:])
        for c in d.get("choices", []):
            if c["delta"].get("content"): ev.append(time.time())
        tf.update(d.get("tensorfold") or {}); sp.update(d.get("speculative") or {}); usage.update(d.get("usage") or {})
    n = usage["completion_tokens"]; dec = max(ev[-1]-ev[0], 1e-6); r = sp.get("rounds") or tf.get("rounds") or 0; a = sp.get("accepted") or tf.get("accepted") or 0; dr = sp.get("drafted") or tf.get("drafted") or 0
    if not r: print("  (no round stats; keys:", sorted(tf), sorted(sp), ")")
    res = {"tps": (n-1)/dec, "tpr": n/r if r else float("nan"), "ms": dec/r*1000 if r else float("nan"), "acc": a/dr if dr else float("nan"), "sha": tf.get("token_sha"), "n": n}
    if out is not None: out[i] = res
    return res
one("Reply with exactly: ready", 8)
for pn, prompt in (("prose", P), ("code", C)):
    rs = [one(prompt, 600) for _ in range(reps)]
    print(f"[{tag}] 1-stream {pn}: tok/s {'/'.join(f'{r['tps']:.1f}' for r in rs)} | tok/round {st.mean(r['tpr'] for r in rs):.2f} | ms/round {st.mean(r['ms'] for r in rs):.1f} | accepted {st.mean(r['acc'] for r in rs)*100:.0f}% | sha {rs[0]['sha']}")
tot = []
for k in range(reps):
    out = {}; th = [threading.Thread(target=one, args=(P + f" Focus {i}: {random.random()}", 400, out, i)) for i in range(4)]
    [t.start() for t in th]; [t.join() for t in th]
    tot.append(sum(o["tps"] for o in out.values()))
    print(f"[{tag}] 4-stream prose, rep {k}: total {tot[-1]:.1f} tok/s | per stream {' '.join(f'{o['tps']:.1f}' for o in out.values())} | tok/round {st.mean(o['tpr'] for o in out.values()):.2f} | ms/round {st.mean(o['ms'] for o in out.values()):.0f} | accepted {st.mean(o['acc'] for o in out.values())*100:.0f}%")
