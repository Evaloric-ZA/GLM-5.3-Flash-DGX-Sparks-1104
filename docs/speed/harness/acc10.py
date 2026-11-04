import json, sys, time, urllib.request, statistics as st
base, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]
TOPICS = ["the history of lighthouses and their keepers", "how glaciers shape valleys", "the economics of medieval fairs",
          "the life cycle of a star", "why cities grew along rivers", "the invention of the printing press",
          "how bees communicate", "the Antarctic Treaty and its politics", "the design of suspension bridges",
          "the history of tea trade routes"]
CODE = ["Write a complete Python module implementing an LRU cache with TTL, type hints and docstrings, plus unit tests.",
        "Write a complete Go program implementing a concurrent rate limiter with tests."]
def one(prompt, max_tokens=400):
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
    n = usage["completion_tokens"]; dec = max(ev[-1]-ev[0], 1e-6); r = sp.get("rounds") or tf.get("rounds"); a = sp.get("accepted") or tf.get("accepted"); dr = sp.get("drafted") or tf.get("drafted")
    return {"tps": (n-1)/dec, "tpr": n/r, "ms": dec/r*1000, "acc": a/dr, "dpr": dr/r}
one("Reply with exactly: ready", 8)
for name, prompts in (("prose", [f"Write a long essay about {t}." for t in TOPICS]), ("code", CODE)):
    rs = [one(p) for p in prompts]
    f = lambda k: f"{st.mean(r[k] for r in rs):.2f}±{(st.stdev(r[k] for r in rs) if len(rs)>1 else 0):.2f}"
    print(f"[{tag}] {name} x{len(rs)}: tok/s {f('tps')} | tok/round {f('tpr')} | drafts/round {f('dpr')} | accepted {f('acc')} | ms/round {f('ms')}")
    print("   per prompt tok/s:", " ".join(f"{r['tps']:.1f}" for r in rs), "| acc:", " ".join(f"{r['acc']*100:.0f}" for r in rs))
