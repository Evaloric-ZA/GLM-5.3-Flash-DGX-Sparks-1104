# Prose decode at temperature 0.7 (sampled, not greedy): 1 stream x reps and 4 streams x reps; same essay prompt as acc.py.
import json, sys, time, urllib.request, threading, random, statistics as st
base, model, tag = sys.argv[1], sys.argv[2], sys.argv[3]; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 2
temp = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
P = "Write a long essay about the history of lighthouses and their keepers."
def one(prompt, max_tokens, out=None, i=0):
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": temp, "top_p": 0.95, "max_tokens": max_tokens,
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
    res = {"tps": (n-1)/dec, "tpr": n/r if r else float("nan"), "ms": dec/r*1000 if r else float("nan"), "acc": a/dr if dr else float("nan"), "n": n}
    if out is not None: out[i] = res
    return res
one("Reply with exactly: ready", 8)
for k in range(reps):
    r = one(P, 600)
    print(f"[{tag}] 1-stream prose T={temp} rep {k}: {r['tps']:.1f} tok/s | tok/round {r['tpr']:.2f} | ms/round {r['ms']:.1f} | accepted {r['acc']*100:.0f}% | {r['n']} tokens", flush=True)
for k in range(reps):
    out = {}; th = [threading.Thread(target=one, args=(P + f" Focus {i}: {random.random()}", 400, out, i)) for i in range(4)]
    [t.start() for t in th]; [t.join() for t in th]
    print(f"[{tag}] 4-stream prose T={temp} rep {k}: total {sum(o['tps'] for o in out.values()):.1f} tok/s | per stream {' '.join(f'{o['tps']:.1f}' for o in out.values())} | tok/round {st.mean(o['tpr'] for o in out.values()):.2f} | ms/round {st.mean(o['ms'] for o in out.values()):.0f} | accepted {st.mean(o['acc'] for o in out.values())*100:.0f}%", flush=True)
