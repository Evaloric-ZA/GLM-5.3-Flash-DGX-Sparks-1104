# A chat stream decodes; 3 s later a cold long prompt arrives. Reports the chat's tok/s and gaps during that prefill, and the long prompt's TTFT.
import json, sys, time, random, threading, urllib.request
URL = "http://127.0.0.1:8888/v1/chat/completions"; TAG = sys.argv[1]; WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
random.seed(time.time_ns())
vocab = "the of and to in is it for on with as by at from that this which these those many some other first last next".split() + [f"w{random.randrange(10**6)}" for _ in range(400)]
long_prompt = "Summarize in one word: " + " ".join(random.choice(vocab) for _ in range(WORDS))
def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)
ev = []  # (t, n_tokens) for the chat stream
def chat():
    r = post({"model": "GLM-5.3-Flash", "stream": True, "temperature": 0, "max_tokens": 8000,
              "messages": [{"role": "user", "content": f"Write a long essay about the history of lighthouses. seed {random.random()}"}]})
    for line in r:
        if line.startswith(b"data: ") and b"[DONE]" not in line:
            d = json.loads(line[6:]); c = d["choices"][0]["delta"].get("content") or ""
            if c: ev.append((time.time(), len(c)))
    ev.append((time.time(), 0))
longr = {}
def long_():
    t0 = time.time(); r = post({"model": "GLM-5.3-Flash", "stream": True, "max_tokens": 4, "temperature": 0,
                              "messages": [{"role": "user", "content": long_prompt}]})
    for line in r:
        if line.startswith(b"data: ") and b"[DONE]" not in line:
            d = json.loads(line[6:])
            if (d["choices"][0]["delta"].get("content") or "") and "ttft" not in longr: longr["ttft"] = time.time() - t0
            if d.get("usage"): longr["usage"] = d["usage"]
    longr["total"] = time.time() - t0
a = threading.Thread(target=chat); a.start(); time.sleep(3.0)
tL = time.time(); b = threading.Thread(target=long_); b.start(); b.join(); tF = tL + longr.get("ttft", 0); a.join()
def phase(name, lo, hi):
    ts = [t for t, _ in ev if lo <= t < hi]
    if len(ts) < 3: print(f"  {name}: {len(ts)} events"); return
    gaps = sorted(ts[i+1]-ts[i] for i in range(len(ts)-1)); q = lambda p: gaps[min(len(gaps)-1, int(p*len(gaps)))]*1000
    chars = sum(n for t, n in ev if lo <= t < hi)
    print(f"  {name}: {len(ts)} events, {chars} chars in {hi-lo:.1f} s = {len(ts)/(hi-lo):.1f} events/s, ~{chars/4/(hi-lo):.1f} tok/s; gaps ms p50 {q(.5):.0f} p95 {q(.95):.0f} max {gaps[-1]*1000:.0f}")
print(f"[{TAG}] long prompt: ttft {longr.get('ttft', float('nan')):.2f} s, usage {longr.get('usage')}")
print(f"[{TAG}] chat stream (one event ~ one token, often several):")
phase("alone, before the long prompt", ev[0][0], tL); phase("during its prefill (while the chat still ran)", tL, min(tF, ev[-1][0])); phase("after its first token", tF, ev[-1][0] + 1e-9)
