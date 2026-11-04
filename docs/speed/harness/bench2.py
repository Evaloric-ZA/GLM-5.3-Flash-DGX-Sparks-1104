# One suite against one server: single-stream decode, cold prefill, 2/4-stream decode, mixed prefill+decode, two prefills at once.
import json, sys, time, random, threading, statistics as st, urllib.request
BASE, MODEL, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
URL = BASE + "/v1/chat/completions"; OUT = open(f"{TAG}.jsonl", "a")
rnd = random.Random(time.time_ns())
VOC = "the of and to in is it for on with as by at from that this which these those many some other first last next".split() + [f"w{rnd.randrange(10**6)}" for _ in range(600)]
def words(n): return " ".join(rnd.choice(VOC) for _ in range(n))
def run(label, content, max_tokens, t_start=None):
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); ev = []; usage = None; sha = None; text = []
    r = urllib.request.urlopen(req, timeout=1800)
    for line in r:
        if not line.startswith(b"data: ") or b"[DONE]" in line: continue
        d = json.loads(line[6:]); t = time.time()
        for c in d.get("choices", []):
            s = c["delta"].get("content") or ""
            if s: ev.append(t); text.append(s)
        if d.get("usage"): usage = d["usage"]
        tf = d.get("tensorfold") or {}; sha = tf.get("token_sha") or sha
    t1 = time.time(); n = (usage or {}).get("completion_tokens"); p = (usage or {}).get("prompt_tokens")
    rec = {"label": label, "prompt_tokens": p, "completion_tokens": n, "ttft": ev[0]-t0 if ev else None, "total": t1-t0,
           "decode_tps": (n-1)/(ev[-1]-ev[0]) if n and len(ev) > 1 and ev[-1] > ev[0] else None,
           "prefill_tps": p/(ev[0]-t0) if ev and p else None, "events": [(t - t0) for t in ev], "sha": sha, "head": "".join(text)[:80]}
    OUT.write(json.dumps({**rec, "events": len(ev)}) + "\n"); OUT.flush(); return rec
def par(fns):
    out = [None]*len(fns)
    def w(i): out[i] = fns[i]()
    th = [threading.Thread(target=w, args=(i,)) for i in range(len(fns))]
    [t.start() for t in th]; [t.join() for t in th]; return out
PROSE = "Write a long essay about the history of lighthouses and their keepers."
CODE = "Write a complete Python module implementing an LRU cache with TTL, type hints and docstrings, plus unit tests."
R = {}
print(f"[{TAG}] {BASE} {MODEL}", flush=True)
run("warm", "Reply with exactly: ready", 8)
# 1. single-stream decode
for name, prompt in (("prose", PROSE), ("code", CODE)):
    rs = [run(f"single_{name}_{i}", prompt, 600) for i in range(2)]
    R[f"single_{name}"] = {"decode_tps": max(r["decode_tps"] for r in rs), "ttft": min(r["ttft"] for r in rs), "sha": rs[0]["sha"], "tokens": rs[0]["completion_tokens"]}
    print(f"  single {name}: {R[f'single_{name}']}", flush=True)
# 2. cold prefill
for nw, name in ((3600, "16K"), (14500, "64K")):
    r = run(f"prefill_{name}", "Summarize in one word: " + words(nw), 4)
    R[f"prefill_{name}"] = {"prompt_tokens": r["prompt_tokens"], "ttft": r["ttft"], "prefill_tps": r["prefill_tps"]}
    print(f"  prefill {name}: {R[f'prefill_{name}']}", flush=True)
# 3. concurrent decode, 2 and 4 streams
for n in (2, 4):
    rs = par([(lambda i=i: run(f"conc{n}_{i}", PROSE + f" Focus {i}: {rnd.random()}", 400)) for i in range(n)])
    R[f"conc{n}"] = {"per_stream_tps": [round(r["decode_tps"], 1) for r in rs], "sum_tps": round(sum(r["decode_tps"] for r in rs), 1), "ttft_max": max(r["ttft"] for r in rs)}
    print(f"  conc{n}: {R[f'conc{n}']}", flush=True)
# 4. mixed: k decoders, then a cold ~50K prompt 3 s later
def mixed(k):
    res = {}
    def dec(i): res[f"d{i}"] = run(f"mix{k}_dec{i}", PROSE + f" Variant {i}: {rnd.random()}", 3000)
    def lng():
        time.sleep(3.0); res["tL"] = time.time(); res["long"] = run(f"mix{k}_long", "Summarize in one word: " + words(11500), 4)
    ths = [threading.Thread(target=dec, args=(i,)) for i in range(k)] + [threading.Thread(target=lng)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]
    L = res["long"]; tL = res["tL"] - t0; tF = tL + L["ttft"]; rows = []
    for i in range(k):
        d = res[f"d{i}"]; ev = [e for e in d["events"]]  # relative to that request's own t0 ~= t0
        during = [e for e in ev if tL <= e < tF]; before = [e for e in ev if e < tL]; after = [e for e in ev if e >= tF]
        gaps = sorted(during[j+1]-during[j] for j in range(len(during)-1)) if len(during) > 2 else [0]
        q = lambda p: gaps[min(len(gaps)-1, int(p*len(gaps)))]*1000
        tpe = d["completion_tokens"]/len(ev) if ev else 1  # tokens per event for this stream
        rows.append({"before_tps": round(len(before)*tpe/max(1e-9, min(tL, ev[-1]) - ev[0]), 1) if before else None,
                     "during_tps": round(len(during)*tpe/max(1e-9, min(tF, ev[-1]) - tL), 1), "gap_p50": round(q(.5)), "gap_p95": round(q(.95)), "gap_max": round(gaps[-1]*1000),
                     "after_tps": round(len(after)*tpe/max(1e-9, ev[-1]-tF), 1) if len(after) > 5 else None})
    out = {"long_prompt_tokens": L["prompt_tokens"], "long_ttft": round(L["ttft"], 2), "decoders": rows}
    R[f"mixed{k}"] = out; print(f"  mixed{k}: {out}", flush=True)
mixed(1); mixed(3)
# 5. two cold ~30K prefills at once, vs one alone
r1 = run("prefill_30K_alone", "Summarize in one word: " + words(6900), 4)
rs = par([(lambda i=i: run(f"prefill_30K_pair{i}", "Summarize in one word: " + words(6900), 4)) for i in range(2)])
R["prefill_pair"] = {"alone_ttft": round(r1["ttft"], 2), "alone_tokens": r1["prompt_tokens"], "pair_ttft": [round(r["ttft"], 2) for r in rs], "pair_tokens": [r["prompt_tokens"] for r in rs]}
print(f"  prefill pair: {R['prefill_pair']}", flush=True)
json.dump(R, open(f"{TAG}.summary.json", "w"), indent=1); print(f"[{TAG}] done", flush=True)
