#!/usr/bin/env python3
"""Mixed prefill + decode, the check a scheduler change needs (Codex's amendment 3).

A long cold prefill is admitted while decode streams are already running. Reports the decoders'
throughput during the overlap, the prefill's own rate, and token_sha for the decoders so exactness is
still checked under contention. Fixed prompts for the decoders (deterministic); fresh content for the
prefill so it cannot hit the prompt cache.

Usage: mixed.py <label>   -> docs/speed/mixed-<label>.json
"""
import json, os, random, statistics, sys, threading, time, urllib.request

BASE = os.environ.get("U", "http://127.0.0.1:8888")
MODEL = os.environ.get("M", "GLM-5.3-Flash")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed")
DECODERS = int(os.environ.get("DECODERS", "2"))
MAXTOK = int(os.environ.get("MAXTOK", "400"))
PREFILL_WORDS = int(os.environ.get("PREFILL_WORDS", "25000"))   # ~95K tokens
TRIALS = int(os.environ.get("TRIALS", "2"))

LONG = ["Write five detailed paragraphs about the history of ocean navigation.",
        "Explain how a modern operating system schedules threads. Be thorough and use several paragraphs.",
        "Describe the engineering of suspension bridges in five paragraphs.",
        "Write five paragraphs on the development of the printing press and its consequences."]

def chat(text, maxtok, timeout=1800):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}],
                       "temperature": 0.0, "max_tokens": maxtok,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    tf, u = r.get("tensorfold", {}), r.get("usage", {})
    return {"wall": time.time() - t0, "completion": u.get("completion_tokens"),
            "prompt": u.get("prompt_tokens"),
            "cached": u.get("prompt_tokens_details", {}).get("cached_tokens"),
            "ttft": tf.get("time_to_first_token"), "prefill_s": tf.get("prefill_seconds"),
            "seconds": tf.get("seconds"), "token_sha": tf.get("token_sha"),
            "spec": r.get("speculative", {})}

def one(trial, label):
    dec = [None] * DECODERS
    pre = [None]
    errs = []
    def run_dec(i):
        try: dec[i] = chat(LONG[i % len(LONG)], MAXTOK)
        except Exception as e: errs.append(f"dec{i}: {e!r}"[:150])
    def run_pre():
        try:
            random.seed(9000 + trial)
            txt = "Summarise in one word:\n" + " ".join(f"w{random.randint(0,99999)}" for _ in range(PREFILL_WORDS))
            pre[0] = chat(txt, 1)
        except Exception as e: errs.append(f"prefill: {e!r}"[:150])
    th = [threading.Thread(target=run_dec, args=(i,)) for i in range(DECODERS)]
    t0 = time.time()
    for x in th: x.start()
    time.sleep(3.0)                      # let the decoders settle into their rounds first
    pt = threading.Thread(target=run_pre); pt.start()
    for x in th: x.join()
    pt.join()
    el = time.time() - t0
    good = [d for d in dec if d]
    if errs or not good or not pre[0]:
        return {"error": errs[:2]}
    tok = sum(d["completion"] for d in good)
    p = pre[0]
    prate = p["prompt"] / p["ttft"] if p["ttft"] else None
    return {"elapsed": el, "decode_tokens": tok, "decode_total_tps": tok / el,
            "decode_shas": [d["token_sha"] for d in good],
            "decode_tps_each": [d["completion"] / (d["seconds"] - (d["prefill_s"] or 0)) for d in good],
            "prefill_tok": p["prompt"], "prefill_ttft": p["ttft"], "prefill_tps": prate,
            "prefill_cached": p["cached"]}

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    print(f"# mixed load: {DECODERS} decoders ({MAXTOK} tok each) with a ~{PREFILL_WORDS//265}K cold prefill admitted mid-flight")
    res = []
    for t in range(TRIALS):
        r = one(t, label)
        res.append(r)
        if "error" in r:
            print(f"  trial {t+1}: ERROR {r['error']}"); continue
        print(f"  trial {t+1}: decode {r['decode_total_tps']:6.1f} tok/s total "
              f"({', '.join(f'{x:.1f}' for x in r['decode_tps_each'])} each)  "
              f"prefill {r['prefill_tok']} tok at {r['prefill_tps']:.0f} tok/s "
              f"(TTFT {r['prefill_ttft']:.1f}s, cached {r['prefill_cached']})  shas {r['decode_shas']}")
    ok = [r for r in res if "error" not in r]
    if ok:
        d = statistics.median(r["decode_total_tps"] for r in ok)
        p = statistics.median(r["prefill_tps"] for r in ok)
        print(f"  -> median decode under load {d:.1f} tok/s, median prefill under load {p:.0f} tok/s")
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(os.path.join(OUT, f"mixed-{label}.json"), "w"), indent=1)
        print(f"# wrote {os.path.join(OUT, f'mixed-{label}.json')}")
