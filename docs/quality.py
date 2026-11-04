#!/usr/bin/env python3
"""Quality gate for a GLM serving config: token NLL on fixed passages + greedy answers.

NLL comes from /v1/completions prompt_logprobs (the engine returns them nonstreaming), so it measures the
model's own distribution over held text and is independent of sampling. Greedy answers are compared as text.
"""
import hashlib, json, os, sys, urllib.request

BASE = os.environ.get("U", "http://127.0.0.1:8888")
MODEL = os.environ.get("M", "GLM-5.3-Flash")

def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=900))
    except urllib.error.HTTPError as e:
        return {"__error__": f"HTTP {e.code}: {e.read()[:200].decode('utf-8','replace')}"}

PASSAGES = [
 "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. Mitochondria "
 "generate most of the cell's supply of adenosine triphosphate, used as a source of chemical energy.",
 "In computing, a hash table is a data structure that implements an associative array, also called a dictionary, "
 "which maps keys to values. A hash table uses a hash function to compute an index into an array of buckets.",
 "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs)//2]\n"
 "    left = [x for x in xs if x < pivot]\n    mid = [x for x in xs if x == pivot]\n"
 "    right = [x for x in xs if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n",
 "Treaties between sovereign states derive their binding force from the principle pacta sunt servanda, "
 "which holds that agreements must be kept. Customary international law supplements treaty obligations.",
]

QUESTIONS = [
 "What is 7 * 8 + 13? Answer with the number only.",
 "What is the capital of Australia? One word.",
 "Write a Python one-liner that reverses a string s. Code only.",
 "Which is larger, 0.9 or 0.11? Answer with just the number.",
 "Name the process by which plants convert light into chemical energy. One word.",
 "What is the 7th prime number? Number only.",
 "Spell 'necessary' backwards. Letters only.",
 "In one short sentence, state what a mutex protects against.",
 "What is 144 divided by 12? Number only.",
 "Give the chemical symbol for potassium. Symbol only.",
]

def nll():
    tot_lp, tot_n, per = 0.0, 0, []
    for i, text in enumerate(PASSAGES):
        r = post("/v1/completions", {"model": MODEL, "prompt": text, "max_tokens": 1,
                                     "temperature": 0, "prompt_logprobs": 0, "echo": False})
        if "__error__" in r:
            return None, r["__error__"], []
        pl = r["choices"][0].get("prompt_logprobs")
        if not pl:
            return None, "no prompt_logprobs in the reply", []
        lps = []
        for slot in pl:
            if not slot:
                continue                                  # the first token has no predecessor
            if isinstance(slot, dict):
                # prompt_logprobs:0 returns exactly one entry a slot: the token actually in the prompt
                if len(slot) != 1:
                    return None, f"expected one logprob a slot, got {len(slot)}", []
                v = next(iter(slot.values()))
                lps.append(v["logprob"] if isinstance(v, dict) else float(v))
        if lps:
            per.append(-sum(lps) / len(lps))
            tot_lp += sum(lps); tot_n += len(lps)
    return (-tot_lp / tot_n if tot_n else None), None, per

def answers():
    out = []
    for q in QUESTIONS:
        r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": q}],
                 "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}})
        if "__error__" in r:
            out.append("ERR:" + r["__error__"]); continue
        out.append((r["choices"][0]["message"].get("content") or "").strip())
    return out

if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "config"
    n, err, per = nll()
    a = answers()
    h = hashlib.sha256("\x1f".join(a).encode()).hexdigest()[:12]
    print(json.dumps({"label": label, "nll": n, "nll_error": err, "nll_per_passage": per,
                      "answers_sha": h, "answers": a}, indent=1))
