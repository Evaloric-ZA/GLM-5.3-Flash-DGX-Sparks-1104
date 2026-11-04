#!/usr/bin/env python3
"""Prompts straddling the 2,048 and 4,096-row chunk boundaries, plus tails, with DFlash2 on.
Each must answer without error; also reports draft acceptance for a decode."""
import json, os, random, re, subprocess, urllib.request
BASE=os.environ.get("U","http://127.0.0.1:8888"); MODEL=os.environ.get("M","GLM-5.3-Flash")
def chat(text, maxtok=24):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":text}],"temperature":0,
        "max_tokens":maxtok,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    try:
        r=json.load(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",body,
            {"Content-Type":"application/json"}),timeout=1200))
        return r["usage"]["prompt_tokens"], (r["choices"][0]["message"].get("content") or "").strip()[:40], None
    except urllib.error.HTTPError as e:
        return None, None, f"HTTP {e.code}: {e.read()[:140].decode('utf-8','replace')}"
    except Exception as e:
        return None, None, repr(e)[:140]
print("# chunk-boundary prompts (DFlash2 on): token counts around 2048/4096 rows and well past")
random.seed(5)
W=[f"w{random.randint(0,99999)}" for _ in range(40000)]
bad=0
for nw in (300, 520, 540, 1080, 1100, 2200, 2800, 6000, 12000):
    p,ans,err = chat("Reply with the single word ok. Ignore this text:\n"+" ".join(W[:nw]))
    if err: print(f"  {nw:6d} words -> ERROR {err}"); bad+=1
    else:   print(f"  {nw:6d} words -> {p:7d} prompt tok, reply {ans!r}")
print(f"# failures: {bad}")
print("# decode + draft acceptance, 1 stream")
import time
t0=time.time()
body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":"Write three paragraphs about the history of sailing ships."}],
    "temperature":0,"max_tokens":300,"chat_template_kwargs":{"enable_thinking":False}}).encode()
r=json.load(urllib.request.urlopen(urllib.request.Request(BASE+"/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900))
el=time.time()-t0; c=r["usage"]["completion_tokens"]
print(f"  decode: {c} tok in {el:.2f}s -> {c/el:.1f} tok/s")
log=subprocess.run(["docker","logs","--tail","80","glm53-spark"],capture_output=True,text=True).stdout
m=[x for x in re.findall(r'accepted=(\d+)/(\d+)', log)]
if m:
    a=sum(int(x) for x,_ in m); b=sum(int(y) for _,y in m)
    print(f"  draft acceptance (last {len(m)} requests): {a}/{b} = {a/b:.3f}")
else:
    print("  draft acceptance: no accepted= line found")
