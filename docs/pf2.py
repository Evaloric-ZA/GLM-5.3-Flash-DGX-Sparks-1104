import json,os,time,urllib.request,random
URL=os.environ.get("U","http://127.0.0.1:8888")+"/v1/chat/completions"
MODEL=os.environ.get("M","GLM-5.3-Flash")
def one(text):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":text}],"temperature":0.0,
        "max_tokens":1,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    t0=time.time()
    r=json.load(urllib.request.urlopen(urllib.request.Request(URL,body,{"Content-Type":"application/json"}),timeout=1800))
    return time.time()-t0, r["usage"]["prompt_tokens"], r["usage"].get("prompt_tokens_details",{}).get("cached_tokens")
print(f"# {MODEL} at {URL}  (fresh random content each trial: no prefix-cache hit)")
for nwords,label in ((6200,"~23K"),(25000,"~95K")):
    for trial in range(3):
        random.seed(1000+trial*17+nwords)          # different content every trial
        text="Summarise in one word:\n"+" ".join(f"w{random.randint(0,99999)}" for _ in range(nwords))
        el,p,cached = one(text)
        print(f"  {label} trial {trial+1}: {p} tok, TTFT {el:.2f}s -> {p/el:.0f} tok/s (cached_tokens={cached})")
