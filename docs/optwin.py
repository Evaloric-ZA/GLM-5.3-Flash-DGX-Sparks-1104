#!/usr/bin/env python3
"""From a measured verify line and per-position landing rates, the window the engine's rule would pick.

Replicates draft_rule.zig exactly so the prediction and the engine agree:

  offer(chances, lambda, a, b):            # draft_rule.zig:155-169
      reach = 1; value = 1 - lambda*a; best = value; k = 0
      for n, p in enumerate(chances, 1):
          reach *= clamp(p, 0, 1)
          value += reach - lambda*b
          if value > best: best, k = value, n
      return k

`chances[j]` is the conditional chance that draft j+1 lands given every earlier one did - which is
what trace.py's position table measures. `lambda` is the stream's tokens a ms (Lean.lambda, floored by
TF_GLM_DRAFT_LAMBDA, default 0.15).

Also prints, for each width k, the throughput the measured numbers imply:
    tokens(k) = 1 + sum of the chain products to k
    ms(k)     = a + b*k + draft_ms
    tok/s     = tokens(k) / ms(k) * 1000
so the best width by measured throughput can be read off directly and compared with what the rule
would choose. Usage: optwin.py <a_ms> <b_ms> <draft_ms> <p1> [p2 ...]   (conditional rates)
"""
import sys

def offer(chances, lam, a, b):
    reach, value = 1.0, 1 - lam * a
    best, k = value, 0
    for n, p in enumerate(chances, 1):
        reach *= min(max(p, 0.0), 1.0)
        value += reach - lam * b
        if value > best:
            best, k = value, n
    return k

def table(a, b, draft, cond):
    print(f"# verify line a = {a:.2f} ms, b = {b:.2f} ms/row (b/a = {b/a:.4f}), drafter {draft:.2f} ms")
    print(f"# conditional landing rates: {', '.join(f'{p:.3f}' for p in cond)}")
    print(f"\n{'k':>3s} {'rows':>5s} {'E[tokens]':>10s} {'ms':>8s} {'tok/s':>8s} {'vs k=0':>8s}")
    base = None
    rows = []
    reach, toks = 1.0, 1.0
    out = []
    for k in range(0, len(cond) + 1):
        if k > 0:
            reach *= min(max(cond[k-1], 0.0), 1.0)
            toks += reach
        ms = a + b * k + draft
        tps = toks / ms * 1000
        if base is None: base = tps
        out.append((k, toks, ms, tps))
        print(f"{k:3d} {k+1:5d} {toks:10.4f} {ms:8.2f} {tps:8.2f} {(tps/base-1)*100:+7.1f}%")
    bestk = max(out, key=lambda r: r[3])
    print(f"\n# best width by measured throughput: k = {bestk[0]} drafts ({bestk[0]+1} rows), "
          f"{bestk[3]:.2f} tok/s, {(bestk[3]/base-1)*100:+.1f}% against k=0 (no drafts)")
    return out, bestk

if __name__ == "__main__":
    a, b, draft = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    cond = [float(x) for x in sys.argv[4:]]
    if not cond:
        cond = [0.70, 0.62, 0.57, 0.52, 0.48, 0.45, 0.42]
        print("# (no rates given: using placeholders, NOT measured)")
    out, bestk = table(a, b, draft, cond)
    print("\n# what the engine's rule would offer, by lambda (tokens a ms):")
    for lam in (0.0353, 0.05, 0.10, 0.15, 0.30):
        k = offer(cond, lam, a, b)
        print(f"   lambda {lam:.4f} -> offer k = {k}")
    print(f"\n# the cut rule for comparison: stops where the CHAIN product falls under the cut.")
    chain = 1.0
    for j, p in enumerate(cond, 1):
        chain *= p
        print(f"   after draft {j}: chain {chain:.4f}" + ("   <- cut 0.7 stops here" if chain < 0.7 and chain*1/p >= 0.7 else ""))
