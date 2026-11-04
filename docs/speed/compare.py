#!/usr/bin/env python3
"""Compare decode legs against a baseline capture, with the gate the panel agreed.

Gate (AGREEMENT.md G1): a leg passes at a load only if its delta exceeds BOTH +2% and 2x the
baseline's observed trial spread at that load, with no worse than -1% at the other loads, and with
the token_sha contract identical on every corpus prompt.

Because the prompts are fixed and greedy, an exact policy change keeps the reply - and so the token
count - identical, which makes `rounds` the honest signal: fewer rounds for the same tokens.

Usage: compare.py <baseline-label> <leg-label> [<leg-label> ...]
"""
import json, os, sys, statistics

D = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def load(label):
    p = os.path.join(D, f"dec-{label}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def contract_ok(label, base="baseline"):
    pb = os.path.join(D, f"contract-{base}.json"); pl = os.path.join(D, f"contract-{label}.json")
    if not (os.path.exists(pb) and os.path.exists(pl)): return None, []
    a, b = json.load(open(pb)), json.load(open(pl))
    bad = [k for k in a if a[k].get("token_sha") != b.get(k, {}).get("token_sha")]
    return (not bad), bad

def main():
    base_label, legs = sys.argv[1], sys.argv[2:]
    base = load(base_label)
    if not base: sys.exit(f"no dec-{base_label}.json")
    print(f"# baseline: {base_label}")
    for n, r in sorted(base.items(), key=lambda kv: int(kv[0])):
        print(f"   {n} stream(s): {r['median_total_tps']:.2f} tok/s  spread {r['spread_pct']:.2f}%  "
              f"round {r.get('median_round_ms') or float('nan'):.2f} ms  "
              f"rounds {r.get('rounds')}  tokens {r.get('tokens')}")
    for label in legs:
        leg = load(label)
        print(f"\n## leg {label}")
        if not leg:
            print("   (no capture)"); continue
        ok, bad = contract_ok(label)
        verdict = []
        for n in sorted(leg, key=int):
            if n not in base: continue
            b_, l_ = base[n], leg[n]
            d = (l_["median_total_tps"] / b_["median_total_tps"] - 1) * 100
            need = max(2.0, 2 * b_["spread_pct"])
            rb = b_.get("median_round_ms"); rl = l_.get("median_round_ms")
            dr = ((rl / rb - 1) * 100) if (rb and rl) else float("nan")
            tb, tl = b_.get("rounds"), l_.get("rounds")
            print(f"   {n} stream(s): {l_['median_total_tps']:.2f} tok/s  {d:+.2f}%  "
                  f"(gate needs +{need:.2f}%)  round {rl or float('nan'):.2f} ms {dr:+.2f}%  "
                  f"rounds {tb}->{tl}  tokens {b_.get('tokens')}->{l_.get('tokens')}")
            verdict.append((n, d, need))
        wins = [v for v in verdict if v[1] >= v[2]]
        regress = [v for v in verdict if v[1] < -1.0]
        print(f"   contract: {'IDENTICAL' if ok else ('BROKEN on ' + ','.join(bad)) if ok is False else 'unknown'}")
        if ok is False:
            print("   VERDICT: FAIL - output changed, revert regardless of speed")
        elif wins and not regress:
            print(f"   VERDICT: PASS at {', '.join(w[0] for w in wins)} stream(s), no load worse than -1%")
        elif regress:
            print(f"   VERDICT: FAIL - regression at {', '.join(r[0] for r in regress)} stream(s)")
        else:
            print("   VERDICT: INCONCLUSIVE - no load clears its gate")

main()
