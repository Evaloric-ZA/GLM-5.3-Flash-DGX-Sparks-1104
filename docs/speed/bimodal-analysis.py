#!/usr/bin/env python3
# Bimodality analysis: same windows taking longer, or a different window mix?
# Joins dec-bimodal-c4.json trials to timestamped dft records and the sampler csv.
import json, re, statistics as st, sys, os

D = "docs/speed"
dec = json.load(open(f"{D}/dec-bimodal-c4.json"))["4"]
trials = dec["trials"]

# trial time windows from T0 (first argv) + cumulative elapsed
t0 = float(sys.argv[1])
bounds, acc = [], 0.0
for t in trials:
    bounds.append((t0 + acc, t0 + acc + t["elapsed"]))
    acc += t["elapsed"]

rec = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z?) .*?(dft r=(\d+) a=(\d+) k=(\w) g=(\d) v=([\d.]+) d=([\d.]+) s=(\d+) w=(\d))")
def epoch(s):
    s = s.rstrip("Z").split("+")[0]
    head, _, frac = s.partition(".")
    import calendar, datetime
    t = datetime.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S")
    return calendar.timegm(t.timetuple()) + float("0." + (frac or "0")[:6])

by_trial = [[] for _ in trials]; loose = 0
for line in open(f"{D}/dft-bimodal.log", errors="replace"):
    m = rec.search(line)
    if not m: continue
    ts = epoch(m.group(1))
    hit = next((i for i, (a, b) in enumerate(bounds) if a - 1 <= ts <= b + 1), None)
    if hit is None: loose += 1; continue
    by_trial[hit].append(dict(r=int(m.group(3)), v=float(m.group(7)), d=float(m.group(8)), w=int(m.group(10)), a=int(m.group(4))))

print(f"trials: {len(trials)}  matched records: {sum(len(x) for x in by_trial)}  unmatched: {loose}")
print(f"{'trial':>5} {'tok/s':>6} {'round ms':>8} | {'n':>4} {'med r':>5} {'med v':>7} {'p90 v':>7} {'med d':>7} {'%w=1':>5}")
for i, (t, recs) in enumerate(zip(trials, by_trial)):
    if not recs:
        print(f"{i+1:>5} {t['total_tps']:6.1f} {'?':>8} | no records"); continue
    vs = sorted(r["v"] for r in recs)
    w1 = sum(1 for r in recs if r["v"] and r["w"] == 1) / len(recs) * 100
    print(f"{i+1:>5} {t['total_tps']:6.1f} {st.mean(t['round_ms']) if t.get('round_ms') else 0:8.1f} | {len(recs):4d}"
          f" {st.median([r['r'] for r in recs]):5.1f} {st.median(vs):7.2f} {vs[int(.9*len(vs))]:7.2f}"
          f" {st.median([r['d'] for r in recs]):7.2f} {w1:5.0f}")

# v at matched r: the decisive compare, fast half vs slow half by trial tps
med = st.median(t["total_tps"] for t in trials)
fast = [i for i, t in enumerate(trials) if t["total_tps"] >= med]
slow = [i for i, t in enumerate(trials) if t["total_tps"] < med]
print(f"\nfast trials {[(i+1, round(trials[i]['total_tps'],1)) for i in fast]}")
print(f"slow trials {[(i+1, round(trials[i]['total_tps'],1)) for i in slow]}")
for r in (2, 3):
    fv = [x["v"] for i in fast for x in by_trial[i] if x["r"] == r]
    sv = [x["v"] for i in slow for x in by_trial[i] if x["r"] == r]
    if fv and sv:
        print(f"  r={r}: v fast med {st.median(fv):6.2f} (n={len(fv)})  slow med {st.median(sv):6.2f} (n={len(sv)})"
              f"  delta {st.median(sv)-st.median(fv):+6.2f} ms")

# sampler correlation at trial midpoints
sp = f"{D}/bimodal-sampler.csv"
if os.path.exists(sp):
    rows = []
    for ln in open(sp).read().splitlines()[1:]:
        p = ln.split(",")
        if len(p) >= 7:
            try: rows.append((float(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6])))
            except ValueError: pass
    print("\nsampler at trial midpoints (free_mb avail_mb gpuMB smclock temp util):")
    for i, (a, b) in enumerate(bounds):
        mid = (a + b) / 2
        near = min(rows, key=lambda r: abs(r[0] - mid)) if rows else None
        if near: print(f"  t{i+1:<2} {near[1]:>6} {near[2]:>6} {near[3]:>7} {near[4]:>5} {near[5]:>3} {near[6]:>4}")
