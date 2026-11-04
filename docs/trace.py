#!/usr/bin/env python3
"""Fit GB10's real verify-cost line from TF_GLM_DFLASH2_TRACE records.

The engine logs one line a verified window (cuda_lanes.zig observe()):
    dft r=<rows> a=<landed> k=<h|c|n> g=<sampled> v=<verify ms> d=<drafter ms> s=<slot> w=<alone> lam=<..> c=<chances,>
`w=1` is a lone window (the only stream), `w=0` a shared round.

What `v=` is, verified in the source (this decides whether comparing it with the compile-time tables
is like-for-like, which the panel refused to assume):
  lone   cuda_lanes.zig:452-454  t0 = now(); x.verify(ids[0..rows]); observe(..., since(t0), true)
                                 -> v is the wall ms of ONE verify forward of `rows` rows.
  shared cuda_lanes.zig:489-493  t0; x.verifyShared(parts); ms = since(t0)
                                 -> v is ONE shared forward's ms, and the SAME ms is then recorded
                                    against every lane in it.
And lanes/config.zig:30,37 define the tables as "forward ms by exact window width" and "shared
forwards' ms by total rows". So v and the tables price the same thing, and the comparison is valid.

Consequence for the shared fit: a shared record's `r=` is that lane's own rows, while its `v=` belongs
to the whole forward. Shared records are therefore grouped back into forwards (consecutive w=0
records carrying the identical ms) and fitted against the forward's TOTAL rows, which is what
`shared_costs.get(rows)` is indexed by (lanes/depth.zig:305). Fitting shared v against one lane's rows
would be wrong.

Usage: trace.py <logfile|-> [label]      (reads `docker logs` output or a saved file)
Prints: per-rows medians and counts, the least-squares line a + b*(rows-1) with residuals, the
drafter's pass ms, and the per-position conditional landing rates derived from the `a=` counts.
"""
import json, os, re, statistics, sys

LINE = re.compile(r"dft r=(\d+) a=(\d+) k=(\w) g=(\d) v=([\d.]+) d=([\d.]+) s=(\d+) w=(\d) lam=([\d.]+) c=([\d.,]*)")

def fit(pts):
    """Least squares on (rows-1, ms) -> a + b*(rows-1), plus R^2 and per-x residuals."""
    if len({x for x, _ in pts}) < 2: return None
    n = len(pts); sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    if abs(den) < 1e-9: return None
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    ybar = sy / n
    sst = sum((y - ybar) ** 2 for _, y in pts)
    sse = sum((y - (a + b * x)) ** 2 for x, y in pts)
    return {"a": a, "b": b, "r2": (1 - sse / sst) if sst > 0 else None, "n": n,
            "rmse": (sse / n) ** 0.5}

def shared_forwards(recs):
    """Regroup shared records into forwards: consecutive w=0 records with the identical ms are one."""
    fwd, cur = [], None
    for r in recs:
        if r["alone"]:
            if cur: fwd.append(cur); cur = None
            continue
        if cur is not None and abs(cur["v"] - r["v"]) < 1e-9:
            cur["rows"] += r["rows"]; cur["lanes"] += 1
        else:
            if cur: fwd.append(cur)
            cur = {"rows": r["rows"], "v": r["v"], "d": r["d"], "lanes": 1,
                   "landed": r["landed"], "alone": False}
    if cur: fwd.append(cur)
    return fwd

def report(recs, kind, out):
    if kind == "shared":
        sel = shared_forwards(recs)
        if sel:
            lanes = {}
            for f in sel: lanes[f["lanes"]] = lanes.get(f["lanes"], 0) + 1
            print(f"\n(regrouped {sum(f['lanes'] for f in sel)} shared records into {len(sel)} "
                  f"forwards; lanes per forward: {dict(sorted(lanes.items()))})")
    else:
        sel = [r for r in recs if r["alone"]]
    if not sel:
        print(f"\n## {kind}: no records"); return
    print(f"\n## {kind} {'forwards' if kind == 'shared' else 'windows'}: {len(sel)} observations"
          + ("  (rows = TOTAL rows in the forward)" if kind == "shared" else ""))
    print(f"{'rows':>5s} {'n':>6s} {'median v ms':>12s} {'mean v ms':>11s} {'p10':>8s} {'p90':>8s}")
    by = {}
    for r in sel: by.setdefault(r["rows"], []).append(r["v"])
    for rows in sorted(by):
        v = sorted(by[rows])
        p = lambda q: v[min(len(v) - 1, int(q * len(v)))]
        print(f"{rows:5d} {len(v):6d} {statistics.median(v):12.2f} {statistics.mean(v):11.2f} "
              f"{p(0.1):8.2f} {p(0.9):8.2f}")
    # fit on medians (robust to scheduling outliers) and on all points
    med_pts = [(rows - 1, statistics.median(by[rows])) for rows in sorted(by)]
    all_pts = [(r["rows"] - 1, r["v"]) for r in sel]
    fm, fa = fit(med_pts), fit(all_pts)
    for name, f in (("medians", fm), ("all points", fa)):
        if f:
            print(f"  fit on {name:10s}: a = {f['a']:.2f} ms, b = {f['b']:.2f} ms/row  "
                  f"(b/a = {f['b']/f['a']:.3f}, r2 = {f['r2'] if f['r2'] is None else round(f['r2'],4)}, "
                  f"rmse {f['rmse']:.2f}, n={f['n']})")
    if fm:
        print("  residual check (median v minus fitted):")
        for x, y in med_pts:
            print(f"    rows {x+1}: {y - (fm['a'] + fm['b']*x):+7.2f} ms")
    d = [r["d"] for r in sel if r["d"] > 0]
    if d: print(f"  drafter pass: median {statistics.median(d):.2f} ms, mean {statistics.mean(d):.2f} ms, n={len(d)}")
    out[kind] = {"fit_medians": fm, "fit_all": fa,
                 "by_rows": {str(k): {"n": len(v), "median": statistics.median(v)} for k, v in by.items()},
                 "drafter_ms_median": statistics.median(d) if d else None}

def positions(recs, out):
    """P(draft at position k lands | all before it landed), from the landed counts."""
    print("\n## landing by chain position (windows with >= k drafts offered)")
    print(f"{'pos':>4s} {'offered':>8s} {'landed':>7s} {'cond. rate':>11s}")
    rows = {}
    for k in range(1, 8):
        off = [r for r in recs if r["rows"] - 1 >= k and r["landed"] >= k - 1]
        if not off: continue
        land = [r for r in off if r["landed"] >= k]
        rows[k] = {"offered": len(off), "landed": len(land), "rate": len(land) / len(off)}
        print(f"{k:4d} {len(off):8d} {len(land):7d} {len(land)/len(off):11.3f}")
    out["positions"] = rows

if __name__ == "__main__":
    src = sys.argv[1]
    text = sys.stdin.read() if src == "-" else open(src, errors="replace").read()
    recs = []
    for m in LINE.finditer(text):
        recs.append({"rows": int(m.group(1)), "landed": int(m.group(2)), "kind": m.group(3),
                     "sampled": m.group(4) == "1", "v": float(m.group(5)), "d": float(m.group(6)),
                     "slot": int(m.group(7)), "alone": m.group(8) == "1",
                     "lam": float(m.group(9)),
                     "chances": [float(x) for x in m.group(10).split(",") if x]})
    print(f"# {len(recs)} dft trace records from {src}")
    if not recs:
        print("# NO TRACE RECORDS - TF_GLM_DFLASH2_TRACE was not on, or the log was rotated.")
        sys.exit(1)
    out = {"records": len(recs)}
    report(recs, "lone", out)
    report(recs, "shared", out)
    positions(recs, out)
    print(f"\n# sm_120 seed for comparison: a = 9.40 ms, b = 1.31 ms/row (b/a = 0.139), draft 1.37 ms")
    print(f"# gb10 lane tables:  window 52.0 + 6.8/row (b/a = 0.131); shared 76.0 + 4.1/row (b/a = 0.054)")
    if len(sys.argv) > 2:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speed", f"trace-{sys.argv[2]}.json")
        json.dump(out, open(p, "w"), indent=1, default=str); print(f"# wrote {p}")
