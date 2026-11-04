#!/usr/bin/env python3
"""Aggregate TF_GLM_PROF=1 reports: where a forward's time goes, and how much of it is not GPU.

The engine logs, every 64 calls (cuda_prof.zig:96-98):
    glm prof <label> (<n> calls): wall <w> ms, gpu <g> ms a call; gate .. embed .. kda .. mla ..
    gather_a .. dense .. router .. routed .. shared .. gather_b .. head .. draw .. mtp .. commit ..
`label` is "verify 1 row".."verify 8 rows" for decode windows, or a prompt label for prefill.

wall - gpu is the part of the call that is NOT GPU work on the profiled stream: host time, syncs and
anything not bracketed by the section events. That difference is the quantity we are hunting.

Profiling inserts events, so treat these as attribution, not as a timing gate.

Usage: prof.py <logfile|-> [label-substring]
"""
import re, sys, statistics
from collections import defaultdict

HEAD = re.compile(r"glm prof (.+?) \((\d+) calls\): wall ([\d.]+) ms, gpu ([\d.]+) ms a call;(.*)")
SEC = re.compile(r"([a-z_]+) ([\d.]+)")

def main():
    text = sys.stdin.read() if sys.argv[1] == "-" else open(sys.argv[1], errors="replace").read()
    want = sys.argv[2] if len(sys.argv) > 2 else None
    rows = defaultdict(lambda: {"wall": [], "gpu": [], "sec": defaultdict(list), "calls": 0, "reports": 0})
    for m in HEAD.finditer(text):
        label, calls, wall, gpu, rest = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), m.group(5)
        if want and want not in label: continue
        r = rows[label]
        r["wall"].append(wall); r["gpu"].append(gpu); r["calls"] += calls; r["reports"] += 1
        for s in SEC.finditer(rest):
            r["sec"][s.group(1)].append(float(s.group(2)))
    if not rows:
        print("# no `glm prof` reports found - TF_GLM_PROF=1 was not set, or the log was rotated.")
        sys.exit(1)
    for label in sorted(rows):
        r = rows[label]
        wall, gpu = statistics.median(r["wall"]), statistics.median(r["gpu"])
        print(f"\n## {label}  ({r['reports']} reports, {r['calls']} calls)")
        print(f"   wall {wall:8.2f} ms a call")
        print(f"   gpu  {gpu:8.2f} ms a call   ({gpu/wall*100:5.1f}% of wall)")
        print(f"   NOT GPU {wall-gpu:6.2f} ms      ({(wall-gpu)/wall*100:5.1f}% of wall)  <- host time, syncs, unbracketed")
        tot = sum(statistics.median(v) for v in r["sec"].values())
        print(f"   {'section':10s} {'ms':>8s} {'% of gpu':>9s} {'% of wall':>10s}")
        for name, vals in sorted(r["sec"].items(), key=lambda kv: -statistics.median(kv[1])):
            ms = statistics.median(vals)
            print(f"   {name:10s} {ms:8.2f} {ms/gpu*100:8.1f}% {ms/wall*100:9.1f}%")
        print(f"   {'(sum)':10s} {tot:8.2f} {tot/gpu*100:8.1f}% {tot/wall*100:9.1f}%")

main()
