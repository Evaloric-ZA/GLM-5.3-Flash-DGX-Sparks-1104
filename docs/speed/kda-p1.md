# kda P1 - dispatch map, byte model, occupancy, and the falsifiable priors (CPU-only, 2026-11-03 ~11:40Z)

Scope, bounded first: this memo resolves **the decode verify path only** - `kda_wide` defaults TRUE
(`TF_GLM_KDA_DECODE_WIDE`, cuda_forward.zig:225 "the Python family's default"), so verify rows run the
**three-kernel chain** (prep -> step -> out), NOT the fused single kernel (grid heads, 1024 thr - untimed here,
flagged), and not chainSegs (shared windows; n/a at 2 rows). wide.zig's kda_rows is the METAL family - not this
engine; struck from the map. The 12.99 ms bucket = chainRows' three launches + inter-launch gaps, 34 layers.

## Constants (source, not inferred)

Per rank at TP2: **heads = 32** (config 64, `kda_heads /= ranks`), dk = dv = 128, conv 4. State per layer per
rank: 32 x 128 x 128 fp32 = **2 MiB**. Verify 2 rows: stepWarps(2) -> wp=4 (per the doc comment; the
rows->warps table has a test), step grid {32, dv/4/4=8} = 256 blocks x 128 thr; prep grid {2,32} = 64 x 512;
out {2,32} x 128. No dynamic smem on any of the three launches (static/none) - the 48 KiB opt-in question does
not arise; thread occupancy at step = 32K threads = ~1/3 of the 98K the 48 SMs hold, prep/out far sparser.

## Byte model per layer per rank at 2 rows (density assumptions stated)

Dominant: state_in read + state_out write, fp32: **4.0 MiB**. Saves (k fp32, v bf16, g/b fp32) + q/y wide
scratch + p reads: ~0.15 MiB. Total ~4.2 MiB/layer -> 34 layers = **143 MiB per forward per rank** (x2 ranks in
parallel; the profiled rank's 12.99 ms is one rank's). Achieved if 12.99 ms were bandwidth: **11 GB/s = 4% of
the 273 GB/s box.** Even doubling for re-reads and save traffic: single-digit percent. Cf the out-proj finding
(15-41%): kda is an order of magnitude further from bandwidth-limited.

## The priors (falsifiable, ranked by what would kill them)

P1 - **latency/occupancy-bound serial scan, not bandwidth** (the 4% figure): the delta-rule chain is sequential
in the key index inside step's blocks; at 2 rows the machine is mostly idle (256 small blocks, 1/3 thread
occupancy, dependent-loop latency). Kill by: ncu showing high SM throughput at low DRAM BW - would flip the
story to compute/latency-structural and point at the algorithm, not the schedule.

P2 - **the rows-amortization signature is already in the record and consistency-checks**: b=5.68 ms/marginal
row vs a=38.73 fixed - rows share each key step's loads; the fix that helps 1-row helps the shipped shape less.
Any proposal must state its gain at rows=2..4, not at rows=1.

## Candidate list (the memo's deliverable; no plan, no code)

C1 - **Merge prep into step** (or prep+out): 102 launches -> 68 per forward; if per-launch fixed cost and tail
gaps are material at 128 us/layer-average, this is pure scheduling, bitwise-safe by the dispatch-map standard
(same kernels' math, order preserved), and the cheapest thing that can move the number. Prior: small-to-moderate;
the 12.99 includes 3-kernel gaps nobody has measured - P2-later reads them from event pairs before anything.

C2 - **Chunked/WY parallel form of the scan** (parallelize the key loop): the real algorithmic prize - turns
the serial chain into block-parallel prefixes. Big rewrite, numerics-bearing (reassociation of the decay
products - the router_ks doctrine applies: a FIXED chunking chosen once, documented, contract-arbitrated).
Only after C1 and an ncu pass; flagged now so nobody mistakes C1's ceiling for the algorithm's.

C3 - **State precision** (fp32 -> bf16/tf32 state, 2x traffic cut): a NUMERICS change - measure-only, user
approval required, the paired-NLL instrument decides. Listed for completeness; not part of the scheduling case.

C4 - **Two rows in one step block** where the layout allows (row-vectorized inner step): raises arithmetic
intensity per block without touching the math per se; dispatch-map test required (m-thresholds, the b16
lesson generalized: any rows-conditional kernel-family switch re-earns the check).

## Standing rules carried

Constants from source and factors checked before conclusions (the smem lesson); scope = the wide 3-kernel path
actually resolved, fused/segs paths explicitly untime'd; the contract precondition outranks every number here;
the decay series keeps the pair - this memo is CPU-only and commits with its own bundle.
