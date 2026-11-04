# The lab's agent, round 1

## The frame: where each phase actually sits against its roof

**Decode is at 84% of the memory roof. There is almost nothing left in decode kernels.**
6.5 GB/rank/token / 28.3 ms = 230 GB/s of the box's ~273 GB/s. Nobody gets past ~85-90% of LPDDR5X
peak. So every remaining decode lever is one of:
  (a) fewer bytes per token (quantisation -> numerics gate, cannot ship tonight);
  (b) **more accepted tokens per weight read** (speculation -> exact by construction);
  (c) not stealing the same bandwidth with CPU-side work.
A faster MoE kernel, better PDL, graphs, prefetch: all dead on this machine, and the sm_120 campaign
already found them dead on a machine with 6x the bandwidth. Stop looking there.

**Prefill is nowhere near any roof I can identify, and I do not know what binds it.**
At 1,735 tok/s with 4,096-row chunks: weights are read once a chunk, so 83.58 GiB / (4096/1735 s)
= ~35 GB/s - 13% of the memory roof. Arithmetic: ~8B active params/rank -> ~16 GFLOP/token/rank ->
65 TFLOP a chunk in 2.36 s = ~28 TFLOP/s, a small fraction of GB10's bf16 tensor throughput. So
prefill is bound by neither DRAM nor the tensor cores. It responds to arithmetic changes (W8A8
+10.5%, ABSORB_MMA +8%), which points at the SIMT/ALU side: EXL3 trellis dequantisation, `kda_step`
(8% of prefill on sm_120 and the largest SIMT kernel there), the indexer, and glue. **This is an
unprofiled machine and that is the single biggest gap in our knowledge.** `TF_GLM_PROF` exists and
needs no rebuild.

## A. Candidates

### A1. Re-price speculative decoding for GB10. [decode, exact, env-only to measure]
**Mechanism.** A verify round reads the same ~6.5 GB of weights whether it verifies 1 row or 8; only
activation arithmetic scales with rows. So on GB10 a verify row is nearly free *relative to the
round*, while on sm_120 (6x the bandwidth, same FLOPs) a row was 14% of a round. Every piece of the
drafting policy in this build is priced with the sm_120 ratio:
 - `draft_rule.zig:178` seeds `verify_a=9.4 ms`, `verify_b=1.31 ms/row` ("measured: sm_120 x8 400 W");
 - `cuda_lanes.zig:24-25` prices GB10 windows at `52.0 + 6.8*row` and shared forwards at `76.0 + 4.1*row`
   - both keep a row at ~13% of a round, i.e. the sm_120 shape, and the round itself (52 ms) is ~1.9x
   our measured ~28 ms;
 - the **default rule is `cut`** (`cuda_lanes.zig:592-598`) with `TF_GLM_DRAFT_CONFIDENCE` **0.7**: a
   chain stops when the product of its chances falls under 0.7. With per-draft chances around 0.8
   that stops after **one or two drafts** (0.8, 0.64 < 0.7).
**The arithmetic that makes this the night's headline.** The cost-aware rule offers a draft while its
chain chance exceeds `lambda * b` (Dinkelbach, `draft_rule.zig:155-169`). With lambda = 35.3 tok/s =
0.0353 tok/ms and a true GB10 row price b of ~1-2 ms, the bar is **0.04-0.07**, not 0.7. An exact
rule should be offering **all 7** held drafts almost always. If conditional landing is ~0.7, E(tokens)
per round = 1 + 0.7 + 0.49 + ... ~= 3.1 at a cost of a + 7b ~= 24 + 10 = 34 ms -> ~90 tok/s against
today's 35.3. Even half of that is the largest number on this page, and **speculation cannot change
the output**: `draft_rule.zig` line 7, "Which rows are offered never changes the tokens: every reply
is the target's picks."
**Measure (no rebuild).** (1) `TF_GLM_DFLASH2_FORCE=1`,`=2`,`=4`,`=8` with `TF_GLM_DFLASH2_TRACE=1`
gives the true verify line a + b*rows and the drafter's ms directly -> fit (a,b,d). (2) Then compare
`cut` at confidence 0.7 / 0.3 / 0.0 and `TF_GLM_DRAFT_RULE=tpm` with `TF_GLM_VERIFY_MS=<a,b,d>` from
(1). All env-only, one container restart each, ~6 min a leg.
**Then patch.** Re-seed `draft_rule.zig:178-180` and the two `cuda_lanes.zig` tables from the fit; if
the economics say "offer everything", also consider raising `dflash.BLOCK`/`max_held` past 7 and
`engine.max_drafts` (3) for the MTP path.
**Kill criterion.** Greedy reply sha != `11ceddde19d4` on any leg (would mean a real bug, since this
is exact by construction), or forced-8 decode slower than forced-1 per accepted token.
**Exactness.** (i) bit-exact by construction.

### A2. The shared-forward cost table, for multi-stream throughput. [decode, exact, patch]
**Mechanism.** 1 -> 4 streams scales only 2.05x. Dense/head bytes (4.4 GB of 6.5) are read once per
*forward*, not per stream, so batching more streams' rows into one shared forward is nearly free on
this machine - exactly the 4,096-row-chunk mechanism, one level up. `gb10_shared_costs` prices a
32-row shared forward at 76 + 4.1*31 = **203 ms**; if the truth is ~35-45 ms, the lane core believes
wide shared forwards are catastrophic and serialises streams that should ride together.
**Measure.** `TF_GLM_SHARED=0/1` and the 2- and 4-stream decode suite; with `TF_GLM_DFLASH2_TRACE=1`
read the real shared-forward ms by rows, then patch `cuda_lanes.zig:25` and re-measure.
**Expected.** If shared forwards are mispriced by ~4x, 4-stream total goes from 72.5 toward 100-130
tok/s. Confidence medium: I have not verified how `lanes.config.Cost` is consumed by the lane core.
**Exactness.** (i).

### A3. Profile prefill on GB10 at all. [measurement only, env-only]
`TF_GLM_PROF` per-category timings at 23.5K and 95K. We are optimising prefill blind: every share we
have is sm_120's, and the one time we tested an sm_120 verdict here (4,096 rows) it inverted. Cheap,
and it decides whether any further prefill work is worth a night. **No gain claimed.**

### A4. The byte cuts we have never measured here: `HEAD=q8` and `DENSE=q6`. [decode+prefill, numerics]
`HEAD=bf16` is 0.63 GB of the 6.5 GB/token = **9.7%** of decode bytes; q8 halves it -> ~+5% decode.
On sm_120 the same change measured +1-2% because that machine was not byte-bound - a textbook
sm_120-verdict flip. `DENSE=q6` cuts 3.8 GB to 2.85 -> **-14.6% of bytes** -> ~+15% decode, and q6 was
never measured on GB10 either ("no gain measured" on sm_120, same reason).
**These change numerics, so tonight they are measured and reported, never shipped.** Harness exists:
`docs/quality.py` (NLL + 10 greedy answers), `docs/nllpaired.py` + `cmp.py` for the paired per-token
test that settled bf16-vs-q8. Judge per passage and paired, never on an aggregate.
**Exactness.** (iii). Both are env-only (`HEAD`, `DENSE` in the recipe), no rebuild.

### A5. Stop the CPU stealing decode's bandwidth. [decode, exact, config]
On a discrete card host work is free; here CPU traffic and GPU traffic are the same 273 GB/s. GB10 has
10 Cortex-X925 + 10 A725 cores; the engine's hot thread, the HTTP server, the RDMA IRQs and the page
cache all contend. Untried: pin the rank's engine thread to performance cores away from the NIC's IRQ
cores (`--cpuset-cpus`), and measure decode with and without the desktop/other containers. Expected
0-5%, low effort, exact.

### A6. `prefill_rows` 4,096 -> 8,192. [prefill, exact, patch + rebuild]
The +7% at 4,096 came from amortising trellis decode over rows, but `cuda_exl3.zig` names 4,096 as
upstream's amortisation point (PR #1576), so the same mechanism should be ~exhausted. Costs scratch,
and scratch already squeezed the prompt cache at 4,096 (3.1 -> 0.4 GiB at reserve 12). Test only if
A3 says per-chunk fixed cost is still visible. Expected 0-3%.

### A7. Map, don't copy, the host prompt tier on a unified device. [TTFT, exact, bigger]
`HOST_CACHE_GIB=0` because a state there "costs a copy to restore" - but on GB10 that copy is
DRAM-to-DRAM in the same physical memory. If the integrated path can hand the engine a pointer
instead, the prompt cache becomes ~free and large, which is TTFT on every follow-up turn in an
agentic session (today a 23.5K follow-up that misses the 4.4 GiB cache re-prefills in ~14 s). Not a
tok/s number; a latency number. Multi-day, exact. Tonight: only confirm whether the tier's restore
path is a `cudaMemcpy` on an integrated device.

### A8. Draft quality rather than draft depth.
Acceptance is 0.525. If A1 makes rows cheap, the binding constraint becomes *how far the drafter
stays right*, and the lever becomes the drafter (bf16 DFlash2 via `DRAFTER_BITS=0`, +0.6 GiB/rank but
better chains; or the MTP head *beside* DFlash2). Only worth it after A1 lands, because A1 changes
what "good enough" means. Env-only to try `DRAFTER_BITS=0`.

## B. Highest conviction: A1.
It is the only item on the page that is simultaneously large (potentially 1.5-2.5x decode),
**exact by construction** (so it clears the user's quality bar trivially), measurable with no rebuild,
and provably mis-tuned today rather than speculatively improvable: the constants in the file say
"sm_120" in a comment, and the GB10 table prices our 28 ms round at 52 ms.

## C. What the others will get wrong.
Both will propose kernel work - fused glue, better MoE tiling, CUDA graphs, prefetch, FlashInfer-style
attention. On a machine running decode at 84% of its DRAM roof those are rounding errors, and the
sm_120 campaign already measured most of them as neutral-to-negative with far more compute headroom
available. I also expect at least one to propose q4/W8A8-style byte cuts without registering that the
user has ruled them out, or to propose MTP depth increases without noticing that the *policy*, not the
depth, is what is throttling drafting today.

## D. The dropped item that flips on GB10.
**`cut` vs `tpm` (and with it the whole draft-window policy).** On sm_120 `tpm` was +0.1%/+1.5% and
was correctly dropped: with a row at 14% of a round there was little to gain from choosing rows well.
On GB10 a row is ~5% of a round, the optimum window is far deeper, and the fixed `cut` at 0.7 cannot
express that at any setting of its own knob - the cut is on chain probability, which falls
geometrically regardless of how cheap rows are. Second flip: **`q8` LM head and `q6` dense** (+1-2%
and "no gain" on sm_120 purely because that machine was not byte-bound).
