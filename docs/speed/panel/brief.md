# Expert panel: find untried decode + prefill speed on GLM-5.3-Flash, 2x DGX Spark

You are one of three experts (the lab's agent, Codex, GLM-5.3-Flash). Round 1: answer independently.
Rounds 2-3: you will see the others' answers and we converge on one agreed plan, which is then
executed unattended tonight on the real pair.

## The system

GLM-5.3-Flash (large MoE, 288 routed experts, first_k_dense_replace 3, KDA/DSA attention, 1,048,576-token
window), served from a 4 bpw EXL3 checkpoint by TensorFold's native Zig/CUDA engine
(`tensorfold-native` 1.1.9 = upstream f350cd3e + 95 local patches), built for sm_121.

Hardware: **two NVIDIA DGX Sparks** (GB10), tensor-parallel 2, one rank per Spark, cabled on one
ConnectX-7 QSFP port (two PCIe links -> two netdevs, two RDMA devices). spark3 = rank 0 (HTTP API,
vision tower), spark4 = rank 1. NCCL over the cable; the engine's own RoCE one-shot RDMA all-gather
carries small exchanges (<= 4 MiB shards).

**GB10 is unified memory: 128 GB LPDDR5X shared by CPU and GPU, ~273 GB/s total.** This is the single
most important difference from the machine most of the prior work was done on (2x RTX PRO 6000, sm_120,
1.71 TB/s GDDR7 per card). Weights, KV pages, prompt cache, page cache and every other process on the
box draw on the same pool and the same bandwidth.

Live config: `DENSE=q8` (TF_GLM_DENSE_BITS=8), `HEAD=bf16`, `KV=fp8` on shared 512-row pages,
`PARALLEL=4`, `CONTEXT=1048576`, `MEMORY_RESERVE_GIB=8`, `HOST_CACHE_GIB=0`, `ABSORB_MMA=1`, `W8A8=0`,
DFlash2 drafter at 4 bits, thinking off. Per rank: model 83.58 GiB, 4,098 KV pages (12.47 GiB,
2,098,176 tokens), 417 MiB fixed per stream, 4.4 GiB device prompt cache, 18.1 GiB room after reserve.
Rank 0 binds (sits ~0.4 GiB over its reserve); spark4 has ~5 GiB the pair cannot use.

## Measured now (3 cold trials/point, fresh content so no prefix-cache hit, reproduces to 0.2%)

| | value |
| --- | --- |
| decode, prose, greedy, 1 / 2 / 4 streams | 35.3 / 53.4 / 72.5 tok/s total (35.3 / 26.7 / 18.1 per stream) |
| prefill @23.5K / @95K | 1,735 / 1,759 tok/s (TTFT 14.6 s / 57.5 s at the older 1,609) |
| DFlash2 draft acceptance | 0.525 (157/299) |
| NLL (4 held passages) / 10-greedy-answer sha | 0.7523 / `11ceddde19d4` |
| load | ~120 s |

Decode is **bandwidth-bound**. Byte budget per rank per token, from the sm_120 profile and consistent
with our timings: int8 dense/attn projections ~3.8 GB + routed EXL3 experts ~2.1 GB + bf16 LM head
~0.63 GB ~= 6.5 GB. At 35.3 tok/s (28.3 ms/step) that is ~230 GB/s of the ~273 GB/s the box has.
Stream scaling confirms the frame: 1 -> 4 streams is only 2.05x because dense/head bytes are shared
across streams while each stream's routed experts diverge (4.4 + 4*2.1 = 12.8 GB for 4 tokens).

So on this machine: **time = bytes moved**, and the two ways to win are (a) move fewer bytes per
token, (b) get more accepted tokens out of each weight read.

## Hard constraints

1. **Quality must not regress.** This is the user's firm position after being burned twice:
   - `W8A8=1` (int8 activations in the prompt dense GEMMs): +10.5% prefill, but per-passage NLL moves
     +-0.03 and it answered "spell necessary backwards" as `yrassen` where the baseline gives the
     correct `yrassecen`. **Stashed by the user. Not an open option.** Patch 0064 stays inert.
   - `DENSE=q4`: +7% prefill but NLL 0.8005 vs 0.7535, same spelling failure. **Rejected.**
   Prefer changes that are **exact by construction** (bit-identical output, or output that is the
   target model's picks by definition). Anything numeric needs a per-passage, paired NLL gate, and
   the user must approve it before it ships - it cannot ship unattended tonight.
2. Keep the 1,048,576-token window and 4 streams. Do not trade context or streams for speed.
3. One pair of Sparks, and they are the user's workstations. A rebuild is ~7 min (412 s image build)
   plus ~2 min load. The server must be back up and healthy by morning.
4. Unattended tonight: every step needs its own measurable gate and a rollback.

## Already tried. Do NOT re-propose without new evidence.

Measured on **GB10, this pair**:
- 4,096-row prompt chunk (patches 0094/0095, `prefill_rows` 2048 -> 4096, dflash `MAX_CTX` to match):
  **+7%, shipped, bit-exact.** Note this was **0.918x (a loss) on sm_120** - see "sm_120 verdicts do
  not transfer" below.
- `ABSORB_MMA=1`: +8.0%, byte-identical answers, shipped.
- Dense width: bf16 1,288/1,262 tok/s at 87.00 GiB and **half the KV pool**, quality indistinguishable
  from q8 (paired per-token over 1,051 tokens of 12 domains: mean delta -0.00394, 95% CI -0.0153..+0.0074).
  bf16 dominated; q4 rejected on quality. **q8 settled. q6 has never been measured on GB10.**
- No prefill gap vs the older v1.0.7 Spark build at equal precision (-0.7% @23.5K, +2.1% @95K). Its
  prompt-path patches (MSA attention, own-front, microbatch, bf16 exchange, overlap2) need no porting.
- The cable is ~3% of TTFT. Not a prefill lever.
- `MEMORY_RESERVE_GIB`: the KV pool grows only in whole-window steps of 2,048 pages (~6.4 GiB), so
  pool and prompt cache are mutually exclusive at the next step. 8 is the chosen point.
- `TF_GLM_PREFILL_REDUCE=split`, `TF_GLM_PROMPT_PAIR`, `TF_GLM_INDEX_SPLIT=2048`,
  `TF_GLM_OVERLAP_PIECES=3`, `TF_GLM_MM` are all already on by default.

Measured on **sm_120 (2x RTX PRO 6000)**, extensive overnight campaigns, all with numbers against them:
- CUDA graphs for the decode chain: idle is only 0.39 ms/step, launch lead p50 2.3 ms, a drafter graph
  recovered <= 0.02-0.05 ms; the big gaps are host syncs around sampling/commit. PDL (programmatic
  dependent launch) is the winning form of this idea and is already in (patches 0067-0091).
- L2 prefetch / evict-first / persisting-L2: in-kernel bulk prefetch +14-24% **slower**; pre-GEMV-wait
  prefetch worse everywhere.
- Expert-parallel MoE at TP=2: byte-neutral (half of 8 sharded experts = 4 whole experts), adds imbalance.
- Drafter-side micro-optimisation: GPU selector, row tiles, drafter CUDA graphs, sum folds, sparse
  drafter head (79-89% coverage vs 99.3% break-even): all <= measurement spread or negative.
- Fused rot_in+gate/up+epilogue: +6.6-9.2% slower. Deeper register/cp.async rings, V4 staging,
  evict_first, register caps, 2x unroll, one-line PDL triggers on kda_out/expand_v: all neutral or worse.
- Staged router, TPM=128 expert items (0.918x), KDA step row overlap, twin copy-engines (0-1%).
- Uneven TP split / expert rebalancing by clock: changes the down-projection's sum order (numerics).
- FlashInfer / vLLM / SGLang / TRT-LLM kernel drop-ins: nothing matches an EXL3 mcg trellis engine.
  DCP for speed, tree verification on KDA (no exact kernel), spin-barrier top-k at batch 1,
  cooperative grid barriers under TP2, Machete (Hopper-only), DeepGEMM fp8/fp4 indexer logits
  (SM90/100 only): all checked and rejected for this stack.
- `TF_GLM_INDEX_CUDA` at prefill: +1.0-1.5%, within 2x the spread.
- q8 LM head on sm_120: +1-2% only (it was at the bandwidth roof there). **On GB10 the same change is
  ~4.7% of the per-token byte budget - never measured here.**

### sm_120 verdicts do not transfer, and that is the biggest opening

The 4,096-row chunk was a **loss on sm_120 (0.918x) and a +7% win on GB10**, because the RTX cards are
near a compute/occupancy roof while GB10 is starved for bandwidth and amortising a weight read over
more rows is nearly free. **Any sm_120 verdict whose mechanism was "we are at the compute or occupancy
roof" or "the extra bytes/rows are not free" deserves re-examination on GB10, and vice versa.** Say
explicitly which of the dropped items above you think flip on GB10, and why, in mechanism terms.

## Findings from my own read of the engine source tonight (facts, not a plan)

The tree is upstream f350cd3e + the 95 patches, at `/home/zurih/TF/.src/tf`
(`zig/src/families/glm/` is the GLM family, `zig/kernels/cuda/` the kernels).

1. **The speculative-decode window rule is priced with sm_120 constants.**
   `draft_rule.zig:178`: `verify_a = 9.4` ms, `verify_b = 1.31` ms/row, `draft_ms = 1.37`, with the
   comment "measured: sm_120 x8 400 W". Settable at runtime via `TF_GLM_VERIFY_MS=a,b,d`.
2. **The lane scheduler's GB10 cost tables look mispriced for this build.**
   `cuda_lanes.zig:24-25`: `gb10_window_costs = costTable(max_rows, 52.0, 6.8)` and
   `gb10_shared_costs = costTable(shared_rows, 76.0, 4.1)`, commented "a window's ms by rows on two
   GB10s (EXL3, TF_GLM_PROF: 86 to 107 ms at 5 to 8)". These are **compile-time** and feed the lane
   core at `cuda_lanes.zig:128-129`. Our measured plain decode round is **~28 ms**, not 52. Both
   tables keep a row at ~13% of a round, the same ratio as the sm_120 seeds - i.e. they look scaled,
   not independently measured for a bandwidth-bound machine where extra verify rows should be far
   cheaper relative to the round.
3. **The default window rule is `cut`, not the cost-aware one.** `cuda_lanes.zig:592-598`: a fixed
   chain-confidence cut, `TF_GLM_DRAFT_CONFIDENCE`, **default 0.7**, no cost awareness at all. The
   cost-aware `tpm` rule (`TF_GLM_DRAFT_RULE=tpm`, Dinkelbach on E(k) - lambda*T(k) with online
   calibration of landing rates and an online fit of the verify line) was evaluated **on sm_120** and
   rejected there as +0.1% lone / +1.5% shared. On GB10's cost curve that comparison is void.
   `max_drafts = 3` (`cuda_engine.zig:28`), `max_held = dflash.BLOCK - 1 = 7`, `shared_rows = 32`.
4. **Speculation is exact by construction** - `draft_rule.zig` header: "Which rows are offered never
   changes the tokens: every reply is the target's picks." So anything in this area is quality-safe.
5. On a unified-memory device the pinned host prompt-state tier is disabled (`HOST_CACHE_GIB=0`)
   because "a state in it costs a copy to restore" - but on GB10 that copy is DRAM-to-DRAM inside the
   same physical memory. Whether the engine can map instead of copy on an integrated device has not
   been looked at.
6. Available off-switches / knobs already in the stack: `TF_GLM_DSEL`, `TF_GLM_PDL_CHAIN`,
   `TF_GLM_MM_TWO`, `TF_GLM_TOPK_SUM`, `TF_GLM_SHARED_SEGS`, `TF_EXL3_SHFL`, `TF_EXL3_PDL`,
   `TF_GLM_KDA_STEP_WARPS`, `TF_GLM_HC_SPLIT`, `TF_GLM_SHARED_WRITES`, `TF_GLM_HC_PART`,
   `TF_GLM_DSCORES`, `TF_GLM_PROF`, `TF_GLM_DFLASH2_TRACE`, `TF_GLM_DFLASH2_FORCE` (force a window's
   rows, for a cost sweep), `TF_GLM_DRAFT_RULE`, `TF_GLM_DRAFT_LAMBDA`, `TF_GLM_SHARED`.

Treat these five as context, not as the answer. I want your independent ideas first.

## What I am asking you for

Look across the **whole** inference stack: kernels, memory movement and placement, quantisation,
MoE execution, attention, speculative decoding, tensor parallelism, communication, CUDA graphs,
scheduling, synchronisation, batching, launch overhead, CPU overhead, algorithmic design, the
serving/admission layer - and anything not on that list. Nothing is off limits except the two
constraints above (quality, and context/streams).

Deliver, in markdown:

**A. Up to 8 candidates, ranked by (expected gain x confidence) / effort.** For each:
   - mechanism, in the terms of *this* machine (bytes/token, bandwidth, launch counts, overlap);
   - which file(s)/function(s) it touches in the tree above, as precisely as you can;
   - expected gain on decode and on prefill, with the arithmetic that gives you that number;
   - **exactness class**: (i) bit-exact by construction, (ii) exact but needs a byte/contract check,
     (iii) changes numerics -> needs the user's gate, cannot ship tonight;
   - cost: env-only / patch + rebuild / multi-day;
   - **how to measure it in one night**, and the **kill criterion** that stops it.
**B. Your single highest-conviction pick** and why.
**C. What you think the other two experts will get wrong.**
**D. Which previously-dropped item you think flips on GB10, with the mechanism.**

Be concrete and quantitative. Say "I do not know" rather than guessing a number. Order the night so
that env-only measurements that need no rebuild come first, and so the pair is serving by morning.
