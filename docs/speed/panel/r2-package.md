# Round 2: converge. Answer with FINAL: AGREE or FINAL: AGREE WITH AMENDMENTS.

All three round-1 answers are below, then the points of agreement, the six open disagreements with my
proposed resolution, and the concrete night plan. **No tool calls are needed or wanted for this round**
(Codex: answer from the text; you already verified the source facts). Keep it under ~1,200 words.
End with a line `FINAL: AGREE` or `FINAL: AGREE WITH AMENDMENTS` followed by a numbered amendment list
(each amendment must be actionable tonight, not a research direction).

## Where all three of us already agree

1. **Top lever: re-price speculative decoding for GB10.** All three ranked it #1, independently.
   Mechanism: a verify round re-reads the same ~6.5 GB of target weights whether it verifies 1 row or
   8, so on a bandwidth-starved machine the marginal verify row is far cheaper *relative to the round*
   than on sm_120 - and every constant in the policy is sm_120's (`draft_rule.zig:178`) or an
   unverified GB10 scaling of it (`cuda_lanes.zig:24-25`), while the default `cut` rule
   (`TF_GLM_DRAFT_CONFIDENCE=0.7`) has no cost awareness at all.
2. **Second lever: the lane scheduler's shared-window cost table**, for the 1->4 stream scaling of 2.05x.
3. **A measurement must come first** (Codex and GLM both insisted; I accept): the phase breakdown and
   the real verify cost line, because the 28 ms end-to-end step and the 52 ms table intercept are
   **not established to be the same quantity**, and 0.525 is an aggregate acceptance, not the
   conditional landing chance by chain position.
4. Numerics items (`HEAD=q8`, `DENSE=q6`) are **measured and reported tonight, never shipped**; the
   user approves them or they die.
5. The pair must be serving a healthy, approved configuration by morning, with a protected recovery
   window and a tested rollback.

## The six open disagreements, with my proposed resolution

**D1. How big is the speculation win?** I said "potentially 1.5-2.5x decode"; Codex said "0-8%, a test
range not a prediction"; GLM said "+4-10% plus an unknown scheduler component".
*Resolution: I withdraw my number.* It assumed a conditional landing rate of ~0.7 per position, which
nothing in our data supports - 0.525 is an aggregate over positions and prompts. The plan is identical
under all three estimates, so we commit to the **protocol**, not to a number, and the fit decides.
Agreed?

**D2. Exactness class of the speculation work.** I claimed (i) bit-exact by construction, citing
`draft_rule.zig`'s "which rows are offered never changes the tokens". Codex says (ii): the contract is
the algorithm's intent, but changed batching and state paths still need a real contract test.
*Resolution: adopt (ii).* Gate = identical **token IDs** (not just the 10-answer sha) against the
baseline on greedy prose, code, a chunk-boundary set and one long-context prompt, plus `boundary.py`
clean. Agreed?

**D3. How many rebuilds tonight?** Codex: at most one. GLM: two batched windows.
*Resolution: at most one rebuild*, chosen after Phase 1 from the evidence, carrying only the re-seeded
constants. `prefill_rows` 8192 is **deferred** unless Phase 0's profile shows per-chunk fixed cost is
still a visible share of prefill. Agreed?

**D4. Are the lane tables pricing the same thing as our 28 ms step?** Unresolved by all three of us.
*Resolution: we do not patch `cuda_lanes.zig:24-25` on the 52-vs-28 argument alone.* Phase 0.3 must
produce the realized rows, lone-vs-shared flag and ms per window from `TF_GLM_DFLASH2_TRACE`, and the
patch only happens if the measured shared line disagrees materially with the table in the **same
units the table is in**. If the trace cannot establish that, the table patch is dropped tonight.
Agreed?

**D5. What binds prefill?** My arithmetic says prefill is at neither the DRAM roof (~35 GB/s of 273)
nor the tensor-core roof (~28 TFLOP/s), so something else binds it - plausibly EXL3 trellis
dequantisation and the SIMT kernels (`kda_step` was the largest SIMT kernel in prefill on sm_120).
Neither of you addressed this.
*Resolution: `TF_GLM_PROF` in Phase 0 settles it, and prefill gets no further work tonight unless the
profile names a target.* Prefill is already at parity with the older build; decode at 35 tok/s is the
user-visible problem. Agreed?

**D6. GLM's candidate 3, the broad knob sweep** (`KDA_STEP_WARPS`, `SHARED_SEGS`, `SHARED_WRITES`,
`TOPK_SUM`, `DSEL`, `PDL_CHAIN`, `EXL3_SHFL`, `HC_SPLIT/PART`).
*Resolution: demoted to a time-permitting tail, and bounded.* Each costs a restart, decode sits at 84%
of the DRAM roof so occupancy knobs have little room, and the sm_120 campaign measured most of these
at neutral-or-worse. If the tail runs, at most four knobs, one at a time, each kept only at >= +2%
with identical token IDs. Codex's `TF_GLM_SHARED_SIDE=0` (bandwidth contention from the shared-expert
side stream) is the one I would put first, because its mechanism is unified-memory-specific and
genuinely untested. Agreed?

## The night plan to ratify

Facts for costing: exported `TF_*` variables reach **both** ranks (rank 0 by name, rank 1 through an
env file: `start.sh:147-180`), so every env leg is `export ... && ./start.sh restart`, about 3-4 min
including load and health. A rebuild is ~7 min plus the restart. Harness in `docs/`: `pf2.py` (3 cold
prefill trials at 23.5K and 95K with fresh random content), `quality.py` (NLL + 10 greedy answers),
`nllpaired.py` + `cmp.py` (paired per-token NLL over 12 domains), `boundary.py` (chunk boundaries +
`accepted=N/M` from the container log). **There is no decode benchmark - I will write `docs/dec.py`
(1/2/4 concurrent streams, 3 trials, fresh prompts) as step 0.0.** The live image
`tensorfold-glm53-spark:503f271d5532` stays on disk as the rollback.

- **Phase 0, measure (~60 min, no rebuild).** 0.0 write `dec.py`. 0.1 baseline: decode 1/2/4,
  prefill 23.5K/95K, `quality.py` sha (must be `11ceddde19d4`), startup line, memory low-water.
  0.2 `TF_GLM_PROF=1` + `TF_GLM_DFLASH2_TRACE=1`: phase breakdown and per-window records.
  0.3 forced-width sweep `TF_GLM_DFLASH2_FORCE=1,2,3,4,5,6,7,8` interleaved, trace on, several fresh
  prose and code prompts: realized rows, landed-by-position, verify ms, draft ms, lone vs shared.
  Fit (a,b,d) and **check residuals rather than assuming linearity**.
  **Gate G0:** if the fitted line is within ~15% of the sm_120 shape, the mispricing thesis is dead;
  stop Phase 1-2, go to Phase 3.
- **Phase 1, env-only policy (~90 min, exact, shippable).** `cut` at 0.7 / 0.55 / 0.4 / 0.0, then
  `tpm` with `TF_GLM_VERIFY_MS=<fit>`. Trace **off** for timing. Decode 1/2/4, 3 trials.
  **Gate G1:** >= +2% at one intended load, no > 1% regression at the others, token IDs identical,
  `boundary.py` clean. Keep the winner; otherwise restore the default.
- **Phase 2, at most one rebuild (~45 min).** Re-seed `draft_rule.zig:178-180` and, only if D4's
  condition is met, `cuda_lanes.zig:24-25`, as patch `0096`. **Gate G2:** G1's gates plus the context
  fit unchanged (4,098 pages, 1,048,576 window), both ranks agreeing, `/health` and smoke clean.
- **Phase 3, numerics, measure only (~60 min).** `HEAD=q8` then, if time, `DENSE=q6`: `quality.py`,
  `nllpaired.py` paired against the q8/bf16 baseline, decode 1/2/4. **Restore the approved precision
  afterwards regardless of result.** Report for the user at breakfast; ship nothing.
- **Phase 4, protected restore and verify (>= 30 min reserved).** Serve the best configuration that
  passed an exact gate, else the original. Verify `/health`, `/v1/models`, 4 streams, the full window,
  4,098 pages, a greedy smoke test and one fresh cold prefill. Write everything to
  `docs/speed/` and a single agreement+results document.

Rules throughout: one change at a time; three trials a point; a leg that fails its gate is reverted
immediately; nothing numeric ships; the user's other containers are left alone.

---
# ===== ROUND 1: agent-r1 =====
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

# ===== ROUND 1: codex-r1 =====
# Round 1

The strongest overnight opportunity is to **measure and retune speculative verification on GB10**. It can increase accepted tokens per target-weight read without changing the target model’s intended picks, and the first experiments require no rebuild.

The estimates below are hypotheses, not measured results. In particular, the reported 0.525 acceptance is an aggregate ratio; it does **not** tell us the conditional probability that a second or third draft lands.

## A. Ranked candidates

### 1. Retune the speculative window on GB10

**Mechanism.** A longer verification window may amortize the roughly 6.5 GB target-weight read over more emitted tokens. The relevant threshold is the *conditional* chance that the next draft lands versus the incremental cost of verifying its row. GB10’s lower bandwidth makes that row-cost ratio worth measuring again.

The default confidence cut is 0.7. Test cuts around it and the cost-aware `tpm` rule, seeded from measured GB10 costs. The rule’s online fit uses lone-window observations; shared windows instead pass calibrated probabilities to the lane scheduler. Those are distinct paths and should be judged separately.

**Files.** [cuda_lanes.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_lanes.zig:580), [draft_rule.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/draft_rule.zig:1).

**Expected gain.** Decode: **0–8% is a plausible test range, not a prediction**. For illustration, if a window currently emits 1.50 tokens in 30 ms, and an added row increases that to 1.65 tokens while raising time to 31 ms, throughput changes from 50 to 53.2 tokens/s, or **+6.5%**. Actual landed counts and costs are unknown. Prefill: approximately **0%**, except for first-token or mixed-load effects.

**Exactness:** **(ii), exact by target-pick contract; verify the implementation.** Compare greedy token IDs with speculation disabled and with the baseline, including boundary and long-context cases. For sampled requests, check the documented sampling contract separately; a greedy hash alone cannot establish it.

**Cost:** Environment-only, with restarts.

**One-night test and kill criterion.**

1. With `TF_GLM_DFLASH2_TRACE=1`, use interleaved `TF_GLM_DFLASH2_FORCE=1,2,3,4,5,6,7,8` and collect enough windows across several fresh prose and code prompts. Record actual offered rows, landed-by-position, verify time, draft time, and whether windows were lone or shared. Forced widths are requests, not a guarantee: held drafts and scheduler limits may reduce realized widths.
2. Fit lone verify cost from the observed widths, checking residuals rather than assuming it is linear. Measure shared rounds independently.
3. Test `TF_GLM_DRAFT_CONFIDENCE=0.4`, `0.55`, and `0.85`, plus `TF_GLM_DRAFT_RULE=tpm` with `TF_GLM_VERIFY_MS=a,b,d` taken from the fit. Do not substitute the 28 ms end-to-end decode interval for `a`: it includes work beyond the measured verify component.
4. Compare tracing-off, paired 1-, 2-, and 4-stream runs against a fresh baseline. Require at least **+2% at one intended load**, no more than **1% regression at the others**, correct token contracts, and three trials per point. Otherwise restore the default cut. Investigate a larger change in behavior before shipping.

The existing aggregate acceptance of 157/299 is useful context, but should not be used as the probability for each position in the chain.

---

### 2. Calibrate the shared lane scheduler’s GB10 cost tables

**Mechanism.** The scheduler chooses how to pack competing windows using compile-time costs. If incremental shared rows are cheaper than the tables imply, the scheduler may leave useful weight amortization unused. Conversely, the tables may be pricing a larger operation than a single decode step; the **52 ms table intercept and 28 ms observed step are not necessarily like-for-like**.

**Files.** [cuda_lanes.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_lanes.zig:24), especially the table consumers and lane-core inputs at lines 128–129.

**Expected gain.** Decode: **unknown; target +2–5% at 2–4 streams**. For scale, adding 0.05 emitted tokens per stream to a 1.5-token round at unchanged round time gives 1.55/1.50 − 1 = **3.3%**. More rows also cost time, so this is an illustrative accounting identity, not a forecast. Lone decode and cold prefill: approximately **0%**, unless scheduling affects mixed traffic.

**Exactness:** **(ii), target-pick contract plus byte/output and scheduling checks.** Packing can change numerical execution paths or expose state-order bugs even when the algorithm intends exact target picks.

**Cost:** Patch and rebuild.

**One-night test and kill criterion.** Only proceed if candidate 1 produces enough shared-round observations to identify a material discrepancy in the same quantities the scheduler prices. Fit separate lone and shared tables, including fixed cost and row slope; inspect nonlinearity and stream count. Build one table revision and run paired 2- and 4-stream decode, greedy token comparisons, and mixed prefill/decode. Kill it if the scheduler does not actually change decisions, if throughput fails the **+2% / no >1% regression** gate, or if any correctness or stability check fails. Restore the saved image and settings.

---

### 3. Increase the prefill chunk beyond 4,096 rows

**Mechanism.** If a larger chunk reuses weights over more prompt rows, it can reduce bytes read per prompt token. The prior 2,048→4,096 result is direct GB10 evidence that this mechanism can win here. Larger chunks also require more scratch, can reduce occupancy or increase compute time, and can hurt decode latency under mixed load.

**Files.** [cuda_state.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_state.zig:19), [cuda_exl3.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_exl3.zig:224), [cuda_prompt_pair.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_prompt_pair.zig:12), and the DFlash context bound changed by patches 0094/0095. Audit all dependent buffer sizes and kernel limits before building.

**Expected gain.** Prefill: **unknown; test +3–7%**, not an extrapolated promise. A +5% result would turn 1,735 into about **1,822 tok/s**, or reduce the prefill portion of a 23.5K prompt by roughly **0.65 s**. Decode: ideally **0%**, but mixed-load latency could regress.

**Exactness:** **(ii), requires a byte/contract check.** Different chunk boundaries can change execution order and cached state.

**Cost:** Patch and rebuild; potentially more than one iteration if buffer assumptions fail.

**One-night test and kill criterion.** Attempt **8,192 only if** a buffer and memory audit shows four streams, the full window, the existing KV pool, and reserve remain viable on rank 0. Test fresh 23.5K and 95K prompts, prompt lengths around chunk boundaries, token IDs, cache reuse, and concurrent decode latency. Kill on allocation pressure, reduced admission capacity, incorrect output, >1% mixed-load regression, or less than **+3%** cold prefill. Revert to the proven 4,096 image. Do not attempt this and candidate 2 in the same unreviewable patch.

---

### 4. Tune shared-expert overlap and existing execution switches on GB10

**Mechanism.** A second stream can overlap shared-expert work with the routed path, but on unified memory simultaneous readers compete for the same bandwidth. Disabling overlap could improve total throughput if contention exceeds useful overlap. Other switches may trade launches or on-chip work against memory traffic. Their direction is empirical.

**Files.** [cuda_engine.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_engine.zig:340), [cuda_forward.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_forward.zig:55), and the relevant kernel dispatches.

**Expected gain.** Decode: **unknown, likely small**; test for **≥2%**. If an overlap choice reduces a 28.3 ms interval by 0.6 ms, throughput rises by 28.3/27.7 − 1 = **2.2%**. Prefill: unknown and possibly zero.

**Exactness:** **(ii)** unless a particular switch explicitly preserves the same operations and output; check tokens and state regardless.

**Cost:** Environment-only, but each configuration needs a restart.

**One-night test and kill criterion.** Start with `TF_GLM_SHARED_SIDE=0`; it directly tests bandwidth contention. Then, time permitting, test **one switch at a time** among the existing shared-segment/write and kernel controls, selecting only switches whose actual implementation has been inspected. Use the same paired 1/4-stream tests and mixed-load gate. Kill any setting below **+2%**, any >1% regression at another load, or any output mismatch. Do not combine neutral changes in hope of an additive win.

---

### 5. Reduce target LM-head reads with `HEAD=q8`

**Mechanism.** The 0.63 GB bf16 head is read per rank per decode token. Halving its weight bytes saves roughly 0.315 GB from a 6.5 GB budget: **4.8% fewer bytes**. An ideal bandwidth-only speedup is 6.5/6.185 − 1 = **5.1%**, before quantization metadata, kernel efficiency, and non-bandwidth time. Prefill normally evaluates the target head far less often per prompt token, so its throughput gain should be small.

**Files.** [cuda_ranks.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_ranks.zig:23), [cuda_tile.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_tile.zig:96), and target head dispatch in [cuda_forward.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_forward.zig).

**Expected gain.** Decode: **0–5% idealized ceiling from this budget**, not a measured GB10 result. Prefill: **I do not know**, likely small.

**Exactness:** **(iii), numerical change.** It can change logits, greedy picks, and NLL.

**Cost:** Environment/config-only (`HEAD=q8`) and restart.

**One-night test and kill criterion.** This is a measurement candidate only: run paired per-passage NLL and the existing answer suite, plus 1/2/4-stream performance. Stop if it is slower, unstable, or produces an unacceptable paired quality result. **Restore bf16 before morning regardless of the result; user approval is required before shipping.** Do not infer quality from the aggregate NLL alone.

---

### 6. Test `DENSE=q6` on GB10

**Mechanism.** Reducing the approximately 3.8 GB q8 dense/attention weight budget by an ideal quarter saves up to **0.95 GB/token** before scales and packing overhead. The ideal bandwidth-only ratio is 6.5/5.55 = **1.17×**. Real savings and kernel cost may be materially worse. Prefill may also benefit from less weight traffic, although its compute and row reuse differ.

**Files.** [cuda_ranks.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_ranks.zig:23), [cuda_tile.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_tile.zig:96), and the corresponding packed-matmul kernels.

**Expected gain.** Decode: **unknown, bounded only loosely by the 17% idealized byte calculation**; do not budget the ceiling as an expectation. Prefill: **unknown**.

**Exactness:** **(iii), numerical change.** The q4 failure makes a quality gate especially important; it does not establish whether q6 passes.

**Cost:** Environment/config-only (`DENSE=q6`) and restart.

**One-night test and kill criterion.** Lower priority than exact changes and only if time remains. Measure paired per-passage NLL over the broader 12-domain set, the known spelling case, the greedy suite, memory allocation, and cold performance. Kill on meaningful per-passage quality regressions or insufficient performance. **Restore q8 before morning even if it looks good; shipping requires user approval.**

---

### 7. Share immutable prompt/KV pages instead of copying restored state

**Mechanism.** For repeated prefixes, alias immutable pages and use copy-on-write when a request diverges. A reused prefix then avoids both its prefill and potentially a full state-copy pass. This is a serving/admission improvement, not a cold-prefill improvement. Mapping pinned host memory alone does **not** eliminate shared DRAM traffic or make live mutable state safely shareable.

**Files.** [cuda_snapshot_pages.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_snapshot_pages.zig:1), [cuda_snapshots.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_snapshots.zig), [cuda_lanes.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_lanes.zig:373), and the KV-page ownership and KDA/DFlash state paths.

**Expected gain.** Cold decode/prefill: **0%**. Repeated-prefix TTFT: **I do not know**. The theoretical value depends on the hit rate and state-copy time; measure those before designing it. For example, avoiding 1 s of restore for 20% of requests would save 0.2 s per request on average, if nothing else changes.

**Exactness:** **(ii), byte and ownership contract.** Partial pages, KDA state, DFlash context, eviction, rank agreement, cancellation, and concurrent writers all need correct lifetime handling.

**Cost:** Multi-day. Do not implement unattended tonight.

**One-night test and kill criterion.** Inspect existing keep/restore logs and benchmark a representative repeated prefix, recording hit rate, restore milliseconds and bytes, and subsequent decode correctness. Kill the project as a performance priority if restore is a small fraction of repeated-prefix TTFT or actual reuse is rare. Keep `HOST_CACHE_GIB=0` during these tests unless there is separately verified memory headroom; pinning more memory does not create capacity on GB10.

## B. Highest-conviction pick

**Candidate 1: measure and tune speculative windows.** It is the best combination of a machine-specific reason, a target-pick contract, existing runtime controls, and a quick rollback. The highest-confidence *action* is the measurement, not a particular setting: the current data do not establish that longer windows or `tpm` will win.

I would test the fixed cuts alongside `tpm`. A fitted rule has value only if its observations and scheduler decisions correspond to the real GB10 workload, particularly at four streams.

## C. What the other experts may get wrong

- **Treat 0.525 acceptance as the chance that every deeper draft lands.** The relevant quantity falls along the chain and depends on the prompt and selector confidence. Measure landed-by-position and actual offered rows.
- **Treat 28.3 ms as the verify intercept.** It is an end-to-end observation; the trace’s verify, draft, and shared-scheduling intervals must be measured in matching units.
- **Assume a byte reduction translates one-for-one into speedup.** Quantization adds metadata and changes kernel work; a speculative row adds activation and expert traffic. The 273 GB/s figure is shared with the rest of the system.
- **Call speculation automatically bit-identical in this implementation.** The algorithm promises target picks, but changed batching and state paths still need contract tests, especially for sampling and long contexts.
- **Count unified memory twice.** Pinned host cache, mapped memory, KV, weights, and workstation processes all consume the same physical pool. A mapped state is not a free extra cache.
- **Spend the night rebuilding or pursue numerical wins as shippable.** A q8 head or q6 dense result can be informative, but neither can ship without the user’s approval.

## D. Previously dropped item most likely to flip

**The cost-aware speculative window rule** is my pick. On sm_120 it gained only 0.1% lone and 1.5% shared. On GB10, target weight reads occupy a larger fraction of round time, so an extra verification row may buy an accepted token at a smaller *relative* cost. The correct comparison is the measured incremental row cost against the position-specific probability of emitting another token. This is the same machine-dependent mechanism that makes the larger prefill chunk result credible, but it does not prove the speculative rule wins.

The **q8 LM head** is the clearest numerical candidate that might also flip: its approximately 0.315 GB/token ideal saving is material against GB10’s bandwidth budget. It remains a quality experiment, not an unattended deployment.

## Overnight order and rollback

1. **Preserve the known-good image and full launch configuration.** Establish health, KV-page count, available memory, current greedy token IDs, and fresh baseline timings. Ensure the restart procedure propagates the exported `TF_*` knobs to both ranks; the recipe’s `start.sh` explicitly handles that.
2. **Run forced-window tracing**, then confidence cuts and `tpm`, with tracing disabled for final timings. Retain a setting only if it passes candidate 1’s performance and token-contract gates.
3. **Run the shared-expert overlap switch** and, if useful, one or two other individually inspected switches. Roll back each failure immediately.
4. **Choose at most one rebuild:** scheduler tables if trace evidence is strong, otherwise the larger prefill chunk if its memory audit is clean. Stop rebuild work early enough to load and validate the saved image if needed.
5. Only if substantial time remains, **measure q8 head or q6 dense**, always restoring the approved precision afterward.
6. Leave a protected final recovery window of **at least one rebuild plus two loads**, based on the supplied times: roughly **11 minutes minimum**, with additional margin for failed startup and checks. Restore the last passing configuration, confirm `/health`, both ranks, four-stream admission, full context configuration, KV-page count, a short completion, and a fresh prefill. If any gate is inconclusive, serve the original configuration by morning.
tokens used
27,604
# Round 1

The strongest overnight opportunity is to **measure and retune speculative verification on GB10**. It can increase accepted tokens per target-weight read without changing the target model’s intended picks, and the first experiments require no rebuild.

The estimates below are hypotheses, not measured results. In particular, the reported 0.525 acceptance is an aggregate ratio; it does **not** tell us the conditional probability that a second or third draft lands.

## A. Ranked candidates

### 1. Retune the speculative window on GB10

**Mechanism.** A longer verification window may amortize the roughly 6.5 GB target-weight read over more emitted tokens. The relevant threshold is the *conditional* chance that the next draft lands versus the incremental cost of verifying its row. GB10’s lower bandwidth makes that row-cost ratio worth measuring again.

The default confidence cut is 0.7. Test cuts around it and the cost-aware `tpm` rule, seeded from measured GB10 costs. The rule’s online fit uses lone-window observations; shared windows instead pass calibrated probabilities to the lane scheduler. Those are distinct paths and should be judged separately.

**Files.** [cuda_lanes.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_lanes.zig:580), [draft_rule.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/draft_rule.zig:1).

**Expected gain.** Decode: **0–8% is a plausible test range, not a prediction**. For illustration, if a window currently emits 1.50 tokens in 30 ms, and an added row increases that to 1.65 tokens while raising time to 31 ms, throughput changes from 50 to 53.2 tokens/s, or **+6.5%**. Actual landed counts and costs are unknown. Prefill: approximately **0%**, except for first-token or mixed-load effects.

**Exactness:** **(ii), exact by target-pick contract; verify the implementation.** Compare greedy token IDs with speculation disabled and with the baseline, including boundary and long-context cases. For sampled requests, check the documented sampling contract separately; a greedy hash alone cannot establish it.

**Cost:** Environment-only, with restarts.

**One-night test and kill criterion.**

1. With `TF_GLM_DFLASH2_TRACE=1`, use interleaved `TF_GLM_DFLASH2_FORCE=1,2,3,4,5,6,7,8` and collect enough windows across several fresh prose and code prompts. Record actual offered rows, landed-by-position, verify time, draft time, and whether windows were lone or shared. Forced widths are requests, not a guarantee: held drafts and scheduler limits may reduce realized widths.
2. Fit lone verify cost from the observed widths, checking residuals rather than assuming it is linear. Measure shared rounds independently.
3. Test `TF_GLM_DRAFT_CONFIDENCE=0.4`, `0.55`, and `0.85`, plus `TF_GLM_DRAFT_RULE=tpm` with `TF_GLM_VERIFY_MS=a,b,d` taken from the fit. Do not substitute the 28 ms end-to-end decode interval for `a`: it includes work beyond the measured verify component.
4. Compare tracing-off, paired 1-, 2-, and 4-stream runs against a fresh baseline. Require at least **+2% at one intended load**, no more than **1% regression at the others**, correct token contracts, and three trials per point. Otherwise restore the default cut. Investigate a larger change in behavior before shipping.

The existing aggregate acceptance of 157/299 is useful context, but should not be used as the probability for each position in the chain.

---

### 2. Calibrate the shared lane scheduler’s GB10 cost tables

**Mechanism.** The scheduler chooses how to pack competing windows using compile-time costs. If incremental shared rows are cheaper than the tables imply, the scheduler may leave useful weight amortization unused. Conversely, the tables may be pricing a larger operation than a single decode step; the **52 ms table intercept and 28 ms observed step are not necessarily like-for-like**.

**Files.** [cuda_lanes.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_lanes.zig:24), especially the table consumers and lane-core inputs at lines 128–129.

**Expected gain.** Decode: **unknown; target +2–5% at 2–4 streams**. For scale, adding 0.05 emitted tokens per stream to a 1.5-token round at unchanged round time gives 1.55/1.50 − 1 = **3.3%**. More rows also cost time, so this is an illustrative accounting identity, not a forecast. Lone decode and cold prefill: approximately **0%**, unless scheduling affects mixed traffic.

**Exactness:** **(ii), target-pick contract plus byte/output and scheduling checks.** Packing can change numerical execution paths or expose state-order bugs even when the algorithm intends exact target picks.

**Cost:** Patch and rebuild.

**One-night test and kill criterion.** Only proceed if candidate 1 produces enough shared-round observations to identify a material discrepancy in the same quantities the scheduler prices. Fit separate lone and shared tables, including fixed cost and row slope; inspect nonlinearity and stream count. Build one table revision and run paired 2- and 4-stream decode, greedy token comparisons, and mixed prefill/decode. Kill it if the scheduler does not actually change decisions, if throughput fails the **+2% / no >1% regression** gate, or if any correctness or stability check fails. Restore the saved image and settings.

---

### 3. Increase the prefill chunk beyond 4,096 rows

**Mechanism.** If a larger chunk reuses weights over more prompt rows, it can reduce bytes read per prompt token. The prior 2,048→4,096 result is direct GB10 evidence that this mechanism can win here. Larger chunks also require more scratch, can reduce occupancy or increase compute time, and can hurt decode latency under mixed load.

**Files.** [cuda_state.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_state.zig:19), [cuda_exl3.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_exl3.zig:224), [cuda_prompt_pair.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_prompt_pair.zig:12), and the DFlash context bound changed by patches 0094/0095. Audit all dependent buffer sizes and kernel limits before building.

**Expected gain.** Prefill: **unknown; test +3–7%**, not an extrapolated promise. A +5% result would turn 1,735 into about **1,822 tok/s**, or reduce the prefill portion of a 23.5K prompt by roughly **0.65 s**. Decode: ideally **0%**, but mixed-load latency could regress.

**Exactness:** **(ii), requires a byte/contract check.** Different chunk boundaries can change execution order and cached state.

**Cost:** Patch and rebuild; potentially more than one iteration if buffer assumptions fail.

**One-night test and kill criterion.** Attempt **8,192 only if** a buffer and memory audit shows four streams, the full window, the existing KV pool, and reserve remain viable on rank 0. Test fresh 23.5K and 95K prompts, prompt lengths around chunk boundaries, token IDs, cache reuse, and concurrent decode latency. Kill on allocation pressure, reduced admission capacity, incorrect output, >1% mixed-load regression, or less than **+3%** cold prefill. Revert to the proven 4,096 image. Do not attempt this and candidate 2 in the same unreviewable patch.

---

### 4. Tune shared-expert overlap and existing execution switches on GB10

**Mechanism.** A second stream can overlap shared-expert work with the routed path, but on unified memory simultaneous readers compete for the same bandwidth. Disabling overlap could improve total throughput if contention exceeds useful overlap. Other switches may trade launches or on-chip work against memory traffic. Their direction is empirical.

**Files.** [cuda_engine.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_engine.zig:340), [cuda_forward.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_forward.zig:55), and the relevant kernel dispatches.

**Expected gain.** Decode: **unknown, likely small**; test for **≥2%**. If an overlap choice reduces a 28.3 ms interval by 0.6 ms, throughput rises by 28.3/27.7 − 1 = **2.2%**. Prefill: unknown and possibly zero.

**Exactness:** **(ii)** unless a particular switch explicitly preserves the same operations and output; check tokens and state regardless.

**Cost:** Environment-only, but each configuration needs a restart.

**One-night test and kill criterion.** Start with `TF_GLM_SHARED_SIDE=0`; it directly tests bandwidth contention. Then, time permitting, test **one switch at a time** among the existing shared-segment/write and kernel controls, selecting only switches whose actual implementation has been inspected. Use the same paired 1/4-stream tests and mixed-load gate. Kill any setting below **+2%**, any >1% regression at another load, or any output mismatch. Do not combine neutral changes in hope of an additive win.

---

### 5. Reduce target LM-head reads with `HEAD=q8`

**Mechanism.** The 0.63 GB bf16 head is read per rank per decode token. Halving its weight bytes saves roughly 0.315 GB from a 6.5 GB budget: **4.8% fewer bytes**. An ideal bandwidth-only speedup is 6.5/6.185 − 1 = **5.1%**, before quantization metadata, kernel efficiency, and non-bandwidth time. Prefill normally evaluates the target head far less often per prompt token, so its throughput gain should be small.

**Files.** [cuda_ranks.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_ranks.zig:23), [cuda_tile.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_tile.zig:96), and target head dispatch in [cuda_forward.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_forward.zig).

**Expected gain.** Decode: **0–5% idealized ceiling from this budget**, not a measured GB10 result. Prefill: **I do not know**, likely small.

**Exactness:** **(iii), numerical change.** It can change logits, greedy picks, and NLL.

**Cost:** Environment/config-only (`HEAD=q8`) and restart.

**One-night test and kill criterion.** This is a measurement candidate only: run paired per-passage NLL and the existing answer suite, plus 1/2/4-stream performance. Stop if it is slower, unstable, or produces an unacceptable paired quality result. **Restore bf16 before morning regardless of the result; user approval is required before shipping.** Do not infer quality from the aggregate NLL alone.

---

### 6. Test `DENSE=q6` on GB10

**Mechanism.** Reducing the approximately 3.8 GB q8 dense/attention weight budget by an ideal quarter saves up to **0.95 GB/token** before scales and packing overhead. The ideal bandwidth-only ratio is 6.5/5.55 = **1.17×**. Real savings and kernel cost may be materially worse. Prefill may also benefit from less weight traffic, although its compute and row reuse differ.

**Files.** [cuda_ranks.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_ranks.zig:23), [cuda_tile.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_tile.zig:96), and the corresponding packed-matmul kernels.

**Expected gain.** Decode: **unknown, bounded only loosely by the 17% idealized byte calculation**; do not budget the ceiling as an expectation. Prefill: **unknown**.

**Exactness:** **(iii), numerical change.** The q4 failure makes a quality gate especially important; it does not establish whether q6 passes.

**Cost:** Environment/config-only (`DENSE=q6`) and restart.

**One-night test and kill criterion.** Lower priority than exact changes and only if time remains. Measure paired per-passage NLL over the broader 12-domain set, the known spelling case, the greedy suite, memory allocation, and cold performance. Kill on meaningful per-passage quality regressions or insufficient performance. **Restore q8 before morning even if it looks good; shipping requires user approval.**

---

### 7. Share immutable prompt/KV pages instead of copying restored state

**Mechanism.** For repeated prefixes, alias immutable pages and use copy-on-write when a request diverges. A reused prefix then avoids both its prefill and potentially a full state-copy pass. This is a serving/admission improvement, not a cold-prefill improvement. Mapping pinned host memory alone does **not** eliminate shared DRAM traffic or make live mutable state safely shareable.

**Files.** [cuda_snapshot_pages.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_snapshot_pages.zig:1), [cuda_snapshots.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_snapshots.zig), [cuda_lanes.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_lanes.zig:373), and the KV-page ownership and KDA/DFlash state paths.

**Expected gain.** Cold decode/prefill: **0%**. Repeated-prefix TTFT: **I do not know**. The theoretical value depends on the hit rate and state-copy time; measure those before designing it. For example, avoiding 1 s of restore for 20% of requests would save 0.2 s per request on average, if nothing else changes.

**Exactness:** **(ii), byte and ownership contract.** Partial pages, KDA state, DFlash context, eviction, rank agreement, cancellation, and concurrent writers all need correct lifetime handling.

**Cost:** Multi-day. Do not implement unattended tonight.

**One-night test and kill criterion.** Inspect existing keep/restore logs and benchmark a representative repeated prefix, recording hit rate, restore milliseconds and bytes, and subsequent decode correctness. Kill the project as a performance priority if restore is a small fraction of repeated-prefix TTFT or actual reuse is rare. Keep `HOST_CACHE_GIB=0` during these tests unless there is separately verified memory headroom; pinning more memory does not create capacity on GB10.

## B. Highest-conviction pick

**Candidate 1: measure and tune speculative windows.** It is the best combination of a machine-specific reason, a target-pick contract, existing runtime controls, and a quick rollback. The highest-confidence *action* is the measurement, not a particular setting: the current data do not establish that longer windows or `tpm` will win.

I would test the fixed cuts alongside `tpm`. A fitted rule has value only if its observations and scheduler decisions correspond to the real GB10 workload, particularly at four streams.

## C. What the other experts may get wrong

- **Treat 0.525 acceptance as the chance that every deeper draft lands.** The relevant quantity falls along the chain and depends on the prompt and selector confidence. Measure landed-by-position and actual offered rows.
- **Treat 28.3 ms as the verify intercept.** It is an end-to-end observation; the trace’s verify, draft, and shared-scheduling intervals must be measured in matching units.
- **Assume a byte reduction translates one-for-one into speedup.** Quantization adds metadata and changes kernel work; a speculative row adds activation and expert traffic. The 273 GB/s figure is shared with the rest of the system.
- **Call speculation automatically bit-identical in this implementation.** The algorithm promises target picks, but changed batching and state paths still need contract tests, especially for sampling and long contexts.
- **Count unified memory twice.** Pinned host cache, mapped memory, KV, weights, and workstation processes all consume the same physical pool. A mapped state is not a free extra cache.
- **Spend the night rebuilding or pursue numerical wins as shippable.** A q8 head or q6 dense result can be informative, but neither can ship without the user’s approval.

## D. Previously dropped item most likely to flip

**The cost-aware speculative window rule** is my pick. On sm_120 it gained only 0.1% lone and 1.5% shared. On GB10, target weight reads occupy a larger fraction of round time, so an extra verification row may buy an accepted token at a smaller *relative* cost. The correct comparison is the measured incremental row cost against the position-specific probability of emitting another token. This is the same machine-dependent mechanism that makes the larger prefill chunk result credible, but it does not prove the speculative rule wins.

The **q8 LM head** is the clearest numerical candidate that might also flip: its approximately 0.315 GB/token ideal saving is material against GB10’s bandwidth budget. It remains a quality experiment, not an unattended deployment.

## Overnight order and rollback

1. **Preserve the known-good image and full launch configuration.** Establish health, KV-page count, available memory, current greedy token IDs, and fresh baseline timings. Ensure the restart procedure propagates the exported `TF_*` knobs to both ranks; the recipe’s `start.sh` explicitly handles that.
2. **Run forced-window tracing**, then confidence cuts and `tpm`, with tracing disabled for final timings. Retain a setting only if it passes candidate 1’s performance and token-contract gates.
3. **Run the shared-expert overlap switch** and, if useful, one or two other individually inspected switches. Roll back each failure immediately.
4. **Choose at most one rebuild:** scheduler tables if trace evidence is strong, otherwise the larger prefill chunk if its memory audit is clean. Stop rebuild work early enough to load and validate the saved image if needed.
5. Only if substantial time remains, **measure q8 head or q6 dense**, always restoring the approved precision afterward.
6. Leave a protected final recovery window of **at least one rebuild plus two loads**, based on the supplied times: roughly **11 minutes minimum**, with additional margin for failed startup and checks. Restore the last passing configuration, confirm `/health`, both ranks, four-stream admission, full context configuration, KV-page count, a short completion, and a fresh prefill. If any gate is inconclusive, serve the original configuration by morning.

# ===== ROUND 1: glm-r1 =====
# Round 1 — independent answer

## Framing I'll commit to

One accounting ambiguity drives everything, and I can't resolve it from the numbers given: **if the 4-stream figure (12.8 GB / 72.5 tok/s) implies one weight pass per ~55 ms round, then the 1-stream figure (6.5 GB / 28.3 ms) is consistent only if the 35.3 tok/s measurement is a *plain* decode round — or if spec-on rounds run ~55 ms with only ~118 GB/s effective during the weight phase, i.e. ~50% duty cycle lost to draft/host/comm.** Those two readings imply completely different night plans (tune speculation vs. hunt 25 ms of host overhead). So measurement zero is a phase breakdown. Everything below is priced conditional on that.

---

## A. Candidates, ranked by (gain × confidence) / effort

### 0. Phase breakdown on GB10 (prereq, not a candidate)
`TF_GLM_PROF=1` + `TF_GLM_DFLASH2_TRACE=1`, 1/2/4 streams, 3 trials each. Record: verify ms, draft ms, host-sync gaps, per-round RDMA/NCCL time, effective BW during the weight-streaming phase. ~30 min, env-only, zero risk. **Every number below is conditioned on this.** Also: quiesce desktops/services on both boxes first — on unified memory, a browser is a direct thief of the same 273 GB/s. Free 1–3%, maybe.

### 1. Re-price speculation with measured GB10 constants
- **Mechanism.** In a bandwidth-bound round, verifying k rows re-reads the *same* 6.5 GB of weights; the marginal row costs only activation traffic + expert divergence (sublinear — k tokens × top-k experts overlap on 288 experts). The sm_120 constants (`verify_a=9.4, verify_b=1.31`) and the lane tables (`52.0 ms` vs. our measured ~28 ms round) price the marginal row at ~2× the correct relative cost, so the `cut@0.7` rule offers too few rows and the scheduler under-packs lanes. Speculation is exact by construction (draft_rule.zig header), so the only gate is speed + sha sanity.
- **Files.** Runtime: `TF_GLM_VERIFY_MS`, `TF_GLM_DRAFT_RULE=tpm`, `TF_GLM_DRAFT_CONFIDENCE`, `TF_GLM_DFLASH2_FORCE` (cost sweep). Patch: `draft_rule.zig:178` defaults, `cuda_engine.zig:28` `max_drafts 3→5` once marginal-row cost is known.
- **Measurement.** `TF_GLM_DFLASH2_FORCE` at rows = 1…8 → fit GB10's actual (a, b). Set `TF_GLM_VERIFY_MS` to the fit; A/B `tpm` vs `cut` and confidence 0.7 vs 0.55/0.6 at fixed seeds; tok/s at 1/2/4 streams + sha `11ceddde19d4`.
- **Gain.** If the fit says marginal row ≤ ~1 ms on a 28 ms round: E(tokens)/round at p=0.525 rises from ~1.95 (k=3) toward ~2.1–2.3 (k=5, decaying acceptance) → **+4–10% decode**; plus whatever the lane scheduler recovers at 2/4 streams (**I do not know** that number — it could be 0 or double digits; the 52-vs-28 ms table error is a 1.9× misprice).
- **Exactness.** (i) bit-exact by construction; sha check is a tripwire, not a gate.
- **Cost.** Env-only for the rule/`VERIFY_MS`; one 7-min rebuild if we bump `max_drafts`.
- **Kill.** If the measured GB10 verify line comes out within ~15% of the sm_120 constants, the mispricing thesis is dead — revert to defaults and stop.

### 2. Fix `gb10_window_costs` / `gb10_shared_costs` in cuda_lanes.zig
- **Mechanism.** Compile-time tables (`cuda_lanes.zig:24–25`) feed lane packing at `:128–129`. A 52 ms assumed window on a 28 ms round means the scheduler believes rows are ~13% of a round when they are likely ~3–5% → it refuses windows it should take, in shared mode especially.
- **Gain.** Unknown until candidate 1's sweep lands; plausibly most of the 1→4-stream inefficiency (2.05× where shared-bytes arithmetic permits more). **I do not know the ceiling.**
- **Exactness.** (i) — scheduling only, token stream unchanged.
- **Cost.** Patch + rebuild, constants taken from candidate 1's fit.
- **Kill.** No 2/4-stream gain ≥ 3% after the patch → revert.

### 3. One-at-a-time env sweep of never-tested-on-GB10 knobs
`TF_GLM_KDA_STEP_WARPS`, `TF_GLM_SHARED_SEGS`, `TF_GLM_SHARED_WRITES`, `TF_GLM_TOPK_SUM`, `TF_GLM_DSEL`, `TF_GLM_PDL_CHAIN`, `TF_EXL3_SHFL`, `TF_GLM_HC_SPLIT/PART`. Mechanism: every one of these was tuned on a compute-roof machine; on a bandwidth-roof machine, occupancy/latency-hiding knobs (warps per KDA step) and memory-movement knobs have different optima — the 4096-row flip is the precedent. 3 trials × ~30 s per point, ~90 min for the full grid. Expected **0–3% aggregate**, class (i)/(ii) (sha-check each), env-only. Kill: any knob < 2% or sha mismatch.

### 4. q8 LM head
- **Mechanism.** Head is 0.63 GB of the 6.5 GB round = 9.7%. q8 → ~0.32 GB, saves 4.8% of bytes. On sm_120 the gain was only 1–2% because that machine wasn't byte-limited at the head; here time ≈ bytes/BW_eff, so it should realize nearly fully: 28.3 → ~27.0 ms → **35.3 → ~37 tok/s (+4.8%)**. Prefill: negligible.
- **Files.** LM-head load/quant path in `zig/src/families/glm/` + EXL3/mcq GEMV dispatch.
- **Exactness.** (iii) — logits shift, picks can flip. Tonight: patch, run paired per-passage NLL + the 10-greedy sha. **Ship only after user approval tomorrow.**
- **Kill.** Paired NLL CI includes regression, or sha changes on any of the 10 prompts without a compensating NLL win.

### 5. Prefill chunk 4096 → 8192 rows
- **Mechanism.** Same one that made 2048→4096 a +7% GB10 win: amortise the per-chunk weight read over more rows while the machine is bandwidth-starved. Diminishing returns: expect **+2–4% prefill** (1735 → ~1780–1800). Activation footprint doubles — check it fits with 4 streams live.
- **Files.** The `prefill_rows` constant behind patches 0094/0095 (one line) + `dflash MAX_CTX` if coupled.
- **Exactness.** (i) — chunking is over rows; per-row reduction order unchanged. Verify with the sha anyway.
- **Cost.** Patch + rebuild. Kill: <+1% or activation OOM at 4 streams.

### 6. Dense/scaled-dim q6 (never measured on GB10)
- **Mechanism.** Dense/attn = 3.8 GB of 6.5 = 58% of the budget. q8→q6 ≈ 0.95 GB saved = **14.6% fewer bytes → up to +8–12% decode** if the BW-bound model holds. q4 failed on quality (0.8005, spelling); q6 sits between and the failure mode was graded, not cliff-like — genuinely unknown.
- **Exactness.** (iii). Tonight: build both, paired NLL on the 12-domain harness + spelling checks, hand the user a gate report at breakfast. Do not enable.
- **Kill.** NLL delta worse than −0.01 mean or any spelling regression vs q8 baseline.

### 7. Host prompt-state: map, don't copy, on integrated memory
- **Mechanism.** With `HOST_CACHE_GIB=0`, stream state restore is a copy; on GB10 that's DRAM→DRAM over the *same* 273 GB/s the weight stream needs. If the engine can map the pinned tier instead (integrated device = same physical pages), restore bytes vanish from the critical path. Mostly affects 2/4-stream scaling.
- **Files.** Host-cache tier logic near the `HOST_CACHE_GIB` handling; `TF_GLM_HC_SPLIT/PART` env knobs first.
- **Gain.** **I do not know** — depends on how many restore-copies/round the trace in candidate 0 shows. If <1 ms/round, drop it.
- **Exactness.** (ii) — byte-identical state, needs a contract check that mapped pages are coherent with GPU writes.
- **Cost.** Patch, medium. Kill: no measurable 4-stream gain, or any coherence weirdness.

**Night order:** 0 → 1 → 3 → 2 (patch) → 5 (patch) → 4 → 6 → 7 if time. Env-only first; two rebuild windows (~00:00, ~01:00) batched; final soak + revert script from ~03:00; serving and healthy well before morning.

## B. Highest-conviction pick

**Candidate 1 (+2): re-price speculation and the lane tables from GB10 measurements.** It is the only lever that is (a) exact by construction — no user gate needed, ships tonight; (b) env-only to validate; (c) aimed at a *documented* 1.9× constant mispricing, not a hypothesis; and (d) the same measurement (verify-cost sweep) produces the constants that fix the scheduler. Worst case is a null result at the cost of one hour. Everything else is either gated on the user (4, 6) or speculative (3, 7).

## C. Where I think the other two will go wrong

- **Codex** will reach for kernel exotica — deeper cp.async rings, TMA, fused epilogues, launch-graph work. All of that was just measured neutral-or-worse, and its failure modes were sm_120 compute-roof arguments; but "re-run on GB10" is not a plan, it's a fishing trip, and several (fused rot_in+gate/up) lose for bandwidth reasons too (extra intermediate traffic). He'll also underweight that the single highest-EV change is **editing two constants**, which feels too small.
- **GLM** will propose a quantisation sweep — fp8→fp4 KV, W4A8, more aggressive expert quant — and treat the NLL gate as a post-hoc check rather than the user's hard process constraint, i.e. exactly the burn pattern from W8A8/DENSE-q4. May also burn the night on comm/RDMA tuning: the cable is 3% of TTFT and decode exchanges are small; unless my candidate-0 trace shows >2 ms/round of comm, that's a dead end.
- **Both** will likely trust the 35.3/28.3 ms framing at face value and never resolve which phase the time is actually in. If there really is ~25 ms of non-weight-stream time per round hiding in spec-on mode, that dwarfs every kernel tweak on the board.

## D. Previously-dropped item that flips on GB10

**q8 LM head** — explicitly flagged as never run here. Mechanism, in bytes: on sm_120 the head read was 0.63 GB against a 1.71 TB/s pipe with compute/launch floors, so halving it bought 1–2%. On GB10 the round *is* the byte stream: head = 9.7% of bytes, so halving it is a ~4.8% step-time cut, realized at near-full efficiency because there is no other roof to hide under. It's the cleanest instance of the general rule the 4096-row chunk already proved.

Runner-up: the **sparse drafter head**. Its 99.3% break-even coverage was priced against sm_120 drafter-head costs; on a bandwidth-bound box the full-head pass is relatively more expensive, so the break-even coverage falls. I don't believe it falls enough to close the 79–89%-vs-99.3% gap — worth one env-config data point at most, not a campaign. The **tpm draft rule** flip is candidate 1 itself, so I won't double-count it here.
