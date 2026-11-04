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
