**Drop the fusion work; keep PDL as a small candidate; defer the disruptive build to the next planned image.** Your measurements contradict the handoff’s expected gain. But I would correct two parts of your interpretation before registering the gate.

The killing measurement is **33.4 → 26.5 µs at two rows**: 6.9 µs × 34 = **0.235 ms/round**. The shared-window deltas predict 0.224–0.299 ms. At 104 ms/round, saving 0.25 ms predicts approximately **80 → 80.2 tok/s**. Reaching 82–84 at unchanged accepted tokens would require roughly **2.5–5 ms**.

Fusion has no compelling result: the two-launch version loses, and the single kernel’s 32.4 versus 32.9 µs at one row is within your reported repeatability; it loses elsewhere.

Two qualifications matter:

- **The shared benchmark permits state-cache reuse.** [run_segs](/home/zurih/c1bench/bench_kda_c1.cu:429) assigns segment states from `Ls[(li+i)%LAYERS]`. Consecutive layer-chains therefore reuse three of four state pairs. Four genuinely cold segments transfer **16 MiB**, whose bandwidth floor is approximately **61.5 µs**, rather than 15 µs. Your 50.3 µs result is consistent with cache reuse. Re-run with distinct state pairs for every layer × segment before treating the shared delta as an engine estimate. The reuse also overwrites intermediate segment `state_out` results before the [snapshots](/home/zurih/c1bench/bench_kda_c1.cu:579), weakening that particular correctness check.
- **0.3 ms is a measured expectation for this PDL change, not a ceiling for graphs.** The inferred engine chain costs 2.5 ms, whereas the two-row standalone costs 1.136 ms across 34 layers. That leaves approximately **1.36 ms unexplained**. It could include instrumentation, host submission gaps, or imperfect attribution. Graphs might recover some of it, but this benchmark neither demonstrates nor excludes that.

**A. The proposed dependency placement looks correct under your stated ordering invariant.** The distinction is that a trigger grants scheduling eligibility; the wait establishes completion and memory visibility. Triggering alone does not publish results. [CUDA PTX contract](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)

- **Step’s pre-wait state load:** safe if both possible writers belong to the ordered prefix that prep waits on. Prep waits before triggering, so step becomes eligible only after that prefix has completed. “Many launches earlier” is not the proof; the dependency path is. A replay on another CUDA stream without an explicit completion dependency would invalidate it. Check the commit-to-next-forward boundary and ping-pong pointer selection.

- **Out’s pre-wait gate load:** safe through the same reasoning. Step triggers only after waiting for prep, and prep waits for the preceding projection. Thus out becomes eligible after the gate producer has completed—even if that producer originally triggered prep early. Immutable norm weights are straightforward.

- **Waiting CTAs:** a credible performance failure mode. They consume residency resources, and step’s state preloads can compete for memory bandwidth. The rows=14 observation warrants replication, but one unchanged result does not establish this mechanism. PDL overlap is opportunistic; correctness must work with serialized execution too. [CUDA PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

- **The preceding `glm_mm`:** neither permitted file establishes its trigger location. Being launched as a dependent does not itself establish an early outgoing trigger. If it triggers early, prep’s entry wait protects correctness, but 512-thread prep CTAs may sit waiting while the matmul finishes. The useful ablation is **ordinary launch ordering into prep, retaining prep’s outgoing trigger and step/out PDL**. That isolates the projection→prep edge. The benchmark’s `REF3_PDL_SO` also removes prep’s outgoing trigger, so it does not isolate that edge.

- **Profiler events:** ordinary event records between kernels can impose completion boundaries and suppress the overlap being measured. If events only surround the entire KDA block, the internal prep→step→out edges can survive. Establish the actual placement before interpreting a failed gate. Events also add submission work, potentially changing launch starvation.

- **Ranks and streams:** request segments selected by `blockIdx.z` share a grid-level dependency; they are not separate CUDA streams. PDL supplies no cross-rank or unrelated-stream synchronization. Preserve communication completion dependencies and scratch-buffer lifetimes. Measure both ranks: the useful saving follows the critical rank and subsequent exchange waits, not the sum of rank-local savings. Persistent communication kernels or cross-stream waits can also change resource contention.

Preserve the benchmark’s wait discipline for masked segment CTAs, cover every step specialization, and retain **no explicit trigger in out**. Omitting it prevents an intentional early outgoing release; the downstream projection still needs its existing wait.

**B. A build is defensible as integration validation, but weak as a standalone performance investment.**

For it: the patch is small, reversible, and has positive standalone evidence. Integration testing can establish whether the existing PDL mechanism safely extends to recurrent state.

Against it: the expected throughput improvement is approximately 0.2 tok/s, far below the standing variation. You incur two-host disruption, while the shared-state benchmark and profiler placement still need clarification.

**Recommendation:** fix those cheap measurement issues first, then bundle PDL into the next scheduled build, initially default off. I would not stop the reference pair solely to pursue this gain.

**C. Keep the 0.15 ms target, but change the measurement and decision rules.**

**Yes, 0.15 ms is resolvable in principle.** CUDA documents event timing resolution around 0.5 µs; your threshold corresponds to approximately 4.4 µs per layer. Timer granularity is not the main obstacle. Instrumentation, drift, normalization, and shape composition are. Summing 64 calls does not remove those biases. [CUDA event timing](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-driver-api/group__CUDA__EVENT.html)

I would pre-register:

1. **One image, identical other PDL members.** Compare the existing member list against that same list plus `kda`; avoid accidentally disabling other chains. Use warmed **A–B–B–A** blocks for two runs of each configuration.

2. **Exact shapes.** Keep verify-two-rows, but separate shared `(2,2,2,2)` and `(3,2,3,2)`. An aggregate “8–10 rows” bucket can move through a changed mixture. Include an uneven larger shape for correctness and regression coverage.

3. **A scheduling-preserving primary interval.** Put one event pair outside the affected region—last input projection through `o_proj`—with no records inside it. Report milliseconds per forward, with the call-count normalization explicit. Retain the existing KDA bucket as supporting attribution.

4. **Profile-off round timing.** Measure warmed round latency and tails with `TF_GLM_PROF=0`. The unchanged projections are also a useful control: just **1.4% drift in their 10.5 ms** can imitate your entire 0.15 ms gate.

5. **Correctness beyond token hashes.** Require prose `64c908210a9e` and code `315341215112`, but also compare representative engine state and replay saves across commit and the following round. Token equality can conceal state differences that have not yet changed an emitted token.

For **GO**, require a mean saving ≥0.15 ms per forward at the registered shapes, uncertainty excluding zero, consistent paired results, and no repeatable round-latency or tail regression.

For **default off**, include inconclusive timing, inadequate precision, and regressions—not just failure of the KDA bucket threshold. A profiler that suppresses PDL cannot decide production benefit.

For **KILL**, include tensor mismatches, hangs, and dependency-related CUDA errors alongside hash changes.

The standing 77–83 tok/s band is a smoke check, not evidence of non-regression: **83 off and 77 on would both pass it despite a substantial slowdown**.

**D. The projections justify the next measurement; they do not yet establish another exact win.**

Measure the actual q8 input-projection set at decode rows 2 and shared rows 8/14, across 34 distinct layer-weight sets. Test launch/layout/vector-load changes that preserve each output’s arithmetic and reduction order, and byte-compare projection outputs.

A measured reduction from **0.308 → 0.288 ms/layer** would save **0.68 ms/round**, already exceeding PDL.

For scale, 50 MB at 273 GB/s takes approximately **0.183 ms**, leaving an idealized 4.25 ms/round between present timing and the bandwidth bound. That is headroom, not a forecast. The prompt-stage 60% bandwidth figure does not establish an attainable decode optimization. Require the decode-shaped timing and bit comparison before prioritizing a build around it.