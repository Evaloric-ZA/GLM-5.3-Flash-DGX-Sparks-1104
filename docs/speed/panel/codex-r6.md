## 1. Router conclusion: no—the label is misleading

**Confidence: high for the attribution error; medium for its exact size on the measured workload.**

The 1,720 ms figure is **not a measurement of router computation**. In the source tree I inspected, `mark()` assigns all elapsed time **since the previous mark** to the named section. The pieced prefill loop runs mixer work, exchanges and glue before entering the FFN, but does not put the corresponding section marks around that work. The next mark is `.router`, after the actual router and selection calls. Consequently, the reported `router` interval can absorb preceding KDA/MLA, projections, communication, glue and waits as well as routing itself.

The relevant boundaries are in [cuda_prof.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_prof.zig:71), [cuda_forward.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_forward.zig:425), and [cuda_moe.zig](/home/zurih/TF/.src/tf/zig/src/families/glm/cuda_moe.zig:95). The source checkout should be matched to the binary before treating it as definitive for that run.

**The single most decisive check is already in your raw log:** inspect the separately bracketed `glm prompt stage ... stage=router` spans and their neighboring mixer/exchange spans for the exact chunks in the aggregate. The first 4,096-row chunk in [profraw-p0096.log](/home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig/docs/speed/profraw-p0096.log) has **34.64 ms summed in the explicit router spans**, against a 2,590.81 ms chunk wall time. That is one chunk, not a replacement estimate for the 21-report median; the spans can also include interference and must not all be summed as independent wall time. But it is strong direct evidence against assigning 1,720 ms to the router kernels.

The different amortisation curves do not rule out waits or misattributed work: the interval includes multiple operations with different row scaling, and a wait need only follow the *unhidden portion* of a dependency, not the routed section’s total duration.

**Next action:** Match the explicit stage spans to the exact chunks and ranks in the report, including actual row counts. Then capture an Nsight Systems timeline of the production prefill path on both ranks **without `TF_GLM_PROF`**, because the repository says that profiling disables prompt pairing. Use it to identify the critical path and cross-stream waits; use the detailed stage spans to identify candidate kernels. Validate any proposed gain with unprofiled interleaved measurements.

## 2. What “router” contains

**Confidence: high for the source boundaries; low for the largest critical-path contributor until the timeline is examined.**

There are two different things called router here:

- The **explicitly bracketed MoE router span** contains the sliced router projection, slice sum and top-k selection in the prompt path. The source selects the grouped/fused selection path only for **decode**, so `TF_GLM_TOPK_SUM=0` is not a direct prefill-router test in this implementation. The projection uses a tiled or fallback kernel and eight K slices; it is not automatically an ideal tensor-core GEMM.
- The **aggregate `.router` event interval** also inherits the unmarked preceding pieced-layer work. That is the plausible explanation for most of the 9.3 µs per row-layer—not an unusually expensive top-k operation.

Expert route-plan construction and expert execution occur **after** the `.router` mark, in the reported `routed` interval for the prompt EXL3 path. Cross-rank exchanges and piece synchronization occur in the surrounding pieced forward and can therefore enter the broad aggregate interval. I would not claim a particular histogram, permutation, or exchange dominates without following its actual kernels and dependencies in the trace.

The proposed 4 ms arithmetic roofline is at most a **hardware lower bound**, not the expected runtime of this implementation: kernel choice, precision, tiling, intermediate traffic, launch overhead and contention matter. More fundamentally, it is being compared with the wrong interval, so the 400× claim should be withdrawn.

## 3. Priority

**Confidence: medium; the order depends on whether the immediate trace review reveals a large actionable bottleneck.**

1. **(a) Investigate prefill attribution, reframed as “find the actual production critical path.”** The detailed spans already exist, making the first diagnostic cheap. A large exposed mixer, exchange or expert cost could produce a gain comfortably above the drift floor. Do not begin a router rewrite on the aggregate label.

2. **(e) Profile decode’s exposed KDA and expert paths, then pursue an exact optimization.** The decode measurements make these substantial candidates, but identify kernel time, memory traffic, synchronization and overlap before choosing one. An exact kernel or scheduling improvement is compatible with the quality constraint. The side-stream-off leg is useful evidence, but judge its end-to-end effect.

3. **(b) `DENSE=q6`, as a measurement-only experiment.** Its stated byte share makes it worth sizing, but bytes are not an end-to-end speedup: overlap, kernel behavior and the actual affected tensors decide the benefit. Check quality against the established baseline and get the user’s explicit approval before shipping any numeric change. Do not infer q6 quality from the rejected q4 result.

4. **(c) Finish the 0096 A→B→A gate, but time-box it.** Preserve the unresolved status. Interleaving controls some drift; it does not by itself resolve a sub-1% effect. Use repeated, randomized or balanced blocks, record host state, and report an uncertainty interval. If the result remains smaller than the uncertainty, leave it inconclusive rather than spending the night forcing a keep/revert decision. The estimate of ~140 restarts per arm is itself tentative given the single drift observation and its assumptions.

5. **(d) Expert-traffic / lag-1 predictor.** Keep it as a research measurement, not the next implementation project. First establish predictive hit rate, lead time, potential bytes avoided or hidden, and the cost of wrong predictions on real workloads. Preserve exact expert selection and outputs; speculative prefetch may help, while skipping or approximating experts would violate the quality constraint. Promote it only if its measured upper bound clears the drift floor by a useful margin.

## 4. Other claims I would revise

- **“Decode has no host-time term” is not established by GPU = 99.9% of wall.** Event intervals can include GPU idle time waiting for host submission, stream dependencies and contention; the wall/GPU difference is not a clean host-time decomposition. A timeline is needed. **Confidence: high.**
- **“87% of a marginal row is routed-expert bytes” is too strong.** The measured attribution is to the *routed interval*. It is consistent with expert traffic, but also contains kernel and scheduling effects. Validate bytes with hardware counters before calling it a byte attribution. **Confidence: high.**
- **KDA is approximately flat from two to three rows in this measurement.** That supports a fixed-cost hypothesis locally, not across all widths, contexts or concurrency levels. The one-row breakdown is an extrapolation, not a measured profile. **Confidence: high.**
- **The shared expert is plausibly hidden, but the byte-model discrepancy does not prove it.** The side-stream-off test is the right falsification; a slower result supports the overlap hypothesis, though interference and changed scheduling can also contribute. **Confidence: medium.**
- **No-drafter throughput is flat across your tested stream counts.** That establishes no aggregate scaling in that experiment; it does not establish that the engine never batches or that every multi-stream gain with a drafter is exclusively due to shared verify forwards. Confirm with forward-width and lane traces. **Confidence: high.**
- **Check the bucket denominators.** The last bucket includes all chunks above 1,024 rows despite its stale “to 2048” label; patch 0094 documents this. Likewise, the “to 64” bucket can contain fewer than 64 rows—the raw log includes a 24-row chunk. Unless the exact reports were independently verified to contain only 4,096- and 64-row chunks, the per-row comparison is not established. **Confidence: high on bucket behavior; unknown on the composition of your selected reports.**
- **The speculation result closes the tested deeper-window and cut-rule variants, not every conceivable speculative-decoding design.** Given the measured losses and the quality constraint, I agree they should be closed for this campaign. **Confidence: high.**
