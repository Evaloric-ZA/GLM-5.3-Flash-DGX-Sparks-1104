# Brief: KDA C1 on GLM-5.3-Flash (TensorFold Zig CUDA engine, 2x DGX Spark GB10, TP2) - a measurement that contradicts the plan

You are consulted as a second opinion before a build. Answer from the numbers below; you may read `bench_kda_c1.cu` and `glm_kda.cu` in this directory (the benchmark and the shipped kernels) if you need the code - please do not spend your budget reading anything else. Be concrete: failure modes, the killing measurement, and a recommendation.

## Setting (all previously measured, greedy, token hashes identical in every run)

- Decode round: ~54 ms at 1 stream (2.1 tokens/round), ~104 ms at 4 streams (shared window, 8-14 rows). Rank-local work; two ranks run the same kernels in parallel, one rank's numbers below.
- The engine's own event profiler (TF_GLM_PROF=1; CUPTI activity is empty on this driver, nsys unusable) attributes per forward: routed experts 17-75 ms (rows-proportional), **kda 13.0 ms at 2 rows / 16.4 ms at shared 8 rows / 17.4 ms at shared 14 rows, nearly flat in rows**. The kda bucket = the KDA block's input projections (dense q8 linears, ~50 MB a layer a rank, ~0.308 ms a layer => ~10.5 ms for 34 layers) + the KDA chain (prep -> step -> out, three kernel launches a layer; at 4 streams the `_segs` forms: the same three launches covering every stream's segment, blockIdx.z = segment). So the chain is ~2.5 ms of a 1-stream round and perhaps 3-6 ms of a 4-stream round, including whatever launch gaps the engine has.
- The handoff I inherited lists "KDA C1: merge the verify chain's 3 launches into 1-2, cuda graphs; +1-2 ms/round; bit-exact" as the top remaining item, expecting 4-stream throughput 80 -> 82-84 tok/s.
- Kernel facts (source): per rank heads=32, dk=dv=128, state fp32 32x128x128 = 2 MiB a layer, read + written each forward (4 MiB; 15 us at the box's 273 GB/s). prep: grid {rows, heads} x 512 threads (conv+SiLU, decay, beta, q/k L2 norms -> global scratch). step: grid {heads, dv/4/WARPS} x WARPS*32 (WARPS 2 at 1 row, 4 at 2-8 rows, 8 above; the delta rule, serial over rows, each warp owns 4 value rows of the state). out: grid {rows, heads} x 128 (gated RMSNorm). None of the three has PDL (griddepcontrol) today; other kernels in the engine do (glm_mm matmuls, MoE router glue, hc glue, exl3 expert chain), under an env TF_GLM_PDL_CHAIN with members reduce,hc,p2p,moe (default all on).

## What I measured today (standalone, idle GB10 sm_121, driver 580.178.04, nvcc 13.0, -O3 --fmad=false, same flags as the engine's kernels)

34 layers with their own 2 MiB state_in/state_out and inputs (so no L2 residency across layers), random inputs; mean us per layer-chain over 60 reps x 34 layers; two passes agreed within ~1 us except where noted. "bits": byte-compare of out, state_out, k/v/g/beta saves against the shipped three-kernel chain, every layer.

Lone window (1 stream, chainRows):

| rows | shipped prep+step+out | + PDL on all three (same kernels + wait/trigger) | prep fused into step (2 launches) | fused + PDL | single fused kernel tf_glm_kda_chain (32 blocks x 1024) |
|---|---|---|---|---|---|
| 1 | 32.9 | 25.9 | 42.5 | 38.8 | 32.4 |
| 2 | 33.4 | 26.5 | 39.2 | 34.6 | 36.7 |
| 3 | 34.5 | 27.5 | 45.7 | 41.2 | 41.0 |
| 4 | 35.7 | 28.6 | 52.4 | 47.9 | 45.3 |
| 8 | 44.1 | 36.1 | 79.1 | 74.6 | 62.5 |
| 11 | 45.8 | 38.1 | 57.3 | 52.3 | 76.2 |
| 14 | 49.9 | 42.8 (pass 1), 49.9 (an earlier run) | 71.6 | 60.3 | 88.9 |
| 16 | 53.5 | 44.7 | 70.9 | 65.9 | 97.4 |

Shared window (4 streams, chainSegs forms, rows split across 4 segments):

| total rows | shipped | + PDL | fused (2 launches) | fused + PDL |
|---|---|---|---|---|
| 8 (2,2,2,2) | 56.9 | 50.3 | 57.7 | 52.3 |
| 10 (3,2,3,2) | 59.3 | 51.4 | 59.8 | 54.4 |
| 14 (4,3,4,3) | 64.2 | 55.4 | 68.9 | 69.2 |
| 16 (4,4,4,4) | 66.6 | 58.4 | 72.8 | 66.8 |

Every candidate is bit-identical to the shipped chain in every cell (and the segs forms to the lone chain).

PDL placement tested: prep waits+triggers at entry (nothing of its own before the wait); step loads its 4x4 state slice (state_in: written by a kernel many launches earlier), then waits, then triggers; out loads the gate row and the norm weight (the projection's output and a weight, both complete before the chain's first kernel began, by the engine's wait-before-trigger doctrine), then waits, no trigger (the o_proj matmul after it is launched as a dependent today and must not start early).

## My reading

1. "Merge prep into step" (the handoff's C1) loses: every step block (8 a head at WARPS 4) recomputes the head's prep; at 4 segments x WARPS 8 it is a wash. Dead.
2. The single fused kernel is no better than three launches at any row count. Dead.
3. The chain's floor is ~15 us (state bytes); the shipped chain is 33-66 us; PDL recovers 6-9 us a layer at every served shape (rows 1-11, segs 8-16) => **~0.2-0.3 ms a round** (34 layers), i.e. 0.3-0.5% of a 1-stream round and ~0.25% of a 4-stream round. The handoff's +1-2 ms is the chain's entire cost, not something a schedule can give back.
4. So KDA C1's honest ceiling is ~0.3 ms/round, invisible to the throughput harness (4-stream band 77-83 tok/s run to run). The only exact, measured-positive piece is "PDL on the three chain kernels", a pure scheduling change under the existing TF_GLM_PDL_CHAIN knob (new member `kda`).

## Questions

A. Failure modes of PDL on this chain inside the engine that the standalone bench cannot show. Specifically: (i) the step kernel's pre-wait load of state_in - is there any way state_in could still be in flight (writers: the previous round's step into the other state copy, or the replay kernel at commit, both many launches earlier; the engine's rule is every dependent waits before it triggers, so everything before the primary is complete)? (ii) the out kernel's pre-wait load of the gate rows written by the projection matmul two launches back; (iii) waiting CTAs squatting SM slots and slowing the primary (the rows=14 lone cell once showed no gain), (iv) the primary before prep is a glm_mm decode matmul launched as a dependent itself (it waits before writing; does it trigger early? if it does, prep waits at entry so correctness holds, but is there a perf trap?), (v) TF_GLM_PROF event records between launches cutting the programmatic edge, (vi) anything with the 2-rank exchange or streams I have not thought of.

B. Is a build justified for ~0.2-0.3 ms a round? The cost: stop the reference pair on the build host for the image build, ship to both Sparks, restart the pair under test twice (A/B). The gain is exact and small. Argue both ways briefly, then recommend.

C. The gate I intend to pre-register (critique it): same image, TF_GLM_PDL_CHAIN with and without `kda`, each twice: (1) token hashes unchanged (prose 64c908210a9e, code 315341215112), (2) TF_GLM_PROF=1 kda section at "verify 2 rows" and "shared 8-10 rows" lower by >= 0.15 ms with kda on, (3) 1-stream and 4-stream tok/s not worse than the standing band. GO: all three twice -> default on. Ship-off: (1) holds but (2) fails -> keep the patch with the member default off, record. KILL: (1) fails -> the patch is wrong, do not ship. Is 0.15 ms resolvable by an event profiler that sums 64 calls? What would you measure instead?

D. Given the attribution (projections 10.5 ms of the 13 ms kda bucket, at ~60% of bandwidth per the prompt-stage trace), is there anything exact in the KDA block worth more than this? Answer only if you can name a measurement, not a hope.
