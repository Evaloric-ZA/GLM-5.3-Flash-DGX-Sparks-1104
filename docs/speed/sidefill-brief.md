# Design brief: the prefill side lane is starved - what should it carry, and how?

You are one of two advisors (Codex, GLM) on the GLM-5.3-Flash / 2x DGX Spark pair (GB10, TP=2, 128 GB unified
LPDDR5X ~273 GB/s shared CPU/GPU per box). TensorFold native Zig/CUDA engine, 45 layers, 34 KDA + 11 MLA,
288-expert MoE top-8, EXL3 4 bpw weights, DFlash2 drafts at 4 bits. Prefill: 1,735 / 1,759 tok/s at 23.5K / 95K
- at parity with the older specialized build; the next prefill gain must come from structure, not tuning.

**Every number below is measured on this pair. Treat as given; do not re-derive. Answer from this brief; if you
must read source, the paths are given - Codex: at most 8 tool calls, then write the deliverable. Max ~500 words.**

## The measured structure (engine's own bracketed per-stage trace; profiled run, ~2% tax, structural claims only)

Rank 0, 4,096-row prompt chunks, 23 passes, stage sums:

- main stream: 12.14 s/chunk total. Largest: ffn/experts 2.39 + 2.62 s, mixer projections 1.37 s, mla 24.7 ms/call
  (990 calls), kda 6.69 ms/call (2,856), ffn+mixer output_projection 1.53 s, shared_expert 0.89 s, glue 0.59 s.
- side stream: 8.71 s/chunk total, of which **wait_filled 6.79 s (78%)** - the side lane is idle-starved, waiting
  on the main pipeline's per-layer outputs. Side useful work: ~1.9 s/chunk (partials 1.70+1.34 ms/call x11,340,
  glued 0.43, all_gather...).
- reverse waits (main blocked on side): 0.42 s/chunk. The side lane is NOT the critical path; the main pipeline is.
- Known knob: `TF_GLM_SHARED_SIDE=0` (side lane off) costs +1.4/+1.0/-... i.e. the side lane's CURRENT work is
  worth ~1-1.4% - so today's overlap is nearly complete. The lever is not fixing starvation; it is MOVING main
  work into the 6.8 s/chunk of side idle.

## The candidate designs (attack these, or replace them)

A. **Work relocation**: move whole main-stream stages onto the side lane's idle slots where dependencies allow -
   candidates by size: mixer projections, the shared-expert GEMM, output projections' partial reduce, gluings,
   next-layer KDA/EXL3 trellis prep, weight/trellis prefetch for layers ahead.
B. **Cross-layer double buffer**: main enqueues layer outputs to a shallow device ring (N deep); side consumes
   one-plus layers behind. Memory = N x per-layer activation slab; quantify the slab for 4,096 rows and the
   sync structure (events per slab? per layer?). Failure mode: ring fills => main blocks => serialization returns.
C. **Split differently**: give side the second half of each layer's experts (row split) instead of the current
   helper role; changes the TP split exchange pattern.
D. Your proposal.

## Hard constraints

- Numerics: pure scheduling only - same kernels, same operands, same math => results must stay bitwise identical;
  the 8-prompt token_sha contract is the gate (all currently identical; q8-head decode shipped on it).
- 128 GB/rank unified, model 83.29 GiB resident, ~20 GiB headroom, KV pool 6.1k pages; reserve 8 GiB. A ring or
  staging slabs come out of headroom - size them or say how to.
- One engineer, days not weeks. The engine is Zig+CUDA, comptime-generated kernels; `families/glm/cuda_prompt*`
  and the stream/event plumbing around `cuda_forward*.zig` are the relevant quarter (read-only reference at
  /home/zurih/TF/.src/tf - zen mode: do not re-derive the whole engine).
- Anything numeric is measure-only until the user approves; the deliverable is a design, not a patch.

## The questions, ranked

1. Which relocation set (A/B/C/D) plausibly yields the most prefill, and what is its ceiling given the 12.14 s
   critical path and what actually serializes on what? Show the dependency reasoning, not hope.
2. What is the cheapest decisive PROBE (existing env knobs, trace instrumentation, a throwaway benchmark) that
   separates "dependencies permit" from "would help", BEFORE any engine change? We have prof-run stage traces,
   a working nsys capture rig (NVTX yes, CUPTI activity broken platform-wide), ncu untested, and full container
   control. The bimodality caution: single-run timing on this box wanders; probes must be interleaved or thick.
3. Memory: worst-case slab/ring footprint for 4,096 rows at hidden 4,096, and does it fit beside the KV pool
   growth the reserve analysis cares about?
4. Kill criteria: what measured outcome at the probe stage should stop this line entirely?

Verdict format: ranked recommendation with the probe plan, the memory bill, and the kill criteria. If the honest
answer is "the ceiling is ~2% and the engineering is a month", say that - a null recommendation is a result.
