# Verdict

## 1. Ranked recommendation

**Rank 1 — A-lite: relocate the two chunk-level independent stages, not per-layer stages.**

Dependency reality from the trace: 6.79 s/chunk of side idle is *per-layer gap waiting* — it recurs 23×45 times and only ever widths ~10–50 µs of stream slack per gap. Per-layer A (mixer proj, shared expert) buys almost nothing because main's layer sequence is strictly serial; the side lane can't start layer n+1's projections before layer n's mixer output exists.

The stages that **don't chain on the layer loop**:

- **next-layer KDA/EXL3 trellis prep + weight/trellis prefetch** (A's last two items): pure weight-side reads, no activation dependency. At 4 bpw, per-layer weight+trellis traffic for 4,096 rows is small but it's currently *inside* the 12.14 s main path as issued reads.
- **output-projection partial reduce** (fp16 partials, 1.53 s class): the partial GEMMs are per-rank; only the final all-reduce needs layer n's operands — partials can be computed as soon as the expert output lands, on side.

**Realistic relocated budget: ~1.5–2.5 s of the 12.14 s path**, but the *ceiling on prefill throughput is bounded by what remains serial*: layer-to-layer dependency (attn→mla→experts gates) can't overlap with itself. Best case, prefill → ~12.14/(12.14−2.0) ≈ **+16% best, +8–12% likely**; the honest arithmetic is: gain ≤ (relocated seconds)/(12.14 − relocated) only if side truly absorbs them with zero added sync, which the 0.42 s reverse-wait says is optimistic.

**Rank 2 — B (ring) is a prerequisite of A-lite, not a separate win**: a 2-deep ring just lets side start layer n−1's relocatable tail while main runs layer n's head. Treat B's memory as A-lite's cost.

**Rank 3 — C**: row-splitting experts changes the reduction pattern → numerics risk for zero clear gain. Skip.

## 2. The cheapest decisive probe (before any engine change)

**Trace-shift probe, one day:** in the existing profiled run, insert NVTX ranges around the two candidate stages and *manually delay* them (cuda event wait, ~200 ms injected per chunk) on the side stream. If prefill doesn't move ≥ the injected delay's overlap fraction, dependencies — not starvation — are the binding constraint. Interleave 3 runs each way (bimodality).

Second probe, zero code: re-run with `TF_GLM_SHARED_SIDE=0` **plus** the injected stage delay. If delta(main) − delta(side-off) ≈ 0, the stage was never actually overlappable.

## 3. Memory bill

Per-layer activation slab, 4,096 rows × hidden 4,096 × fp16 = **32 MiB/rank**. 2-deep ring = 64 MiB; add event pairs (2/slab, ~KB). Even a generous 4-deep ring with double-staged partial buffers ≈ **192 MiB**. Against 20 GiB headroom minus 8 GiB reserve: **fits trivially; the binding constraint is KV pool growth, not the ring.** Ring is not the risk; sync granularity is (per-layer events, not per-slab, or side stalls a full layer).

## 4. Kill criteria (probe stage)

- Injected-delay probe shows <3% prefill delta for a 2 s/chunk relocation candidate → dependencies bind; stop.
- `SIDE=0` + delay shows no additivity → side lane can't actually absorb more; stop.
- Any token_sha mismatch in the throwaway scheduling harness → numerics contract at risk beyond "pure scheduling"; stop and re-scope.
- Engineering estimate for A-lite exceeds ~4 engineer-days in `cuda_forward*.zig` stream/event plumbing → ceiling/gain ratio fails; null-result the line with the trace attached.

**Null-honest note:** if the probe shows per-layer gaps are dependency-shaped (likely), the ceiling collapses to ~2–4% and a month of work — kill it and spend the week on the 2.62 s expert GEMM shape instead.