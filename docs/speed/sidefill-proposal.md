# Side-fill design proposal (post-advisor synthesis) - 2026-11-03

Inputs: `sidefill-brief.md` (the facts both were given), `sidefill-codex.md`, `sidefill-glm.md`. No code written.
Status: AWAITING USER APPROVAL for the probe stage only (throwaway instruments; no engine change).

## Where the advisors agree (and where I concur)

1. **The 6.79 s/chunk of side idle is not a speedup budget.** It is per-layer gap waiting (23 passes x 45 layers
   of small slack). The lever is relocating *main-path* work into side slots - and only work with a real
   independent interval between input-ready and consumer.
2. **Per-layer stage relocation is nearly worthless** (GLM's sharpening, accepted): the layer loop is strictly
   serial; side cannot start layer n+1's projections before layer n's mixer output. The candidates that survive
   are the **chunk-level independent stages**: next-layer KDA/EXL3 trellis prep and weight prefetch (pure
   weight-side reads, no activation dependency), and the **output-projection partial GEMMs** (per-rank work
   that needs only the expert output, not the cross-rank reduce).
3. **The ring (B) is an enabler, not a win** - a 2-deep slab ring is what lets side work one layer behind main.
   Memory: 32 MiB/slab fp16 at 4,096 rows; even 4-deep with staging ~192 MiB - trivial against headroom; the
   binding resources are sync granularity (per-layer events, never per-slab) and KV-pool growth, both sized.
4. **Expert row-split (C): skip** - changes reduction order, risks the bitwise contract, days-scale, no clear gain.
5. **Contract gate on everything**: 8-prompt token_sha identical; any mismatch in a throwaway harness stops the line.
6. **Kill criteria, pre-agreed** (both advisors, adopted verbatim in substance): probe shows <3% prefill delta for
   a ~2 s/chunk candidate => dependencies bind, stop; no additivity with `TF_GLM_SHARED_SIDE=0` => side cannot
   absorb more, stop; engineering estimate >~4 engineer-days in the stream/event plumbing => ceiling/gain fails,
   record as null with the trace attached.

## Where they disagree - the honest ceiling

- **Codex**: bound per branch; shared-expert branch (0.89 s of 12.14 s) = at most ~7.9% throughput optimistically,
  other branches only where the trace shows independent intervals; concurrent-kernel bandwidth contention unresolved;
  "if the best repeatable bound is ~2%, prefer null".
- **GLM**: relocatable budget ~1.5-2.5 s of the 12.14 s path (prefetch + partial-reduce classes) => **+8-12% likely,
  +16% best**, with the explicit caveat that 0.42 s of reverse-wait says zero-sync absorption is optimistic; and the
  null-honest fallback: if gaps prove dependency-shaped, ceiling collapses to ~2-4% for a month of work - kill it.

My synthesis: GLM's number is the ceiling *if* both chunk-level classes absorb fully at zero sync cost; Codex's is
the per-branch discipline that keeps us honest. The range 2-12% is exactly what the probes exist to collapse. The
2.62 s expert GEMM shape (GLM's parting note) is the parallel workstream if this dies.

## The probe stage - three instruments, cheapest first, each with its kill gate

- **P1, zero code, half a day**: dependency map from the existing 239,690-line stage trace + targeted NVTX
  re-runs: for each candidate stage, input-ready event vs first consumer, per layer, and the distribution of
  true independent intervals. Kill: no candidate with a material independent interval => null, documented.
- **P2, tiny, one day**: injected-delay additivity - delay the candidate ~200 ms/chunk on side in a throwaway
  harness; interleaved x3 (the bimodality rule). Delta(main) vs delta(SIDE=0) additivity tells whether the side
  lane can absorb at all. Kill: <3% response => stop.
- **P3, one day**: two-stream replay microbenchmark - the *same kernels and operands* of the named stages,
  sequential vs concurrent, CUDA-event timed, join included, thick interleaved reps. Measures opportunity
  separately from contention without touching the engine (Codex's instrument; device timings, not NVTX, because
  CUPTI activity is broken platform-wide). Kill: no repeatable joined-makespan win beyond run variation.

Only if P1-P3 clear: a scheduling experiment in `cuda_forward*.zig`'s stream/event plumbing, engine A/B with the
contract gate, interleaved, per the standing measurement rules - separately proposed, separately approved.

## Decision requested

Approve P1-P3 (measure-only, throwaway, no engine change)? On approval I run them and return with either a
one-page engine-change design + measured ceiling, or the null result with the trace that proves it.
