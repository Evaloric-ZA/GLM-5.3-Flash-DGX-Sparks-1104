# P1 - dependency map from the existing trace + source (CPU-only; 2026-11-03, within-boot, no GPU touched)

Inputs: docs/speed/stages-q8.log (239,690 lines, 23 chunk-passes, rank 0, 4,096-row chunks) and
families/glm/cuda_forward.zig / cuda_kda_block.zig at the read-only mirror. No rebuild, no GPU, no restart.

## Finding 1 - the starvation is uniform, not structural by layer

Side `wait_filled` runs ~3.0 s/layer on every KDA layer (2.4 on the 3 dense, ~5.0 on the two MLA boundary
layers), 156.2 s over 23 chunks. There is no "bad layer" to fix: the side lane is idle by the same mechanism
every layer - the main pipeline simply has nothing to hand it.

## Finding 2 - the lead candidate is thinner than the proposal assumed, and bitwise-safe by construction

`Forward.fillRows` (cuda_forward.zig:347) is the `output_projection` stage: **the per-rank partial projection**
("this rank's rows of a piece" into the gathered/exchange buffer), before any cross-rank reduce. Per layer it
runs alone on the main stream between the block output and the exchange - main has no independent work in that
window; that is exactly the 3 s/layer of side idle.

Therefore: **split `fillRows`'s row range across the main and side streams.** Per-row dot products are unchanged
(a row partition, not a reduction split) => bitwise-identical outputs by construction, the strongest possible
standing under the contract. And the pattern already exists in this engine: the prompt path's side stages
(`partials`, `glued`, the wait pair) are precisely a two-stream row-split around the reduce, with the per-layer
event handshake built. The `ring` enabler SHRINKS to nothing for this candidate - no cross-layer buffering, same
layer, same operands, one extra event pair per call.

Scale: mixer+ffn output_projection = 35.1 s / 23 chunks = 1.53 s/chunk of serial main time eligible for ~2x
wall shrink => ceiling ~0.75 s of 12.14 = **+6.2% pre-contention** (inside the pre-registered >=3% band; the
whole question is whether concurrent streams actually shrink the wall - kernels may already fill the machine.
That is P3's measurement, nothing else).

## Finding 3 - the weight/trellis prefetch candidate downgrades to a footnote

KDA/EXL3 weights are consumed by fused kernels with k-step prefetch already designed into the GEMM
(exl3_block_lin2: "with or without k-step prefetch"); there is no separable host-visible weight-load phase for
a side stream to overlap. A side "warmer" would be pure L2 contention with unpriced benefit. Drops out as a
primary candidate; survives only as a P3 variant if time permits.

## What P2/P3 must now be (revised, cheaper)

- P3 (the decisive one): two-stream replay of the actual fill shapes - mixer (hc-wide) and ffn (hidden-wide)
  out-proj GEMMs at 4,096 rows, alone vs row-split across two streams with the join, CUDA-event timed,
  interleaved x3. Measures the contention truth directly. Kill gate as pre-registered (>=3% end-to-end
  equivalent; 2-3% only if trivial; <2% null).
- P2 (additivity) shrinks to a formality: delay half the fill on side and measure main's wall delta in the same
  harness as P3 - no separate engine-side run needed.
- Engine-change sketch (only if gates pass): extend the existing prompt side-handshake to carry half of
  `fillRows`'s range; event pair per call; no new buffers beyond the side stream's existing scratch. Estimated
  touched surface: one function's launch site + the existing event plumbing - the "trivial <=1 engineer-day"
  test the 2-3% band hinges on looks plausible, judged properly at design time.

## Bottom line for the user

P1 collapses the proposal's breadth to ONE candidate with the strongest possible numerics standing (row
partition = bitwise identical), existing plumbing, a +6.2% pre-contention ceiling, and a single decisive
measurement (P3) standing between here and either a one-page engine-change design or the pre-registered null.
The 12.14 s serial layer loop is otherwise confirmed untouchable without the ring - and the ring has no
candidate left to justify it.
