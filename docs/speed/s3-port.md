# S3 port: B's drafter forward into A - the design, the milestones, the gate (2026-11-04, night)

Status: DESIGNED, NOT STARTED. The user approved the port ("go"); this document is its day-1 deliverable: what
follows from the night's source archaeology, in the order it must be built, with the kill points. Nothing in
patches/engine yet; no image built.

## What the night established (all source-verified, both trees read in full)

- A and B drafter structures are THE SAME: dedicated drafter forward (taps -> fc -> norm -> per-layer KV;
  block: embed -> 5 layers of conv/qkv/attention/mlp -> norm -> top16 -> selector projection), ring 2048/cap
  per stream, chain-cut policy on top. A's tap = the hc 4-stream mean at layers {5,14,24,33,42}
  (cuda_taps.zig, df_tap mode 0); B's tap consumption (fc/hidden_norm) matches. Policy: equivalent (0101's
  mimic cell proved the cut/temperature axis moves along, not above, the frontier).
- The difference is the KERNEL IMPLEMENTATIONS: B runs the python engine's own ops
  (glm_* torch-replacement kernels + mm.Q4 linears, sources in B zig/patches 0002/0003, compiled into its
  binary); A runs hand-written CUDA (zig/kernels/cuda/glm_dflash.cu, 352 lines, "after z-lab/dflash") + A's
  q4 linears. Same math on paper; different accumulation orders, tiling and precision paths - measured
  consequences: B's tokens/round +12.6% (1 stream) and +10.6% (4 streams), uniform across depths (its FIRST
  picks land more often too: 0.665 vs 0.600 offered-weighted).
- A has no Triton-AOT dflash path and B has no aot.json: both drafter stacks are compiled-in fatbins. The port
  is a SOURCE vendoring + orchestration port, not a cubin copy.

## The port, in milestones (each ends in a testable artifact; kill points marked)

M1 - the state-diff harness (1 day, de-risks everything): one binary (A's tree, a host test or a CLI verb)
that runs A's drafter forward AND B's (vendored, unmodified math) on IDENTICAL tap rows and identical weights,
and diffs every intermediate (fc out, per-layer K/V, attention out, final hidden, top16 ids, selector projection).
Deliverable: the delta table, layer by layer. KILL: if the deltas are <1e-2 everywhere, the kernels are
equivalent and the whole S3 theory dies - stop, the campaign's verdict stands, ~83 is A's ceiling.

M2 - the konvergence patch: whichever stage owns the dominant delta, port B's version of that stage alone
(likely the q4 linear's accumulation layout or the attention's chunking), behind
TF_GLM_DFLASH2_EXACT=1, default off, A's kernels untouched at 0. Gate: the M1 delta at that stage <1e-3, hashes
unchanged. KILL: no single stage owns the delta (spread thin) -> the full-forward port is the only path; go to M3
only with the user's node (its size doubles).

M3 - the full forward behind the env: B's block/ingest sequences driving A's buffers (the 401-line
cuda_dflash_forward.zig adapted to A's driver/args glue, B's mm.Q4 packer added to A's dflash_load). Paired
/metrics gate on the production pair, pre-registered and now amended with Codex's arithmetic:
  GO (default-on): paired tpr >= 2.25 AND ms/round <= 104, twice, hashes unchanged;
  GO (partial credit, keep env-opt-in): tpr >= 2.18 at <= 104 ms twice;
  else STOP, record, and the campaign's measured verdict stands.

## Facts the implementer needs (night's notes)

- B's orchestration: /tmp/tfwork/b-src/b_dflash_forward_full.zig (401 lines) and b_draft_policy.zig; its torch
  ops and qmm: B zig/patches 0003-zig-glm-kernels (glm_sort/pack/index_copy/rotary/ring/rank_add/...),
  mm.Q4 in 0002's cuda_matmuls.zig (Entry table = AOT matmul variants built by B's 0001 kernel table
  fn_qmm/qmm_group -A: A's equivalents are zig/src/cuda/q4 + mm pieces; the PACKING layout is the thing to
  match, see M1's delta table).
- A's side: zig/src/families/glm/dflash{,_block,_load,_select}.zig, zig/kernels/cuda/glm_dflash.cu, taps in
  cuda_taps.zig; the verify-side tap write is cuda_forward.zig whole()/pieced() -> taps.at (hc mean).
  A's dflash_load packs the DFlash2 checkpoint its own way (the M1 harness needs both packers fed from the
  same safetensors - A's dflash_load has the reader).
- The ring semantics differ in the corners: B ring cap = streams*2176? (config ring 2176: 2048 window + 128
  slack, one buffer, indexCopy'd); A per-slot ring 2048 exactly, ringK/ringV layout, from/first restart rules
  (measured equivalent on the serving path: lost=0/2048). M1 must run both rings from the same history.
- The selector: B cublasLt bf16(4MiB ws, deterministic-algo pick); A's projection path in dflash_block
  (lin...rank0 gather). M1 diffs the 256-projection too; a 1e-3-level difference there shifts every chance.
- Do NOT ship any of this from a serving-host build; the image flow is the standing one (spark1, pair stopped,
  save/load both Sparks, label first).

## Cost and verdict shape

M1 1 day; M2 1-2 days incl. the image cycle and the paired gate; M3 2-3 days more if M2's kill fires. Ceiling if
everything lands: B's tpr on A's latency ~89.2 total at 4 streams (Codex-checked) - parity with B, not 95; the
95 needs B's round cost too (its 102 ms vs A's 104 at equal work is within A's KDA C1 reach).
