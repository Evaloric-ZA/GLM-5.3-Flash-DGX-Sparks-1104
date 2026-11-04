# B's drafter stack - reference sources for the S3 port (docs/speed/s3-port.md M1)

Extracted verbatim from the serving B tree (spark1 ~/newMod/GLM-5.3-Flash-EXL3-TensorFold, branch
zig/python-parity-defaults) on 2026-11-04. Read-only reference: nothing here builds; the M1 harness vendoring
selects from these.

- b_dflash_forward_full.zig - B's drafter forward, complete (cuda_dflash_forward.zig, 401 lines): buffers,
  blockCompute/tapsCompute op order, graph capture, pinned readback, the deterministic-cublasLt selector.
- b_dflash_forward.zig - the first 100 lines, kept for the diff trail.
- b_draft_policy.zig - the chain policy (proven equivalent to A's + knobs; reference only now).
- b_dflash_weights.zig - B's drafter weight loading and mm.Q4 packing of the DFlash2 checkpoint.
- b_matmuls.zig - the 169-line glm-specific glue of cuda_matmuls.zig from 0004 (the Entries table wiring).
- 0002-zig-flashnext-cuda.patch - B's full kernel/ops patch: the Q4 struct and matmul live here
  (pub const Q4 at patch line ~101335, the frag-layout int32 [npad/64][k/32][8][32] packing; matmul at ~89606).
- 0003-zig-glm-kernels.patch - B's torch-replacement CUDA kernels (glm_sort/pack/index_copy/rotary/ring/
  rank_add/scaled_max/mass_pack...), the ops B's forward composes.
- 0004-zig-glm-engine.patch on the B tree (not copied here; 36.5k lines) has every glm_cuda file complete -
  pull pieces from spark1 as M1 needs them.

A's side for the same diff: zig/src/families/glm/dflash{,_block,_load,_select}.zig,
zig/kernels/cuda/glm_dflash.cu, zig/src/families/glm/cuda_taps.zig (the hc-mean tap), cuda_engine.zig
noteVerify call sites.
