# P1c - occupancy half of the roofline, and the bitwise claim rebuilt on the dispatch map (2026-11-03, CPU-only)

## 1. Saturation, completed per the user's correction - memory half alone did not support "underfed"

Device truth from the driver (compiled query): **48 SMs**, warp 32, 100 KiB shared/SM reported, 64K regs/SM.
The fill kernel (cuda_b16.zig:70 path): 256-thread CTAs, tiles 128 rows x 64/128 cols, dynamic smem by the
glue's own formulas = 147 KB (bmm_tile) / 196 KB (bmm_tile1) class - **above the reported 100 KiB/SM, i.e.
opt-in dynamic smem and at most one CTA resident per SM** (exact residency: P3 confirms with occupancy output).

Consequences, stated as arithmetic not mechanism:
- Blocks per status-quo piece call: cdiv(1365,128) x cdiv(4096,64..128) = 11 x 32..64 = 352..704 CTAs
  = **7..15 waves over 48 singly-resident SMs.** No tail-starvation story: the m=1365 -> 4096 merge does not
  change waves-per-SM materially; both saturate the grid. The S2 (coarsening) gain, if any, is in launch
  count and piece-boundary drains, not residency - now expected SMALL, and measured as such.
- **The two-stream arm's co-residency is smem-limited by the same arithmetic**: 147+147 KiB > what one SM
  admits alongside itself at these sizes, so S1 likely interleaves CTAs of the two fills rather than running
  them truly side-by-side - overlap of drain/fill/latency phases, not 2x issue slots. This is the contention
  prior, derived, and it points at the LOW end of the pre-registered band. P3's occupancy counters confirm or
  kill it; until then the record's wording is exactly: **memory leaves headroom (15-41% of 273 GB/s); compute
  headroom unknown; the two-stream arm is the test.**
- A 28% of peak with a full grid is equally consistent with latency-bound issue at 8 warps/SM (1 CTA x 256
  threads) - which the smem-heavy design chooses deliberately for its 3-stage pipeline. Raising wave count is
  NOT available (grid already multiples of 48); the levers are concurrency or shape. Both are P3's arms.

## 2. "Bitwise-identical by construction" withdrawn as a slogan; replaced by the dispatch map (user's correction)

The codebase's own doctrine, verified at the cited lines: glm cuda_glue.zig:209 `router_ks = 8` exists *because*
"the sliced and the single pass differ in bits" - K-partitioning must be FIXED for determinism; and :223-226
dispatches router_tile vs router_part on `x % 16 == 0 and x_stride % 8 == 0` - alignment picks the kernel.
Both applied to `fillRows`'s b16 path, from source:

- `sk = splitK(n, k)` (cuda_b16.zig:16) - **functions of n and k only, not m**: the K-reduction order is
  shape-fixed, the router lesson already embodied.
- Kernel variant keys on `m >= rows_from` with `rows_from = 128` (:13), plus the alignment tuple
  `x%16, x_stride%8, n%8, out%16`. Row strides here are 8192 B (KDA saves) / 8192 B fp32 out - **every
  row-multiple offset preserves all four conditions**, so re-partitioning rows cannot flip the alignment class.
- Therefore: **any partition with every sub-call m >= 128 keeps kernel identity, sk, and alignment - same
  kernel, same per-row K order: bitwise-stable by dispatch-map, not by slogan.** All planned arms (4096 / 2048 /
  ~1365) clear 128 by 10x. Boundary condition recorded: a future partition below 128 rows a sub-call lands in
  the decode/`part` path (split-K partials, d32 variants) and MUST re-earn the check.
- Scope stated: this closes the b16/KDA arm by reading. dsa_block.fill (MLA), mlp.fill, moe.fill get the
  identical two-question test (m-threshold? alignment-keyed variant?) at design review, before any engine
  change - same standard, no inheritance.
- The empirical contract stays the arbiter either way, as a precondition read before numbers (recorded P1b).

## Status

P1 is complete: one candidate (split fillRows), one extra arm (coarsen), a derived contention prior that
lowers the expected ceiling toward the band's floor, a bitwise story with its boundary conditions, and the
pre-registered gates unchanged. Nothing further CPU-side. Next touch of the pair belongs to the decay series
(t+1h due 11:10:43Z; scheduler v2 healthy); P2/P3 after +24h per the locked sequence.

## v2 corrections (2026-11-03, ~11:15Z) - the user caught a factor two; redone from the constants

**smem, corrected** (the trailing *2 in the glue's formulas IS the bf16 byte size - I applied it twice):
bmm_tile = 3*(128+64)*64*2 = **72 KiB**; bmm_tile1 = **48 KiB (fp32-out) / 66 KiB (bf16-out)**. Not 147/196 KB.

**Opt-in mechanism, corrected**: 72 KiB clears the **48 KiB per-block default**, hence the explicit opt-in -
a per-BLOCK attribute, not per-SM capacity pressure. Per-SM (100 KiB reported): 1x72 fits; 2x72=144 does NOT.

**The dispatch resolved for the mixer out-proj (n=4096, k=8192):**
- Storage class FIRST (the fork I glossed): the checkpoint quantizes `scope: glm53_routed_experts_only`,
  `head_bits: 16` - **every non-expert projection is bf16**, so `fillRows`'s mixer arm is the **b16 path**, not
  qmmf/lane. Consequence for the record's bandwidth table: the "41% ffn combine" class was the **MoE/EXL3 family**
  (cuda_q4/mm, its own splitK) - reclassified as a different kernel family, unverified here; the b16-verified
  number is the mixer's 28% (75 GB/s) with a 15% slow-tail.
- `sk = splitK(4096, 8192)`: tiles 64, steps 128 -> the loop settles at **sk=4** (64*4=256 >= 192 stops it).
  `one = (sk==1 and n>256)` = **false** -> **bmm_tile, bn=64, 72 KiB, grid cdiv(m,128) x 64**. The comment's
  "two an SM" belongs to bmm_tile1 (sk==1), which is NOT this call. `wide=true` (fillRows kda passes it) only
  selects fp32 accumulation; `bmm_tile_smem` is not wide-indexed, so it changes nothing here.
- **Residency, corrected: 1 CTA/SM at 72 KiB - but for the right reason** (72 fits 100; 2x does not), not
  "above per-SM capacity". Waves: m=1365 -> 704 CTAs = **14.7 waves**; merged m=4096 -> 42.7 waves. The waves
  conclusion survives on correct numbers: **the merge arm stays SMALL** (grid already saturates; the lever would
  be launch/drain overhead only).
- **Two-stream co-residency, recomputed as directed**: 72+72 > 100 KiB, so two streams cannot pack 2 CTAs/SM
  either - the prior is NOT "co-residency limited to interleaving because smem exceeds the SM" (v1's wrong
  mechanism) but: **full occupancy either way, single CTA/SM per stream's waves; the only overlap available is
  phase interleaving (drain/tail/latency hiding) across the two kernels, plus memory-level parallelism.
  Prior: low-to-moderate, mechanism renamed, falsifiable by P3's occupancy counters.** The prior's DIRECTION
  survives; its v1 reasoning is withdrawn.

**Scope narrowed as directed**: the residency result above is **mixer-arm-only (b16, 45 layers' attention
blocks)**. dsa (MLA, 11 layers), mlp (3 dense - storage class unverified), moe (42 layers, EXL3/mm family) are
distinct kernel families and get the same two-question treatment before any claim. fillRows never calls b16
directly - "the fill kernel" language is corrected to "the b16 path of the mixer arm".

Sidefill scope note: the contract-precedence and series-priority rules stand as recorded in p1b; the decay
series keeps the pair; this file is CPU-only analysis and its correction too.
