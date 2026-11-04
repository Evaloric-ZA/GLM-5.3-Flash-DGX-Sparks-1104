# Round 1 — independent answer

## Framing I'll commit to

One accounting ambiguity drives everything, and I can't resolve it from the numbers given: **if the 4-stream figure (12.8 GB / 72.5 tok/s) implies one weight pass per ~55 ms round, then the 1-stream figure (6.5 GB / 28.3 ms) is consistent only if the 35.3 tok/s measurement is a *plain* decode round — or if spec-on rounds run ~55 ms with only ~118 GB/s effective during the weight phase, i.e. ~50% duty cycle lost to draft/host/comm.** Those two readings imply completely different night plans (tune speculation vs. hunt 25 ms of host overhead). So measurement zero is a phase breakdown. Everything below is priced conditional on that.

---

## A. Candidates, ranked by (gain × confidence) / effort

### 0. Phase breakdown on GB10 (prereq, not a candidate)
`TF_GLM_PROF=1` + `TF_GLM_DFLASH2_TRACE=1`, 1/2/4 streams, 3 trials each. Record: verify ms, draft ms, host-sync gaps, per-round RDMA/NCCL time, effective BW during the weight-streaming phase. ~30 min, env-only, zero risk. **Every number below is conditioned on this.** Also: quiesce desktops/services on both boxes first — on unified memory, a browser is a direct thief of the same 273 GB/s. Free 1–3%, maybe.

### 1. Re-price speculation with measured GB10 constants
- **Mechanism.** In a bandwidth-bound round, verifying k rows re-reads the *same* 6.5 GB of weights; the marginal row costs only activation traffic + expert divergence (sublinear — k tokens × top-k experts overlap on 288 experts). The sm_120 constants (`verify_a=9.4, verify_b=1.31`) and the lane tables (`52.0 ms` vs. our measured ~28 ms round) price the marginal row at ~2× the correct relative cost, so the `cut@0.7` rule offers too few rows and the scheduler under-packs lanes. Speculation is exact by construction (draft_rule.zig header), so the only gate is speed + sha sanity.
- **Files.** Runtime: `TF_GLM_VERIFY_MS`, `TF_GLM_DRAFT_RULE=tpm`, `TF_GLM_DRAFT_CONFIDENCE`, `TF_GLM_DFLASH2_FORCE` (cost sweep). Patch: `draft_rule.zig:178` defaults, `cuda_engine.zig:28` `max_drafts 3→5` once marginal-row cost is known.
- **Measurement.** `TF_GLM_DFLASH2_FORCE` at rows = 1…8 → fit GB10's actual (a, b). Set `TF_GLM_VERIFY_MS` to the fit; A/B `tpm` vs `cut` and confidence 0.7 vs 0.55/0.6 at fixed seeds; tok/s at 1/2/4 streams + sha `11ceddde19d4`.
- **Gain.** If the fit says marginal row ≤ ~1 ms on a 28 ms round: E(tokens)/round at p=0.525 rises from ~1.95 (k=3) toward ~2.1–2.3 (k=5, decaying acceptance) → **+4–10% decode**; plus whatever the lane scheduler recovers at 2/4 streams (**I do not know** that number — it could be 0 or double digits; the 52-vs-28 ms table error is a 1.9× misprice).
- **Exactness.** (i) bit-exact by construction; sha check is a tripwire, not a gate.
- **Cost.** Env-only for the rule/`VERIFY_MS`; one 7-min rebuild if we bump `max_drafts`.
- **Kill.** If the measured GB10 verify line comes out within ~15% of the sm_120 constants, the mispricing thesis is dead — revert to defaults and stop.

### 2. Fix `gb10_window_costs` / `gb10_shared_costs` in cuda_lanes.zig
- **Mechanism.** Compile-time tables (`cuda_lanes.zig:24–25`) feed lane packing at `:128–129`. A 52 ms assumed window on a 28 ms round means the scheduler believes rows are ~13% of a round when they are likely ~3–5% → it refuses windows it should take, in shared mode especially.
- **Gain.** Unknown until candidate 1's sweep lands; plausibly most of the 1→4-stream inefficiency (2.05× where shared-bytes arithmetic permits more). **I do not know the ceiling.**
- **Exactness.** (i) — scheduling only, token stream unchanged.
- **Cost.** Patch + rebuild, constants taken from candidate 1's fit.
- **Kill.** No 2/4-stream gain ≥ 3% after the patch → revert.

### 3. One-at-a-time env sweep of never-tested-on-GB10 knobs
`TF_GLM_KDA_STEP_WARPS`, `TF_GLM_SHARED_SEGS`, `TF_GLM_SHARED_WRITES`, `TF_GLM_TOPK_SUM`, `TF_GLM_DSEL`, `TF_GLM_PDL_CHAIN`, `TF_EXL3_SHFL`, `TF_GLM_HC_SPLIT/PART`. Mechanism: every one of these was tuned on a compute-roof machine; on a bandwidth-roof machine, occupancy/latency-hiding knobs (warps per KDA step) and memory-movement knobs have different optima — the 4096-row flip is the precedent. 3 trials × ~30 s per point, ~90 min for the full grid. Expected **0–3% aggregate**, class (i)/(ii) (sha-check each), env-only. Kill: any knob < 2% or sha mismatch.

### 4. q8 LM head
- **Mechanism.** Head is 0.63 GB of the 6.5 GB round = 9.7%. q8 → ~0.32 GB, saves 4.8% of bytes. On sm_120 the gain was only 1–2% because that machine wasn't byte-limited at the head; here time ≈ bytes/BW_eff, so it should realize nearly fully: 28.3 → ~27.0 ms → **35.3 → ~37 tok/s (+4.8%)**. Prefill: negligible.
- **Files.** LM-head load/quant path in `zig/src/families/glm/` + EXL3/mcq GEMV dispatch.
- **Exactness.** (iii) — logits shift, picks can flip. Tonight: patch, run paired per-passage NLL + the 10-greedy sha. **Ship only after user approval tomorrow.**
- **Kill.** Paired NLL CI includes regression, or sha changes on any of the 10 prompts without a compensating NLL win.

### 5. Prefill chunk 4096 → 8192 rows
- **Mechanism.** Same one that made 2048→4096 a +7% GB10 win: amortise the per-chunk weight read over more rows while the machine is bandwidth-starved. Diminishing returns: expect **+2–4% prefill** (1735 → ~1780–1800). Activation footprint doubles — check it fits with 4 streams live.
- **Files.** The `prefill_rows` constant behind patches 0094/0095 (one line) + `dflash MAX_CTX` if coupled.
- **Exactness.** (i) — chunking is over rows; per-row reduction order unchanged. Verify with the sha anyway.
- **Cost.** Patch + rebuild. Kill: <+1% or activation OOM at 4 streams.

### 6. Dense/scaled-dim q6 (never measured on GB10)
- **Mechanism.** Dense/attn = 3.8 GB of 6.5 = 58% of the budget. q8→q6 ≈ 0.95 GB saved = **14.6% fewer bytes → up to +8–12% decode** if the BW-bound model holds. q4 failed on quality (0.8005, spelling); q6 sits between and the failure mode was graded, not cliff-like — genuinely unknown.
- **Exactness.** (iii). Tonight: build both, paired NLL on the 12-domain harness + spelling checks, hand the user a gate report at breakfast. Do not enable.
- **Kill.** NLL delta worse than −0.01 mean or any spelling regression vs q8 baseline.

### 7. Host prompt-state: map, don't copy, on integrated memory
- **Mechanism.** With `HOST_CACHE_GIB=0`, stream state restore is a copy; on GB10 that's DRAM→DRAM over the *same* 273 GB/s the weight stream needs. If the engine can map the pinned tier instead (integrated device = same physical pages), restore bytes vanish from the critical path. Mostly affects 2/4-stream scaling.
- **Files.** Host-cache tier logic near the `HOST_CACHE_GIB` handling; `TF_GLM_HC_SPLIT/PART` env knobs first.
- **Gain.** **I do not know** — depends on how many restore-copies/round the trace in candidate 0 shows. If <1 ms/round, drop it.
- **Exactness.** (ii) — byte-identical state, needs a contract check that mapped pages are coherent with GPU writes.
- **Cost.** Patch, medium. Kill: no measurable 4-stream gain, or any coherence weirdness.

**Night order:** 0 → 1 → 3 → 2 (patch) → 5 (patch) → 4 → 6 → 7 if time. Env-only first; two rebuild windows (~00:00, ~01:00) batched; final soak + revert script from ~03:00; serving and healthy well before morning.

## B. Highest-conviction pick

**Candidate 1 (+2): re-price speculation and the lane tables from GB10 measurements.** It is the only lever that is (a) exact by construction — no user gate needed, ships tonight; (b) env-only to validate; (c) aimed at a *documented* 1.9× constant mispricing, not a hypothesis; and (d) the same measurement (verify-cost sweep) produces the constants that fix the scheduler. Worst case is a null result at the cost of one hour. Everything else is either gated on the user (4, 6) or speculative (3, 7).

## C. Where I think the other two will go wrong

- **Codex** will reach for kernel exotica — deeper cp.async rings, TMA, fused epilogues, launch-graph work. All of that was just measured neutral-or-worse, and its failure modes were sm_120 compute-roof arguments; but "re-run on GB10" is not a plan, it's a fishing trip, and several (fused rot_in+gate/up) lose for bandwidth reasons too (extra intermediate traffic). He'll also underweight that the single highest-EV change is **editing two constants**, which feels too small.
- **GLM** will propose a quantisation sweep — fp8→fp4 KV, W4A8, more aggressive expert quant — and treat the NLL gate as a post-hoc check rather than the user's hard process constraint, i.e. exactly the burn pattern from W8A8/DENSE-q4. May also burn the night on comm/RDMA tuning: the cable is 3% of TTFT and decode exchanges are small; unless my candidate-0 trace shows >2 ms/round of comm, that's a dead end.
- **Both** will likely trust the 35.3/28.3 ms framing at face value and never resolve which phase the time is actually in. If there really is ~25 ms of non-weight-stream time per round hiding in spec-on mode, that dwarfs every kernel tweak on the board.

## D. Previously-dropped item that flips on GB10

**q8 LM head** — explicitly flagged as never run here. Mechanism, in bytes: on sm_120 the head read was 0.63 GB against a 1.71 TB/s pipe with compute/launch floors, so halving it bought 1–2%. On GB10 the round *is* the byte stream: head = 9.7% of bytes, so halving it is a ~4.8% step-time cut, realized at near-full efficiency because there is no other roof to hide under. It's the cleanest instance of the general rule the 4096-row chunk already proved.

Runner-up: the **sparse drafter head**. Its 99.3% break-even coverage was priced against sm_120 drafter-head costs; on a bandwidth-bound box the full-head pass is relatively more expensive, so the break-even coverage falls. I don't believe it falls enough to close the 79–89%-vs-99.3% gap — worth one env-config data point at most, not a campaign. The **tpm draft rule** flip is candidate 1 itself, so I won't double-count it here.