# Round 2 — convergence

## The six disagreements: ratified as proposed

- **D1** — Agree. Withdraw point estimates; the protocol (fit, then decide) is the commitment. The fit decides, not anyone's prior.
- **D2** — Agree, class (ii). Greedy **token-ID** equality against baseline on prose, code, chunk-boundary, and one long-context prompt, plus `boundary.py` clean. The 10-answer sha is a tripwire, not the contract gate.
- **D3** — Agree. One rebuild, contents chosen from Phase 0 evidence, constants only. `prefill_rows=8192` deferred unless the profile shows per-chunk fixed cost is a visible prefill share.
- **D4** — Agree. No `cuda_lanes.zig` patch on the 52-vs-28 argument alone; the trace must contradict the table in the table's own units, or the hunk dies tonight.
- **D5** — Agree. Prefill is bound by something we haven't named; `TF_GLM_PROF` names it or prefill gets nothing. Decode at 84% of roof with 35 tok/s is the user-visible problem.
- **D6** — Agree. Knob sweep demoted to a bounded tail: max four knobs, one at a time, kept only at ≥ +2% with identical token IDs, `TF_GLM_SHARED_SIDE=0` first.

The night plan is sound: correct ordering (measure → env-only exact → at most one rebuild → numerics measure-only → protected restore), correct gates, correct rollback. I ratify it with five amendments, all executable tonight.

## Amendments

**1. G0 kills the `tpm` leg and Phase 2 — not the `cut` sweep.** If the fitted line lands within ~15% of the sm_120 shape, the *constant-mispricing* thesis is dead and the `tpm`-with-fitted-constants leg plus the Phase 2 rebuild are dropped. But the four `cut` legs (0.7 / 0.55 / 0.4 / 0.0) still run: the confidence cut is a policy knob judged directly against measured throughput and trusts no constant at all, and a 15% error in the b/a ratio still moves the optimal depth under geometric acceptance decay. Cost of insurance: ~4 restarts, ~30–40 min. Cheap against a null night.

**2. Establish the noise floor in Phase 0.1 and bind G1 to it.** `dec.py` must record per-trial spread at each load in the baseline. G1's "+2%" only means something above noise: a leg passes only if its delta at the target load exceeds **2× the observed trial spread** at that load. A +2.1% delta with ±1.5% trial noise is a null, not a win. This is a one-line addition to the gate, decided from data already collected in 0.1.

**3. `dec.py` freezes the contract corpus in 0.0.** One manifest, written once: fixed prompt set (prose, code, chunk-boundary, long-context), greedy/temperature-0 decoding, fixed output length, and per-leg **token-ID capture** — not just tok/s. Every leg in Phases 1–2 runs the identical manifest and diffs token IDs against the 0.1 baseline capture. The 10-answer `quality.py` sha stays as the quality harness; the contract check lives in `dec.py` so it is never skipped under time pressure.

**4. Hard clock checkpoints; the Phase 4 window is inviolable.** Phase 1 must start by **T+2:30**; if Phase 0 overruns, the forced-width sweep shrinks to widths {1, 2, 4, 8} × two prompts — the fit needs the endpoints and one midpoint more than it needs density. Phase 3 starts only if **T+5:00** remains before the protected window; otherwise it is dropped entirely — a numerics report is worth less than a verified-healthy serving state. Phase 4's ≥ 30 min is reserved first, spent last, and never borrowed.

**5. Make D4's "materially" a number, and pre-commit the 0096 fallback.** The `cuda_lanes.zig` hunk ships only if the measured shared line disagrees with the table by **> 20% in the table's own units** (fixed cost and row slope checked separately — a 20% slope error matters more than a 20% intercept error at our widths). If both hunks go into 0096 and G2 fails, the recorded disentanglement is **revert the lanes hunk, keep the `draft_rule.zig` re-seed**, and that second rebuild may only start if it completes before the protected window — otherwise 0096 carries the `draft_rule` hunk alone from the start. Decide this at the Phase 1→2 transition, not mid-rebuild.

One addition to Phase 0.0, 30 seconds of work: dump the full live configuration — exported `TF_*` variables, the `start.sh` invocation, image tag, `/health` output, KV-page count — to `docs/speed/baseline_env.txt` before the first restart. The rollback in Phase 4 restores a configuration, not just an image, and it should restore it from a record, not from memory.

With those five: the plan is final. Measurement first, one exact env lever, at most one rebuild, numerics reported not shipped, pair healthy by morning.

FINAL: AGREE WITH AMENDMENTS

1. On G0-fail, drop only the `tpm` leg and Phase 2; the four `cut` legs still run (~30–40 min, env-only).
2. Record per-trial spread in Phase 0.1; G1's +2% gate requires the delta to exceed 2× that spread.
3. `dec.py` freezes a greedy token-ID contract corpus in 0.0 and every leg diffs against the 0.1 capture.
4. Clock checkpoints: Phase 1 by T+2:30 (shrink sweep to widths 1/2/4/8 × 2 prompts on overrun); Phase 3 only if T+5:00 remains; Phase 4's 30 min never borrowed.
5. D4's threshold is >20% per-coefficient in the table's units; 0096 fallback is lanes-hunk-out / `draft_rule`-in, decided at the Phase 1→2 transition. Plus: snapshot full live env/config to `docs/speed/baseline_env.txt` in 0.0.