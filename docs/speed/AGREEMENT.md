# Agreed plan: decode and prefill speed on 2x DGX Spark (GB10)

**Date:** 2026-11-03. **Panel:** the lab's agent (this agent), Codex (`gpt-6.1-terra`, read-only on the
engine tree), GLM-5.3-Flash (`http://rtx:8888/v1`, the RTX pair's own server).
**Status:** ratified. Both reviewers returned **FINAL: AGREE WITH AMENDMENTS** in round 2; the eleven
amendments are folded into the plan below. Panel record: `docs/speed/panel/`.

**Scope:** find decode and prefill speed that has *not* been tried, on a machine whose prior
optimisation record was written on different hardware (2x RTX PRO 6000, sm_120). Quality may not
regress: `W8A8` and `DENSE=q4` were both rejected on quality and are not open options.

---

## 0. The correction that reframes the night

My round-1 analysis said decode runs at **84% of the memory roof**, and concluded that decode kernel
work was finished on this machine. **That was wrong, and both reviewers independently caught the
error before any work started.** I had divided per-token bytes by the per-*token* interval; the engine
decodes in *rounds* that emit several tokens.

Measured on the live pair (one 200-token greedy reply, `tensorfold` + `speculative` telemetry):

```
tokens_per_second 36.07   seconds 5.517   prefill_seconds 0.149
rounds 99   drafted 191   accepted 100   acceptance_rate 0.5236   tokens_per_round 2.0202
```

So: **a decode round is ~54 ms** (5.368 s of decode / 99 rounds) and emits 2.02 tokens, offering ~1.9
drafts. The per-round weight-stream floor, from the checkpoint's real shape (45 layers, hidden 4096,
288 routed experts at top-8, `moe_intermediate 2048`, EXL3 4-bit on routed experts only, q8 dense,
bf16 head, TP=2), is **~24-35 GB.. ms per rank** - routed experts are 2.12 GB/rank/round exactly, the
non-expert q8 tensors are 4.4-7.2 GB/rank depending on how the KDA projections size (34 KDA layers,
64 heads x 128, is the uncertain term), plus a 0.63 GB bf16 head share. At ~273 GB/s that is a 24-35 ms
floor against a 54 ms round.

**Therefore decode runs at roughly 45-65% of the memory roof, and ~20-30 ms of every round is not
weight streaming.** That is the largest unexplained quantity in the system and nobody has ever
profiled it on GB10. It does not displace speculation as the top lever - it adds a second one.

GLM reached this conclusion from the brief's arithmetic alone, before the measurement:
*"the 1-stream figure is consistent only if ... ~50% duty cycle lost to draft/host/comm. Those two
readings imply completely different night plans."* Codex reached it independently
(*"the 52 ms table intercept and 28 ms observed step are not necessarily like-for-like"*).

## 1. What all three experts agreed on, unprompted

1. **Top lever: re-price speculative decoding for GB10.** A verify round re-reads the same target
   weights whether it verifies 1 row or 8, so the marginal verify row is far cheaper *relative to the
   round* here than on sm_120 - and every constant in the policy is sm_120's or an unverified scaling
   of it:
   - `draft_rule.zig:178`: `verify_a = 9.4 ms`, `verify_b = 1.31 ms/row`, `draft_ms = 1.37`, commented
     **"measured: sm_120 x8 400 W"**. Overridable at runtime by `TF_GLM_VERIFY_MS=a,b,d`.
   - `cuda_lanes.zig:24-25`: `gb10_window_costs = costTable(max_rows, 52.0, 6.8)`,
     `gb10_shared_costs = costTable(shared_rows, 76.0, 4.1)` - compile-time, consumed at `:128-129`.
     Both hold a row at ~13% of a round, the sm_120 *shape*.
   - The **default rule is `cut`** (`cuda_lanes.zig:592-598`), a fixed chain-confidence cut,
     `TF_GLM_DRAFT_CONFIDENCE`, **default 0.7**, with no cost awareness at all. The cost-aware `tpm`
     rule exists (Dinkelbach on `E(k) - lambda*T(k)`, online calibration, online cost fit) and was
     measured **on sm_120** at +0.1% lone / +1.5% shared and dropped. That verdict does not transfer.
   - Live evidence it is throttling: `drafted/rounds = 1.93` offered drafts a round,
     `tokens_per_round = 2.02`, against `max_held = 7` available.
2. **Second lever: the lane scheduler's shared cost table**, for the 1->4 stream scaling of only 2.05x.
3. **Measurement comes first.** The 54 ms round vs the 52 ms table intercept are not established to be
   the same quantity, and `0.5236` is an aggregate acceptance, not the conditional landing chance by
   chain position - which is what the window rule actually optimises.
4. **Numerics are measured and reported, never shipped**: `HEAD=q8` (0.63 GB of the round, ~4.8% of
   bytes, measured only +1-2% on sm_120 *because that machine was not byte-bound* - a textbook
   verdict-flip) and `DENSE=q6` (never measured on GB10). The user approves them or they die.
5. **The pair serves an approved configuration by morning**, with a tested rollback and a protected
   recovery window.

### Why this is quality-safe
Speculative decoding cannot change the reply: `draft_rule.zig` line 7 - *"Which rows are offered never
changes the tokens: every reply is the target's picks."* The whole top lever is therefore outside the
blast radius that burned `W8A8` and `q4`. It is nonetheless gated as class **(ii)**, not (i), at
Codex's insistence: the algorithm's contract is intent, and changed batching and state paths still
need a real output check.

### The output gate we can actually run
Chat `logprobs` are refused by this backend (`"logprobs are not supported by this model or backend"`),
so per-token logprob capture is unavailable. **But the engine returns
`tensorfold.token_sha` / `token_sha256` - a hash of the generated token IDs - on every reply.** That is
the token-ID contract check both reviewers asked for, and it is stronger than comparing reply text.
Every leg records `token_sha` per prompt.

## 2. Ranked candidates

| # | Candidate | Phase | Exactness | Cost | Expected |
| --- | --- | --- | --- | --- | --- |
| 1 | Re-price speculation: measure GB10's verify line, then sweep `cut` cuts and `tpm` | 0,1 | (ii) | env-only | unknown; the fit decides |
| 2 | Account for the ~20-30 ms/round of non-weight time (`TF_GLM_PROF`) | 0.2 | measurement | env-only | names the next target |
| 3 | Re-seed `draft_rule.zig` constants (+ `cuda_lanes.zig` tables if D4's test passes) | 2 | (ii) | 1 rebuild | follows from 1 |
| 4 | `HEAD=q8` | 3 | (iii) | env-only | ~5% decode ceiling, gated |
| 5 | `DENSE=q6` | 3 | (iii) | env-only | ~15% byte ceiling, gated |
| 6 | `TF_GLM_SHARED_SIDE=0` and <=3 further knobs, one at a time | tail | (ii) | env-only | 0-3% |
| 7 | `prefill_rows` 4096 -> 8192 | deferred | (ii) | 1 rebuild | 0-4%, only if 0.2 justifies |
| 8 | Map-don't-copy host prompt tier on integrated memory | not tonight | (ii) | multi-day | TTFT only |

**No point estimate is committed for #1.** My round-1 "1.5-2.5x" is withdrawn: it assumed a ~0.7
conditional landing rate per position, which `0.5236` aggregate does not support. The protocol is the
commitment, not a number.

## 3. The night plan

Mechanics: exported `TF_*` variables reach **both** ranks (rank 0 by name, rank 1 via a mode-600 env
file; `start.sh:147-180`), so an env leg is `export ... && ./start.sh restart`, ~3-4 min with load and
health. A rebuild is ~7 min plus the restart. Rollback image: `tensorfold-glm53-spark:503f271d5532`.
Harness: `docs/pf2.py`, `quality.py`, `nllpaired.py`+`cmp.py`, `boundary.py`, and new `docs/dec.py`.

**Phase 0 - measure (no rebuild).**
- **0.0** Write `docs/dec.py`: 1/2/4 concurrent streams, 3 trials, fresh content, greedy, recording
  per-trial tok/s **and per-trial spread**, plus `token_sha`, `tokens_per_round`, `acceptance_rate`,
  `rounds`, `drafted` per request. It freezes the **contract corpus** (prose, code, chunk-boundary,
  long-context) that every later leg re-runs. Dump the full live configuration - exported `TF_*`, the
  image tag, `/health`, the startup line, KV-page count - to `docs/speed/baseline_env.txt`
  **before the first restart**, and test the rollback restore *before* any experiment.
- **0.1** Baseline: `dec.py` 1/2/4, `pf2.py` 23.5K/95K, `quality.py` (sha must be `11ceddde19d4`),
  memory low-water. **Record the noise floor.**
- **0.2** `TF_GLM_PROF=1` + `TF_GLM_DFLASH2_TRACE=1`: the phase breakdown, and where the 20-30 ms/round
  of non-weight time sits. **First confirm the trace actually exposes** realized rows, landed-by-
  position, verify ms, draft ms and the lone/shared flag; any absent field is recorded as unknown and
  never inferred.
- **0.3** Forced-width sweep `TF_GLM_DFLASH2_FORCE=1,2,3,4,5,6,7,8` interleaved, trace on, several
  fresh prose and code prompts. Fit `(a,b,d)` and **check residuals instead of assuming linearity**.
  Forced widths are requests, not guarantees - record realized widths.

**Gate G0 (as amended by both reviewers):** if the fitted line lands within ~15% of the sm_120 shape,
the *constant-mispricing* thesis is dead and both the `tpm`-with-fitted-constants leg and the Phase 2
rebuild are dropped - **but the four `cut` legs still run**, because the confidence cut is judged
directly against measured throughput and trusts no constant.

**Phase 1 - env-only policy, exact, shippable.** `cut` at 0.7 / 0.55 / 0.4 / 0.0, then `tpm` with
`TF_GLM_VERIFY_MS=<fit>`. Trace **off** for timing. Baseline and candidate **interleaved on the same
prompt sets** with fresh content across trial pairs.
**Gate G1:** delta at the target load exceeds **2x the observed trial spread** *and* >= +2%; no > 1%
regression at the other loads; `token_sha` identical against both the baseline **and** drafts-disabled;
`boundary.py` clean. A borderline 1-2% result is repeated or recorded inconclusive.

**Phase 2 - at most one rebuild, conditional and attributable.** Patch `0096` re-seeds
`draft_rule.zig:178-180`. The `cuda_lanes.zig:24-25` hunk is included **only if** the measured shared
line disagrees with the table by **> 20% per coefficient, in the table's own units** (intercept and
slope judged separately). Decided at the Phase 1->2 transition, not mid-rebuild. Pre-committed
fallback if G2 fails: **revert the lanes hunk, keep the `draft_rule` re-seed**, and only if that
second rebuild finishes before the protected window - otherwise `0096` carries `draft_rule` alone.
**Gate G2:** G1's gates, plus the context fit unchanged (4,098 pages / 1,048,576 window), both ranks
agreeing, `/health` and smoke clean, and mixed prefill+decode checked.

**Phase 3 - numerics, measure only.** `HEAD=q8`, then `DENSE=q6` if time: `quality.py`, `nllpaired.py`
paired against the baseline capture, `dec.py` 1/2/4. **Restore the approved precision afterwards
regardless of the result.** Judge per passage and paired, never on an aggregate.

**Phase 4 - protected restore and verify, >= 30 min, reserved first and spent last, never borrowed.**
Serve the best configuration that passed an exact gate, else the original, restored **from the
`baseline_env.txt` record rather than from memory**. Verify `/health`, `/v1/models`, 4 streams, the
full window, 4,098 pages, a greedy smoke test, one fresh cold prefill.

**Clock checkpoints.** Phase 1 starts by **T+2:30**; on Phase 0 overrun the sweep shrinks to widths
{1,2,4,8} x two prompts (the fit needs endpoints and a midpoint more than density). Phase 3 starts
only if **T+5:00** of margin remains before the protected window, else it is dropped - a numerics
report is worth less than a verified-healthy serving state.

**Standing rules.** One change at a time. Three trials a point. A leg that fails its gate is reverted
immediately. Nothing numeric ships. The user's other containers (`compass-pg`) are left alone.
Results land in `docs/speed/RESULTS.md`.

## 4. Amendment ledger

Codex: (1) G0 must not kill the `cut` sweep; (2) verify trace coverage before relying on it; (3)
paired interleaved gates, repeat borderline results, check mixed load for scheduler changes; (4)
compare token IDs against baseline *and* drafts-disabled, sampled contract separately; (5) rebuild
conditional and attributable, not "because it is available"; (6) test rollback early, hard stop >= 30
min before the deadline.
GLM: (1) G0-fail drops only `tpm` + Phase 2, the four `cut` legs still run; (2) bind the +2% gate to
2x the measured trial spread; (3) `dec.py` freezes the token-ID contract corpus in 0.0; (4) hard clock
checkpoints with a pre-committed degraded sweep; (5) D4's threshold is > 20% per coefficient, with the
`0096` fallback pre-committed; plus snapshot the live config before the first restart.
The agent: withdrew the "84% of roof" framing and the 1.5-2.5x estimate; accepted exactness class (ii).

---

# Revision 2 (2026-11-03, after the measurement): the thesis is refuted, and the plan is re-cut

Rounds 3 and 4 of the panel, held after Phase 0 measured and after the user lifted the deadline
("You have no limited time to complete this. Do what is needed."). Full text in `panel/`.

## What the measurement did to the plan

**The unanimous #1 candidate is dead.** `b/a = 0.147` measured against sm_120's 0.139: a verify row
costs the same share of a round on both machines, because each row draws its own top-8 of 288 routed
experts and that traffic scales per row. Forcing deeper windows measured **-17% / -19%** at 1 and 2
streams. The `cut` rule at its shipped 0.7 is on a flat optimum: 0.85 is +0.2/-0.5/+1.2% (inside the
gate) and 0.40 is **-3.9/-6.3/-3.5%**. **G0 fired; the `tpm` leg and its rebuild are dropped.**
The policy question is closed in all three directions and should not be reopened without new evidence.

**One lever survived, and its mechanism was measured before any build.** The absolute prices are
wrong (lone intercept +34%, shared intercept +60%, shared slope -19%), which meets D4's pre-set
>20%-per-coefficient test, and `lanes/depth.zig:304-306` learns the scheduler's overhead as
`ms - shared_costs.get(rows)` clamped at zero. Of the 139 measured two-lane forwards, **4 could fire
that learner at the old prices against 133 at the measured ones**. Starved, not pinned. Both
reviewers authorised the one rebuild on that evidence; Codex wanted the `draft_rule.zig` re-seed kept
out of it and GLM wanted it in as a separate commit, resolved as a separate **patch file** so `0096`
stays independently revertible. Mechanism test, agreed as sufficient: if the corrected tables change
a scheduling decision, `drafted/round` at 2 and 4 streams must move; if it does not, `0096` is inert
and is reverted and reported as such.

## Corrections the panel made to my claims, for the record

- The 38.73 ms figure is one **verify forward** at one row; the 24-35 GB..ms byte estimate is for a
  **round**. Codex: these need matching boundaries before they are compared, and the profile's
  wall-minus-gpu is "a useful lead, not automatically host overhead".
- My "rows already overlap about a quarter" (5.68 measured against a ~7.8 ms full-divergence
  estimate) is confounded by L2 residency and shared-expert caching, which produce the same
  signature. GLM: motivation, never a parameter.
- `token_sha` identical on 8 prompts is empirical exactness evidence **for those runs**, not a
  general theorem. Both said so independently.
- Do not pre-announce a "decode is at its floor" conclusion and then profile to confirm it. And the
  floor world is not "10% left": the drafter pass (5.92 ms against a ~2.2 ms byte floor, 11% of a
  53.5 ms round) plus host time is plausibly 12-18% of the round on its own.
- GLM found the flaw in my own objection to expert-overlap-aware drafting: routing for position `t`
  at layer `n` needs a hidden state that does not exist yet, but the layer-`n` state for `t-1`
  **does**, so a lag-1 routing predictor is implementable and its hit rate is measurable offline from
  logged routing. Build nothing until that hit rate justifies it; do not build a token-identity
  predictor (288 experts, the per-token distribution is too flat to fit).

## The re-cut order, with no deadline (merged from both FINAL lists)

1. **Land `0096` + `0097`**, observe the learner through `drafted/round`, revert if inert. *In flight.*
2. **`TF_GLM_PROF=1` is the trunk.** It logs per-section GPU ms (`gate embed kda mla gather_a dense
   router routed shared gather_b head draw mtp commit`) **and wall vs gpu a call**, by window width.
   One profile run triages the three candidate big projects below. Attribution only, never a gate;
   decisions are read from uninstrumented runs.
3. **The drafter probe**: 5.92 ms a round against a ~2.2 ms byte floor. Both put this as the one
   honest remaining win if decode is otherwise near its floor.
4. **Expert-traffic characterisation** - the agreed larger project, and both arrived at the same first
   milestone from different directions: `gather_a` / `gather_b` section ms at 1 vs 4 rows from the
   profile. If the delta scales with union size the prize is real, and milestone 2 is a standalone
   grouped-GEMM microbench with per-row expert lists before touching the model.
5. **`HEAD=q8`** and **`DENSE=q6`**, paired per passage, restored afterwards, shipped never.
6. **`num_experts_per_tok` 8 -> 6**, measure-only, **after** the profile confirms the expert-byte term
   dominates - it is the only idea that moves the marginal row cost, the speculation ceiling and
   stream scaling with one number, but if the 5.68 ms is mostly gather launch overhead its payoff
   shrinks.
7. **`prefill_rows` 4096 -> 8192**, conditional on the profile showing per-chunk fixed cost.
8. **`TF_GLM_SHARED_SIDE=0`** because it is a structural hypothesis, not a knob. **No knob sweeps:**
   the `cut` result is the template for the whole knob surface - broad plateau, shipped value on it -
   so thoroughness means characterising a plateau once, not walking it. One representative per family.
9. **Lag-1 routing predictability**, offline from logged routing; build nothing on it yet.
10. **KDA prefill** only if the profile shows KDA above roughly 30% of prefill; **map-don't-copy
    prompt tier** only if prompt copying is a material share of a representative end-to-end workload.
11. If the floor is confirmed, the deliverable is **a bounded, evidenced ceiling analysis** with the
    drafter and host-time wins enumerated inside it - a defensible "no more here" is a real result,
    but only after the profile, never before.
