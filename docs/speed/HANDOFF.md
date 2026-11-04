# Handoff: GLM-5.3-Flash decode/prefill speed campaign

**Working folder:** `/home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig`
**Engine source (read-only reference, not built from here):** `/home/zurih/TF/.src/tf`
**Full record:** `docs/speed/RESULTS.md` (554 lines, chronological, includes retractions)
**Ratified plan:** `docs/speed/AGREEMENT.md` -- **sections of this are now superseded; RESULTS.md wins.**

The hardware is 2x NVIDIA DGX Spark (GB10), TP=2, 128 GB unified LPDDR5X a box at ~273 GB/s shared by CPU and GPU,
cabled on one QSFP port. spark3 = rank 0 (the API, at `127.0.0.1:8888`), spark4 = rank 1 (`zurih@10.0.34.4`).
45 layers, hidden 4096, 34 KDA + 11 full-attn MLA layers, 288 routed experts top-8, `first_k_dense_replace` 3,
vocab 154,880. DFlash2 drafter at 4 bits. Engine `tensorfold-native 1.1.9` for `sm_121`, upstream `f350cd3e` + 95
patches (0001-0093 carried from the RTX PRO 6000 recipe, 0094-0095 this pair's own) + two candidate patches below.

---

## 1. HARD CONSTRAINTS -- do not relax these

- **Quality must not regress.** `W8A8` and `DENSE=q4` were both rejected by the user on quality and are **closed**.
  Do not re-open either without an eval that clears it.
- **Anything numeric is measure-only and needs the user's explicit approval before shipping.** This includes
  `HEAD=q8` (see section 4), which is currently the leading candidate and is *not* approved.
- **Judge quality per passage and paired, never on an aggregate NLL.** `W8A8`'s aggregate matched baseline exactly
  while every passage had moved. `DENSE=q6` reproduced the trap tonight (section 4).
- **Banned sources** (the lab's standing rule, still in force): never copy, port, adapt or rewrite code from
  github.com/tonyd2wild, github.com/r0b0tlab, github.com/vcruz305. If a task needs it, stop and say the source is
  banned; write from scratch instead.
- **`token_sha` is the output contract.** `docs/dec.py <label> --contract` runs 8 frozen prompts
  (`prose code spelling math zig json boundary longctx`) and prints a token-ID-level sha a prompt. Baseline:
  `87fa94aff290 2e49486f633d 39fcf5dc3ba3 690b2a33a2d6 60b55b74b419 29a950f3434c fbe213fd977e fbe213fd977e`.
  Chat `logprobs` is unsupported by this backend; NLL goes through `/v1/completions` `prompt_logprobs: 0`.

---

## 2. LIVE STATE AT HANDOFF (2026-11-03 07:30Z)

**A job is running.** `docs/speed/chain2.sh` (pid 49715) -> `docs/speed/abba.sh` (pid 777368), currently in coverage
pass `b1`, started 07:27:57Z. Expected finish **~07:56Z**.

> **CRITICAL:** `abba.sh` switches arms by **moving** `0096`/`0097` between `patches/engine/` and
> `docs/speed/held/`. Right now they are in **`docs/speed/held/`** and the server runs the 93-patch baseline image
> `503f271d5532`. Pass `a2` moves them back and restarts on `eeb98fe202ff`.
> **If you kill the job, you must restore them yourself:**
> `mv docs/speed/held/009[67]-*.patch patches/engine/ && bash docs/safestart.sh`
> Otherwise the repo looks like the two patches were deleted, and the next start builds/selects the wrong image.

Images on disk: `eeb98fe202ff` (95 patches + 0096 + 0097) and `503f271d5532` (95 patches, the rollback).
Switching arms needs **no rebuild** -- moving the patch files changes `PATCHES_HASH`, which selects the already-built
image by label.

Git: branch `master`, clean history, but everything in `docs/speed/`, the new `docs/*.py`/`*.sh` harness, and
`patches/engine/0096`+`0097` are **untracked**. Nothing has been committed for this campaign. `README.md` **has
uncommitted edits** (section 6).

---

## 3. THE HARNESS (all written for this campaign; use it, do not rewrite it)

| file | what it does |
| --- | --- |
| `docs/safestart.sh` | **Always restart through this, never `./start.sh restart` directly.** Verified stop (polls `docker inspect` on both ranks until the container is provably absent, refuses to start otherwise), then restart, then waits for `/health`. Exit 4 = refused, 5 = not healthy in 10 min. Tested: a restart is ~2m13s. |
| `docs/leg.sh <label> [VAR=val ...]` | One measured leg: export env -> `safestart` -> decode sweep -> contract corpus -> `token_sha` diff vs baseline. |
| `docs/dec.py <label> [--contract]` | Decode sweep on **fixed** prompts. `LOADS=1,2,4 TRIALS=n`, `CTXWORDS=n` prepends seeded filler for long-context. Reports tok/s, round ms p50/p95/max, tokens_per_round, acceptance, drafted/round, `token_sha`, mean+-se, spread, and a tokens/rounds identity check. Writes `docs/speed/dec-<label>.json`. |
| `docs/speed/coverage.sh <label>` | One coverage pass against whatever is serving (does **not** restart): 10-trial 4-stream, long-context at 1 and 4 streams, mixed prefill+decode, contract corpus. |
| `docs/speed/abba.sh` | A -> B -> A over the 0096 image and the baseline image. See the patch-moving warning above. |
| `docs/pf2.py` | Cold prefill, **fresh random content a trial** so no prefix-cache hit. Always check `cached_tokens == 0`. |
| `docs/quality.py` | 4-passage NLL + sha over 10 greedy replies. Reference `answers_sha` = `11ceddde19d4`. |
| `docs/nllpaired.py <label>` + `docs/cmp.py <a> <b>` | Paired **per-token** NLL over 1,051 tokens of 12 domains, and the paired comparison with a 95% CI. This is the instrument that must decide any numerics question. |
| `docs/prof.py`, `docs/profleg.sh` | `TF_GLM_PROF=1` parser. **Read section 5 before trusting its output.** |
| `docs/trace.py`, `docs/optwin.py` | Fits the verify line from `dft` trace records (lone and shared forwards separately); replicates `draft_rule.zig`'s `offer()` and tabulates implied throughput by window width. |
| `docs/mixed.py` | Mixed prefill+decode load. |

Measurement protocol that actually works here: **fixed prompts, 3+ trials, and interleaved A/B** (section 5).
Random-nonce prompts gave 12.16% spread at 1 stream (content luck in `tokens_per_round`) and are useless.

---

## 4. WHAT IS ESTABLISHED (with evidence) -- do not re-litigate these

**The leading candidate, needs user approval + A->B->A confirmation:**

- **`HEAD=q8`: +3.1% / +0.9% / +1.2%** at 1/2/4 streams (38.9 / 54.3 / 70.6 tok/s), round @1 52.64 -> 51.50 ms.
  Trial spreads 1.40 / 0.19 / 0.01%. **All 8 contract hashes identical**, **10 greedy answers identical**
  (`11ceddde19d4`), paired per-token NLL delta **-0.00050, 95% CI -0.00156..+0.00056**, `quality.py` NLL 0.752643 vs
  0.752338. Measured 06:54:50-06:59:18Z, which is **clean** (before the contamination window in section 5).
  The head is 4.8% of the 6.58 GB/rank forward; halving it predicts -2.4% and -2.2% was measured -- the only
  byte-budget prediction confirmed end-to-end. The 1-vs-4-stream asymmetry is the expected signature of a **fixed
  per-forward** cost (like KDA), not a per-row one.
  **Next step: A->B->A it, then ask the user.** +3.1% is the only effect in this campaign above the drift band.

**Closed, measured, do not retry:**

- **Deeper speculation is dead.** Verify cost is `a + b*(rows-1)` with a=38.73, b=5.68 ms, so `b/a = 0.147` -- worse
  than sm_120's 0.139. Forced deep windows measured **-17% / -19%**. The shipped `cut` rule at 0.70 is bracketed in
  all three directions (0.40 -> -3.9/-6.3/-3.5%; 0.85 -> flat). `optwin.py` puts the optimum at k=2 against a shipped
  k=1.83. Mechanism: each row draws its own top-8 of 288 experts, so routed-expert bytes scale per row.
- **`DENSE=q6` fails quality.** 10 greedy answers changed (`81de7e0a307d`), **3 of 8** contract hashes changed
  (prose, math, zig -- the zig answer went 48 -> 120 tokens with a different function name), `quality.py` NLL
  **+0.0187**. Its paired *aggregate* NLL looked marginally better, which is the W8A8 trap. q4 was +0.053, so q6 does
  ~35% of q4's damage: same failure mode, smaller. Its throughput was contaminated and is not worth re-measuring.
- **Patch 0096 is unresolvable, not keep and not revert.** It corrects genuinely mis-priced GB10 cost tables (the
  shared table was so over-priced that `depth.zig`'s overhead learner could fire on only 4 of 139 forwards vs 133/139
  at measured prices). The mechanism fires as predicted (`drafted/round` moves at 2 and 4 streams, bit-identical at 1),
  but throughput is +0.20/+0.26/+0.65% -- **inside the drift band**. `0097` re-seeds `draft_rule.zig`'s cost seed and
  is inert under the default `cut` rule. Pre-committed gate: `docs/speed/PRECOMMIT-0096.md`.
- **Host-state drift is ~1.5% and dominates small effects.** The *same* config measured **+1.52/+1.78/+1.49%** faster
  after spark4 was power-cycled, while within-run trial spreads were 0.14-0.69% and tokens bit-identical. So
  **only interleaved A->B->A is valid at the 1-2% level.** Rough sample sizes against sigma~1.5%: 0.5% effect needs
  ~140 restarts an arm, 2% needs ~9, 5% needs ~2. **sigma rests on one observation** -- the ABA now running is as much
  a measurement of sigma as a gate on 0096. **Stop chasing sub-1% patches.**
  Untested free lever: if long uptime costs ~1.5% (spark4 had been up 4 days), a scheduled restart beats every patch
  tried. Clean test = cycle spark3 (never restarted) and re-measure. **Do not reboot spark3 without asking the user.**
- **Drafter settings are confirmed optimal.** `DRAFTER=none` 26.57 tok/s vs 37.73 shipped, so **speculation is worth
  +41%**; and with no drafter **1->4 stream scaling is exactly 1.00x** (rounds 36.5/74.3/149.2 ms), so all
  multi-stream throughput comes from the lane scheduler sharing verify forwards. `DRAFTER_BITS=0` (bf16) is **-3.7%**
  (acceptance improves 0.591->0.610 but the round grows 4.07 ms). `DRAFTER=mtp` is a wash at 1 stream and
  **-15%/-17.5%** at 2/4 -- though at 1 stream it runs 10% shorter rounds at higher acceptance (0.661) with 29% fewer
  drafts, so a cut threshold re-tuned for MTP at low load is an open, unexplored idea.
- **`TF_GLM_SHARED_SIDE=0` is -1.4/-1.0/-0.7%**, i.e. the shared-expert side stream is working. This was run as a
  falsification test of the byte accounting and the accounting survived.
- **The pair survives a hard power cycle bit-identically** (all 8 hashes, decode `sha@1` `5a0be163e6f3`).
- **Prefill has no gap to the v1.0.7 build at equal dense width** (-0.7% at 23.5K, +2.1% at 95K). The residual is q8
  vs its q4 default, a quality choice. Nothing to port.

---

## 5. TRAPS -- every one of these cost real time tonight. Read before running anything.

**Measurement contamination:**

1. **`~/.codex-shim/models.json` points at `http://127.0.0.1:8888/v1` -- the benchmark server.** Running
   `codex exec` to consult an advisor puts load on the pair being measured. This destroyed the `DENSE=q6` throughput
   numbers and the `restored` leg (31.6 tok/s at 2 streams vs 53.8 at the same config, 25-57% spreads).
   **Never consult an advisor against the measured endpoint.** Sequence it outside a leg, or re-point the shim. The
   user's GLM advisor is at `http://rtx:8888/v1`, a different host.
2. **`TF_GLM_PROF=1` disables prompt pairing** (`cuda_prompt_pair.zig:83`: `or e.prompt_profs[3] != null) return
   false`). **No prefill attribution from a profiled run represents production.** It also costs ~2% of prefill
   (1,700 vs 1,735 tok/s). Use Nsight Systems on the unprofiled path for any prefill critical-path claim.
3. **Check `cached_tokens == 0`** on every prefill measurement. A long prompt sharing a prefix with an earlier request
   resumes it and reports a rate that is not prefill. An early draft of the repo's own table read 2,005 tok/s this way.

**Instrument semantics:**

4. **`Prof.mark()` is cumulative since the previous mark, not a bracketed span.** `cuda_prof.zig:71`:
   *"The work queued on `s` since the last mark belongs to `section`."* The pieced prompt path never marks the mixer,
   attention, output projections or cross-rank exchanges, so the next mark -- `.router` -- **absorbs all of it**.
   I read 1,720 ms of 1,982 ms wall as "the router is 400x off roofline" and published it as the headline finding.
   **It was wrong and is retracted.** Codex caught it.
5. **Use the fine per-stage trace instead**, which *is* properly bracketed: `glm prompt stage rank= slot= start= rows=
   layer= piece= site= stage= stream= ms=` lines in the raw log. Filtered to exactly 4,096-row chunks on rank 0 (23
   passes), ms a pass: `mixer wait_filled side 3755.91`, `ffn wait_filled side 2963.37`, `ffn output_projection main
   933.00`, `ffn shared_expert main 882.13`, `ffn partials side 875.39`, `mixer partials side 689.93`,
   `mixer output_projection main 576.89`, `mixer glue main 294.47`, `ffn glue main 292.69`, `ffn glued side 213.00`,
   `mixer glued side 208.07`, `ffn wait_shared main 127.96`, `mixer wait_shared main 122.17`,
   `mixer wait_gathered main 121.64`, `ffn wait_gathered main 101.20`.
   Main-stream total 3,452 ms vs 1,982 ms wall, so **these overlap and are not additive -- relative weights only.**
   Gating is ~17% of main-stream time, not 87%. The large terms are **output projections (1,510 ms)** and the
   **shared expert (882 ms)**, with **473 ms (13.7%) of main-stream waits** while the side stream sits in
   `wait_filled` ~6.7 s, i.e. largely idle. **That idle side stream is the most interesting unexplored prefill signal.**
6. **The coarse prof buckets have mixed denominators.** The last bucket takes anything **above 1,024 rows** (its
   "to 2048" label is stale; patch 0094 documents this) and the raw log shows it mixing 3064, 3073, 3100 and 4096-row
   chunks; the "to 64" bucket held a **24-row** chunk. Any per-row figure from those buckets is invalid.

**Operational (these are now fixed in the harness, but know why):**

7. **`start.sh restart` has no verified stop.** On 2026-11-03 05:07Z the worker's `docker stop` failed
   (`cannot delete running task ... failed precondition`, `DeadlineExceeded`) and `docker run` was issued 3 s later
   anyway -- two ~100 GiB containers on a 128 GB host. No OOM kill, no panic: root cron kept running while `zurih`
   ssh logins stalled `[preauth]`, the journal stopped, and **spark4 needed a physical power cycle**. Ping and TCP/22
   answered throughout, so it looked alive. `docs/safestart.sh` now guards every restart in the harness; **fixing
   `start.sh` itself is still open** and is a change to the user's startup path, so ask first. Open question: hard
   `kill` after a stop timeout, or abort for a human?
8. **Never wrap a long job in a pipe or a self-referential poll.** `while pgrep -f "docs/mixed.py"` matched its own
   shell's command line and spun for **5h19m**, losing the night. `timeout 900 ./start.sh restart | tail -25` never
   saw EOF because a child held the pipe, so a healthy server looked like it was still loading for **41 minutes**.
   Redirect to a file with `>`, and check **ground truth** -- `docker ps`, `/health`, the server's own log -- never
   your own wrapper's output. Give every unattended wait a deadline that prints on expiry.

---

## 6. OPEN WORK, in the order I would do it

1. **Let the running ABA finish (~07:56Z) or stop it and restore the patch files** (section 2). Read `|A1 - A2|`
   first: that is the drift between two identical configs and it bounds every claim. If it is large, extend to
   A-B-A-B-A rather than pretending the gate resolved anything. Record 0096 as **inconclusive** and move on -- do not
   spend more restarts forcing a keep/revert on a 0.65% effect.
2. **A->B->A `HEAD=q8`**, then present it to the user for approval with the section-4 evidence. This is the only
   thing measured here that might actually ship.
3. **Prefill critical path, properly.** Nsight Systems on the **unprofiled** path on both ranks. The fine trace
   (trap 5) says to look at the output projections, the shared expert, and the 6.7 s of side-stream `wait_filled`.
   Do not start from the coarse buckets.
4. **Decode's two big exposed terms**, from the one attribution that is trustworthy (verify forwards, GPU 99.9% of
   wall): at 2 rows, `routed` 17.09 ms and `kda` 12.99 ms of 44.50 ms. `kda` is **flat in rows** (12.99 -> 13.17 from
   2 to 3 rows) so it is a fixed per-forward cost -- and it is the single largest fixed term. Nobody has looked at it.
   Caveat from Codex, accepted: these are *interval* attributions, not verified byte attributions; confirm traffic
   with hardware counters before calling anything bandwidth-bound.
5. **The uptime/drift lever** (section 4). Potentially bigger than every patch tried. Needs the user's consent to
   reboot spark3.
6. **Expert-traffic / lag-1 routing predictor**: research only until its measured upper bound clears the drift floor.
   Milestone 1 is logging top-8 per position to see whether routing correlates across adjacent positions. A marginal
   row costs 5.68 ms against 7.74 ms of disjoint expert bytes while random overlap explains only 2.8 points of that
   gap -- so either routing correlates (predictor viable) or the gather runs nearer peak. Prefetch is exact; skipping
   or approximating experts is **not** and would violate the quality constraint.
7. **`README.md` has uncommitted corrections** I made: patch count 93 -> 95 in three places plus an 0094-0095 table
   row, prefill 1,609/1,652 -> 1,735/1,759 and TTFT 14.6/57.5 -> 13.6/54.1 s, the "~11% behind v1.0.7" claim replaced
   with the measured parity-at-equal-dense-width result, reserve 12 -> the shipping 8, and `ABSORB_MMA` no longer
   described as untimed. **Review before committing** -- nothing in this campaign has been committed at all.

## 7. Advisors

The user's standing instruction was to consult **both** advisors before important decisions and not decide alone.
GLM is at `http://rtx:8888/v1`. Codex is the `codex` CLI (`codex exec --skip-git-repo-check --sandbox read-only
"$(cat brief.md)"`) -- **but see trap 1 about its endpoint.** Past rounds are in `docs/speed/panel/`
(`brief.md`, `agent-r1.md`, `codex-r1.md`, `glm-r1.md`, `r2-package.md`, `codex-r2.md`, `glm-r2.md`,
`brief-r6.md`, `codex-r6.md`). Codex's round 6 is the most useful single document: it caught the `mark()` artifact,
the mixed bucket denominators, and correctly weakened four of my other claims. Brief them with facts marked as
already-verified and cap their tool calls -- an early Codex round spent its entire budget reading source and returned
nothing.
