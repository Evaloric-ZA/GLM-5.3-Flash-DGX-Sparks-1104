# Handoff 2: what is left after HEAD=q8 shipped

**Supersedes `docs/speed/HANDOFF.md` for state and priorities. Its sections 1 (constraints), 3 (harness) and 5
(traps) are still fully in force -- read them.** `docs/speed/RESULTS.md` (~650 lines) remains the record and wins
over `AGREEMENT.md` wherever they disagree.

Working folder: `/home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig`
Engine source (read-only reference): `/home/zurih/TF/.src/tf`

## Current shipped state (2026-11-03 ~08:4xZ)

- **`HEAD=q8` is the serving default**, user-approved and verified: `scripts/local.sh:46`, rank banner reads model
  **83.29 GiB**, contract corpus **8/8 identical** to the bf16 baseline (`contract-shipped-q8.json`).
  Evidence: interleaved A->B->A **+3.69 / +2.21 / +2.01%** at 1/2/4 streams with `|A1-A2|` of 0.04-0.22%, round @1
  53.4 -> 51.9 ms (-2.8% against a -2.4% byte-budget prediction, now confirmed twice), 10/10 greedy answers
  `11ceddde19d4`, paired per-token NLL delta -0.00050 with 95% CI -0.00156..+0.00056 **re-measured on the ship
  image itself**. Rollback: `HEAD=bf16` in `scripts/local.sh`, then `bash docs/safestart.sh`.
- **Patches 0096 + 0097 are in `patches/engine/` and in the serving image `eeb98fe202ff`.** 0096 is recorded
  **inconclusive on throughput** -- a correctness fix (the mis-priced shared cost table starved `depth.zig`'s
  overhead learner: 4 of 139 forwards could fire it vs 133/139 at measured prices) whose throughput effect is
  below the instrument's noise. Contract half of `PRECOMMIT-0096.md` passed outright. **Spend no more restarts on
  it.** 0097 stays, inert under the default `cut` rule.
- `docs/leg.sh` with no overrides now measures **q8** (local.sh wins). A bf16 comparison must pass `HEAD=bf16`
  explicitly. `contract-baseline.json` was captured at bf16 and is still the valid reference -- the hashes are
  identical under q8, which is the point.
- **Nothing is committed.** 18 files: `docs/speed/` entire, the `docs/*.py` + `docs/*.sh` harness,
  `patches/engine/0096`+`0097`, `scripts/local.sh`, and `README.md` (+37/-26).

## Closed -- do not re-open

Deeper speculation (`b/a = 0.147`, forced deep windows -17/-19%, `cut` bracketed in all three directions);
`W8A8` and `DENSE=q4` (user-rejected on quality); **`DENSE=q6`** (10 greedy answers changed, 3 of 8 contract
hashes changed, `quality.py` NLL +0.0187 -- the W8A8 aggregate-cancellation trap reproduced); drafter settings
(`DRAFTER=none` -41%, `DRAFTER_BITS=0` -3.7%, `DRAFTER=mtp` -15/-17.5% at load); `TF_GLM_SHARED_SIDE=0` (-1.4/-1.0/
-0.7%, so the side-stream overlap works); prefill parity with the v1.0.7 build at equal dense width.

## WHAT IS LEFT, in the order I would do it

### 1. Commit the campaign. Do this first.

18 files, untracked or modified, including the **shipped numerics change** and the entire measurement record. One
`git clean` or a bad day loses all of it. Suggested split: (a) the harness (`docs/*.py`, `docs/*.sh`), (b) the
record (`docs/speed/`), (c) patches 0096+0097 with the inconclusive verdict in the message, (d) the ship act
(`scripts/local.sh` + `README.md`). Review the README diff before committing -- it carries corrections made
mid-campaign (patch count 93->95, prefill 1,735/1,759, the "~11% behind v1.0.7" claim replaced by measured parity,
reserve 8, and the q8 decode table).

### 2. The bimodal 4-stream round time. This is the biggest unexplored number in the project.

The coverage bench is **bimodal within a single pass**: round-ms modes near **110 ms and 150-185 ms**, per-trial
throughput swinging **26-68 tok/s**, within-pass spreads of 52-104%. The clean leg sweep after the same restarts is
0.01-3% stable (108.4 ms constant). So the engine has a slow mode at 4 streams that the leg sweep does not trigger
and the coverage bench does.

Why this matters more than any patch tried: the slow mode is ~40-70% worse than the fast mode. If it can be
identified and eliminated, that dwarfs every effect measured in this campaign (the best was +3.7%). It is also a
**p95 latency bug that real serving would feel**, not just a benchmark artifact.

Leads, untested: the KV-pool fit itself wanders with free host memory (**4,098 pages in a1 vs 6,062 in a2**,
identical config and image, 28 min apart) -- "same config" does not mean "same memory fit" on a unified device.
Also candidates: lane-scheduler mode switching, prompt-cache pressure, page migration between CPU and GPU, thermal.
Start by correlating per-trial round ms against the pool size and free host memory in the same pass, then at the
`dft` trace level (`docs/trace.py` parses those records) to see whether the slow mode is a different window/rows
regime or the same window taking longer.

### 3. The monotonic degradation, and the uptime lever.

Within the 0096 ABA the 4-stream means fell **49.2 -> 46.4 -> 43.1 tok/s (-12%) over 38 minutes**, one-directional,
identical config. Separately, the *same* config measured **+1.5%** faster after spark4 was power-cycled from 4 days
of uptime. Both point at a slow, accumulating component -- plausibly unified-memory fragmentation, since weights,
KV pages and the prompt cache all draw on one 128 GB pool.

If real, a scheduled restart is worth more than every patch in this campaign combined. Probably the same root cause
as item 2 and worth investigating together. The clean test needs **rebooting spark3** (rank 0, the API, never
restarted during this campaign) -- **ask the user before doing that.** A cheaper first step: log `MemAvailable`,
the pool size and fragmentation indicators alongside a long decode run and look for the correlation.

### 4. Prefill critical path, with the right instrument.

**Do not use `TF_GLM_PROF` for this** -- it disables prompt pairing (`cuda_prompt_pair.zig:83`), so its prefill
path is not production's, and `Prof.mark()` is cumulative-since-last-mark, which is what produced the retracted
"router is 400x off roofline" claim. Use **Nsight Systems on the unprofiled path, on both ranks.**

The properly bracketed per-stage trace (HANDOFF sec 5, item 5) says where to look, as relative weights only:
output projections **1,510 ms/pass**, shared expert **882 ms**, main-stream waits **473 ms (13.7%)**, and the side
stream idle in `wait_filled` for **~6.7 s**. That idle side stream against a waiting main stream is the most
interesting prefill signal nobody has chased.

### 5. KDA -- the largest fixed decode term, never examined.

From the one trustworthy attribution (verify forwards, GPU 99.9% of wall): at 2 rows, `kda` is **12.99 ms of
44.50 ms**, and it is **flat in rows** (12.99 -> 13.17 from 2 to 3 rows), so it is a fixed per-forward cost -- the
largest one. 34 of 45 layers are KDA (64 heads x 128, rank-128 gates). Nobody has looked at its kernels.
Accepted caveat from Codex: these are *interval* attributions, not verified byte attributions -- confirm traffic
with hardware counters before calling anything bandwidth-bound.

### 6. Expert traffic / lag-1 routing predictor -- research only.

A marginal verify row costs 5.68 ms against 7.74 ms of disjoint expert bytes, while random overlap between two
rows' top-8 of 288 explains only 2.8 of the 27-point gap. So either routing correlates across adjacent positions
(predictor viable) or the gather runs nearer peak. Milestone 1: log top-8 per position and measure the correlation.
Promote only if the measured upper bound clears the drift floor by a useful margin. **Prefetch is exact; skipping
or approximating experts is not and would violate the quality constraint.**

### 7. `start.sh`'s missing verified stop -- still open, needs a user decision.

`docs/safestart.sh` guards every restart in the harness and held on all three ABA transitions, but `start.sh`
itself still fires `docker run` without confirming the previous container died -- the race that physically wedged
spark4 (journal evidence in RESULTS.md). Fixing it touches the user's startup path, so **ask first.** Open
question for the user and the advisors: hard `kill` after a stop timeout, or abort and leave it for a human?

## Harness notes from the last session

- `dec.py`'s summary and noise-floor prints are now None-safe (`n/a`) -- a degenerate load used to raise and cost
  the rest of a pass.
- `docs/speed/aba.py` and `docs/speed/head-aba.sh` are the interleaved A->B->A driver and its evaluator; reuse them
  rather than writing a new one. `aba.py` prints IDENTITY warnings if tokens or hashes diverge between arms.
- **Effect sizes come only from the clean leg sweep, interleaved.** Coverage passes are for contract and contention
  coverage, never for effect size.
