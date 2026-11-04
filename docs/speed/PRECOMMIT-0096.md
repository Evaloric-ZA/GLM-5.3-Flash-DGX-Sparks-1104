# Pre-committed decision rule for 0096, written before the coverage results were seen

Written 2026-11-03, after the 3-trial A/B (+0.20 / +0.26 / +0.65% at 1/2/4 streams, all below the
+2% gate, contract identical, mechanism confirmed by `drafted/round` moving at 2 and 4 streams only)
and **before** any coverage leg reported. The panel split: GLM KEEP, Codex REVERT.

## Why this is not a gate relaxed after the fact

0096's own header, written before the build, says: *"Whether that improves throughput is a question
for the A/B, not a claim of this patch."* The patch was pre-registered as a correctness fix to a cost
model whose constants were wrong by 34% and 60%, with an unknown throughput effect. The A/B's job was
to show it does not regress, not to justify it. That framing is pre-registered, not post-hoc, which is
what Codex's objection was aimed at. His risk-coverage objection stands on its own and is what the
legs below are for.

## The rule

**0096 is REVERTED if any of the following holds**, on the coverage suite below:

1. Any `token_sha` in the contract corpus differs, at any load, including under contention.
2. Any tested load regresses by more than **1.0%** in median total tok/s against the paired baseline.
3. p95 round ms at 4 streams, or at 8 streams under `PARALLEL=8`, is more than **5%** worse than the
   paired baseline (a scheduler trading drafts for round time can hide a worse tail; mean tok/s
   masks it).
4. Mixed prefill+decode shows either the decoders or the admitted prefill more than **1.0%** worse.
5. Long-context decode (about 45K tokens of context) regresses by more than 1.0%.
6. The pair fails to serve, or `/health`, `/v1/models` or the smoke test fails on the 0096 image.

**0096 is KEPT if none of the above holds.** It is then recorded as **a correctness fix with a
null-to-marginal throughput result** - both reviewers were explicit that "marginal win" overstates
the evidence and that no +2% language may be used. 0097 is kept in either case: it is inert under the
default `cut` rule and both reviewers wanted the stale sm_120 seed and comment gone.

If 0096 is reverted, the measured tables are still recorded in `RESULTS.md` so the next person has
the numbers without the behaviour change.

## The coverage suite, paired and bracketed

Run as **A -> B -> A** (0096, baseline, 0096 again) so drift across the night is visible, which is
the repo's own house rule for a same-slot comparison. Per pass:

- 4 streams, **10 trials**, fixed prompts: median, mean, standard error, p50/p95/max round ms.
  Three trials cannot resolve 0.65%; ten can say whether it is a shift or three lucky draws.
- Long-context decode: `CTXWORDS=12000` (about 45K tokens in the window), 1 and 4 streams.
- Mixed prefill+decode: 2 decoders with a ~95K cold prefill admitted 3 s in, `token_sha` checked
  under contention.
- `PARALLEL=8` stress at 8 concurrent streams. Note this changes the served configuration and its
  memory fit, so it is a stress probe, not the shipped shape; the shipped shape is `PARALLEL=4`.
- Contract corpus every pass.
