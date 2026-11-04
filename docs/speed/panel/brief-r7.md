# Round 7: HEAD=q8 cleared the interleaved ABA -- check the evidence package before it goes to the user

You are one of the advisors on the GLM-5.3-Flash / 2x DGX Spark (GB10, TP=2) speed campaign. Standing rules:
quality must not regress; anything numeric is measure-only and needs the user's explicit approval before shipping.

**Every number below is measured on this pair. Treat them as given; do not re-derive or re-measure them.**
Answer from this brief alone. Codex: at most 6 shell calls, do NOT read engine source; judge the evidence package.
Max ~400 words. The single question: **is there a hole in this package, or is it ready to present for ship approval?**

## The candidate

`HEAD=q8`, env-only (same image, same patches). The output head is 0.634 GiB of the 6.58 GiB/rank forward (4.8%);
q8 halves it, predicting about -2.4% on a 1-row forward round.

## Evidence A -- first pass, 06:54-06:59Z, clean window, non-interleaved

- Decode +3.1 / +0.9 / +1.2% at 1/2/4 streams (38.9 / 54.3 / 70.6 tok/s); round@1 52.64 -> 51.50 ms (-2.2%).
- Trial spreads 1.40 / 0.19 / 0.01%.
- Contract corpus: all 8 token_sha identical to baseline. 10 greedy answers identical (`11ceddde19d4`).
- Paired per-token NLL, 1,051 tokens, 12 domains: delta -0.00050, 95% CI -0.00156..+0.00056.
- quality.py 4-passage NLL 0.752643 vs 0.752338 (+0.0003).

## Evidence B -- interleaved A->B->A, 07:54-08:07Z, all legs on the 0096 image

| load | A1 | B (q8) | A2 | drift \|A1-A2\| | effect B vs pooled A |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 stream | 37.42 | 38.59 | 37.40 | 0.04% | +3.69% |
| 2 streams | 52.92 | 54.14 | 53.04 | 0.22% | +2.21% |
| 4 streams | 68.89 | 70.20 | 68.90 | 0.02% | +2.01% |

- Round ms @1: 53.41/53.37 -> 51.92 (-2.8%), vs the -2.4% byte prediction. Second independent confirmation.
- Tokens and per-trial shas identical across all three arms; contract 8/8 in every leg.
- Arm identities verified from the engine banner: model 83.58 / 83.29 / 83.58 GiB (B really loaded the halved head).
- B's slowest 1-stream trial (37.06) was its first after restart; A arms show the same cold-first-trial pattern.

## Context you need

- Same night, the SAME configs 26 min apart wandered 18% on the 10-trial coverage bench (49.6 vs 40.5 med), and
  means declined monotonically 49.2 -> 46.4 -> 43.1 across 40 min. The leg-sweep instrument (fixed prompts, 3
  trials, right after a verified restart) is the stable one: A arms above agreed within 0.25%.
- The quality evidence (greedy identity, paired CI) is from the 06:54 pass; tonight's legs re-confirmed the
  contract hashes but did not repeat the paired NLL (config unchanged, same image class).
- 06:54 effects at 2/4 streams were +0.9/+1.2; tonight +2.2/+2.0. Same direction, larger tonight.

## The ask

Ship-approval readiness only: name concrete holes (missing load points? evidence too narrow? a check that is cheap
tonight and material?), or say ready. If you propose an additional check, it must be runnable in under 30 min on
this rig and must not be another sub-1% measurement.
