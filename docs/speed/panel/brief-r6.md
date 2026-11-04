# Round 6: the profile landed, and it says something none of us predicted

You are one of three advisors (the lab's agent = me, Codex = you, GLM) on speeding up GLM-5.3-Flash decode and prefill on a
2x DGX Spark (GB10) pair, TP=2, 128 GB unified LPDDR5X at ~273 GB/s shared by CPU and GPU, served by TensorFold's
native Zig/CUDA engine. Standing constraint: **quality must not regress.** `W8A8` and `DENSE=q4` were both rejected
on quality by the user and are closed. Anything numeric is measure-only and needs the user's explicit approval.

**Every number below is measured by me on this pair tonight. Treat them as given; do not re-derive them.**
Repo root is the cwd. Budget: **at most 10 shell/tool calls, then write the full deliverable.** A previous round of
yours spent its entire budget reading source and produced no answer, which wasted the round. Do not repeat that.

## 1. What the panel believed, and what measurement did to it

Round 1-2: all three of us ranked "go deeper on speculative decoding" first. **Refuted.** The verify-cost line on GB10
is `a + b*(rows-1)` with a=38.73 ms, b=5.68 ms, so `b/a = 0.147` -- slightly *worse* than sm_120's 0.139. A draft row
is NOT cheap here. Forced deep windows measured -17%/-19%. The shipped `cut` rule at 0.70 is bracketed in all three
directions (0.40 -> -3.9/-6.3/-3.5%; 0.85 -> flat). That lever is closed.

Mechanism: each row draws its own top-8 of 288 routed experts, so routed-expert bytes scale per row.

## 2. Host-state drift is larger than the effects we were measuring

spark4 wedged (a restart race: `docker stop` failed, `docker run` was issued anyway, two ~100 GiB containers on a
128 GB host) and was power-cycled. The pair came back **bit-identical** (all 8 contract `token_sha` match, decode
`sha@1` unchanged). But the *same config* measured **+1.52 / +1.78 / +1.49%** faster after the reboot at 1/2/4 streams.
Within-run trial spreads are 0.14-0.69% and tokens are bit-identical, so this is host state, not trial noise.

Consequence: patch 0096 (a cost-table correction) measured +0.20/+0.26/+0.65%, which is **inside the drift band**. It
is unresolved -- not keep, not revert. Only interleaved A->B->A measurement is valid at the 1-2% level, and sub-1%
effects need ~140 restarts an arm to resolve (sigma~1.5%, one observation, so sigma itself is poorly known).

## 3. THE PROFILE (the new thing; this is what I want your judgement on)

`TF_GLM_PROF=1`. Per **4,096-row prompt chunk** (21 reports, 84 calls), wall 1,981.78 ms a call:

| section | ms/call | % of wall |
| --- | ---: | ---: |
| **router** | **1,720.38** | **86.8%** |
| routed | 641.07 | 32.3% |
| head | 12.13 | 0.6% |
| embed, gate, draw, mtp, commit | < 1.3 | ~0% |

GPU totals 120.5% of wall; "NOT GPU" is -20.5%. So sections overlap on concurrent streams and the column sums past 100.

Per-row, comparing the 64-row bucket against the 4,096-row bucket:

| section | 64 rows | 4,096 rows | amortisation |
| --- | ---: | ---: | ---: |
| routed | 1.511 ms/row | 0.157 ms/row | 9.6x |
| router | 0.727 ms/row | 0.420 ms/row | 1.7x |

**My reading**: the router needs ~4 ms of arithmetic a chunk (gating is a 4096x288 GEMM a layer, ~217 GFLOP over 45
layers; top-8 of 288 over 184K rows is trivial) and is taking 1,720 ms -- ~400x off roofline. It did not show up before
because patch 0094's wider chunk amortised `routed` 9.6x and left the router as the wall, capping that patch at +7%.
I argue it is not merely blocking on the expert stream, because a wait would track `routed`'s amortisation curve and
it does not.

Decode attribution from the same profile (GPU = 99.9% of wall, so **decode has no host-time term**):

| section | 2 rows | 3 rows | marginal |
| --- | ---: | ---: | ---: |
| routed | 17.09 | 22.66 | **+5.57** |
| kda | 12.99 | 13.17 | +0.18 |
| gather_a+b | 1.85 | 2.36 | +0.51 |
| mla / router / head / dense / shared | 12.57 | 12.71 | +0.14 |
| total | 44.50 | 50.89 | +6.39 |

So **87% of a marginal verify row is routed-expert bytes**, and **KDA is flat in rows** (a fixed per-forward cost).
Extrapolated to 1 row (~38.09 ms): kda 33.6%, routed 30.2%, mla 12.9%, router 8.6%, head 7.2%, gathers 3.5%,
dense 3.0%, shared 0.9%. My independent byte budget from the checkpoint's shape predicted 35.8 / 32.1 / 10.9 / 9.6 /
3.4% -- so the byte model holds, except `shared` (predicted 8.0% of bytes, measures 0.9%) which is being hidden on its
side stream. `TF_GLM_SHARED_SIDE=0` is therefore predicted to be *worse*; that leg is running now as a falsification test.

Drafter probes, 1/2/4 streams tok/s:

| leg | 1 | 2 | 4 | round ms @1 | tok/round | accept | drafted/round |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shipped (DFlash2 4-bit) | 37.73 | 53.79 | 69.79 | 52.64 | 2.083 | 0.591 | 1.83 |
| DRAFTER=none | 26.57 | 26.45 | 26.40 | 36.46 | 1.004 | 0 | 0 |
| DRAFTER_BITS=0 (bf16) | 36.18 | 50.60 | 67.60 | 56.71 | 2.155 | 0.610 | 1.88 |
| DRAFTER=mtp | 37.47 | 45.62 | 57.60 | 47.37 | 1.866 | 0.661 | 1.30 |

Speculation is worth +41%. **Without a drafter, 1->4 stream scaling is exactly 1.00x** (rounds 36.5/74.3/149.2 ms), so
all multi-stream throughput comes from the lane scheduler sharing verify forwards, not from batching.

## 4. What I am asking you

1. **Is the router conclusion sound, or am I over-reading an overlap artifact?** Given GPU=120.5% of wall, name the
   single most decisive confirmation I can run or read. I have over-claimed twice tonight and want this one attacked.
2. **What does this engine's prefill `router` section most plausibly contain?** If it is more than gating logits +
   top-k (e.g. dispatch index building, the permutation of hidden states into expert-contiguous order, a histogram /
   prefix-sum, a cross-rank exchange), say which, and what the 9.3 us per row-layer is most likely being spent on.
   Relevant: the engine has off switches `TF_GLM_TOPK_SUM=0`, `TF_GLM_DSEL=0`, `TF_GLM_MM_TWO=0`,
   `TF_GLM_SHARED_SEGS=0`, `TF_EXL3_SHFL=0`, `TF_EXL3_PDL=0`, `TF_GLM_PDL_CHAIN=0` (patches 0067-0091).
3. **Priority call.** Rank: (a) investigate the prefill router, (b) `DENSE=q6` (~14.6% of forward bytes, measure-only,
   needs user approval to ship), (c) finish the A->B->A gate on 0096, (d) the expert-traffic / lag-1 routing predictor
   project, (e) something else you name. Justify in terms of expected value against the ~1.5% drift floor.
4. **Anything in sections 1-3 you think is wrong.** Say so plainly.

Write a direct deliverable with headings, concrete next actions, and your confidence on each. Disagree where you
disagree -- agreement I have not earned is worthless to me.
