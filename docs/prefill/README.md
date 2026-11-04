# Prefill on this pair

Measured 2026-11-02 on spark3 + spark4, engine `tensorfold-native 1.1.9` for `sm_121`. Every prefill figure is the
median of three cold trials with **fresh random content**, so no prefix cache is hit (`cached_tokens` is 0 in each
reply); trials reproduce to within 0.2%. NLL is over four held passages via `/v1/completions` `prompt_logprobs: 0`;
"answers" is a sha over ten greedy replies. `quality.py` and `pf2.py` in `../` are the harness.

| Config | @23.5K | @95K | NLL | answers |
| --- | ---: | ---: | ---: | --- |
| `q8`, numerics off | 1,503 | 1,530 | 0.7535 | reference |
| `q8` + `ABSORB_MMA`, 2,048 rows | 1,609 | 1,652 | 0.7523 | identical to reference |
| **`q8` + `ABSORB_MMA`, 4,096 rows (shipping)** | **1,735** | **1,759** | **0.7523** | **identical to reference** |
| `q8` + `ABSORB_MMA` + `W8A8` (**stashed**) | 1,789 | 1,828 | 0.7535 | 1 of 10 differs, and wrong |
| `q4` + `ABSORB_MMA` | 1,747 | 1,781 | 0.8005 | 1 of 10 differs, and wrong |
| v1.0.7 Spark build (`q4`, 8 streams) | 1,858 | 1,835 | not measured | not measured |

## The 4,096-row prompt chunk: +7%, exact, shipped (patches 0094 + 0095)

`cuda_state.zig`'s `prefill_rows` was 2,048. A call to the common EXL3 experts decodes every expert's weights and a
prompt pass makes one call a chunk, so that decode amortises over the chunk's rows; `cuda_exl3.zig` records that at
1,024 rows "every trellis tile was decoded twice a pass" and names 4,096 as where upstream's PR #1576 puts it.
`cuda_exl3.zig:281` loops the experts call in `chunk_rows` increments, so 4,096 rows at a 2,048 setting is two calls.
Prompt pairing does **not** help here: it exists to hide each chunk's exchanges under the other's work, with twin
buffers on a side stream, and the drafter ingests the two chunks separately.

`0094` takes `prefill_rows` to 4,096. On its own that breaks the drafter: the pair loads, smoke-tests, and the first
real prefill ends both ranks with `DrafterContextTooWide`. `0095` takes `dflash.zig`'s `MAX_CTX` to 4,096 and leaves
`WIN` at 2,048, which is sufficient because `promptRun` already keeps only the rows from `len - WIN` on and returns
null for a chunk ending before the window -- `MAX_CTX` is a tap-buffer capacity, not a window. The guard cannot merely
be relaxed: a forward pass writes a chunk's taps into `taps_prefill` before `promptRun` runs
(`cuda_engine.zig:454`), so `MAX_CTX` has to size the buffers.

Measured with DFlash2 on, against the 93-patch image, at `MEMORY_RESERVE_GIB=10`:

| | 93 patches (2,048 rows) | 95 patches (4,096 rows) |
| --- | ---: | ---: |
| prefill @23.5K | 1,609 tok/s | **1,735 (+7.8%)** |
| prefill @95K | 1,652 tok/s | **1,759 (+6.5%)** |
| NLL | 0.7523 | 0.7523 |
| 10 greedy replies | `11ceddde19d4` | `11ceddde19d4` |
| decode, 1 stream | 35.3 tok/s | 35.7 tok/s |
| draft acceptance | not measured | 0.525 (157/299) |
| model | 81.80 GiB | 83.58 GiB |
| prompt cache | 3.1 GiB | 2.5 GiB |
| window / KV pool | 1,048,576 / 2,098,176 | unchanged |

Nine boundary prompts from 1,169 to 45,599 tokens, straddling both the 2,048 and 4,096-row boundaries, answered with
no failures.

**The reserve matters here.** At the recipe's old `MEMORY_RESERVE_GIB=12` the wider chunk's scratch took the prompt
cache from 3.1 GiB to 0.4, which trades multi-turn resume for cold prefill: a 23.5K follow-up that used to resume
instantly would re-prefill in ~14 s. At 10 the cache returns to 2.5 GiB with prefill and replies unchanged, which is
why `scripts/local.sh` sets 10. That is ~2 GiB less cushion than the engine's own unified default (~11.9) on a Spark
that also runs a desktop and other containers; `MEMGUARD` (floor 2.5 GiB) is the backstop, and 11 is the middle option.

### What is still unproven

- Draft acceptance was not measured on the 2,048-row baseline, so acceptance parity rests on decode speed being
  unchanged rather than on a direct comparison.
- `cuda_forward_pair.zig` and `cuda_moe.zig` were not in the set read during review, so the pair-to-expert call chain
  is inferred from `cuda_exl3.zig:281` and the pair's own comments.
- The +7% is not isolated to expert amortisation: chunk width also moves buffer sizes, scheduling and pair eligibility.
- Identical NLL and reply hashes come from a sampled set, which is evidence of exactness, not proof for all prompts.

## What is not the answer

- **The link.** Hidden 4,096 over 45 layers puts a chunk's split exchange at 16 MiB of fp32 (half the rows), about
  0.72 GiB a chunk, roughly 3% of a 15 s TTFT at 23.5K. The v1.0.7 build's bf16 partial exchange would buy ~1.4%.
- **Any default being off.** `TF_GLM_PREFILL_REDUCE=split`, `TF_GLM_PROMPT_PAIR`, `TF_GLM_INDEX_SPLIT=2048`,
  `TF_GLM_OVERLAP_PIECES=3` and `TF_GLM_MM` are all on already.
- **Dense width.** `q4` is ~1.8 GiB of 81.8 here and buys 7% while costing real quality (NLL 0.8005).

## W8A8: stashed

Set aside on 2026-11-02 at the user's call, so it is not an open option. It is the fastest thing measured here and it
reaches the v1.0.7 build's prefill, but it is the only change in this whole exercise that made the model worse on a
task it had been getting right. Engine patch 0064 stays in the stack, inert, with `W8A8=0`; removing it from the image
would cost a rebuild and gains nothing while the flag is off. Re-open it only behind an eval that clears it, and judge
that eval per passage -- its aggregate NLL matched baseline exactly while every passage had moved.

## Dense width settled: q8, with numbers

All three widths measured on this pair, 4,096-row chunk, `ABSORB_MMA=1`, reserve 10:

| dense | prefill @23.5K | @95K | model | KV pool | quality |
| --- | ---: | ---: | ---: | ---: | --- |
| bf16 | 1,288 | 1,262 | 87.00 GiB | 2,052 pages (6.24 GiB) | indistinguishable from q8 |
| **q8 (shipping)** | **1,735** | **1,759** | 83.58 GiB | 4,098 pages (12.47 GiB) | reference |
| q4 | 1,863 | 1,885 | 81.81 GiB | 4,098 pages | NLL +0.053, answers a task wrong |

**bf16 is dominated**: 26% slower, 3.42 GiB larger, half the KV pool, and no measurable quality gain.
A paired per-token comparison over 1,051 tokens of 12 domains (English technical and narrative prose, Python, Zig,
mathematics, legal, dialogue, Chinese, JSON, medical, history, instructions; `nllpaired.py` + `cmp.py`, raw in
`nll-q8.json` / `nll-bf16.json`):

```
mean per-token logprob delta (bf16 - q8):  -0.00394   se 0.00580   95% CI -0.0153 .. +0.0074
bf16 better on 517 tokens (49.2%), worse on 528 (50.2%), identical on 6
aggregate NLL: q8 1.50609, bf16 1.51002      per passage: 5 of 12 favour bf16, 7 favour q8
```

The point estimate favours q8 and the interval spans zero, so any difference is smaller than about 0.015 NLL a token
-- the resolution 1,051 tokens buys. Resolving a 0.005 effect would need roughly ten times the corpus.

A narrower four-passage run had bf16 ahead by 0.0048, which did not survive widening the corpus: its passages pointed
both ways (q8 better on 2 of 4) and the mean was their cancellation. **Judge dense width, like numerics, per passage
and paired, never on a narrow aggregate.** The one greedy answer that differs between q8 and bf16 is the same correct
`yrassecen` with different wording of its check -- not a regression, unlike q4's `yrasssecen` or W8A8's `yrassen`.

## No prefill gap against the v1.0.7 build

Re-measured with the 3-trial protocol, since the earlier single-shot figures spanned 12% at one point:

| at equal q4 dense | @23.5K | @95K |
| --- | ---: | ---: |
| this engine (q4 + 4,096 rows + absorb) | 1,863 | **1,885** |
| v1.0.7 Spark build (q4, its default) | **1,877** | 1,847 |

Parity, crossing over with prompt length: -0.7% at 23.5K, +2.1% at 95K. The ~7% that remained after the 4,096-row
chunk was q8 against its q4 default, not anything missing from this engine's prompt path. Nothing to port.

## Memory: the reserve, the KV pool and the prompt cache

The pool grows only in **whole-window steps of 2,048 pages (~6.4 GiB)**, so the reserve does not trade smoothly
against it. Measured on this pair with the 4,096-row chunk, vision on, 4 streams, a 1,048,576-token window
(`reserve-vs-kv-pool.log`); low-water is MemAvailable after a 95K-token prefill:

| reserve | room | KV pool | prompt cache | prefill @23.5K | spark3 low-water | spark4 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 10 | 16.1 GiB | 4,098 pages / 2.10M tok | 2.5 GiB | 1,735 | 10.4 GiB | 15.3 |
| **8 (shipping)** | 18.1 GiB | 4,098 pages / 2.10M tok | **4.4 GiB** | 1,742 | 8.3 GiB | 13.3 |
| 6 | 20.2 GiB | **6,224 pages / 3.19M tok** | **none** | 1,723 | 6.3 GiB | 11.2 |

Because of the step, 10 -> 8 buys the pool nothing and the 2 GiB goes to the prompt cache instead; 10 -> 6 buys a
third window but spends the cache entirely ("the KV pages took it"). Keeping a 2.5 GiB cache *and* a third window
would need about 25 GiB of room, i.e. a reserve near 1 on rank 0, so **pool and cache are mutually exclusive at the
next step**. With vision and a prompt cache both required, reserve 8 is the best point. All three settings were stable;
6.3 GiB is still well above MEMGUARD's 2.5 GiB floor.

**Rank 0 binds, and the pair is unbalanced.** spark3 (rank 0: vision tower, HTTP server, the larger weight share, plus
a desktop and other containers) sits ~0.4 GiB above its reserve while spark4 holds ~5 GiB spare, and the engine admits
by the smaller room of the two ranks, so spark4's surplus is unusable. Running the recipe *from* spark4
(`WORKER=...:10.0.34.3`) would put the heavy rank on the emptier Spark and balance them, worth an estimated ~2.4 GiB
of binding room for free -- at the cost of moving the API to spark4. Not done.

`PROMPT_CACHE_GIB` is left empty, so the cache is the **residual** after the pages, which is why it has read 0.4, 2.5,
4.4 and 4.5 GiB under different settings. Pin it to a number if a predictable cache matters more than maximal pages.
