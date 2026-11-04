# Question: is a 4,096-row prompt chunk really gated on reworking the DFlash2 drafter?

You are reviewing TensorFold's GLM-5.3-Flash CUDA engine (Zig). `src/` holds the relevant files, copied from a
checkout of upstream `f350cd3e` with this recipe's engine patches 0001-0066 applied (patches 0067-0093 exist but are
not in these copies; none of them changes the constants below). Read-only: do not edit anything. Answer in `answer.md`.

## What I measured on two DGX Sparks (GB10, TP=2, one rank each)

`src/cuda_state.zig` has `pub const prefill_rows = 2048`. `src/cuda_exl3.zig` sets
`chunk_rows = cuda_state.prefill_rows` with a note that a call to the common EXL3 experts decodes **every** expert's
weights, so a prompt pass makes one call a chunk and that decode amortises over the chunk's rows; it records that at
1,024 rows "every trellis tile was decoded twice a pass" and names 4,096 as where upstream PR #1576 puts it.

I changed only that constant to 4,096 (`0094-...patch`) and measured, three cold trials a point, fresh random prompt
content each trial so no prefix cache is hit, `DRAFTER=none` on both sides so the drafter is out of the comparison:

| chunk rows | prefill @23.5K tok | @95K tok | NLL (4 passages) | sha of 10 greedy replies |
| --- | --- | --- | --- | --- |
| 2,048 | 1,626 tok/s | 1,648 tok/s | 0.7523 | 11ceddde19d4 |
| 4,096 | 1,743 tok/s (+7.2%) | 1,757 tok/s (+6.6%) | 0.7523 | 11ceddde19d4 |

So +7%, and bit-exact (identical NLL and identical greedy replies).

With the real config (`DRAFTER=dflash2`, the shipped default) the pair loads, passes a smoke test, and then the first
genuine prefill ends both ranks:

```
error: rank 0: rank 1 stopped (DrafterContextTooWide); this rank stops too
```

`src/dflash.zig` has `MAX_CTX = 2048` ("context rows one ingest takes (a prompt chunk's)") and `WIN = 2048`
("the context window (and a ring's slots)"), with the checks at lines ~282, ~370 and ~385.

## My current conclusion, which I want you to check

I concluded that getting the +7% needs a change to the drafter's ingest path (letting it take a wide chunk in
`MAX_CTX`-sized slices, since it only needs the last `WIN` rows), because `WIN` is the trained drafter's attention
window with rotary tables built for it, and so is not a buffer I can simply widen.

## Specific questions

1. **Is `MAX_CTX` alone the binding constraint?** If `MAX_CTX` were raised to 4,096 and `WIN` left at 2,048, would the
   result be correct, or does `WIN` genuinely bound how many rows of one chunk the drafter can ingest? Trace what
   `MAX_CTX` sizes (`dflash_load.zig` allocations, `gs`) versus what `WIN` governs (which taps/positions are read, the
   ring's slots, rotary). Cite line numbers.
2. **Does a paired prompt step already amortise the expert trellis decode over 4,096 rows?** `src/cuda_prompt_pair.zig`
   (TF_GLM_PROMPT_PAIR, on by default, `default_min_len = 2 * prefill_rows + 1`) makes a prompt step take two chunks.
   Does the EXL3 expert call then cover both chunks' rows in one call, or one call per chunk? If pairing already covers
   two chunks, why would raising `prefill_rows` still give +7%? This is the answer I most want.
3. **Is there a cheaper path to the same win** -- amortising the expert trellis decode over more rows *without*
   changing the chunk size the drafter ingests? For instance, making the experts call span a paired step's two chunks,
   or decoding the trellis once for two consecutive calls.
4. If a drafter-side change is genuinely required, what is the smallest correct one, and what would it risk (draft
   acceptance, `BadPromptContext` at ~385, the ring's slots)?

Be concrete and cite file:line. Say plainly if you think my conclusion is wrong.
