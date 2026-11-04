I would raise the reserve to **9.5 GiB**, hold **0103 for a parser guard**, and canary **0104 after cache retention works**. I read only the four requested files; this is static review and arithmetic, with no live testing.

**The cache cliff is real.** With eight admitted lanes, your rounded constants give:

- Two-window branch: cache ≈ `room − 2×6.23 − 7×0.41`.
- Three-window branch: cache ≈ `room − 3×6.23 − 7×0.41`.

At room 19.8, the latter is −1.76 GiB, saturated to zero. At reserve 9.5, room becomes approximately 18.3 and the two-window branch leaves **2.97 GiB**. The threshold is approximately **19.51 GiB**, giving **1.21 GiB** of upward room margin. Your predictions are consistent with the code and rounded inputs. [Planner arithmetic](/home/zurih/c2codex/kv_pool_plan.txt:33)

There are several qualifications to the reserve change:

- `windows=3` is a reservation heuristic, **not a guarantee of three full windows**: it is calculated without charging all seven extra streams’ fixed costs. Today’s 5,580 pages provide only 2.72 window equivalents.
- Approximately 4,096 pages provide 2,097,152 KV rows in a shared pool. “Two × 1M” or “eight × 262K” describes aggregate capacity before additional rows, page rounding, and any run-page requirements. Count prompt **and generated** tokens. Eight short requests completing does not establish eight long-context capacity.
- The native caller passes `window + F.window_rows`; the planner also includes `runPages()`. Those omitted constants can explain small differences such as 4,096 versus 4,098 pages. Cache splitting and the caller’s sub-minimum-state fallback can further change the allocation. [Native caller](/home/zurih/c2codex/cuda_kv_pool_plan.txt:28)
- **The claimed 1.5 GiB increase in available memory needs measurement.** It reduces the planned allocation ceiling. Because pages are allocated on demand and the restored cache now consumes memory, the same short workload need not gain 1.5 GiB of actual available memory. Measure both Sparks after the cache has filled. The previous 8.3 GiB low-water was 5.8 GiB above memguard’s threshold, but it did not exercise this allocation mix.

Lowering parallelism or context can be better when those capabilities are expendable. Your four-lane prediction is correct: it still selects three windows and leaves zero cache. Two lanes cap the window count at two and would leave roughly **6.9 GiB cache** at today’s room, at the cost of concurrency. The 524K context setting leaves approximately 1.4 GiB while reducing the maximum context. Neither knob gives monotonic cache improvement across window thresholds. Given your stated requirements, **9.5 GiB reserve is the better immediate tradeoff**.

A fixed reserve does **not** pin room: another approximately 1.2 GiB becoming available at startup could cross the threshold again. `--prompt-cache-gib` cannot prevent that because `held()` is calculated first. For an operational fix, check the resulting pages and cache budget at every startup and alert if they miss the accepted floor. The caller also mentions `TENSORFOLD_CUDA_MEMORY_LIMIT_GB` as a room bound; a correctly configured ceiling could prevent upward crossing, but its accounting is not shown sufficiently here to prescribe a value.

The durable planner fix is an explicit **minimum cache reservation**, accounted for before choosing additional whole windows, with `holds()` using the same policy. Setting only a larger desired cache cannot accomplish that.

**0103 has a concrete false-call case worth fixing before porting.** An opener alone inside a valid, offered outer JSON call is safe: successful initial parsing bypasses the retry. The problem appears when the argument contains a complete literal call:

```text
<tool_call>{"name":"lookup","arguments":{"q":"<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>"}}</tool_call>
```

Under the earliest-opener/first-closer rule described by the patch, the first closer truncates the outer JSON inside `q`. The outer parse fails. The retry then extracts:

```text
lookup<arg_key>n</arg_key><arg_value>7</arg_value>
```

That is exactly the valid inner-call shape covered by the patch’s positive tests. It can invoke `lookup({"n":7})` instead of passing the literal string as `q`. The offered-tool check accepts it because `lookup` is offered. This turns an existing delimiter failure into a wrong invocation. [Retry implementation](/home/zurih/c2codex/0103-server-a-call-after-prose-that-quotes-its-opener-is-the-call.patch:45)

I would add that regression for both unlimited and single-call parsing, and prevent recovery from selecting candidates inside JSON string arguments. Making delimiter discovery aware of JSON strings and escapes is the fuller fix; conservatively refusing recovery inside an apparent JSON payload is another option.

The four-candidate bound does keep recovery linear. It also deliberately leaves false negatives: after a prose opener, four literal openers inside the real call’s argument can push the real opener beyond the retry budget. The added quoting test covers GLM argument markup with one opener, rather than JSON arguments or that bound.

**0104 is a reasonable canary once the cache exists.** It addresses a separate problem: restored cache capacity lets an unchanged loop prefix hit; warming creates checkpoints for the rendering after reasoning is cleared by the next user message. Reserve alone should not be expected to deliver the 8,192-token next-user hit.

For TP2, differing local queue observations are harmless **if rank 0’s shared cancellation decision governs both ranks’ next compute step and interrupted-state retention**. A rank taking a local early exit instead would risk mismatched collectives, a hang, or unequal retained prefixes. The patch documents the shared-gate invariant and changes its predicate; the gate implementation is outside the supplied material, so I cannot certify it from these hunks. RoCE mainly makes the cost and timing of cancellation more important. [Cancellation change](/home/zurih/c2codex/0104-glm-warm-turns-the-next-user-message-resumes-past-the-agent-loop.patch:147)

The fill-queue addition closes a real scheduling hole: admitted foreground prefills now block warming even before decoding begins. But preemption remains at chunk gates. **150 ms is a start delay, not a foreground latency bound.** If a warm chunk were 4,096 tokens, your approximately 1,750 tok/s prefill rate suggests roughly 2.3 seconds of work before the next gate, plus interruption/cleanup costs. Measure the actual chunk duration on the Sparks.

Consequently:

- Back-to-back requests arriving during the quiet wait should suppress warming, but rendering/tokenization and cancellation overhead can still affect TTFT or request throughput while decode tok/s stays unchanged.
- A second agent arriving during a warm chunk can wait for that chunk. Sustained foreground traffic should largely suppress warming, so the quiet-engine cache target is not a busy-engine guarantee.
- Warm checkpoints can compete with another conversation’s reusable checkpoints. No cache isolation is added here. Expected harm is eviction and later re-prefill; active stream KV has its separately planned pool budget.
- With zero cache there is no retained benefit. Restore and verify retention before enabling this behavior.

I would retain your proposed gates, with these additions and refinements:

1. **Test reserve alone first.** Require approximately 4,096 pages, approximately 3 GiB usable cache, eight lanes, and no minimum-state warning. The unchanged-prefix step inside the agent loop must get a grid-aligned hit—typically at least 4,096 tokens for a genuinely shared 6K prefix. Continued zero hits veto the change.

2. **For quiet warming, wait for completion rather than a fixed delay.** Then require cached ≥8,192 of approximately 9,335 and measure TTFT. Moving from 4,096 to 8,192 saves approximately **2.30–2.36 seconds** of foreground prefill at your reported rate. A “done” log alone does not prove retention.

3. **Make an arrival during an active warm chunk the decisive 0104 test.** Cover arrival during the quiet wait and after warming demonstrably starts, including the proposed 0.3-second case. Require foreground latency consistent with at most the measured remaining chunk plus cleanup, both ranks completing, and a subsequent request successfully reusing retained state. The added host test establishes cancellation before any warm chunk; it does not establish this TP2 path.

4. **Compare cold, completed-warm, and interrupted-warm execution of the same final prompt.** Check greedy outputs as well as cached counts. Unchanged kernels do not by themselves prove correct checkpoint selection and restoration.

5. **Add cache pressure from two conversations and measure memory after warming.** Alternate chats until eviction occurs, then check hit rates, completion, and available-memory low-water on both hosts. Exercise long-context page pressure separately from the eight-short-stream gate; verify the pool’s actual exhaustion behavior.

Keep the decode bands **38–39 / 56–57 tok/s**, four-stream **77–83**, hashes **64c908210a9e / 315341215112**, cold prefill **1,737–1,780 tok/s**, and mixed chat **≥11 tok/s**. Ensure cold-prefill cases report `cached=0`. For back-to-back and competing-agent traffic, add TTFT and maximum token gap: average decode throughput can conceal the principal warm-pass regression.