# Brief: GLM-5.3-Flash on 2x DGX Spark (TensorFold Zig engine, TP2) - a dead prompt cache, and a two-patch port

Second opinion before two changes to the serving pair. Answer from the numbers; `kv_pool_plan.txt` in this directory is
the planner (core/kv_pool.zig plan/held/windows/split) and `cuda_kv_pool_plan.txt` its native caller; the two patches
to port are `0103-*.patch` and `0104-*.patch`. Please do not read anything else. Be concrete: failure modes, the
killing measurement, and a recommendation on each.

## 1. The prompt cache is zero on the serving pair

Rank 0's startup line today: "model 83.29 GiB; 8 streams at once, the engine's own and 7 more taking 5580 pages of 512
rows (16.98 GiB) as they need them (enough for one at a 1048576-token window), 417 MiB fixed each, of 19.8 GiB left
after a 8.0 GiB reserve; ... no prompt cache", preceded by "warning: GLM CUDA prompt cache budget 0 bytes cannot keep
one state: the minimum planned state needs 0.254 GiB; lower --context or --parallel, or TENSORFOLD_MEMORY_RESERVE_GIB
(--prompt-cache-gib only caps the budget the weights and streams leave)". A replay of a small agent loop (system +
user 6.0K tokens -> tool call -> tool result 9.3K -> next user message) shows cached=0 on EVERY turn, including the
loop's own next step whose prompt extends the previous one; each turn re-prefills 9.3K tokens, ~6 s for a 10-token
answer. Yesterday's record at the same reserve (8 GiB) had "4,098 KV pages (12.5 GiB) + a 4.5 GiB prompt cache" at a
room of 18.2 GiB; today the room is 19.8 GiB (the engine now counts 0.53 GiB of free per-CPU page lists, and the box
has less else running).

My reading of the planner: cache = min(want, left - held), held = max(fit pages, (windows-1) x pages(window)) x page,
windows = min(lanes, 1 + (room - own) / (own + fixed)). With own = 2048 pages x 3.12 MiB = 6.23 GiB and fixed = 0.41
GiB: at room 18.2 windows = 2 (held one window, cache ~4 GiB); at room 19.8 windows = 3 (held two windows, cache 0).
The threshold is room >= 6.23 + 2 x 6.64 = 19.5 GiB. A model of the planner with these constants predicts today's
5569 pages / 0 cache (observed 5580 / 0). Predictions:

| reserve GiB | parallel | window | pages | pool GiB | cache GiB |
|---|---|---|---|---|---|
| 8 (now) | 8 | 1,048,576 | 5569 | 16.95 | 0 |
| 8 | 4 | 1,048,576 | 6105 | 18.58 | 0 |
| 9 | 8 | 1,048,576 | 4096 | 12.46 | 3.5 |
| 9.5 | 8 | 1,048,576 | 4096 | 12.46 | 3.0 |
| 10 | 8 | 1,048,576 | 4096 | 12.46 | 2.5 |
| 8 | 8 | 524,288 | 5120 | 15.58 | 1.4 |

Proposed: TENSORFOLD_MEMORY_RESERVE_GIB 8 -> 9.5 (recipe setting, no engine change): two whole 1M windows of pages
(2.1M tokens, the same pool yesterday's shipped state had), ~3 GiB of prompt cache, 1.1 GiB of margin below the
threshold, and 1.5 GiB more free memory under a memguard that kills the pair at 2.5 GiB available (the low-water after
a 95K-token prefill at reserve 8 was 8.3 GiB).

Questions: (a) failure modes of raising the reserve (fewer pages for 8 concurrent long contexts: 2 x 1M at once, or
8 x 262K; prompt-cache states and KV pages competing; anything in the planner I misread); (b) is lowering --context
or --parallel ever better here; (c) is there a way to make the cache robust to the room crossing a window threshold
(--prompt-cache-gib only caps), or is pinning the reserve the honest fix; (d) the killing measurement: I plan the
replay's cached counts (loop step must cache its prefix), the startup line (pages, cache GiB, no warning), the decode
harness (1-stream 38-39 / 56-57 tok/s, 4-stream 77-83, greedy hashes 64c908210a9e/315341215112 unchanged), cold
prefill 16K/64K (1737-1780 tok/s), a chat beside a 50K prefill (>= 11 tok/s), and 8 concurrent streams still served.

## 2. Porting two patches from the sister recipe (2x RTX PRO 6000, same engine pin, same 93 base patches)

The user wrote and verified these there with a council (the lab's agent, Codex, GLM); both apply cleanly on our 102-patch base
(individually and together); the cold host suites give the base tree's failure set and the patches' own steps pass
(test-tool-parse-host, test-tool-visibility-http-host, test-json-answer-host, check-native-cuda-host: 37/37).

- 0103 (their 0096): server/tool_parse.zig - a tool call after prose that quotes a tool-call opener parses as the
  call (retry from the last opener inside a rejected envelope, up to four back). Server-side only.
- 0104 (their 0097): GLM warm turns - after a reply, the server prefills the next turn's rendering as a background
  prompt-only pass (warm.zig existed for Nemotron; GLM now declares warm_quiet 150 ms); the chunk gate yields to real
  work; a tool-call reply warms the tool turn; the warm message carries reasoning_content; lane_host's yield checks
  also see requests admitted into the fill queue. Verified there: next user message after a loop cached 8192 of 9335
  (was 4096). Touches core/lane_host.zig (every family), families/glm/cuda_lanes.zig and cuda_native.zig, native/cuda.zig,
  server/warm.zig and openai.zig.

Our pair differs from theirs: TP2 across two Sparks over RoCE (not two GPUs in one box), PARALLEL=8, a decode-throughput
benchmark that sends back-to-back single requests (a warm pass after each reply would start just as the next request
arrives and must step aside), and - until section 1 is fixed - no prompt cache at all, so 0104 would keep nothing.

Questions: (e) failure modes of 0104 on a two-rank pair: can a warm pass on rank 0 and a foreground request disagree
across ranks (the gate's shared cancel is "by rank 0's word")? Can the pass's kept states evict a live chat's states
from a 3 GiB cache? Does the pass change decode throughput for back-to-back requests (the harness) or for a second
agent's traffic? (f) anything in 0103's retry that could misparse a legitimate reply (quoted opener inside a JSON
string argument)? (g) the gate: hashes and the decode band unchanged (the patches touch no numerics), the replay's
next-user-message row cached >= 8192 of ~9.3K on a quiet engine, rank 0's log showing "next turn: background prefill of
N tokens done" after an answer and "stepped aside" when a request follows within 0.3 s, prefill and mixed numbers
unchanged. What would you add or drop?
