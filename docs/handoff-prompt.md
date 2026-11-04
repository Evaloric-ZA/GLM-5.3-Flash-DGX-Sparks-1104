GLM-5.3-Flash EXL3 on 2x DGX Spark, TensorFold native engine. Recipe:
/home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig (git, no remote).
Port of the 2x RTX PRO 6000 recipe to GB10. Donor for the two-host bits:
/home/zurih/TF/glm53-spark-zig-ref (MiaAI-Lab/...-2x-DGX-Sparks-TensorFold @ zig-engine).

HARDWARE  spark3 = rank 0 + API, spark4 = rank 1 (WORKER=zurih@10.0.34.4).
Linked on both PCIe links of one QSFP port: 10.0.34.3/.103 <-> .4/.104,
RDMA rocep1s0f1 + roceP2p1s0f1, RoCE v2 GID index 3 on both. Discovery is automatic.

RUNNING   ./start.sh | ./stop.sh | DRY_RUN=1 ./start.sh | ./start.sh restart
Port 8888, model id GLM-5.3-Flash, no API key. Engine tensorfold-native 1.1.9,
TF f350cd3e + 95 patches, sm_121 aarch64, image tensorfold-glm53-spark:<patches hash>.
Config in scripts/local.sh: q8 dense, bf16 head, fp8 KV on pages, PARALLEL=4,
CONTEXT=auto -> 1,048,576, MEMORY_RESERVE_GIB=8, HOST_CACHE_GIB=0, ABSORB_MMA=1,
W8A8=0, MEMGUARD=1 (floor 2.5 GiB). model 83.58 GiB a rank, 2,098,176-token KV pool,
4.4 GiB prompt cache, loads in ~120 s.
MEMORY: the KV pool grows only in whole-window steps of 2,048 pages (~6.4 GiB), so reserve
10->8 gave the 2 GiB to the prompt cache (2.5 -> 4.4) and left the pool alone; reserve 6 buys a
third window (3.19M tokens) but zeroes the cache. Pool and cache are exclusive at the next step.
Rank 0 binds: spark3 sits ~0.4 GiB over its reserve, spark4 has ~5 GiB the pair cannot use.
Running from spark4 instead would balance it (~+2.4 GiB) and move the API there. PROMPT_CACHE_GIB
is empty, so the cache is the residual after pages -- pin it for a predictable size.

MEASURED (3 cold trials a point, fresh content so no prefix-cache hit, +-0.2%)
  prefill @23.5K / @95K tok/s:  q8 1,735 / 1,759   q4 1,863 / 1,885   v1.0.7 1,877 / 1,847
  decode 1/2/4 streams: 35.3 / 53.4 / 72.5 tok/s.  draft acceptance 0.525
  NLL (4 passages) / 10-greedy-answer sha: q8 0.7523 / 11ceddde19d4, q4 0.8005 / 28d50949f647
  dense width settled: bf16 1,288 / 1,262 tok/s at 87.00 GiB and half the KV pool, and
  indistinguishable from q8 in quality (paired, 1,051 tokens over 12 domains: mean per-token
  delta -0.00394, 95% CI -0.0153..+0.0074, sign test 49.2/50.2). bf16 is dominated; q8 stands.

SETTLED, do not re-propose
- There is no prefill gap vs the v1.0.7 Spark build at equal precision: ours is -0.7% at
  23.5K and +2.1% at 95K. The ~7% was q8 vs its q4 default. Nothing in its prompt path
  (0101 MSA, 0102 own-front, 0104 microbatch, 0106 bf16 exchange, 0033 overlap2) needs porting.
- W8A8=0, stashed by the user. +10.5% prefill but rounds activations: per-passage NLL +-0.03
  and it answered "spell necessary backwards" wrong where q8 is right. Patch 0064 stays inert.
- DENSE=q4 rejected on quality (NLL 0.8005), not merely unused.
- ABSORB_MMA=1 is free on GB10: +8.0% prefill, byte-identical answers, NLL 0.7523 vs 0.7535.
- Patches 0094 (prefill_rows 4096) + 0095 (dflash MAX_CTX 4096, WIN stays 2048): +7% prefill,
  bit-exact. MAX_CTX is a tap-buffer capacity, not a window -- promptRun already clips a chunk
  to the last WIN tokens. The guard cannot just be relaxed: a forward pass writes taps into
  taps_prefill before promptRun runs (cuda_engine.zig:454).
- The link is ~3% of TTFT, so it is not a prefill lever. prefill_rows is compile-time because
  the drafter is dimensioned to it.

METHOD, learned the hard way
- Measure cold prefill with FRESH content per trial and check usage.prompt_tokens_details
  .cached_tokens == 0. A long prompt sharing a prefix with an earlier one resumes it and the
  rate is not prefill. This invented a false crossover once.
- Judge numerics AND dense width per passage and paired, never on a narrow aggregate: W8A8's
  aggregate NLL matched baseline exactly while every passage had moved, and a 4-passage run had
  bf16 ahead by 0.0048 which vanished at 12 passages paired.
- Single-shot numbers are not comparable to 3-trial medians; a v1.0.7 point spanned 12%.
- Never wait on a process pattern that your own launcher's command line contains (pgrep/pkill
  self-match); wait on a log sentinel.
- Pre-verify a patch hunk with git apply --check in a throwaway repo before paying a build.

HARNESS  docs/quality.py (NLL + 10 greedy answers), docs/pf2.py (cold prefill),
docs/boundary.py (chunk boundaries + draft acceptance), docs/prefill/ (results, Codex review).
Codex: setsid nohup ~/.local/bin/codex exec -s read-only -C <dir> --skip-git-repo-check
--ephemeral -c model_reasoning_effort=xhigh - < brief.md (it cannot write under read-only;
read its stdout log).

OPEN  Nothing outstanding. If prefill is revisited, the only untested lever left is
TF_GLM_OVERLAP_PIECES (default 3, range 1-8), worth maybe a couple of percent.
