# Handoff: make spark3+4 beat spark1+2 at 4-stream decode (written 2026-11-03 ~22:00 local)

## The goal, stated by the user

Pair A (spark3+4, this recipe) must decode faster than pair B (spark1+2) at 4 streams, without giving up what A
already wins (prefill, chat beside a long prefill, code at 1 stream). Exact patches (identical greedy token ids)
ship as defaults; a patch that changes bits is opt-in and needs the paired-NLL instrument (docs/nllpaired.py) and
the contract (docs/speed/RESULTS.md's rules). The user's standing orders in RESULTS.md apply: measured numbers or
blockers only.

## The numbers to beat (all measured 2026-11-03 from the rtx box, greedy, thinking off, same prompts)

| test | A: spark3+4 (this recipe, 98 patches) | B: spark1+2 |
| --- | --- | --- |
| 1 stream prose, 10 topics | 39.8 +- 1.3 tok/s, 2.15 tok/round, 54 ms/round, 60% drafts land | 40.5 +- 2.1, 2.27 tok/round, 56 ms, 64% |
| 1 stream code | 54-57 tok/s, 3.4-3.8 tok/round | 50.6, 3.2 tok/round |
| 2 streams total | 60.3 | 63.4 |
| 4 streams total | 78-84 tok/s; per stream ~20 at 2.07-2.11 tok/round, 103-106 ms/round, 1.7 drafts/stream/round, 61-64% land | 93-95; per stream 23-24 at 2.34-2.47 tok/round, 102 ms/round, 2.0 drafts/stream/round, 68-71% land |
| cold prefill 16K / 64K | 1,715 / 1,780 tok/s | 1,624 / 1,632 |
| chat beside a 50K prefill | 14.5 tok/s (2.7 s worst pause) | frozen (0.8 tok/s) |

B's round at 4 streams is NOT shorter than A's. B gets ~0.3 more landed tokens a round from ~0.3 more drafts a
stream and ~7 points more landing. That is the whole gap.

What B is: /home/mia/newMod/GLM-5.3-Flash-EXL3-TensorFold on spark1, branch `zig/python-parity-defaults`
(TensorFold v1.0.7 Zig port in zig/patches/0001-0005 + 0006-0030), image tensorfold-glm53:zig-v1.0.7, started
with `PATH=$HOME/.local/bin:$PATH DENSE=fp8 ENGINE=zig ./start.sh` (user mia). 8-stream arena, fp8 dense, bf16
drafter, chain rule fnc7:0.3 compiled in (zig/src/families/glm_cuda/cuda_draft_policy.zig in patch 0004: chain of
up to 7, stop when the product of per-draft confidences < 0.3, confidence = softmax((score+noise)/beta=2), score =
value + 0.6*edge), copy drafts up to 15, no draft pricing, prompts fill one at a time. Its per-reply stats
(`tensorfold.rounds/drafted/accepted`) may count differently from A's (`speculative` block): compare tok/s and
tok/round, not acceptance percentages across engines.

## Where A's 4-stream round goes (TF_GLM_PROF=1, patch 0099 adds the shared labels)

| forward | wall ms | routed | kda | mla | router | gathers | head | dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| verify 2 rows (1 stream) | 43.8 | 17.2 | 13.0 | 5.0 | 3.3 | 2.5 | 1.5 | 1.2 |
| shared 10 rows (4 streams, ~2.5 rows each) | 93.1 | 58.5 | 16.8 | 5.8 | 3.7 | 4.6 | 1.5 | 1.2 |
| shared 14 rows | 110.8 | 74.6 | 17.4 | 6.2 | 4.0 | 5.0 | 1.5 | 1.2 |

Routed: 4.2-5.8 ms a row = 8 fresh experts a row a layer at 4 bpw, 1.4 GiB a rank, i.e. the bytes at 273 GB/s.
KDA mixers: flat in rows; projections (50 MB a rank a layer q8) at ~60% of bandwidth, the chain the rest
(docs/speed/kda-p1.md has the dispatch map and candidates C1-C4). Drafter pass (TF_GLM_DFLASH2_PROF=1): ingest 0.5,
block 4.5 (7.2 at 4 streams), select 0.05 ms; the trace's d= (8.4 ms) exceeds these by ~3 ms of rank exchange.
A 4-stream round = forward 93 + drafter 8 + ~3 = 104 ms. CUPTI/nsys kernel activity is EMPTY on this driver
(RESULTS.md 2026-11-03 11:00Z); the engine's own brackets are the only attribution tool.

How drafts are chosen on A: rank 0's selector (zig/src/families/glm/dflash_select.zig) picks each draft among 16
candidates by logit + codebook edge; chances chained and cut at TF_GLM_DRAFT_CONFIDENCE (0.7) in
cuda_lanes.zig; the lane core's allocator (zig/src/core/lanes/allocate.zig allocateDrafts) then adds rows across
streams while expected tokens / (shared_costs(rows) + overhead) rises, with the GB10 cost tables from patch 0096
(shared 47.38 + 5.05 ms a row) and the overhead learner in zig/src/core/lanes/depth.zig. Trace one window a
line with TF_GLM_DFLASH2_TRACE=1 (docker logs: `dft r= a= k= g= v= d= s= w=`), force widths with
TF_GLM_DFLASH2_FORCE=1,2,...; the forced-width landing table (lone rows=2: 0.74 of 1 draft lands; rows=4: 1.68 of
3; shared rows=4: 1.22 of 3) is in this session's record below.

## Tried today, all exact (token hashes prose 64c908210a9e / code 315341215112 unchanged), none a gain

| lever | 1 stream prose / code | 4 streams |
| --- | --- | --- |
| baseline | 39.0 / 57.3 | 78-80 |
| DRAFTER_BITS=0 (bf16 drafter) | 36.9 / 53.3 | 76.7 |
| TF_GLM_DRAFT_CONFIDENCE=0.5 / 0.3 | 37.5 / 55.3, 34.6 / 52.0 | 78, 76-77 (deeper rows, longer rounds) |
| TF_GLM_DRAFT_RULE=tpm | 38.7 / 55.9 | 80 |
| 0098 TF_GLM_DFLASH2_EDGE=0.6 | 39.0 / 56.8 | 81-82 |
| 0100 TF_GLM_MM_STAGES=5/6/8 | 38.1-38.5 / 56.6-56.8 | 79-84 |
| DENSE=bf16 | 30.5 / 46.5 | 70 |
| stock TensorFold v1.2.1 (no patches; refuses fp8 KV on pages) | 35.8 / 53.3 | 67-71; prefill 1,058-1,086 |

0098 and 0100 are in docs/speed/candidates/ (apply with the rest if wanted; they change no bits).

## Directions not yet tried, in the order I would take them

1. **PARALLEL=8 on A** (config only; CONTEXT=auto refits the window). B runs an 8-stream arena; A's 4-stream
   total may simply be a 4-stream number. Measure total and per-stream at 4 and 8 concurrent requests with
   docs/dec.py or the harness below. Window shrinks; the user's serving shape rule (one request to ~1M, 4 streams)
   must still hold, so this may be a measurement, not a shipping option.
2. **Why A offers 1.7 drafts a stream a round at 4 streams and B 2.0.** The allocator's stop is the rate test
   with shared_costs (0096) and the learned overhead; print or trace allocateDrafts' decisions (it is host code,
   zig/src/core/lanes/allocate.zig; add a log line, rebuild). If the chain cut (0.7) is what zeroes the third
   draft, the 0.5 run says the extra rows land too rarely to pay; so look at calibration: the selector's chance
   (softmax at dr.temp 1.0 over 16 candidates) against measured landing per depth from TF_GLM_DFLASH2_TRACE (a=
   vs r=). B's confidence is softmax((score+noise)/2): flatter. A calibrated chance would let the cut and the
   allocator price depth correctly. Exact: drafts only propose.
3. **KDA chain** (docs/speed/kda-p1.md C1 merge launches, C4 two rows a block; C2 chunked/WY changes bits): ~6 ms
   a round at 1 stream, 16-17 ms a shared forward; a 10-20% KDA gain is 1-3% of a round. Small but exact for C1/C4.
4. **Port B's drafting whole** (cuda_draft_policy.zig and its selector): risky, and B's prefill freeze must not
   come with it; only if 2 shows A's selector or chances are the difference.
5. **TF_GLM_CHUNK_SOLO** (upstream v1.2.1, #1823): not a decode lever, but the one thing from v1.2.1 worth
   re-porting (large prompt chunk only while nothing decodes: removes A's 2.7 s pause with no prefill loss).

## How to work on this pair without hurting it

- Build images on spark1 with its pair stopped (`cd /home/mia/newMod/GLM-5.3-Flash-EXL3-TensorFold && ./stop.sh`),
  never on spark3/spark4: a build on the serving host trips the memguard and stops the pair (happened 16:59Z).
  Build folder /home/mia/tf-build/recipe (docker/ + patches/engine/ copied from this repo); the exact command is in
  this session's record and in /home/mia/tf-build/build-98.log's first lines; label = `scripts/prepare.sh --label`
  computed on spark3 with the intended patches/engine set; tag tensorfold-glm53-spark:<label>, label
  tf.patches=<label>. Ship: `ssh spark1 docker save tensorfold-glm53-spark:<label> | ssh spark3 docker load`, same
  to spark4. Then `./start.sh restart` here finds it ("Image is ready").
- Keep candidate patches OUT of patches/engine on spark3 until their image is on both Sparks: start.sh computes
  the label from the folder and builds here if no image matches (happened twice today).
- The local toolchain's `zig build test-glm-tp3-host test-glm-fp8-kv-host test-glm-exl3-host` (toolchain at
  /home/zurih/MiaAI_Lab/TensorFold/toolchain/zig-0.17.0/zig on the rtx box, tree at the session's scratchpad
  tf/ clone, branch recipe-97 + the new patches) catches Zig errors in a minute; the mocks take []const u8 kernel
  names where the real module takes [:0]const u8, so that one slips through to the image build.
- Exactness gate: the token hashes above on the same prompts (acc.py prints sha); bench2.py (full suite: 1 stream,
  prefill 16K/64K, 2/4 streams, mixed 1+long and 3+long, two prefills at once) and acc.py / acc10.py (tok/round,
  ms/round, acceptance; 10 topics) are committed in docs/speed/harness/ (also in the rtx box scratchpad for this session:
  <the session's scratchpad>/
  (bench2.py, acc.py, acc10.py, mix.py; results spark34.summary.json, spark12zig.summary.json,
  spark34-v121stock.summary.json). Copy them somewhere permanent; /tmp does not survive a reboot.
- Profiling: `bash docs/profleg.sh <label> [KEY=VALUE...]` restarts with TF_GLM_PROF=1 and aggregates with
  docs/prof.py; shared windows report as "shared N rows" since 0099. TF_GLM_DFLASH2_PROF=1 for the drafter phases.
- The decay series (docs/speed/decay-*.log, /tmp/decay-sched.sh) was invalidated by today's restarts.
- spark3's repo has no git remote. Commit by path; never `git add -A` (decay logs and raw profiles are dirty).

## This session's record

docs/speed/RESULTS.md "Decode attribution campaign on the pair, and the spark1+2 comparison (2026-11-03)";
CHANGELOG.md 2026-11-03 (0099); docs/speed/candidates/ (0098, 0100); docs/speed/kda-p1.md; the rtx box's memory
notes (spark-pairs-comparison, no-builds-on-serving-sparks, prefill-share).
