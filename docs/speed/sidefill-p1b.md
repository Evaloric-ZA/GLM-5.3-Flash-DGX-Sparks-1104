# P1b - saturation check, CPU-only, from the existing trace (2026-11-03; user-directed before P3)

Question: does one stream already saturate the pair during `fillRows`? If yes, the +6.2% ceiling is unreachable
and the candidate dies without a harness.

## The piece structure P1 missed (found by counting calls)

`output_projection` does NOT run as one 4,096-row GEMM: the trace shows **3 piece-calls per layer per site**
(pieces 0/1/2, ~1,365 rows each - the exchange-overlap split), p50 1.06 ms mixer / 1.89 ms ffn, ~2.96 max.
11,320 calls per site over 23 chunks. Small-kernel regime, not one big GEMM.

## Achieved bandwidth per call (bytes from the config: hidden 4096, 64 heads -> K=8192 mixer out-proj;
DENSE=q8 ~1 B/weight; acts fp16; fp32 accumulate), against the 273 GB/s box:

| call | bytes (W+act+out) | ms p50 | achieved | of 273 GB/s |
| --- | ---: | ---: | ---: | ---: |
| mixer out-proj piece | ~80 MB | 1.06 | ~75 GB/s | **28%** |
| mixer, slow pieces | ~80 MB | 2.0 | ~40 GB/s | 15% |
| ffn combine-class piece | ~213 MB | 1.89 | ~113 GB/s | **41%** |

Caveats stated: byte sizes assume q8 dense weights and the K=8192 reading of the head layout; the ffn stage's
exact operator mix is from structure, not source-verified. The conclusion survives the error bars.

## Verdict: far from roofline - the candidate lives, and the lever is sharper than "two streams"

At 15-41% of box bandwidth, with nothing else queued on main (serial layer loop) and the side lane idle by
construction, the out-proj windows underfeed the device. Two attack shapes, both launch-side only and both
row-partition/bitwise-safe:

- S1 (as proposed): run two piece-calls' worth of rows concurrently on main+side per layer.
- S2 (new, from the piece structure): **coarsen** - one whole-chunk call instead of three, on main. The pieces
  exist to overlap the exchange with the next piece's compute; whether that overlap beats feeding the GEMM
  properly is an empirical question of exactly the same class.

P3 therefore grows one arm: (a) status-quo replay, (b) S1 two-stream, (c) S2 merged single call,
(d) S2+two-stream. Same kernels, same operands, per-row dots unchanged; CUDA-event makespans, interleaved x3,
against the pre-registered gates. The 2.96 ms max calls (15% BW) are the interesting tail: whatever makes them
slow is the contention prior personified.

## Standing rules recorded with the user's corrections

- **Contract precedence: 8/8 token_sha is a precondition, read before any throughput number - not a co-equal
  gate.** Precedent DENSE=q6 (aggregate NLL better, hashes moved). A broken contract is not tradeable against
  a percentage; the empirical contract stays the arbiter of every "bitwise by construction" claim.
- **Priority: the decay/incidence series outranks side-fill for pair time.** +6.2% here is ~3.4 s of the 54.1 s
  TTFT at 95K - real, modest; the p95-cliff question potentially dwarfs it. Series first, always.
