**Pick C-C, timeboxed through tomorrow’s diagnosis; then C-D unless it identifies a small, testable port. No A-only drafting move earns an 88+ bet.**

**1. Calibration is cleanup, with two arithmetic corrections.**

- If **5.05 ms = one additional shared verify row**, break-even joint survival is **\(4×0.0202×5.05=0.408\)**, not 0.102. Use the measured shared marginal cost at the actual row count.
- Granting **+0.135 tokens/stream and +2.2 ms for the whole round** gives **84.0–84.2 total tok/s**, not 86. If 2.2 ms was calculated per stream, charge **8.8 ms**, giving approximately **79.3**.
- At **104 ms**, 88 requires **2.288 tokens/stream/round**; 93 requires **2.418**. Baseline 80.6 implies **2.096**: deficits **+0.192 / +0.322**, before additional cost.

A calibrated allocator may improve allocation. The aggregate ×0.65 factor supplies neither new proposal quality nor a calibrated score curve. Apply joint corrections to joint survival; do not multiply joint-depth factors together. These experiments reject a compelling widening path; they do **not** prove global unreachability.

**2. Ranking for this objective**

| Rank | Candidate | Decision |
|---|---|---|
| **1** | **C-C** | Buy diagnosis first; authorize further effort from its result. |
| **2** | **C-D** | Stop under the current design/budget if no small transplant emerges. |
| **3** | **C-A** | From 80.6, saving 1–2 ms gives **81.4–82.2**. |
| **4** | **C-B** | Saving 2–4 ms gives **82.2–83.8**, with greater implementation risk. |

Those projections hold yield fixed. Neither KDA change closes this gap.

**3. Minimal useful B instrument: two hooks, rank 0, buffered.**

**Policy hook**, copying existing host readback:

- All available depths’ **top-16 candidate IDs/raw values + full selector projections**, before cutting; anchor ID/position; effective selector/cut/beta/cost settings.
- Actual picks, conditional chances, cumulative chances, requested depth, cut depth/reason, final allocated depth.
- Preserve enough inputs to replay **both selectors offline**, including changed predecessor tokens. Selected-token scores alone are insufficient.

**Lane hook**, after acceptance/emission:

- `round_id, request_id, absolute_prefix_position/hash, active_streams`.
- Proposal source; actual offered IDs/count **after trimming**; accepted prefix **excluding anchor**; actual newly emitted IDs/count; committed rows; EOS/cap flags.
- Drafter context end before/after, ring span, and committed tap source positions/layers.
- Whole-round wall time recorded **once**, shared row count, verification/drafter time.

Include **zero-draft windows**. Ordinary untruncated windows should reconcile emitted tokens with **\(1+\) accepted drafts**, consistent with [prefix accounting](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).

**One ordinary B trace cannot settle all three causes.**

- Counts settle bookkeeping and confirm the real yield advantage. Counter semantics cannot erase an independently measured throughput gap.
- Replaying A’s selector on B’s inputs isolates selector differences **on B’s inputs**.
- Attributing the remaining difference specifically to conditioning requires **matched-prefix A/B payloads or a controlled conditioning swap**. Otherwise label it “upstream unresolved.”

In the [supplied B policy](/tmp/tfwork/b-src/b_draft_policy.zig:95), greedy noise is zero; beta changes confidence, not picks.

**4. Pre-register C-C failures and stopping rules.**

- **Trace perturbation:** buffer existing readbacks; no new per-window synchronization. Check hashes and trace-on/off round behavior.
- **Population mismatch:** identical prompts/hashes do not align round anchors. Compare absolute prefixes; label censored depths.
- **False attribution:** candidate/projection differences can reflect numerics, candidate generation, or conditioning.
- **Unpaid gain:** require projected **≥88 after charging shared-row cost**; matching B’s acceptance percentage is insufficient.
- **Tomorrow’s cutoff:** require reconciled counts and a localized, small port with credible headroom. Otherwise choose **C-D**, reporting “no demonstrated path under the current drafting design,” rather than “proved unreachable.”