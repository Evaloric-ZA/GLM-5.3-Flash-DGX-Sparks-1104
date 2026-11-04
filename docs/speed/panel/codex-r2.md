I agree with the resolutions to D1–D4 and D6, and with profiling prefill before choosing further work. For D5, the profile may identify where time goes without proving the underlying hardware bottleneck; use it to select a concrete target, not as conclusive roofline evidence.

The plan is sound with these changes to its gates and recovery procedure:

FINAL: AGREE WITH AMENDMENTS

1. **Revise G0.** A verify-cost shape within 15% of sm_120 does not rule out an env-only policy win: acceptance and the existing confidence cut still matter. If the fit is adequate, run at least the baseline, one alternative cut and `tpm`. If the trace cannot support a fit, skip `tpm` calibration and the constants patch, but retain the cut comparison if time allows.

2. **Check trace coverage before relying on it.** Confirm the records actually expose the proposed timing boundaries, realized rows, landing positions and lone/shared classification. If a field is absent or ambiguous, record it as unknown; do not infer scheduler prices from unlike measurements.

3. **Make performance gates paired.** Interleave baseline and candidate runs using the same prompt sets, with fresh content across trial pairs. Record per-trial results and repeat a borderline 1–2% result if time permits; otherwise treat it as inconclusive. For a scheduler patch or overlap switch, also check mixed prefill/decode performance.

4. **Strengthen the output gate.** Compare greedy token IDs against both the existing baseline and speculation disabled, including the stated boundary and long-context cases. Check the documented sampled-request contract separately before claiming sampled behavior is preserved.

5. **Make the rebuild conditional and attributable.** Re-seed only constants supported by the measurements. If both policy seeds and scheduler tables change in the one build, test each against its corresponding runtime baseline where possible; retain the build only if the combined configuration passes G2. Do not use the rebuild merely because it is available.

6. **Test rollback early and reserve the clock.** Save the image *and* complete launch configuration, then test their restoration before experiments. Set a hard stop at least 30 minutes before the deadline; on any inconclusive final gate, restore that known-good configuration and complete Phase 4.
