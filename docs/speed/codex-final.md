1. **A already ingests only kept rows.** `noteVerify` records the whole verify window; [`keptRuns`](/tmp/tfwork/a-src/dflash.zig:305) selects exactly its first `q.n` rows, and [`ingest`](/tmp/tfwork/a-src/dflash.zig:278) copies only those. Rejected tail rows are excluded. With causal verification, kept states also cannot depend on later rejected tokens. “Ingest only accepted” therefore changes nothing.

   The supplied B patch likewise [commits `taps[:keep]`](/tmp/tfwork/b-src/0030-glm-multi-stream-engine.patch:1533), then applies [`fc → norm → per-layer KV`](/tmp/tfwork/b-src/0027-glm-multi-dflash2.patch:342). That is equivalent accepted-prefix conditioning in structure, rather than token replay through the drafter’s full transformer. Numerical equivalence remains unproven: B’s supplied Zig forward ends midway through initialization, and A’s attention kernels are absent.

2. **93 has no demonstrated route through further policy tuning.** The plateau supports stopping those sweeps, not declaring every possible drafts-side fix impossible. S2 remains a concrete mechanism: the metadata failure sets `from=q.pos`, excluding prior history, and increments `lost`. A tap-lifetime fix could shift the frontier without rewriting conditioning. S3 remains untested; identical output hashes do not establish hidden-state equality.

   Also, at **104 ms**, 93 total requires **2.418 tpr**, or **+0.322** over A. Even B’s measured 2.319 would yield approximately 89.2 at that latency.

3. **Tomorrow’s last small-change experiment: eliminate tap-lifetime restarts.** Compare unchanged A against A that preserves each stream’s kept taps until ingestion, or ingests them exactly once before buffer reuse. Freeze policy and precision; record `lost` deltas and actual context spans; run the paired protocol twice. If baseline already has zero losses and intact spans, S2 is excluded without that patch.

   Pre-register:

   - **Go:** unchanged hashes and **≥93 measured total tok/s in both repetitions**, with ordinary timing including ingestion overhead.
   - **Stop this small-change campaign:** S2 is absent, or eliminating it fails that gate. A larger forward port then needs separate evidence and scope.

   Your **2.25 tpr / 104 ms** threshold is reasonable for funding another bounded step, but represents only **86.5 total**. For 93, require approximately **2.42 tpr at 104 ms**, or **2.25 at ≤96.8 ms**.