I provisionally rank **H2 > H3’s conditioning variant > H1 as an explanation of the whole gap**. The evidence favors better useful proposals over simple widening, but does not identify selector quality versus conditioning. H3’s commit/accounting variant is separately testable.

**A profitable calibration change remains possible. A is not proven to be at its selector’s frontier.**

1. **The free/forced contrast establishes selection bias, not depth-2 quality.**

   First normalize A’s counters. Your histogram contains 718 offered drafts across 400 windows:

   - Per drafted window: \(718/371=1.935\).
   - Per window overall: \(718/400=1.795\).
   - If 1.19 accepted is also per drafted window, overall yield is \(1+1.19(371/400)=2.104\), matching measured tok/round.

   Thus 1.94 versus B’s “2.0” is not yet a comparable exposure count.

   Let \(L\) be accepted-prefix length. For a two-draft window,
   \[
   E[L]=P(L\ge1)+P(L\ge2).
   \]
   This follows the usual [accepted-prefix accounting](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).

   Free \(r=3\) gives only their sum, 1.30. Its second-row marginal could lie anywhere from **0.30 to 0.65**. You cannot subtract the **different population’s** first-draft rate, 0.576, to estimate it.

   Likewise, conditional depth-2 accuracy among windows whose first draft landed is not the economic marginal:
   \[
   P(L\ge2)=P(L\ge1)\,P(L\ge2\mid L\ge1).
   \]

   Free two-draft windows landing 1.30 versus forced three-draft windows landing 1.22 shows that the existing gate selects easier windows. It does **not** measure what another row would land in the currently truncated population. It supports caution about broad widening; it proves neither H2 nor a selector frontier.

   H1 ranks last as the complete explanation because lowering the cut already failed, and even **0.3 additional drafts at a true marginal 0.55–0.60 yield only 0.165–0.18 additional tokens**, before charging their rows. The bf16 result also weakens H2’s arithmetic subcase specifically.

2. **Some currently single-draft windows could support a profitable second row, but the supplied aggregates cannot tell you which ones.**

   The missing measurement is
   \[
   q_{2,\mathrm{stop1}}
   =P(L\ge2\mid\text{baseline would offer exactly one draft}).
   \]

   Using your 0.47 threshold, the pooled population with first-draft landing 0.576 needs conditional continuation accuracy above
   \[
   0.47/0.576 \approx 0.816.
   \]
   A conditional depth-2 rate of 0.60–0.70 would instead give an economic marginal of **0.346–0.403** in that population.

   Under pure H1—extending leaves existing proposals unchanged—the strongest possible local bound is:

   - Adding a second draft to all 43% of one-draft windows yields at most \(0.43\times0.576=\mathbf{0.248}\) tokens/stream/round.
   - Those rows add \(4\times0.43\times5.05=\mathbf{8.69}\) ms/round.
   - Starting from 2.1 tokens/stream and 104 ms, even that perfect-continuation limit gives approximately **83.3 total tok/s**, versus 80.8 initially.

   Therefore that population alone cannot explain +0.30 at equal ms. Other stop-depth strata or row reallocation could contribute. But dropping only fourth-and-later drafts offers little budget: the trace contains just **39 such rows/400 windows = 0.0975 rows/stream**.

   Uniform cut sweeps failing does not rule out a profitable subset. Conversely, finding a profitable subset does not establish enough headroom to reach B.

   One denominator check matters: actual throughput near 81 tok/s implies a marginal threshold near \(81\times0.00505=0.41\), whereas 0.47 corresponds to roughly 93 tok/s. Reconcile the allocator’s “~10 tokens” numerator with measured ~8.4 emitted tokens before treating 0.47 as authoritative.

3. **My first measurement would be a matched-prefix, four-stream, forced-three-draft A/B landing trace, labelled with A’s would-have-cut width.**

   Match request and absolute token offset; identical prompts and final hashes do not make round anchors identical. The initial decode window after identical prefills is a straightforward matched prefix.

   | Hypothesis | Cheapest discriminating measurement | Result that kills the claimed explanation |
   |---|---|---|
   | **H1: truncated offers explain the gap** | Re-bin existing forced traces by baseline would-cut width, if chances/cut decisions are retained. Otherwise add that label to the per-depth trace. Measure joint prefix survival in the blocked strata. | Upper confidence bounds show insufficient profitable marginal landing—or insufficient gain at the existing total row budget—to supply +0.25–0.30. A low pooled mean alone does not exclude a favorable subset. |
   | **H2: B produces better deep proposals** | Matched-prefix forced-width A/B traces, comparing \(P(L\ge2)\) and \(P(L\ge3)\) within relevant stop-width strata. | The appropriately weighted landing advantage is too small, with uncertainty bounded, to explain the gap. This rules out the required proposal-quality advantage on that population. |
   | **H3: different commit/round structure** | Audit actual prefix-position advance against \(1+L\), excluding terminal windows, with rounds tied to actual verify calls. | Both engines obey identical accounting and commit the same verified prefix for the same proposals. This kills the commit/accounting branch. |
   | **H3: different drafter conditioning** | One controlled conditioning crossover in A, keeping selector, width and verify fixed; assess with forced per-depth traces. | Changing to the suspected B conditioning produces no landing gain large enough to explain the gap. |

   The forced A/B experiment separates **offer policy from realized proposal quality**. If B wins there, separating selector from conditioning requires the controlled crossover; ordinary landing traces cannot attribute that difference.

   Paired-NLL is useful for localizing numerics under identical conditioning. **Equal mean NLL does not kill an argmax/acceptance hypothesis.** Nor do token hashes: they gate output correctness, not proposal efficiency.

   Also check whether forcing width changes subsequent drafter-state feeding. If it does, a long forced run mixes H1 with H3; matched initial-window probes or preserved baseline state updates are needed for a literal extension counterfactual.

4. **A static landing table mainly risks performance regression under your stated correctness invariant.**

   Its principal failure modes are:

   - **Wrong probability:** using conditional survivor accuracy where the allocator needs joint prefix survival.
   - **Censored calibration:** free-run depth tables describe selected windows and omit precisely the extensions you want to evaluate.
   - **Lost selection:** one depth-average can extend difficult prose windows while cutting profitable code windows. The free/forced contrast already demonstrates relevant heterogeneity.
   - **Distribution and policy shifts:** prose/code, active stream count, and changed round boundaries alter the table’s population. Conditioning feedback can amplify this.
   - **Wrong integration point:** replacing allocator chances after the 0.7 cut has zeroed rows cannot recover those offers.
   - **Wrong cost model:** higher tok/round can lose throughput through additional verify rows; the measured row-cost slope must remain applicable.

   Before shipping, fit from forced traces with the relevant stop-width labels, then evaluate on held-out prose/code at both one and four **active** streams. Require physically consistent prefix-survival estimates, \(q_{d+1}\le q_d\), and retain uncertainty bounds where samples are sparse. Bootstrap by request rather than treating adjacent windows as independent.

   Finally, run the proposed table end to end against baseline and compare **total emitted tokens / total wall time**, round cost and latency tails, with the hash gate. Offline estimates establish whether enough headroom exists; the rollout establishes whether it survives changed round boundaries.

   Selector changes need the same landing-and-wall-time check: an improved isolated score or NLL can worsen deep prefix survival, and any existing confidence table becomes suspect. The measured bf16 action has already shown its failure mode—unchanged acceptance with additional milliseconds.