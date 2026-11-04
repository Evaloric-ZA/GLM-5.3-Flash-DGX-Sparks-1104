There is one material hole before ship approval: **the quality evidence has not been tied to a defined no-regression criterion on the exact image used for the interleaved run.**

The paired NLL result is reassuring, but its confidence interval includes a small increase, and the four-passage result also increased slightly. Identical greedy answers and contract hashes show output stability for those cases; they do not establish that the quantized head preserves probabilities or quality across domains.

A useful check tonight would be to rerun the existing paired NLL corpus on the **0096 image**, baseline versus q8, and report the aggregate interval and per-domain results alongside a pre-agreed acceptable quality threshold. This reuses an existing corpus and should fit the 30-minute constraint if the harness is ready. If no nonzero tolerance is authorized, present the interval honestly: the evidence cannot establish literally zero regression.

The speed evidence is otherwise ready to present. The interleaved A→B→A result supports a gain at all three tested loads, and the round-time change is consistent with the expected direction and magnitude. The larger 2/4-stream gains tonight add uncertainty about the precise benefit, but do not by themselves require another load point or performance run.

**Recommendation:** complete the quality check, then present the measured gains and quality limits for explicit ship approval. Keep q8 measure-only until that approval.
