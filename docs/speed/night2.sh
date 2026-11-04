#!/usr/bin/env bash
# The remaining measurement programme, in the order the panel agreed (AGREEMENT.md revision 2).
# Everything here is measurement or a reversible env change; nothing numeric is left enabled.
#   night2.sh
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
LOG=docs/speed/night2.log; : > "$LOG"
say() { printf '\n%s\n' "$*" | tee -a "$LOG"; }

say "######## 1. PROFILE - the trunk. Attribution only, never a gate. ########"
bash docs/profleg.sh p0096 2>&1 | tee -a "$LOG"

say "######## 2. DRAFTER PROBE ########"
say "# DRAFTER=none gives a plain-decode round: a pure 1-row forward with no drafter pass, which is"
say "# the cleanest datum for the floor analysis. DRAFTER_BITS=0 doubles the drafter's weight bytes:"
say "# if its 5.92 ms is bandwidth-bound the pass should roughly double; if launch-bound it will not"
say "# move. That distinguishes the two hypotheses in one leg. Both keep the target's picks, so"
say "# token_sha must stay identical - except for DRAFTER=none, where there are no drafts at all and"
say "# the reply is still the target's own greedy picks, so it must ALSO be identical."
bash docs/leg.sh drafter_none DRAFTER=none   2>&1 | tee -a "$LOG"
bash docs/leg.sh drafter_bf16 DRAFTER_BITS=0 2>&1 | tee -a "$LOG"
bash docs/leg.sh drafter_mtp  DRAFTER=mtp    2>&1 | tee -a "$LOG"

say "######## 3. STRUCTURAL KNOB: the shared-expert side stream ########"
say "# On unified memory two simultaneous readers contend for the same 273 GB/s, so overlapping the"
say "# shared expert onto a side stream may cost more than it hides. A structural hypothesis, not a"
say "# knob sweep - and the only one of its family being tested, per the panel."
bash docs/leg.sh shared_side0 TF_GLM_SHARED_SIDE=0 2>&1 | tee -a "$LOG"

say "######## 4. NUMERICS, MEASURE ONLY - never left enabled ########"
say "# Paired per-token NLL baseline on the shipped precision first, so the comparisons are paired."
( cd docs && python3 -I nllpaired.py q8base ) 2>&1 | tee -a "$LOG"

say "---- HEAD=q8 (0.634 GB of 6.58, so 4.8% of forward bytes) ----"
bash docs/leg.sh head_q8 HEAD=q8 2>&1 | tee -a "$LOG"
( cd docs && python3 -I quality.py && python3 -I nllpaired.py headq8 && python3 -I cmp.py q8base headq8 ) 2>&1 | tee -a "$LOG"

say "---- DENSE=q6 (the largest byte lever: 58% of the forward is q8, so q6 cuts ~14.6%) ----"
bash docs/leg.sh dense_q6 DENSE=q6 2>&1 | tee -a "$LOG"
( cd docs && python3 -I quality.py && python3 -I nllpaired.py denseq6 && python3 -I cmp.py q8base denseq6 ) 2>&1 | tee -a "$LOG"

say "######## 5. RESTORE the approved precision and verify ########"
bash docs/leg.sh restored 2>&1 | tee -a "$LOG"
( cd docs && python3 -I quality.py ) 2>&1 | tee -a "$LOG"
python3 -I docs/pf2.py 2>&1 | tee -a "$LOG"
say "--- health, model list, startup line ---"
curl -s -m 15 http://127.0.0.1:8888/health | tee -a "$LOG"
curl -s -m 15 http://127.0.0.1:8888/v1/models | tee -a "$LOG"
grep -hE "CUDA sm_121 device 0" logs/current/rank0.log | tee -a "$LOG"
docker ps --filter name=glm53-spark --format 'image={{.Image}} status={{.Status}}' | tee -a "$LOG"
say "######## night2 done $(date -u +%Y-%m-%dT%H:%M:%SZ) ########"
