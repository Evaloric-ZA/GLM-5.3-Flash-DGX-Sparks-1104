#!/usr/bin/env bash
# Run a list of legs sequentially, then compare them all against the baseline.
#   phase1.sh <label>:<KEY=VALUE[,KEY=VALUE...]> ...
# e.g. phase1.sh cut040:TF_GLM_DRAFT_CONFIDENCE=0.40 cut000:TF_GLM_DRAFT_CONFIDENCE=0.0
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
LABELS=()
for spec in "$@"; do
  label=${spec%%:*}; rest=${spec#*:}
  LABELS+=("$label")
  args=()
  if [[ "$rest" != "$spec" && -n "$rest" ]]; then IFS=',' read -ra args <<< "$rest"; fi
  echo "=== leg $label  $(date -u +%H:%M:%SZ) ==="
  bash docs/leg.sh "$label" "${args[@]}" || echo "LEG $label FAILED"
done
echo "=== comparison  $(date -u +%H:%M:%SZ) ==="
python3 -I docs/speed/compare.py base2 "${LABELS[@]}"
