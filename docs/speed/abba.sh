#!/usr/bin/env bash
# A -> B -> A over the 0096 image and the baseline image, so drift across the night is visible.
# Switching is by moving the patch files, which changes PATCHES_HASH and so selects the already-built
# image by its label - no rebuild either way.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
OUT=docs/speed; HELD=$OUT/held; mkdir -p "$HELD"
LOG=$OUT/abba.log; : > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

patches_in()  { for f in "$HELD"/009[67]-*.patch; do [[ -e "$f" ]] && mv -f "$f" patches/engine/; done; true; }
patches_out() { for f in patches/engine/009[67]-*.patch; do [[ -e "$f" ]] && mv -f "$f" "$HELD"/; done; true; }

pass() {  # pass <label> <in|out>
  local label=$1 mode=$2
  say ""; say "########## $label ($mode)  $(date -u +%H:%M:%SZ) ##########"
  if [[ "$mode" == in ]]; then patches_in; else patches_out; fi
  say "  patches/engine now has: $(ls patches/engine/009[67]-*.patch 2>/dev/null | wc -l) of 0096/0097"
  if ! bash docs/safestart.sh >> "$LOG" 2>&1; then
    say "  FAILED to start for $label - aborting the ABBA and restoring 0096 state"
    patches_in; bash docs/safestart.sh >> "$LOG" 2>&1 || say "  RESTORE ALSO FAILED - needs a human"
    return 3
  fi
  docker ps --filter name=glm53-spark --format '  image={{.Image}}' | tee -a "$LOG"
  bash docs/speed/coverage.sh "$label" 2>&1 | tee -a "$LOG"
}

say "### ABBA  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
pass a1 in  || exit 3
pass b1 out || exit 3
pass a2 in  || exit 3
say ""; say "### ABBA done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "### final state: 0096/0097 present ($(ls patches/engine/009[67]-*.patch 2>/dev/null | wc -l) files), serving the 0096 image"
