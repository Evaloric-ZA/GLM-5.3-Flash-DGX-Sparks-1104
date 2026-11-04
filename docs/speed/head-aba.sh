#!/usr/bin/env bash
# A -> B -> A for HEAD=q8, the leading candidate (HANDOFF sec 4): baseline, HEAD=q8, baseline.
# Interleaved arms are the house rule at the 1-3% level (host drift ~1.5%); the +3.1% claim at
# 1 stream must survive this before it goes to the user for approval.
#
# No patch files are moved: both arms run on whatever patches/engine holds at launch (the 0096
# image, after abba.sh), so the comparison is internal to this run and also checks that 0096 did
# not eat the effect. Each leg is a full docs/leg.sh: verified restart, decode sweep 1/2/4 streams
# x3 trials on the fixed prompts, contract corpus, token_sha diff vs contract-baseline.json.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
OUT=docs/speed; LOG=$OUT/head-aba.log; : > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }

leg() {  # leg <label> [KEY=VALUE...]
  local label=$1; shift
  say ""; say "########## leg $label $*  $(date -u +%H:%M:%SZ) ##########"
  if ! bash docs/leg.sh "$label" "$@" >> "$LOG" 2>&1; then
    say "  LEG $label FAILED (rc=$?) - attempting one recovery start, then aborting"
    bash docs/safestart.sh >> "$LOG" 2>&1 || say "  RECOVERY START ALSO FAILED - needs a human"
    return 3
  fi
  grep -hE '^  -> |^# NOISE FLOOR|CONTRACT: ' "$OUT/leg-$label.log" | sed 's/^/  /' | tee -a "$LOG"
}

say "### HEAD=q8 ABA  $(date -u +%Y-%m-%dT%H:%M:%SZ)  (arms: A=baseline env, B=HEAD=q8)"
docker ps --filter name=glm53-spark --format 'serving image={{.Image}}' | tee -a "$LOG"
leg head-a1            || exit 3
leg head-b   HEAD=q8   || exit 3
leg head-a2            || exit 3
say ""; say "### HEAD ABA done  $(date -u +%H:%M:%TZ)"
say "### analyze: dec-head-a1.json vs dec-head-b.json vs dec-head-a2.json; |A1-A2| bounds drift"
