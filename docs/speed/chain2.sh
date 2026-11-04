#!/usr/bin/env bash
# The remaining programme, in one chain so a single completion notification covers it.
# Order: the profile+probes+numerics (night2, the triage the panel ranked first), then the
# 0096 ABBA coverage gate. Nothing here waits on a pgrep pattern.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
echo "### chain2 start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash docs/speed/night2.sh >/dev/null 2>&1
echo "### night2 exit=$? $(date -u +%H:%M:%SZ)"
bash docs/speed/abba.sh  >/dev/null 2>&1
echo "### abba   exit=$? $(date -u +%H:%M:%SZ)"
echo "### chain2 done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
