#!/usr/bin/env bash
# The 0096/0097 experiment: build the new image, A/B it, and roll back if it fails its gate or breaks.
#
# Rollback uses the recipe's own mechanism rather than an IMAGE override: the image tag is a hash over
# docker/ + patches/engine/*.patch + the pins, so moving the two new patch files out of
# patches/engine/ restores the previous hash, and the previous image is then found by its label with
# no rebuild. That makes the rollback a ~3 min restart, not an 11 min one.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
OUT=docs/speed; LOG=$OUT/run0096.log; : > "$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
HELD=$OUT/held
P96=$(ls patches/engine/0096-*.patch 2>/dev/null | head -1)
P97=$(ls patches/engine/0097-*.patch 2>/dev/null | head -1)

rollback() {
  say "--- ROLLBACK: moving 0096/0097 aside and restarting on the previous image ---"
  [[ -z "${P96:-}" ]] || mv -f "$P96" "$HELD/" 2>/dev/null
  [[ -z "${P97:-}" ]] || mv -f "$P97" "$HELD/" 2>/dev/null
  ./start.sh restart >> "$LOG" 2>&1 && say "  rolled back and serving" || say "  ROLLBACK FAILED - needs a human"
  curl -s -m 15 http://127.0.0.1:8888/health >/dev/null && say "  /health OK" || say "  /health NOT OK"
}

say "### 0096+0097 experiment  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "  patches: $P96"
say "           $P97"
say "--- build + restart (new patches hash -> new image; the old one stays on disk) ---"
if ! ./start.sh restart >> "$LOG" 2>&1; then
  say "FAILED: build or start with 0096/0097. Rolling back."
  rollback; exit 3
fi
grep -hE "CUDA sm_121 device 0" logs/current/rank0.log | tee -a "$LOG"
docker ps --filter name=glm53-spark --format '  image={{.Image}}' | tee -a "$LOG"

say "--- A/B: decode 1/2/4, contract, prefill, quality ---"
python3 -I docs/dec.py p0096 2>&1 | tee -a "$LOG"
python3 -I docs/dec.py p0096 --contract 2>&1 | tee -a "$LOG"
python3 -I docs/pf2.py 2>&1 | tee -a "$LOG"
python3 -I docs/quality.py 2>&1 | tee -a "$LOG"

say "--- verdict against base2 ---"
python3 -I docs/speed/compare.py base2 p0096 2>&1 | tee -a "$LOG"

say "--- mechanism: did allocation change? (drafted/round at 2 and 4 streams) ---"
python3 -I - <<'PY' 2>&1 | tee -a "$LOG"
import json, os
d = "docs/speed"
for lab in ("base2", "p0096"):
    p = os.path.join(d, f"dec-{lab}.json")
    if not os.path.exists(p): continue
    r = json.load(open(p))
    for n in sorted(r, key=int):
        t = r[n]["trials"][0]
        dr = sum(x for x in t["drafted"] if x) / sum(x for x in t["rounds"] if x)
        tpr = sum(x for x in t["tokens_per_round"] if x) / len([x for x in t["tokens_per_round"] if x])
        print(f"  {lab:7s} {n} stream(s): drafted/round {dr:.3f}  tokens/round {tpr:.3f}  rounds {r[n]['rounds']}")
print("  If these are identical at 2 and 4 streams, the corrected tables changed no scheduling")
print("  decision and 0096 is inert: revert it and report that, per Codex's amendment 5.")
PY
say "### 0096 experiment done  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "NOTE: this script does not auto-revert on a failed gate - the verdict above is read by a human"
say "      (or by me) and rollback() is called deliberately. It auto-reverts only on a build/start failure."
