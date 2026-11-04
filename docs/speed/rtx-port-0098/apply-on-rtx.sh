#!/bin/bash
# Run on the rtx box in the recipe directory: adds 0098 and the changelog entry, commits only those two paths.
set -euo pipefail
cd /home/zurih/MiaAI_Lab/TensorFold/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold-Zig
P=0098-server-an-untyped-tool-parameter-takes-the-JSON-object-or-array-the-model-wrote.patch
cp /tmp/rtx-port/$P patches/engine/
python3 - <<'PY'
s=open("CHANGELOG.md").read(); head="# Changelog\n\n"; assert s.startswith(head)
open("CHANGELOG.md","w").write(head+open("/tmp/rtx-port/changelog-entry.md").read()+s[len(head):]); print("changelog entry added")
PY
git add patches/engine/$P CHANGELOG.md
git -c user.name=MiaAI_Lab -c user.email=mia.ailab1@gmail.com commit -q -m "Engine patch 0098: an untyped tool parameter takes the JSON object or array the model wrote (issue #180 of the 2x DGX Sparks recipe; ported from the GB10 pair recipe)" -- patches/engine/$P CHANGELOG.md
git log --oneline -1
