#!/usr/bin/env bash
# Fires decay-point.sh at boot++'1h/3h/12h/24h'. One shot per point; safe to re-run; logs to decay-sched.log.
BOOT=1793700643
D=/home/zurih/TF/GLM-5.3-Flash-EXL3-2x-DGX-Spark-TensorFold-Zig
L=$D/docs/speed/decay-sched.log
for spec in 3600:t+1h 10800:t+3h 43200:t+12h 86400:t+24h; do
  off=${spec%%:*}; tag=${spec##*:}
  due=$((BOOT+off)); now=$(date +%s)
  if (( due > now )); then
    echo "$(date -u +%FT%TZ) waiting $((due-now))s for $tag" >> $L
    sleep $((due-now))
  fi
  if grep -q "decay point $tag done" $D/docs/speed/decay-log.txt 2>/dev/null; then
    echo "$(date -u +%FT%TZ) $tag already done, skip" >> $L; continue
  fi
  echo "$(date -u +%FT%TZ) firing $tag" >> $L
  bash $D/docs/speed/decay-point.sh "$tag" "$BOOT"
done
echo "$(date -u +%FT%TZ) series complete" >> $L
