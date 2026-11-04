#!/usr/bin/env bash
# A restart with a VERIFIED stop, because `start.sh restart` does not have one.
#
# On 2026-11-03 05:07Z the worker's `docker stop` failed ("cannot delete running task ... failed
# precondition", "DeadlineExceeded") and `docker run` was issued anyway, so two ~100 GiB containers
# existed at once on a 128 GB machine. No OOM kill and no panic: the host simply stopped being able
# to fork, ping and TCP/22 kept answering, and it needed a power cycle. See docs/speed/RESULTS.md.
#
# This stops through ./stop.sh -- drain, log archive, the stop->kill->verified ladder -- then refuses
# to start until BOTH ranks' containers are provably absent, and refuses to return
# success until /health answers. Every ssh has a timeout, so a wedged worker fails fast and loudly
# instead of hanging a programme for hours.
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
NAME=glm53-spark
WORKER=$(sed -n 's/^WORKER=\([^ \t#]*\).*/\1/p' scripts/local.sh | head -1)
[[ -n "$WORKER" ]] || { echo "[safestart] WORKER not found in scripts/local.sh"; exit 3; }
sshw() { timeout "${1:?}" ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlPath=none "$WORKER" "${2:?}"; }

gone_local()  { ! docker inspect "$NAME" >/dev/null 2>&1; }
gone_worker() { ! sshw 25 "docker inspect $NAME >/dev/null 2>&1" >/dev/null 2>&1; }

echo "[safestart] $(date -u +%H:%M:%SZ) verifying both ranks are stopped (worker=$WORKER)"
echo "[safestart] $(date -u +%H:%M:%SZ) ./stop.sh (drain, archive, kill on timeout, verified absent)"
stop_out=$(./stop.sh 2>&1); stop_rc=$?
if (( stop_rc == 0 )); then
  [[ "$stop_out" == *Nothing\ to\ stop* ]] || echo "[safestart] $(echo "$stop_out" | tail -1)"
else
  echo "[safestart] stop.sh failed rc=$stop_rc - falling back to a forced remove; its last lines:"
  echo "$stop_out" | tail -3 | sed 's/^/[safestart]   /'
  docker rm -f "$NAME" >/dev/null 2>&1
  sshw 60 "docker rm -f $NAME >/dev/null 2>&1; true" >/dev/null 2>&1
fi

for i in $(seq 1 24); do          # up to 2 min for the task to actually die
  l=no; w=no
  gone_local  && l=yes
  gone_worker && w=yes
  if [[ "$l" == yes && "$w" == yes ]]; then
    echo "[safestart] both ranks absent after ${i}x5s - safe to start"
    break
  fi
  [[ $i -eq 24 ]] && {
    echo "[safestart] REFUSING TO START: container still present (local_gone=$l worker_gone=$w)."
    echo "[safestart] This is the condition that wedged spark4. Needs a human."
    exit 4
  }
  sleep 5
done

echo "[safestart] $(date -u +%H:%M:%SZ) ./start.sh restart"
./start.sh restart > /tmp/safestart-start.$$.log 2>&1 &   # never piped: a pipe can outlive the job
sp=$!
for i in $(seq 1 120); do          # up to 10 min to serve
  if curl -fs -m 8 http://127.0.0.1:8888/health >/dev/null 2>&1; then
    echo "[safestart] HEALTHY after ~$((i*5))s at $(date -u +%H:%M:%SZ)"
    tail -3 /tmp/safestart-start.$$.log 2>/dev/null; rm -f /tmp/safestart-start.$$.log
    exit 0
  fi
  sleep 5
done
echo "[safestart] NOT HEALTHY after 10 min - aborting this leg"
tail -20 /tmp/safestart-start.$$.log 2>/dev/null; rm -f /tmp/safestart-start.$$.log
kill "$sp" 2>/dev/null
exit 5
