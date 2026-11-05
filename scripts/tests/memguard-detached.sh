#!/usr/bin/env bash
# The memory guard start.sh leaves behind must hold none of start.sh's stdin, stdout or stderr: a caller such as
# `ssh spark './start.sh'` waits for the last writer of its pipe, and a guard that kept the pipe open kept the ssh
# session open with it (issue 7). The test runs start_memguard with pipes on all three and reads the guard's
# descriptors from /proc.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
tmp=$(mktemp -d); trap 'rm -rf -- "$tmp"' EXIT
# The function alone, from start.sh (sourcing start.sh would run the start).
fn=$(awk '/^start_memguard\(\) \{/ { on = 1 } on { print } on && /^\}/ { exit }' start.sh)
[[ -n "$fn" ]] || { echo 'start_memguard not found in start.sh' >&2; exit 1; }
{ cat <<'SCRIPT'
set -u
docker() { echo true; }
mem_avail_gb() { echo 50; }
worker_avail_gib() { echo 50; }
at_least() { return 0; }
worker_host() { echo worker; }
log() { :; }
SCRIPT
  printf '%s\n' "$fn"
  cat <<'SCRIPT'
start_memguard
for fd in 0 1 2; do readlink "/proc/$MEMGUARD_PID/fd/$fd" || echo "fd $fd: unreadable"; done
kill "$MEMGUARD_PID" 2>/dev/null; wait "$MEMGUARD_PID" 2>/dev/null || true
SCRIPT
} > "$tmp/run.sh"
# stdin, stdout and stderr are pipes here.
fds=$(echo | MEMGUARD=1 DRY_RUN=0 MEMGUARD_INTERVAL_S=1 MEMGUARD_FLOOR_GIB=2.5 MEMGUARD_HOLD_S=4 \
  RUN_LOG_DIR="$tmp" REPO_ROOT="$tmp" CONTAINER_NAME=test bash "$tmp/run.sh" 2>&1)
status=0
while IFS= read -r line; do
  case "$line" in
    pipe:*|*unreadable*|socket:*) echo "memguard holds the caller's descriptor: $line" >&2; status=1 ;;
  esac
done <<< "$fds"
grep -q 'memguard.log$' <<< "$fds" || { echo "memguard's stdout is not its log: $fds" >&2; status=1; }
(( status == 0 )) && echo 'memguard-detached: pass'
exit "$status"
