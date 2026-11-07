#!/usr/bin/env bash
# An NCCL exclusion list exported for the whole box (NCCL_SOCKET_IFNAME=^lo,docker,... from /etc/profile.d, issue 17)
# must not stop a start: the recipe needs one netdev, so validate_config warns and leaves the link to detect_link.
# A plain netdev is kept, and anything else is still refused.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
fails=0
check() { # check <name> <value> <want: kept|auto|refused>
  local out err rc got
  err=$(mktemp)
  out=$(env -i PATH="$PATH" HOME="$HOME" NCCL_SOCKET_IFNAME="$2" WORKER=user@host bash -c '
    source scripts/prepare.sh
    validate_config >/dev/null
    printf "%s" "$NCCL_SOCKET_IFNAME"' 2>"$err") && rc=0 || rc=$?
  if (( rc != 0 )) && grep -q 'E_CONFIG.*NCCL_SOCKET_IFNAME' "$err"; then got=refused
  elif (( rc != 0 )); then got="error $rc: $(head -1 "$err")"
  elif [[ -z "$out" ]]; then got=auto
  elif [[ "$out" == "$2" ]]; then got=kept
  else got="changed to $out"; fi
  rm -f -- "$err"
  if [[ "$got" == "$3" ]]; then echo "nccl-ifname: $1: pass"; else echo "nccl-ifname: $1: FAIL (got $got, want $3)" >&2; fails=1; fi
}
check 'an exclusion list is auto' '^lo,docker,tailscale,enP7s7,wlP9s9,veth' auto
check 'a netdev is kept' 'enp1s0f0np0' kept
check 'empty is auto' '' auto
check 'a list of netdevs is refused' 'enp1s0f0np0,enp1s0f1np1' refused
check 'shell text is refused' 'eth0;true' refused
exit $fails
