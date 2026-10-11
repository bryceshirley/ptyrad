#!/usr/bin/env bash
# Independent balance + deadline kill-switch (run LOCALLY, in the background,
# BEFORE starting the campaign). Terminates the pod when EITHER
#   * account balance drops below MIN_BALANCE_USD (default $2), or
#   * DEADLINE_HOURS since start elapse (default 6 h),
# regardless of what the pod is doing. This is the round-2 post-mortem fix:
# a hung arm once idled the pod to $0 and RunPod's auto-terminate wiped the
# disk with the 30-iter artifacts on it. A completion watcher is NOT enough.
#
# Usage: bash watchdog_redo.sh [MIN_BALANCE_USD] [DEADLINE_HOURS]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod_api.sh
source "$HERE/runpod_api.sh"
source "$HERE/pod_state.env"   # POD_ID

MIN_BALANCE="${1:-2}"
DEADLINE_HOURS="${2:-6}"
T0=$(date +%s)
LOG="$HERE/watchdog.log"
echo "$(date -u +%FT%TZ) watchdog: pod $POD_ID, min \$$MIN_BALANCE, deadline ${DEADLINE_HOURS}h" | tee -a "$LOG"

kill_pod() {
  echo "$(date -u +%FT%TZ) KILL: $1 — terminating $POD_ID" | tee -a "$LOG"
  for _ in 1 2 3; do
    rp_terminate "$POD_ID" >>"$LOG" 2>&1 && break
    sleep 10
  done
  exit 0
}

while :; do
  BAL="$(rp_balance 2>>"$LOG" || echo "")"
  EL_H=$(awk -v t0="$T0" -v t1="$(date +%s)" 'BEGIN{printf "%.2f", (t1-t0)/3600}')
  echo "$(date -u +%FT%TZ) balance=\$${BAL:-?} elapsed=${EL_H}h" >> "$LOG"
  if [ -n "$BAL" ] && awk -v b="$BAL" -v m="$MIN_BALANCE" 'BEGIN{exit !(b<m)}'; then
    kill_pod "balance \$$BAL < \$$MIN_BALANCE"
  fi
  if awk -v e="$EL_H" -v d="$DEADLINE_HOURS" 'BEGIN{exit !(e>=d)}'; then
    kill_pod "deadline ${DEADLINE_HOURS}h reached"
  fi
  # pod already gone? stop watching
  STATUS="$(rp_pod "$POD_ID" 2>/dev/null | jq -r '.desiredStatus // empty')"
  if [ "$STATUS" = "TERMINATED" ] || [ -z "$STATUS" ]; then
    echo "$(date -u +%FT%TZ) pod gone (status '$STATUS') — watchdog exiting" | tee -a "$LOG"
    exit 0
  fi
  sleep 60
done
