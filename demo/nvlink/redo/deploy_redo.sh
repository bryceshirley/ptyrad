#!/usr/bin/env bash
# Deploy the NVLink-redo pod (default: 4x A100 SXM, NVSwitch) and wait for SSH.
# Writes pod id / ip / port to demo/nvlink/redo/pod_state.env for the other
# scripts. ABORTS unless the account balance covers BUDGET_USD.
#
# Usage: bash deploy_redo.sh [--gpus 4] [--budget 35]
# Round-2 post-mortem rules built in:
#   * refuses to deploy on insufficient balance (no mid-run credit exhaustion)
#   * STALL_MINUTES: a pod stuck in "initializing"/no-SSH is terminated and
#     must be redeployed (last round: 32 min stuck, $0 billed, wasted hour)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runpod_api.sh
source "$HERE/runpod_api.sh"

GPUS=4
BUDGET=35
STALL_MINUTES=12
while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --budget) BUDGET="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

BAL="$(rp_balance)"
echo "balance: \$$BAL | required budget: \$$BUDGET"
if ! awk -v b="$BAL" -v r="$BUDGET" 'BEGIN{exit !(b>=r)}'; then
  echo "ABORT: balance \$$BAL < budget \$$BUDGET — top up before deploying." >&2
  exit 1
fi

PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"
BODY=$(jq -n --argjson g "$GPUS" --arg pk "$PUBKEY" '{
  name: "nvlink-redo-pso",
  imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
  cloudType: "SECURE",
  computeType: "GPU",
  gpuTypeIds: ["NVIDIA A100-SXM4-80GB"],
  gpuCount: $g,
  containerDiskInGb: 80,
  volumeInGb: 0,
  ports: ["22/tcp"],
  env: {PUBLIC_KEY: $pk}
}')
echo "creating pod ($GPUS x A100 SXM)..."
RESP="$(rp_rest POST /pods "$BODY")"
POD_ID="$(echo "$RESP" | jq -r '.id // empty')"
if [ -z "$POD_ID" ]; then
  echo "ABORT: pod create failed:" >&2
  echo "$RESP" | jq . >&2
  exit 1
fi
echo "pod id: $POD_ID"
T0=$(date +%s)
while :; do
  sleep 20
  DEST="$(rp_ssh_dest "$POD_ID" || true)"
  if [ -n "$DEST" ]; then
    IP="${DEST% *}"; PORT="${DEST#* }"
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p "$PORT" \
        "root@$IP" true 2>/dev/null; then
      break
    fi
  fi
  EL=$(( ($(date +%s) - T0) / 60 ))
  echo "waiting for SSH... ${EL}m"
  if [ "$EL" -ge "$STALL_MINUTES" ]; then
    echo "STALLED ${EL}m — terminating and giving up (redeploy manually)." >&2
    rp_terminate "$POD_ID" >/dev/null || true
    exit 1
  fi
done
{
  echo "POD_ID=$POD_ID"
  echo "POD_IP=$IP"
  echo "POD_PORT=$PORT"
  echo "DEPLOYED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$HERE/pod_state.env"
echo "ready: ssh -p $PORT root@$IP  (state in $HERE/pod_state.env)"
echo "NEXT: bash watchdog_redo.sh &   # kill-switch FIRST"
echo "      bash sync_back.sh &       # incremental copy-back"
echo "      bash push_and_run.sh      # data up + campaign"
