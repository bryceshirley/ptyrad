#!/usr/bin/env bash
# Incremental copy-back (run LOCALLY, in the background, alongside the
# watchdog). Every INTERVAL seconds rsync the pod's results + recon logs into
# demo/nvlink/runpod_results/redo/. Small files only — model checkpoints stay
# on the pod (the 0.5 MB/s home downlink cannot move them; pull any keeper
# explicitly at the end). Survives pod death: last synced state is kept.
#
# Usage: bash sync_back.sh [INTERVAL_SECONDS]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pod_state.env"   # POD_IP, POD_PORT
INTERVAL="${1:-120}"
DEST="$(cd "$HERE/.." && pwd)/runpod_results/redo"
mkdir -p "$DEST"
SSH="ssh -o StrictHostKeyChecking=no -p $POD_PORT"
echo "sync_back: root@$POD_IP:$POD_PORT -> $DEST every ${INTERVAL}s"
while :; do
  rsync -az --timeout=60 -e "$SSH" \
    --include='*/' \
    --include='*.log' --include='*.txt' --include='*.md' --include='*.yml' \
    --include='*.png' --include='*.csv' --include='*.json' \
    --exclude='*' \
    "root@$POD_IP:/workspace/ptyrad/demo/nvlink/results_redo*" \
    "root@$POD_IP:/workspace/ptyrad/demo/output" \
    "$DEST/" 2>/dev/null
  date -u +"%FT%TZ synced" >> "$DEST/sync.log"
  sleep "$INTERVAL"
done
