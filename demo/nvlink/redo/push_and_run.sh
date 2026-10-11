#!/usr/bin/env bash
# LOCAL: set the pod up and launch the campaign detached.
#   1. clone nvlink-bench on the pod + install (PEP-668: --break-system-packages)
#   2. upload sample_data_PrScO3.mat (287 MB @ ~0.5 MB/s ~ 10 min) + checksum
#   3. build the 16384-DP tiled dataset ON the pod (4.3 GB — never uploaded)
#   4. nohup run_redo_bench.sh; tail progress via sync_back.sh / ssh
# Prereqs: deploy_redo.sh done (pod_state.env), watchdog + sync_back running.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pod_state.env"
SSH=(ssh -o StrictHostKeyChecking=no -p "$POD_PORT" "root@$POD_IP")
MAT="$(cd "$HERE/../.." && pwd)/data/PSO/sample_data_PrScO3.mat"
MAT_SHA="e5c470797e90a21381b9fdb0a353493f3ead96fcafdd4b9a163bb110059e472a"

echo "== 1. clone + install =="
"${SSH[@]}" bash -s <<'EOF'
set -e
cd /workspace
command -v rsync >/dev/null || { apt-get update -qq >/dev/null; apt-get install -y -qq rsync >/dev/null; }
[ -d ptyrad ] || git clone -b nvlink-bench --depth 1 https://github.com/bryceshirley/ptyrad
cd ptyrad
pip install --break-system-packages -q -e . h5py torchvision 2>&1 | tail -2
mkdir -p demo/data/PSO demo/output
python -c "import ptyrad, torch; print('ptyrad at', ptyrad.__file__, '| torch', torch.__version__)"
EOF

echo "== 2. upload .mat (background-resumable via rsync) =="
rsync -a --partial --progress -e "ssh -o StrictHostKeyChecking=no -p $POD_PORT" \
  "$MAT" "root@$POD_IP:/workspace/ptyrad/demo/data/PSO/"
"${SSH[@]}" "cd /workspace/ptyrad/demo/data/PSO && echo '$MAT_SHA  sample_data_PrScO3.mat' | sha256sum -c -"

echo "== 3. build tiled 128x128 dataset on the pod =="
"${SSH[@]}" "cd /workspace/ptyrad/demo && python -m ptyrad.utils.tile_scan \
  data/PSO/sample_data_PrScO3.mat data/PSO/tiled_PrScO3_128x128.hdf5 --tile 2 2"

echo "== 4. launch campaign (detached) =="
"${SSH[@]}" "cd /workspace/ptyrad/demo && nohup bash nvlink/redo/run_redo_bench.sh \
  > nvlink/campaign.log 2>&1 & echo launched pid \$!"
echo "tail with: ssh -p $POD_PORT root@$POD_IP tail -f /workspace/ptyrad/demo/nvlink/campaign.log"
