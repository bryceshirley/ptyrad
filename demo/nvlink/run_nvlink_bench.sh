#!/usr/bin/env bash
# NVLink validation of the multi-GPU Born benchmarks on the real PSO sample.
# Run on a rented 2-4x A100/H100 SXM (NVLink/NVSwitch) box. See NVLINK_PROMPT.md
# for the full context, data-transfer manifest, and expected outcomes.
#
# Usage:  bash demo/nvlink/run_nvlink_bench.sh [--skip-recon]
#   --skip-recon  skip the two ~25 min 20-iteration reconstructions
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="$(dirname "$HERE")"
cd "$DEMO"
PY="${PY:-$(command -v python)}"
[ -x "$DEMO/../.venv/bin/python" ] && PY="$DEMO/../.venv/bin/python"
RES="$HERE/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RES"
export PTYRAD_DISABLE_ISS_PRECOND=1   # MANDATORY for batch-1 Born runs
echo "python: $PY | results: $RES"

log() { echo -e "\n=== $* ===" | tee -a "$RES/summary.log"; }

# ---- 0. environment report + preflight ------------------------------------
log "0. environment"
{ nvidia-smi topo -m; nvidia-smi nvlink --status; } > "$RES/topology.log" 2>&1
grep -E "NV[0-9]+" "$RES/topology.log" >/dev/null \
  && echo "NVLink detected (NV* in topo)" | tee -a "$RES/summary.log" \
  || echo "WARNING: no NV* links in topo -- this box may not have NVLink!" | tee -a "$RES/summary.log"
"$PY" - <<'EOF' 2>&1 | tee -a "$RES/summary.log"
import torch
n = torch.cuda.device_count()
print(f"GPUs: {n};", torch.cuda.get_device_name(0))
print("P2P:", [[int(torch.cuda.can_device_access_peer(i, j)) if i != j else 1
                for j in range(n)] for i in range(n)])
EOF
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>100' | wc -l)
[ "$BUSY" -gt 0 ] && { echo "ABORT: $BUSY GPU(s) busy -- benchmarks need idle GPUs."; exit 1; }

CKPT_DIR="output/PSO/20260929_pso_born6_qr_msT_noreg_full_N4096_dp256_random1_p4_1obj_21slice_dz10_plr1e-4_oalr5e-4_oplr5e-4_dpblur1.0_orblur0.4_ozblur1.0_mamp0.03_4.0_oathr0.96_oposc_sng1.0_spr0.1"
CKPT="$CKPT_DIR/model_iter0100.hdf5"
MAT="data/PSO/sample_data_PrScO3.mat"
[ -f "$CKPT" ] || { echo "MISSING $CKPT -- scp it per NVLINK_PROMPT.md"; exit 1; }

# ---- 1. slice-split vs real checkpoint (real transfer + compute floor) ----
log "1. bench_slice_real (real transfer)"
"$PY" bench_slice_real.py 2>"$RES/slice_real.err" | tee "$RES/slice_real.log"
log "1b. bench_slice_real NOTRANSFER=1 (compute floor)"
NOTRANSFER=1 "$PY" bench_slice_real.py 2>/dev/null | grep -E "N=[0-9]+: fwd " | tee "$RES/slice_floor.log"

# ---- 2. slice vs mode vs batch head-to-head --------------------------------
log "2. bench_axes_real (slice/mode/batch, N=1/8/32)"
"$PY" bench_axes_real.py 2>"$RES/axes.err" | tee "$RES/axes.log"

# ---- 2b. deep-stack Nz sweep (the multislice-comparison regime) ------------
log "2b. bench_slice_nzsweep (batch-1 deep stacks, hypothesis 4; M=6/2/1)"
for MM in 6 2 1; do
  echo "--- BORN_M=$MM ---" | tee -a "$RES/nzsweep.log"
  BORN_M=$MM "$PY" bench_slice_nzsweep.py 2>"$RES/nzsweep.err" | tee -a "$RES/nzsweep.log"
done

# ---- 3. end-to-end 20-iter reconstructions (~25 min each) ------------------
if [ "${1:-}" != "--skip-recon" ]; then
  if [ -f "$MAT" ]; then
    log "3a. 20-iter reconstruction: baseline 1-GPU"
    CUDA_VISIBLE_DEVICES=0 "$PY" run_cmp_modesplit.py baseline \
      > "$RES/recon_baseline.log" 2>&1
    log "3b. 20-iter reconstruction: mode-split 2-GPU"
    CUDA_VISIBLE_DEVICES=0,1 "$PY" run_cmp_modesplit.py modesplit \
      > "$RES/recon_modesplit.log" 2>&1
    grep -hE "Iter: (1|5|10|15|20), Total Loss|Finished 20" \
      "$RES/recon_baseline.log" "$RES/recon_modesplit.log" | tee -a "$RES/summary.log"
  else
    echo "SKIP recon: $MAT missing (scp it per NVLINK_PROMPT.md)" | tee -a "$RES/summary.log"
  fi
fi

log "DONE -- key lines"
grep -hE "rel err|N=[0-9]+:|forward speedup|fwd\+bw|Finished" \
  "$RES"/slice_real.log "$RES"/slice_floor.log "$RES"/axes.log 2>/dev/null \
  | tee -a "$RES/summary.log"
echo "All raw logs in $RES"
