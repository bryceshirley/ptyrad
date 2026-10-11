#!/usr/bin/env bash
# POD-SIDE campaign: PSO Fig-S5 full workload (256^2 det, 16384 DPs, 12 probe
# modes, 42 slices) on NVLink A100 SXM — fixed Born M=6/M=8 at 1/2/4 GPUs vs
# multislice on the SAME hardware. See NVLINK_PROMPT.md (2026-10-11 REDO).
#
# Expects: repo at /workspace/ptyrad (branch nvlink-bench), venv deps
# installed, data/PSO/tiled_PrScO3_128x128.hdf5 built (push_and_run.sh does
# all three). Arms are ordered cheap/informative first and each is wrapped in
# `timeout`, so a hung arm cannot idle the pod (the LOCAL watchdog is the
# second, independent line of defense).
#
# Usage: bash run_redo_bench.sh [--skip-micro] [--skip-e2e]
set -uo pipefail
cd /workspace/ptyrad/demo
PY="${PY:-python}"
RES="nvlink/results_redo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RES"
export PTYRAD_DISABLE_ISS_PRECOND=1   # MANDATORY for batch-1 Born runs
NG=$(nvidia-smi -L | wc -l)
log() { echo -e "\n=== $(date -u +%FT%TZ) $* ===" | tee -a "$RES/summary.log"; }

# ---- 0. environment + preflight -------------------------------------------
log "0. environment ($NG GPUs)"
{ nvidia-smi topo -m; nvidia-smi nvlink --status | head -40; } > "$RES/topology.log" 2>&1
grep -E "NV[0-9]+" "$RES/topology.log" >/dev/null \
  || { echo "ABORT: no NV* links — wrong box" | tee -a "$RES/summary.log"; exit 1; }
"$PY" - <<'EOF' 2>&1 | tee -a "$RES/summary.log"
import torch
n = torch.cuda.device_count()
print(f"GPUs: {n};", torch.cuda.get_device_name(0))
print("P2P:", [[int(torch.cuda.can_device_access_peer(i, j)) if i != j else 1
                for j in range(n)] for i in range(n)])
EOF
[ -f data/PSO/tiled_PrScO3_128x128.hdf5 ] \
  || { echo "ABORT: tiled dataset missing (run push_and_run.sh)"; exit 1; }

# ---- 1. micro benches at the Fig-S5 shape (PMODE=12) ----------------------
if [ "${1:-}" != "--skip-micro" ]; then
  log "1a. carry schedule chain-vs-direct on NVSwitch (PCIe verdict: chain wins)"
  for MM in 6 8; do
    for P in 2 4; do
      [ "$P" -le "$NG" ] || continue
      echo "--- BORN_M=$MM NGPU=$P ---" | tee -a "$RES/carry.log"
      timeout 1200 env BORN_M=$MM NGPU=$P PMODE=12 NZ_LIST=42,64,128,512 \
        "$PY" bench_carry_direct.py 2>>"$RES/carry.err" | tee -a "$RES/carry.log"
    done
  done
  log "1b. slice-split Nz sweep, 12 modes, M=6/8, P=2/4"
  for MM in 6 8; do
    for P in 2 4; do
      [ "$P" -le "$NG" ] || continue
      echo "--- BORN_M=$MM NGPU=$P ---" | tee -a "$RES/nzsweep.log"
      timeout 1200 env BORN_M=$MM NGPU=$P PMODE=12 NZ_LIST=21,42,64,128,512 \
        "$PY" bench_slice_nzsweep.py 2>>"$RES/nzsweep.err" | tee -a "$RES/nzsweep.log"
    done
  done
  log "1c. multislice vs born head-to-head, 12 modes"
  for MM in 6 8; do
    echo "--- BORN_M=$MM ---" | tee -a "$RES/msvb.log"
    timeout 1800 env BORN_M=$MM PMODE=12 NZ_LIST=42,64,512 NGPU=$NG \
      "$PY" bench_ms_vs_born_b1.py 2>>"$RES/msvb.err" | tee -a "$RES/msvb.log"
  done
fi
[ "${1:-}" = "--skip-e2e" ] && exit 0

# ---- 2. end-to-end s/iter arms (NITER=1 each, $20-budget variant; cheap first)
# per-arm param copies: distinct prefix => distinct output folder (overwrite gotcha)
arm_yml() { # arm_yml BASE ARM -> params/redo_<ARM>.yml
  sed "s/'prefix'            : '[^']*',/'prefix'            : 'redo_$2',/" \
    "params/$1.yml" > "params/redo_$2.yml"
  echo "params/redo_$2.yml"
}
run_arm() { # run_arm ARM ROLE BASE_YML TIMEOUT_S [ENV...]
  local arm="$1" role="$2" base="$3" tmo="$4"; shift 4
  local yml; yml="$(arm_yml "$base" "$arm")"
  log "E2E $arm (role=$role, $yml, timeout ${tmo}s)"
  timeout "$tmo" env "$@" "$PY" run_cmp_modesplit.py "$role" "$yml" \
    > "$RES/e2e_$arm.log" 2>&1
  local rc=$?
  [ $rc -eq 124 ] && echo "TIMEOUT after ${tmo}s" >> "$RES/e2e_$arm.log"
  [ $rc -ne 0 ] && echo "ARM $arm exited rc=$rc" | tee -a "$RES/summary.log"
  grep -E "Total Loss|averaged iter_t|Refit born_coeffs" "$RES/e2e_$arm.log" | tail -6 \
    | tee -a "$RES/summary.log"
}

run_arm ms_1gpu    baseline  nvl42_ms 2700 CUDA_VISIBLE_DEVICES=0
run_arm m6_slice4  slicedist nvl42_m6 3600 SLICE_P=4
run_arm m6_mode4   modesplit nvl42_m6 3600 MODE_P=4
run_arm m6_1gpu    baseline  nvl42_m6 4800 CUDA_VISIBLE_DEVICES=0
run_arm m6_slice2  slicedist nvl42_m6 3600 SLICE_P=2
run_arm m8_slice4  slicedist nvl42_m8 3600 SLICE_P=4
run_arm m8_1gpu    baseline  nvl42_m8 4800 CUDA_VISIBLE_DEVICES=0
run_arm m8_mode4   modesplit nvl42_m8 3600 MODE_P=4

log "DONE — results in $RES (synced home incrementally)"
