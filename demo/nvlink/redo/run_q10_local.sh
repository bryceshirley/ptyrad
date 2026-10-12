#!/usr/bin/env bash
# LOCAL quality round (SSO.2 follow-up): 10-iter reconstructions at pmode 4,
# full 16384-DP Nz=42 workload, SINGLE GPU per arm (UUID-pinned), outputs KEPT
# (SAVE_ITERS 2: model hdf5 + objp tifs). Timing is NOT valid here (arms
# contend on host PCIe/CPU) — quality/loss only. born order arms: fixed
# M6/M5/M4 + grow-from-3 + grow-from-1 (SSK.4 says M1 start costs contrast;
# owner wants it re-tested at 16384 DPs). OWNER RULE: use only 2 GPUs
# (idx 0+1); GPUs 2+3 stay free.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> demo/
export PTYRAD_DISABLE_ISS_PRECOND=1
export PYTHONPATH="$HOME/ptyrad-nvlink/src"
PY="$HOME/ptyrad/.venv/bin/python"
G0=GPU-03a8aae1-354c-2d7f-d513-2ace87ccacc8
G1=GPU-5c1cb7cd-b1a6-01b3-ef6e-ba1d766f59cb
G2=GPU-b0d04bcb-6a95-8ece-f114-ff510679a5a8
G3=GPU-4bd9ec6a-6fc7-6eff-2fe6-edc66632b0b7
RES="nvlink/results_q10_local_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RES"
log(){ echo -e "\n=== $(date -u +%FT%TZ) $* ===" | tee -a "$RES/summary.log"; }
"$PY" -c "import ptyrad; assert '/ptyrad-nvlink/' in ptyrad.__file__, ptyrad.__file__"

[ -f data/PSO/tiled_PrScO3_128x128.hdf5 ] || \
  "$PY" -m ptyrad.utils.tile_scan data/PSO/sample_data_PrScO3.mat \
        data/PSO/tiled_PrScO3_128x128.hdf5 --tile 2 2

mk(){ sed "$2" params/p4_m6.yml | sed "s/: 'p4_m6',/: '$1',/" \
      | sed "s/'NITER'             : 1,/'NITER'             : 10,/" > "params/$1.yml"; }
mk q10_m6 ""
mk q10_m5 "s/'born_iterations'         : 6,/'born_iterations'         : 5,/"
mk q10_m4 "s/'born_iterations'         : 6,/'born_iterations'         : 4,/"
growsed(){ echo "s/'born_iterations'         : 6,/'born_iterations'         : $1,/;s/'grow_tol'     : null,/'grow_tol'     : 1.0e-2,/;s/'n_limit'      : null,/'n_limit'      : 6,/"; }
mk q10_grow3 "$(growsed 3)"
mk q10_grow1 "$(growsed 1)"

run(){ local arm="$1" gpu="$2"
  log "ARM $arm on $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_cmp_modesplit.py baseline "params/$arm.yml" \
    > "$RES/e2e_$arm.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "ARM $arm exited rc=$rc" | tee -a "$RES/summary.log"
  grep -E "Total Loss|Finished" "$RES/e2e_$arm.log" | tail -4 | tee -a "$RES/summary.log"
}
( run q10_m6 "$G0"; run q10_m4 "$G0" ) &
( run q10_m5 "$G1"; run q10_grow3 "$G1"; run q10_grow1 "$G1" ) &
wait
log "Q10 LOCAL DONE"
