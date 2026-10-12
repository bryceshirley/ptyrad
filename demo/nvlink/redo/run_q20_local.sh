#!/usr/bin/env bash
# LOCAL quality round v2 (owner spec 2026-10-12): 20-iter reconstructions at
# pmode 4, full 16384-DP Nz=42 workload, grow arms capped at M=8 (was 6).
# SINGLE GPU per arm, OWNER RULE: only GPUs idx 0+1; 2+3 stay free. Outputs
# KEPT LOCALLY under demo/output/PSO (no Drive push). Timing NOT valid
# (contention) — quality/loss only. Supersedes run_q10_local.sh (its 10-iter
# m5/m6 results remain in output/PSO as q10_*).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> demo/
export PTYRAD_DISABLE_ISS_PRECOND=1
export PYTHONPATH="$HOME/ptyrad-nvlink/src"
PY="$HOME/ptyrad/.venv/bin/python"
G0=GPU-03a8aae1-354c-2d7f-d513-2ace87ccacc8
G1=GPU-5c1cb7cd-b1a6-01b3-ef6e-ba1d766f59cb
RES="nvlink/results_q20_local_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RES"
log(){ echo -e "\n=== $(date -u +%FT%TZ) $* ===" | tee -a "$RES/summary.log"; }
"$PY" -c "import ptyrad; assert '/ptyrad-nvlink/' in ptyrad.__file__, ptyrad.__file__"

[ -f data/PSO/tiled_PrScO3_128x128.hdf5 ] || \
  "$PY" -m ptyrad.utils.tile_scan data/PSO/sample_data_PrScO3.mat \
        data/PSO/tiled_PrScO3_128x128.hdf5 --tile 2 2

mk(){ sed "$2" params/p4_m6.yml | sed "s/: 'p4_m6',/: '$1',/" \
      | sed "s/'NITER'             : 1,/'NITER'             : 20,/" > "params/$1.yml"; }
mk q20_m6 ""
mk q20_m5 "s/'born_iterations'         : 6,/'born_iterations'         : 5,/"
mk q20_m4 "s/'born_iterations'         : 6,/'born_iterations'         : 4,/"
growsed(){ echo "s/'born_iterations'         : 6,/'born_iterations'         : $1,/;s/'grow_tol'     : null,/'grow_tol'     : 1.0e-2,/;s/'n_limit'      : null,/'n_limit'      : 8,/"; }
mk q20_grow3 "$(growsed 3)"
mk q20_grow1 "$(growsed 1)"

run(){ local arm="$1" gpu="$2"
  log "ARM $arm on $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_cmp_modesplit.py baseline "params/$arm.yml" \
    > "$RES/e2e_$arm.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "ARM $arm exited rc=$rc" | tee -a "$RES/summary.log"
  grep -E "Total Loss|Finished" "$RES/e2e_$arm.log" | tail -4 | tee -a "$RES/summary.log"
}
( run q20_m6 "$G0"; run q20_m4 "$G0"; run q20_grow1 "$G0" ) &
( run q20_m5 "$G1"; run q20_grow3 "$G1" ) &
wait
log "Q20 LOCAL DONE"
