#!/usr/bin/env bash
# POD-SIDE round-4 arms (bench_results SSO.2): born order at pmode 4, slice-2,
# full 16384-DP Nz=42 workload. Derives per-arm params from params/p4_m6.yml.
# Ran 2026-10-11 on 2x A100 SXM NV12 (pod 7l2rfw58c91bw3), torch 2.14.1+cu130.
set -uo pipefail
cd /workspace/ptyrad/demo
export PTYRAD_DISABLE_ISS_PRECOND=1
RES="nvlink/results_redo_p4round2_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RES"
log(){ echo -e "\n=== $(date -u +%FT%TZ) $* ===" | tee -a "$RES/summary.log"; }
{ nvidia-smi topo -m | head -4; python -c "import torch; print(torch.__version__)"; } > "$RES/env.log" 2>&1
mk(){ sed "$2" params/p4_m6.yml | sed "s/: 'p4_m6',/: '$1',/" > "params/$1.yml"; }
mk p4r_m6s2 ""
mk p4r_m5s2 "s/'born_iterations'         : 6,/'born_iterations'         : 5,/"
mk p4r_m4s2 "s/'born_iterations'         : 6,/'born_iterations'         : 4,/"
sed -e "s/'born_iterations'         : 6,/'born_iterations'         : 3,/" \
    -e "s/'grow_tol'     : null,/'grow_tol'     : 1.0e-2,/" \
    -e "s/'n_limit'      : null,/'n_limit'      : 6,/" \
    -e "s/'NITER'             : 1,/'NITER'             : 3,/" \
    -e "s/: 'p4_m6',/: 'p4r_grows2',/" params/p4_m6.yml > params/p4r_grows2.yml
run(){ local arm="$1" tmo="$2"
  log "ARM $arm"
  timeout "$tmo" env SLICE_P=2 python run_cmp_modesplit.py slicedist "params/$arm.yml" > "$RES/e2e_$arm.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "ARM $arm exited rc=$rc" | tee -a "$RES/summary.log"
  grep -E "Total Loss|averaged iter_t|Refit born_coeffs" "$RES/e2e_$arm.log" | tail -8 | tee -a "$RES/summary.log"
}
run p4r_m6s2 1500
run p4r_m5s2 1500
run p4r_m4s2 1500
run p4r_grows2 3300
log "P4ROUND2 DONE"
