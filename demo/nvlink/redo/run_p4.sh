#!/usr/bin/env bash
set -uo pipefail
cd /workspace/ptyrad/demo
export PTYRAD_DISABLE_ISS_PRECOND=1
RES="nvlink/results_redo_p4_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RES"
log(){ echo -e "\n=== $(date -u +%FT%TZ) $* ===" | tee -a "$RES/summary.log"; }
sed "s/'probe_pmode_max'        : 12,/'probe_pmode_max'        : 4,/" params/nvl42_ms.yml | sed "s/'prefix'            : 'nvl42_ms',/'prefix'            : 'p4_ms',/" > params/p4_ms.yml
sed "s/'probe_pmode_max'        : 12,/'probe_pmode_max'        : 4,/" params/nvl42_m6.yml | sed "s/'prefix'            : 'nvl42_m6',/'prefix'            : 'p4_m6',/" > params/p4_m6.yml
grep -c "probe_pmode_max.*: 4," params/p4_ms.yml params/p4_m6.yml >> "$RES/summary.log"
run(){ local arm="$1" role="$2" yml="$3" tmo="$4"; shift 4
  log "P4 $arm (role=$role, $yml)"
  timeout "$tmo" env "$@" python run_cmp_modesplit.py "$role" "$yml" > "$RES/e2e_$arm.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "ARM $arm exited rc=$rc" | tee -a "$RES/summary.log"
  grep -E "Total Loss|averaged iter_t" "$RES/e2e_$arm.log" | tail -3 | tee -a "$RES/summary.log"
}
run m6_slice4_p4 slicedist params/p4_m6.yml 1800 SLICE_P=4
run ms_1gpu_p4   baseline  params/p4_ms.yml 2700 CUDA_VISIBLE_DEVICES=0
run m6_1gpu_p4   baseline  params/p4_m6.yml 2700 CUDA_VISIBLE_DEVICES=0
log "P4 DONE"
