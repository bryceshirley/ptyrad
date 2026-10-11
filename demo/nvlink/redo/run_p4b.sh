#!/usr/bin/env bash
set -uo pipefail
cd /workspace/ptyrad/demo
export PTYRAD_DISABLE_ISS_PRECOND=1
RES=$(ls -td nvlink/results_redo_p4_* | head -1)
log(){ echo -e "\n=== $(date -u +%FT%TZ) $* ===" | tee -a "$RES/summary.log"; }
log "P4 m6_slice2_p4 (role=slicedist, params/p4_m6.yml)"
timeout 2400 env SLICE_P=2 python run_cmp_modesplit.py slicedist params/p4_m6.yml > "$RES/e2e_m6_slice2_p4.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "ARM m6_slice2_p4 exited rc=$rc" | tee -a "$RES/summary.log"
grep -E "Total Loss|averaged iter_t" "$RES/e2e_m6_slice2_p4.log" | tail -3 | tee -a "$RES/summary.log"
log "P4B DONE"
