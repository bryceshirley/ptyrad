#!/usr/bin/env bash
# POD-SIDE 20-iter quality round (owner spec): pmode 4, full 16384-DP Nz=42
# workload, born-order arms fixed M6/M5/M4 + grow-from-3 + grow-from-1, grow
# cap M=8. ONE GPU PER ARM (no contention => timings valid). Outputs kept
# (SAVE_ITERS 2: model hdf5 + objp tifs); tifs/PNGs sync home continuously,
# final model_iter0020.hdf5 pulled explicitly at the end.
# OWNER SPEC (final, 2026-10-12): ONE arm — grow-from-1 cap M=8, 20 iters,
# run as the production SLICE-2 split across BOTH GPUs (no fixed-M6 arm).
# Other configs are still generated; re-enable run lines if budget allows.
# NOTE: no same-workload 20-iter fixed-M6 reference exists for quality
# comparison — judge via loss trajectory, refit residuals, and tifs.
set -uo pipefail
cd /workspace/ptyrad/demo
export PTYRAD_DISABLE_ISS_PRECOND=1
RES="nvlink/results_redo_q20_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RES"
log(){ echo -e "\n=== $(date -u +%FT%TZ) $* ===" | tee -a "$RES/summary.log"; }
{ nvidia-smi topo -m | head -4; python -c "import torch; print(torch.__version__)"; } > "$RES/env.log" 2>&1

mk(){ sed "$2" params/p4_m6.yml | sed "s/: 'p4_m6',/: '$1',/" \
      | sed "s/'NITER'             : 1,/'NITER'             : 20,/" > "params/$1.yml"; }
mk q20_m6 ""
mk q20_m5 "s/'born_iterations'         : 6,/'born_iterations'         : 5,/"
mk q20_m4 "s/'born_iterations'         : 6,/'born_iterations'         : 4,/"
growsed(){ echo "s/'born_iterations'         : 6,/'born_iterations'         : $1,/;s/'grow_tol'     : null,/'grow_tol'     : 1.0e-2,/;s/'n_limit'      : null,/'n_limit'      : 8,/"; }
mk q20_grow3 "$(growsed 3)"
mk q20_grow1 "$(growsed 1)"

log "ARM q20_grow1 (slicedist SLICE_P=2, both GPUs)"
timeout 10800 env SLICE_P=2 python run_cmp_modesplit.py \
  slicedist params/q20_grow1.yml > "$RES/e2e_q20_grow1.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "ARM q20_grow1 exited rc=$rc" | tee -a "$RES/summary.log"
grep -E "Total Loss|Refit born_coeffs|Finished" "$RES/e2e_q20_grow1.log" | tail -25 \
  | tee -a "$RES/summary.log"
log "Q20 POD DONE"
