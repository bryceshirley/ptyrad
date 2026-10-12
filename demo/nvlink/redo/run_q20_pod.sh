#!/usr/bin/env bash
# POD-SIDE 20-iter quality round (owner spec): pmode 4, full 16384-DP Nz=42
# workload, born-order arms fixed M6/M5/M4 + grow-from-3 + grow-from-1, grow
# cap M=8. ONE GPU PER ARM (no contention => timings valid). Outputs kept
# (SAVE_ITERS 2: model hdf5 + objp tifs); tifs/PNGs sync home continuously,
# final model_iter0020.hdf5 pulled explicitly at the end.
# BUDGET CUT (owner, 2026-10-12): only the two decision arms run —
#   GPU0: m6 (reference)   GPU1: grow1 cap 8      wall ~2.7 h on A100 SXM.
# m5/m4/grow3 configs are still generated; re-enable their run lines if the
# budget ever allows the full five.
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

run(){ local arm="$1" gpu="$2" tmo="$3"
  log "ARM $arm on GPU $gpu"
  timeout "$tmo" env CUDA_VISIBLE_DEVICES="$gpu" python run_cmp_modesplit.py \
    baseline "params/$arm.yml" > "$RES/e2e_$arm.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && echo "ARM $arm exited rc=$rc" | tee -a "$RES/summary.log"
  grep -E "Total Loss|Finished" "$RES/e2e_$arm.log" | tail -3 | tee -a "$RES/summary.log"
}
run q20_m6 0 12000 &
run q20_grow1 1 11000 &
wait
log "Q20 POD DONE"
