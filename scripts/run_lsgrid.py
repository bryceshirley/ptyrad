"""
Exact-line-search advantage grid on tBL-WSe2 (Nguyen et al.), geometry of the
paper's Figs 3/4/6.  STEP 2 of the overnight job.

Three engines, each reconstructing the SAME data / init / seed:
    A  ISS  + Adam            (python -m ptyrad run, solver_type=born)
    B  multislice  + Adam            (python -m ptyrad run, solver_type=multislice)
    C  ISS  + EXACT line search (run_linesearch.py, ls_damp=0.5 FIXED)

Grid (total sample thickness held fixed at TOTAL_THK, subdivided into N slices --
finer slicing, NOT more sample):
    N   in {2, 4, 8, 12, 24}   at |B| = 32
    |B| in {1, 8, 32, 128}     at N   = 12
Order: (N=12,|B|=32) first, then the N sweep, then the |B| sweep. The shared
(N=12,|B|=32) cell is run once.

Prediction under test: the fixed Adam step is capped by the Gershgorin bound
alpha < 2/N, so as N grows engine A must shrink its step and stalls; the exact
step is uncapped, so C's advantage over A should GROW with N and be absent at N=2.

Controls:
  * same dataset, same random_seed (=> same init object/probe), same probe
    geometry and same GPU (RTX A4000, gpuid 0) for every cell.
  * Adam lr is tuned PER CELL by a short scan (never tuned once and reused) --
    engines A and B only; C has no lr (steps come from ls_damp/alpha/beta).
  * ls_damp is FIXED at 0.5 for every C cell. If a cell needs a different
    ls_damp to be stable, that is recorded as a finding, not silently changed.

TARGET (defined BEFORE any curve is inspected): per cell, target_loss =
(1 + TARGET_TOL) * min_over_iters(loss). iterations-to-target = first iteration
whose loss <= target_loss; wall-to-target = cumulative iter_time at that point.
Also reported: sec/iter (mean iter_time), final loss at fixed iteration budgets
and at fixed wall-clock budgets.

    python run_lsgrid.py            # run the grid
    python run_lsgrid.py --dry      # only generate params files + print the plan

Writes incrementally to  ~/draft_paper/figs-lsgrid/lsgrid_results.json  and echoes
one line per cell. Reconstruction outputs land under demo/output/lsgrid/<tag>/.
"""
import os
import re
import sys
import glob
import json
import time
import subprocess

import numpy as np
import h5py

DEMO = "/home/dnz75396/ptyrad/demo"
PY = sys.executable
OUTROOT = f"{DEMO}/output/lsgrid"
RESULTDIR = "/home/dnz75396/draft_paper/figs-lsgrid"
os.makedirs(OUTROOT, exist_ok=True)
os.makedirs(RESULTDIR, exist_ok=True)
RESULTS = os.path.join(RESULTDIR, "lsgrid_results.json")

DRY = "--dry" in sys.argv

TOTAL_THK = 12.0            # Angstrom, held fixed across N (demo tBL-WSe2 geometry)
SEED = 42
# 2026-09-14 revision (user decision after measuring 61 s/iter on the full
# 16384-pattern set): keep the full dataset, run fewer iterations; trim the lr
# scan; rescale time budgets (old 30-240 s was below one iteration); drop the
# |B|=1 cell.
NITER = 40                  # iteration cap (was 100)
NITER_TUNE = 10             # short scan to pick the per-cell Adam lr (was 30)
SAVE_ITERS = NITER          # one final checkpoint carrying the full loss_iters curve
LR_GRID = [5.0e-4, 2.0e-3]  # (was 4 values)
TARGET_TOL = 0.005          # within 0.5% of the per-cell best loss
ITER_BUDGETS = [10, 20, 40]
TIME_BUDGETS = [120, 300, 900, 1800]   # seconds
LS_DAMP = 0.5               # FIXED for engine C

# engine -> (base params file stem, solver_type, launcher, tune_lr)
ENGINES = {
    "A_born_adam":  ("born",       "born",       "cli",        True),
    "B_ms_adam":    ("multislice", "multislice", "cli",        True),
    "C_born_ls":    ("born",       "born",       "linesearch", False),
}

# cell plan, in execution order, deduplicated on (N, B)
def build_cells():
    seen, cells = set(), []
    def add(N, B):
        if (N, B) not in seen:
            seen.add((N, B)); cells.append((N, B))
    add(12, 32)                       # anchor first
    for N in [2, 4, 8, 12, 24]:       # N sweep at |B|=32
        add(N, 32)
    for B in [8, 32, 128]:            # |B| sweep at N=12 (B=1 dropped: ~14 h
        add(12, B)                    # per Adam engine at 16384 updates/iter)
    return cells

CELLS = build_cells()


def make_params(stem, tag, N, B, niter, lr, outsub):
    """Write a params yml derived from the minimal config, return its path."""
    src = open(f"{DEMO}/params/tBL_WSe2_reconstruct_minimal_{stem}.yml").read()
    dz = TOTAL_THK / N
    y = src
    y = re.sub(r"('obj_Nlayer'\s*:\s*)12", rf"\g<1>{N}", y)
    y = re.sub(r"('obj_slice_thickness'\s*:\s*)1\b", rf"\g<1>{dz}", y)
    y = re.sub(r"('NITER'\s*:\s*)100", rf"\g<1>{niter}", y)
    y = re.sub(r"('SAVE_ITERS'\s*:\s*)10", rf"\g<1>{niter}", y)
    y = re.sub(r"('size'\s*:\s*)32", rf"\g<1>{B}", y)
    if lr is not None:
        y = re.sub(r"('obja'\s*:\s*\{'start_iter'\s*:\s*1,\s*'lr'\s*:\s*)[0-9.eE+-]+",
                   rf"\g<1>{lr}", y)
        y = re.sub(r"('objp'\s*:\s*\{'start_iter'\s*:\s*1,\s*'lr'\s*:\s*)[0-9.eE+-]+",
                   rf"\g<1>{lr}", y)
    # random seed: inject into init_params (add if absent)
    if re.search(r"'random_seed'", y):
        y = re.sub(r"('random_seed'\s*:\s*)[0-9]+", rf"\g<1>{SEED}", y)
    else:
        y = y.replace("init_params : {",
                      "init_params : {\n    'random_seed' : %d," % SEED, 1)
    # output dir
    lines = []
    for ln in y.splitlines(keepends=True):
        if ln.lstrip().startswith("'output_dir'"):
            ind = ln[:len(ln) - len(ln.lstrip())]
            ln = ind + "'output_dir': 'output/lsgrid/%s',\n" % outsub
        lines.append(ln)
    y = "".join(lines)
    pp = f"{DEMO}/params/lsgrid_{tag}.yml"
    open(pp, "w").write(y)
    return pp


def read_curve(outsub):
    """Return (losses, iter_times) from the newest model*.hdf5 under outsub."""
    pat = f"{OUTROOT}/{outsub}/**/model*.hdf5"
    files = sorted(glob.glob(pat, recursive=True), key=os.path.getmtime)
    if not files:
        return None, None
    with h5py.File(files[-1], "r") as f:
        li = f["loss_iters"][()] if "loss_iters" in f else None
        it = f["iter_times"][()] if "iter_times" in f else None
    if li is None:
        return None, None
    li = np.asarray(li, dtype=float)
    losses = li[:, 1] if li.ndim == 2 else li
    itimes = np.asarray(it, dtype=float) if it is not None else None
    return losses, itimes


def launch(engine, pp, tag):
    stem, solver, launcher, _ = ENGINES[engine]
    log = f"{OUTROOT}/{tag}.log"
    t0 = time.time()
    if launcher == "cli":
        cmd = f"{PY} -m ptyrad run --params_path {pp} --gpuid 0 --seed {SEED} > {log} 2>&1"
        rc = subprocess.call(cmd, shell=True, cwd=DEMO)
    else:
        env = dict(os.environ, PTYRAD_LS_PARAMS=pp, PTYRAD_LS_RUNNAME=tag)
        cmd = f"{PY} run_linesearch.py > {log} 2>&1"
        rc = subprocess.call(cmd, shell=True, cwd=f"{DEMO}/scripts", env=env)
    return rc, time.time() - t0, log


def metrics(losses, itimes):
    losses = np.asarray(losses, dtype=float)
    best = float(np.nanmin(losses))
    target = (1 + TARGET_TOL) * best
    hit = np.where(losses <= target)[0]
    it_to_target = int(hit[0] + 1) if len(hit) else None
    if itimes is not None and len(itimes) == len(losses):
        ctime = np.cumsum(itimes)
        wall_to_target = float(ctime[hit[0]]) if len(hit) else None
        sec_per_iter = float(np.mean(itimes))
        loss_at_time = {}
        for T in TIME_BUDGETS:
            within = np.where(ctime <= T)[0]
            loss_at_time[str(T)] = float(np.nanmin(losses[within])) if len(within) else None
    else:
        ctime = None
        wall_to_target = sec_per_iter = None
        loss_at_time = {str(T): None for T in TIME_BUDGETS}
    loss_at_iter = {str(k): (float(losses[k - 1]) if k <= len(losses) else None)
                    for k in ITER_BUDGETS}
    return dict(best_loss=best, target_loss=target, iters_to_target=it_to_target,
                wall_to_target_s=wall_to_target, sec_per_iter=sec_per_iter,
                final_loss=float(losses[-1]), n_iter=len(losses),
                loss_at_iter=loss_at_iter, loss_at_time=loss_at_time,
                loss_curve=[float(v) for v in losses],
                iter_times=[float(v) for v in itimes] if itimes is not None else None)


def tune_lr(engine, N, B):
    """Short per-cell scan; return the lr with the lowest tuning loss."""
    stem = ENGINES[engine][0]
    best_lr, best_loss = None, np.inf
    scan = {}
    for lr in LR_GRID:
        tag = f"{engine}_N{N}_B{B}_tune_lr{lr:.0e}"
        outsub = tag
        pp = make_params(stem, tag, N, B, NITER_TUNE, lr, outsub)
        if DRY:
            scan[f"{lr:.0e}"] = None
            continue
        rc, dt, log = launch(engine, pp, tag)
        losses, _ = read_curve(outsub)
        fl = float(np.nanmin(losses)) if losses is not None else np.inf
        scan[f"{lr:.0e}"] = fl if np.isfinite(fl) else None
        if np.isfinite(fl) and fl < best_loss:
            best_loss, best_lr = fl, lr
    if best_lr is None:
        best_lr = 5.0e-4
    return best_lr, scan


def load_results():
    fresh = dict(meta=dict(total_thk_A=TOTAL_THK, seed=SEED, niter=NITER,
                           niter_tune=NITER_TUNE, lr_grid=LR_GRID,
                           target_tol=TARGET_TOL, ls_damp=LS_DAMP,
                           gpu="NVIDIA RTX A4000", cells_order=CELLS),
                 cells={})
    if os.path.exists(RESULTS):
        old = json.load(open(RESULTS))
        # config guard: never mix cells from a different grid configuration
        om = old.get("meta", {})
        if (om.get("niter") == NITER and om.get("niter_tune") == NITER_TUNE
                and om.get("lr_grid") == LR_GRID):
            return old
        stale = RESULTS + ".stale"
        os.replace(RESULTS, stale)
        print("[config changed] previous results moved to %s" % stale)
    return fresh


def main():
    print("=== line-search advantage grid: %d cells x 3 engines ===" % len(CELLS))
    print("cell order (N,B):", CELLS)
    res = load_results()
    for (N, B) in CELLS:
        for engine in ENGINES:
            stem, solver, launcher, tune = ENGINES[engine]
            key = f"{engine}_N{N}_B{B}"
            if key in res["cells"] and res["cells"][key].get("done"):
                print("[skip] %s already done" % key); continue
            lr, scan = (tune_lr(engine, N, B) if tune else (None, None))
            tag = key
            pp = make_params(stem, tag, N, B, NITER, lr, tag)
            if DRY:
                print("[dry] %s  N=%d B=%d dz=%.3fA lr=%s params=%s"
                      % (key, N, B, TOTAL_THK / N, lr, os.path.basename(pp)))
                continue
            rc, dt, log = launch(engine, pp, tag)
            losses, itimes = read_curve(tag)
            if losses is None:
                cell = dict(done=False, rc=rc, wall_s=round(dt, 1), lr=lr,
                            lr_scan=scan, note="no loss curve found; see log",
                            log=log)
                print("[FAIL] %s rc=%d wall=%.0fs (no curve)" % (key, rc, dt))
            else:
                m = metrics(losses, itimes)
                cell = dict(done=True, rc=rc, wall_s=round(dt, 1), N=N, B=B,
                            engine=engine, solver=solver, dz_A=TOTAL_THK / N,
                            lr=lr, lr_scan=scan, ls_damp=(LS_DAMP if not tune else None),
                            log=log, **m)
                print("[ok] %s | lr=%s | best=%.4e iters->tgt=%s wall->tgt=%s "
                      "s/iter=%s final=%.4e"
                      % (key, lr, m["best_loss"], m["iters_to_target"],
                         m["wall_to_target_s"], m["sec_per_iter"], m["final_loss"]))
            res["cells"][key] = cell
            json.dump(res, open(RESULTS, "w"), indent=1)
    print("=== grid complete -> %s ===" % RESULTS)


if __name__ == "__main__":
    main()
