#!/usr/bin/env python
"""
Fine-band (0.3-0.6 Nyquist) power ratio of the depth-summed object phase for
the Chebyshev 3-slab PSO reconstruction vs the archived ladder, computed
EXACTLY as ~/papers/paper_1/figs-p2/paper2_pso_fine_band.py does (loader,
centred common crop, and band metric adapted verbatim; that file is not
modified). Ladder reference: LT21 benchmark = 1, chin4a 11sl ~ 0.88,
LT11 ~ 0.14.

Run: ~/ptyrad/.venv/bin/python scripts/cheb_fine_band.py [cheb_model.hdf5]
"""

import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

BASE = Path("/home/dnz75396/ptyrad/demo/output/high_order_benchmarks")
WORKTREE = Path(__file__).resolve().parent.parent
CHEB_GLOB = str(WORKTREE / "demo/output/chebyshev/*cheb3_matched_n50*/model_iter00*.hdf5")

RUNS = [
    ("LT 21 (benchmark)",
     BASE / "chin_results", "20260927_lt21_ref_s42_full*", "model_iter0100.hdf5"),
    ("Chin 4A, 11 slices",
     BASE / "chin_bench_b1", "20260925_chin4a_matched_n100*", "model_iter0100.hdf5"),
    ("LT 11",
     BASE / "chin_bench_b1", "20260925_lt11_n100*", "model_iter0100.hdf5"),
]


def load_phase_sum(rundir_glob, fname):
    d = sorted(glob.glob(str(rundir_glob)))[0]
    f = h5py.File(Path(d) / fname, "r")
    cand = {}
    f.visititems(lambda n, o: cand.update({n: o.shape})
                 if hasattr(o, "shape") and "objp" in n.lower()
                 and len(getattr(o, "shape", ())) >= 3 else None)
    name = sorted(cand)[0]
    arr = np.asarray(f[name])
    f.close()
    while arr.ndim > 3:
        arr = arr[0]
    return arr.sum(axis=0), name, str(d)


def fine_band_power(img, lo=0.3, hi=0.6):
    x = img - img.mean()
    F = np.fft.fftshift(np.abs(np.fft.fft2(x)) ** 2)
    ny, nx = x.shape
    fy = np.fft.fftshift(np.fft.fftfreq(ny)) * 2
    fx = np.fft.fftshift(np.fft.fftfreq(nx)) * 2
    R = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return float(F[(R >= lo) & (R < hi)].sum())


def main():
    cheb_files = sorted(glob.glob(sys.argv[1])) if len(sys.argv) > 1 else sorted(glob.glob(CHEB_GLOB))
    runs = list(RUNS)
    for cf in cheb_files:
        it = Path(cf).stem.split("iter")[-1]
        runs.append((f"Chebyshev 3 slabs (iter {int(it)})",
                     Path(cf).parent.parent, Path(cf).parent.name, Path(cf).name))

    panels = []
    for label, base, pat, fname in runs:
        ps, dsname, rundir = load_phase_sum(base / pat, fname)
        panels.append({"label": label, "phase_sum": ps, "dataset": dsname,
                       "run": Path(rundir).name})
        print(f"{label}: {ps.shape} from {Path(rundir).name} [{dsname}]")

    shp = tuple(min(p["phase_sum"].shape[i] for p in panels) for i in (0, 1))
    for p in panels:
        a = p["phase_sum"]
        oy, ox = (a.shape[0] - shp[0]) // 2, (a.shape[1] - shp[1]) // 2
        p["phase_sum"] = a[oy:oy + shp[0], ox:ox + shp[1]]

    powers = [fine_band_power(p["phase_sum"]) for p in panels]
    ratios = [pw / powers[0] for pw in powers]
    print(f"\ncommon crop: {shp}")
    for p, r in zip(panels, ratios):
        print(f"  fine-band (0.3-0.6 Nyq) ratio to benchmark: {p['label']}: {r:.3f}")

    out = WORKTREE / "demo/output/chebyshev/fine_band_ratios.json"
    json.dump({"runs": [{k: p[k] for k in ("label", "dataset", "run")} for p in panels],
               "fine_band_ratios": ratios, "band_nyquist": [0.3, 0.6],
               "crop_shape": shp}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
