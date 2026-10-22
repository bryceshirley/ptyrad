#!/usr/bin/env python
"""chin_aa_fine_band — fine-band power ratios for the chin-antialias recon
arms, computed IDENTICALLY to ~/papers/paper_1/figs-p2/paper2_pso_fine_band.py:
depth-summed object phase at iteration 100, common centred crop across all
panels, annulus 0.3-0.6 Nyquist of |FFT(phase_sum - mean)|^2, quoted as a
ratio to the 20260927_lt21_ref_s42_full benchmark.

Run:  ~/ptyrad/.venv/bin/python scripts/chin_aa_fine_band.py
Writes chin_aa_fine_band.json next to this script.
"""

import glob
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).parent
BASE = Path("/home/dnz75396/ptyrad/demo/output/high_order_benchmarks")

RUNS = [
    ("LT 21 (benchmark)",
     BASE / "chin_results", "20260927_lt21_ref_s42_full*", "model_iter0100.hdf5"),
    ("Chin 4A, 11 slices (archived baseline)",
     BASE / "chin_bench_b1", "20260925_chin4a_matched_n100*", "model_iter0100.hdf5"),
    ("LT 11 (archived)",
     BASE / "chin_bench_b1", "20260925_lt11_n100*", "model_iter0100.hdf5"),
    ("Chin 4A + antialias, raw g (arm i)",
     BASE / "chin_aa", "*chin4a_aa_n100*", "model_iter0100.hdf5"),
    ("Chin 4A + antialias + band-limited g (arm ii)",
     BASE / "chin_aa", "*chin4a_aa_blg_n100*", "model_iter0100.hdf5"),
    ("Chin 4A + antialias, g off (arm iii)",
     BASE / "chin_aa", "*chin4a_aa_g0_n100*", "model_iter0100.hdf5"),
]


def load_phase_sum(rundir_glob, fname):
    dirs = sorted(glob.glob(str(rundir_glob)))
    if not dirs:
        return None, None, None
    d = dirs[0]
    try:
        f = h5py.File(Path(d) / fname, "r")
    except OSError:
        return None, None, str(d)
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
    fy = np.fft.fftshift(np.fft.fftfreq(ny)) * 2  # units of Nyquist
    fx = np.fft.fftshift(np.fft.fftfreq(nx)) * 2
    R = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return float(F[(R >= lo) & (R < hi)].sum())


def main():
    panels = []
    for label, base, pat, fname in RUNS:
        ps, dsname, rundir = load_phase_sum(base / pat, fname)
        if ps is None:
            print(f"{label}: MISSING ({base / pat})")
            continue
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
    for p, r in zip(panels, ratios):
        print(f"  fine-band (0.3-0.6 Nyq) ratio to benchmark: {p['label']}: {r:.3f}")

    json.dump({"runs": [{k: p[k] for k in ("label", "dataset", "run")}
                        for p in panels],
               "fine_band_ratios": ratios, "band_nyquist": [0.3, 0.6],
               "crop_shape": shp},
              open(HERE / "chin_aa_fine_band.json", "w"), indent=1)
    print(f"wrote {HERE / 'chin_aa_fine_band.json'}")


if __name__ == "__main__":
    main()
