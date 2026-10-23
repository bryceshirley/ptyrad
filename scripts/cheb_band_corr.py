#!/usr/bin/env python
"""
Fine-band FIDELITY diagnostic: Pearson correlation of the band-filtered
(0.3-0.6 Nyquist, same annulus as the fine-band power metric) depth-summed
object phase against the LT21 benchmark, same loader and centred common
crop as scripts/cheb_fine_band.py.

The fine-band POWER ratio measures how much energy a reconstruction puts
into the band; an unregularised run can pump NOISE into the band and
inflate the ratio. The band-filtered correlation with the benchmark
separates structure from noise (noise decorrelates, power does not).

Run: ~/ptyrad/.venv/bin/python scripts/cheb_band_corr.py <model.hdf5 glob> ...
"""

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cheb_fine_band import BASE, load_phase_sum  # noqa: E402


def band_filter(img, lo=0.3, hi=0.6):
    x = img - img.mean()
    F = np.fft.fft2(x)
    ny, nx = x.shape
    fy = np.fft.fftfreq(ny) * 2
    fx = np.fft.fftfreq(nx) * 2
    R = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return np.fft.ifft2(F * ((R >= lo) & (R < hi))).real


def main():
    ref, _, refdir = load_phase_sum(BASE / "chin_results" / "20260927_lt21_ref_s42_full*",
                                    "model_iter0100.hdf5")
    files = []
    for pat in sys.argv[1:]:
        files += sorted(glob.glob(pat))
    print(f"benchmark: {Path(refdir).name}")
    for cf in files:
        ps, _, rundir = load_phase_sum(Path(cf).parent, Path(cf).name)
        shp = (min(ref.shape[0], ps.shape[0]), min(ref.shape[1], ps.shape[1]))
        crops = []
        for a in (ref, ps):
            oy, ox = (a.shape[0] - shp[0]) // 2, (a.shape[1] - shp[1]) // 2
            crops.append(a[oy:oy + shp[0], ox:ox + shp[1]])
        br, bp = band_filter(crops[0]), band_filter(crops[1])
        corr = float(np.sum(br * bp) / np.sqrt(np.sum(br**2) * np.sum(bp**2)))
        print(f"  band corr vs benchmark: {corr:+.4f}  "
              f"[{Path(rundir).name}/{Path(cf).name}]")


if __name__ == "__main__":
    main()
