"""The exported probe: k-space aperture measures 30 +- 0.5 mrad (radial
profile half-max) and its real-space intensity is normalized."""

import numpy as np

from generate_data.sampling import PRODUCTION_PLAN
from generate_data.simulate import make_recon_probe


def test_probe_aperture_and_norm():
    plan = PRODUCTION_PLAN
    probe = make_recon_probe(plan)

    # real-space intensity normalized to 1
    assert abs((np.abs(probe) ** 2).sum() - 1.0) < 1e-5

    # k-space aperture radius: half-max area recipe (same as the BF-disk
    # measurement in single_cbed.py; robust to radial-bin truncation)
    pk = np.fft.fftshift(np.fft.fft2(probe))
    intensity = np.abs(pk) ** 2
    c = plan.det_npix // 2
    plateau = intensity[c - 4 : c + 5, c - 4 : c + 5].mean()
    r_eff = np.sqrt((intensity > 0.5 * plateau).sum() / np.pi)
    measured_mrad = r_eff * plan.da_mrad
    assert abs(measured_mrad - plan.semiangle_mrad) <= 0.5, measured_mrad


def test_probe_centered():
    probe = make_recon_probe(PRODUCTION_PLAN)
    intensity = np.abs(probe) ** 2
    n = probe.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    com_y = (yy * intensity).sum()
    com_x = (xx * intensity).sum()
    assert abs(com_y - n // 2) < 1.0
    assert abs(com_x - n // 2) < 1.0
