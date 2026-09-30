"""Pure sampling arithmetic - no simulation.

Verifies that the L/gpts/crop/bin chain reproduces the target detector
calibration for the production sample, and that the reconstruction pixel
follows from the final da.
"""

import math

from generate_data.sampling import PRODUCTION_PLAN


def test_wavelength():
    assert math.isclose(PRODUCTION_PLAN.wavelength_A, 0.0250793, rel_tol=1e-4)


def test_extent():
    assert math.isclose(PRODUCTION_PLAN.extent_A, 92.31, rel_tol=1e-9)


def test_da_chain():
    """crop 512 + bin 4 on the 92.31 A / 1024 gpts grid -> 128 px at 1.0868 mrad."""
    plan = PRODUCTION_PLAN
    assert plan.det_npix == 128
    assert math.isclose(plan.native_da_mrad, 0.2717, abs_tol=2e-4)
    assert math.isclose(plan.da_mrad, 1.0868, abs_tol=5e-4)
    # detector edge ~ +-69.6 mrad
    assert math.isclose(plan.det_max_angle_mrad, 69.6, abs_tol=0.2)


def test_recon_dx_follows_from_da():
    plan = PRODUCTION_PLAN
    assert math.isclose(plan.recon_dx_A, 0.18029, abs_tol=5e-5)
    # explicit formula: la / (2 * 64 * da)
    dx = plan.wavelength_A / (2 * 64 * plan.da_mrad * 1e-3)
    assert math.isclose(plan.recon_dx_A, dx, rel_tol=1e-12)


def test_bf_disk_radius():
    assert math.isclose(PRODUCTION_PLAN.bf_radius_px, 27.6, abs_tol=0.1)


def test_antialias_exceeds_detector_edge():
    """The antialiased max angle of gpts=1024 must exceed the 70 mrad edge."""
    plan = PRODUCTION_PLAN
    assert plan.antialias_max_angle_mrad > 70.0
    assert math.isclose(plan.antialias_max_angle_mrad, 92.8, abs_tol=1.0)


def test_dx_sim():
    assert math.isclose(PRODUCTION_PLAN.dx_sim_A, 0.0901, abs_tol=2e-4)
