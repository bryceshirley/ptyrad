"""Sampling arithmetic for the deep-stack 4D-STEM simulations.

The chain is: lateral extent L fixes the native angular sampling la/L of the
far-field; cropping to `crop_px` native pixels and binning by `bin_factor`
yields the detector grid (`det_npix` px at `da_mrad` mrad/px).  The
reconstruction real-space pixel follows from the FINAL da, not from the
nominal detector cutoff: dx = la / (det_npix * da).
"""

from dataclasses import dataclass

import numpy as np
from abtem.core.energy import energy2wavelength


@dataclass(frozen=True)
class SamplingPlan:
    extent_A: float  # lateral cell size L (assumed square)
    gpts: int  # simulation grid points per side
    crop_px: int = 512  # native pixels kept in the far field (per side)
    bin_factor: int = 4  # binning applied after the crop
    kv: float = 200.0
    semiangle_mrad: float = 30.0
    # probe aberrations (Kirkland/abtem convention: positive defocus =
    # underfocus, i.e. C10 = -defocus); both the simulation probe and the
    # exported PtyRAD recon probe are built from these. Angles are abtem
    # phi_nm in RADIANS (abtem cos convention; the PtyRAD make_stem_probe
    # equivalents are f_a2=C12 @ theta_a2=phi12-pi/4, f_c3=C21 @
    # theta_c3=phi21-pi/2, f_a3=C23 @ theta_a3=phi23-pi/6 - verified by
    # complex correlation).
    defocus_A: float = 0.0
    c3_A: float = 0.0
    c12_A: float = 0.0  # twofold astigmatism
    phi12_rad: float = 0.0
    c21_A: float = 0.0  # coma
    phi21_rad: float = 0.0
    c23_A: float = 0.0  # threefold astigmatism
    phi23_rad: float = 0.0

    @property
    def aberrations(self) -> dict:
        return {
            "defocus": self.defocus_A,
            "C30": self.c3_A,
            "C12": self.c12_A,
            "phi12": self.phi12_rad,
            "C21": self.c21_A,
            "phi21": self.phi21_rad,
            "C23": self.c23_A,
            "phi23": self.phi23_rad,
        }

    @property
    def wavelength_A(self) -> float:
        return energy2wavelength(self.kv * 1e3)

    @property
    def dx_sim_A(self) -> float:
        """Real-space sampling of the simulation grid."""
        return self.extent_A / self.gpts

    @property
    def native_da_mrad(self) -> float:
        """Angular sampling of the raw far-field pixels: la/L."""
        return self.wavelength_A / self.extent_A * 1e3

    @property
    def det_npix(self) -> int:
        return self.crop_px // self.bin_factor

    @property
    def da_mrad(self) -> float:
        """Detector angular sampling after crop + bin."""
        return self.bin_factor * self.native_da_mrad

    @property
    def det_max_angle_mrad(self) -> float:
        """Detector edge (half-width) after the crop."""
        return (self.crop_px / 2) * self.native_da_mrad

    @property
    def antialias_max_angle_mrad(self) -> float:
        """abtem's antialias cutoff: 2/3 of the simulation Nyquist angle."""
        nyquist_mrad = self.wavelength_A / (2 * self.dx_sim_A) * 1e3
        return (2.0 / 3.0) * nyquist_mrad

    @property
    def bf_radius_px(self) -> float:
        """Bright-field disk radius on the binned detector."""
        return self.semiangle_mrad / self.da_mrad

    @property
    def recon_dx_A(self) -> float:
        """PtyRAD real-space pixel: la / (2 * (det_npix/2) * da)."""
        return self.wavelength_A / (self.det_npix * self.da_mrad * 1e-3)


A_SI = 5.43  # Si conventional lattice constant (Angstrom)

# Production Sample A: 17x17 conventional Si(001) cells laterally.
PRODUCTION_PLAN = SamplingPlan(extent_A=17 * A_SI, gpts=1024)

# Milestone-0 smoke sample: 11x11 Si cells laterally (~60 A).
SMOKE_PLAN = SamplingPlan(extent_A=11 * A_SI, gpts=1024)

# Twisted-stack plan (user request 2026-09-20, moderated 3x on feedback:
# "larger radius, clear aberrations" -> "more" -> "way too much" ->
# "slightly less"): same sampling as production, structured probe —
# 100 A underfocus + C3 = 3e4 A (a few soft rings, ~8 A diameter) with
# mild twofold astigmatism (25 A @ 30 deg) and coma (600 A @ 15 deg).
# Aperture-edge phases ~14 / 3.4 / 2.8 / 1.4 rad.
TWIST_PLAN = SamplingPlan(
    extent_A=17 * A_SI,
    gpts=1024,
    defocus_A=70.0,
    c3_A=1.5e4,
    c12_A=15.0,
    phi12_rad=np.deg2rad(30.0),
    c21_A=350.0,
    phi21_rad=np.deg2rad(15.0),
)

# Thick twisted stack (user request: "even thicker - does Born do even
# better?"): 256 slices x 3 A = 768 A. The 30 mrad exit cone then spans
# ~46 A, so the real-space window widens to 46.2 A by halving the binning
# (crop 512, bin 2 -> 256^2 detector @ 0.543 mrad) at the SAME recon dx
# (0.1803 A) - beam-containment marginality (alpha*t = half-window) is the
# same ratio dense128b/t ran at. Probe identical to TWIST_PLAN.
T256_PLAN = SamplingPlan(
    extent_A=17 * A_SI,
    gpts=1024,
    crop_px=512,
    bin_factor=2,
    defocus_A=70.0,
    c3_A=1.5e4,
    c12_A=15.0,
    phi12_rad=np.deg2rad(30.0),
    c21_A=350.0,
    phi21_rad=np.deg2rad(15.0),
)
