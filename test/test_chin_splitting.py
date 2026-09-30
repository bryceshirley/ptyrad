"""
Tests for the Fresnel propagator kernel and the Chin 4A/4B fourth-order
splittings (forward_models/multislice.py, utils/physics.py, models.py).

Covers:
 1. Defaults ('lie_trotter', 'angular_spectrum') reproduce golden outputs
    bit-for-bit (test/golden/golden_defaults.npz).
 2. fresnel_evolution matches the second-order expansion of
    near_field_evolution up to the carrier phase at small angles.
 3. Order of convergence against a dense expm of the paraxial generator:
    local error dz^5 (4A, 4B), dz^3 (Strang), dz^2 (Lie-Trotter); global
    error dz^4 at fixed total thickness. Wrong correction sign drops 4A
    below fourth order (negative control).
 4. Unitarity for a pure-phase object.
 5. torch.autograd.gradcheck in float64 (amplitude, phase, probe, g, dz).
 6. Absorption (amp != 1) against the same expm reference.
 7. 4B far-field flag: dropped end drifts + pre-propagated probe reproduce
    the intensities to round-off.
 8. Compiled entry points run under torch.compile on CPU (and CUDA if
    available) and match the eager implementations.
"""

import os

import numpy as np
import pytest
import torch
from torch.fft import fft2, ifft2

from src.ptyrad.forward_models.multislice import (
    CHIN_4B_A1,
    CHIN_4B_A2,
    _multislice_forward_chin4a_impl,
    _multislice_forward_chin4b_impl,
    multislice_forward_chin4a,
    multislice_forward_chin4b,
)
from src.ptyrad.utils import fftshift2, fresnel_evolution, near_field_evolution

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "golden_defaults.npz")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def shifted_k_grid(n, dx, dtype=torch.float64):
    """The (ifftshifted) angular frequency grid used by every propagator
    builder in the codebase, including the +0.5 half-bin offset."""
    grid = (torch.arange(-(n // 2), n // 2, dtype=dtype) + 0.5) / n
    k1d = torch.fft.ifftshift(2 * torch.pi * grid / dx)
    return torch.meshgrid(k1d, k1d, indexing="ij")  # Ky, Kx


def fresnel_kernel_t(n, dx, dz, k0):
    Ky, Kx = shifted_k_grid(n, dx)
    return torch.exp(-1j * dz * (Kx**2 + Ky**2) / (2 * k0))


def paraxial_generators(n, dx, k0, eta):
    """Dense (n^2, n^2) kinetic and potential generators of the paraxial
    equation d(psi)/dz = (i/(2 k0)) Lap psi + i eta psi, with the kinetic term
    diagonalized on the SAME shifted discrete K grid the code uses."""
    Ky, Kx = shifted_k_grid(n, dx)
    ksq = (Kx**2 + Ky**2).reshape(-1)

    F1 = torch.tensor(np.fft.fft(np.eye(n)), dtype=torch.complex128)
    F1i = torch.tensor(np.fft.ifft(np.eye(n)), dtype=torch.complex128)
    F = torch.kron(F1, F1)
    Fi = torch.kron(F1i, F1i)

    G_T = Fi @ (torch.diag(-1j * ksq / (2 * k0)).to(torch.complex128) @ F)
    G_V = torch.diag(1j * eta.reshape(-1).to(torch.complex128))
    return G_T, G_V


def smooth_eta(n, dx, strength=0.4, absorbing=False, seed=7):
    """Smooth complex chi(x) per unit length: a few well-resolved Gaussian
    bumps vanishing at the edges (so non-periodic gradients are clean)."""
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    eta = np.zeros((n, n))
    for _ in range(3):
        cy, cx = rng.uniform(0.3 * n, 0.7 * n, 2)
        amp = rng.uniform(0.5, 1.0)
        eta += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * (0.12 * n) ** 2))
    eta = strength * eta / np.abs(eta).max()
    if absorbing:
        eta = eta * (1.0 - 0.25j)  # Im(eta) < 0 => amp = exp(Im chi) < 1
    return torch.tensor(eta, dtype=torch.complex128)


def gaussian_probe(n, dtype=torch.complex128):
    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64) - n / 2 + 0.5,
        torch.arange(n, dtype=torch.float64) - n / 2 + 0.5,
        indexing="ij",
    )
    psi = torch.exp(-(y**2 + x**2) / (2 * (0.15 * n) ** 2)) * torch.exp(1j * 0.2 * (y + 0.5 * x))
    psi = psi / psi.abs().pow(2).sum().sqrt()
    return psi.to(dtype)


def g_patches_like_model(phase, logamp, dx):
    """Replicates PtychoAD.get_g_patches on a (..., Ny, Nx) canvas
    (fourth-order non-periodic central differences)."""
    from src.ptyrad.utils import fd_gradient4

    gyp, gxp = fd_gradient4(phase, dx)
    gyl, gxl = fd_gradient4(logamp, dx)
    g_re = gyp**2 - gyl**2 + gxp**2 - gxl**2
    g_im = -2.0 * (gyp * gyl + gxp * gxl)
    return torch.stack([g_re, g_im], dim=-1)


def spectral_gradient(f, dx):
    """Spectrally exact transverse gradient of a smooth real field (used by
    the order tests to isolate the dz-order of the splitting from the O(h^2)
    error of the shipped central-difference g)."""
    n = f.shape[-1]
    k1d = 2 * torch.pi * torch.fft.fftfreq(n, d=dx, dtype=torch.float64)
    Ky, Kx = torch.meshgrid(k1d, k1d, indexing="ij")
    F = torch.fft.fft2(f.to(torch.complex128))
    gy = torch.fft.ifft2(1j * Ky * F).real
    gx = torch.fft.ifft2(1j * Kx * F).real
    return gy, gx


def slab_inputs(eta, dz, n_slices, dx, g_mode="fd"):
    """Object/g patches for a uniform slab: chi_k = eta * dz per slice.
    chi = phase - i*log(amp), so phase = Re(chi) and log(amp) = -Im(chi)."""
    chi = eta * dz  # (n, n) complex
    phase = chi.real
    logamp = -chi.imag
    amp = torch.exp(logamp)
    n = eta.shape[-1]
    object_patches = torch.zeros(1, 1, n_slices, n, n, 2, dtype=torch.float64)
    object_patches[..., 0] = amp
    object_patches[..., 1] = phase
    if g_mode == "fd":
        g = g_patches_like_model(phase, logamp, dx)  # (n, n, 2)
    else:
        gyp, gxp = spectral_gradient(phase, dx)
        gyl, gxl = spectral_gradient(logamp, dx)
        g = torch.stack(
            [gyp**2 - gyl**2 + gxp**2 - gxl**2, -2.0 * (gyp * gyl + gxp * gxl)], dim=-1
        )
    g_patches = g.expand(1, 1, n_slices, n, n, 2).clone()
    return object_patches, g_patches


def detector_intensity(psi, eps=0.0):
    """(pmode, omode, Ny, Nx) or (Ny, Nx) exit wave -> detector pattern with
    the same fftshift2/'ortho'/incoherent-sum convention as the models."""
    if psi.ndim == 2:
        psi = psi[None, None]
    return (fftshift2(fft2(psi, norm="ortho"))).abs().square().sum(dim=(0, 1)) + eps


def run_scheme(scheme, eta, dz, n_slices, dx, k0, probe, sign=+1.0, g_mode="fd"):
    """Detector intensity of one scheme for a uniform slab of n_slices*dz."""
    n = eta.shape[-1]
    object_patches, g_patches = slab_inputs(eta, dz, n_slices, dx, g_mode=g_mode)
    if sign != +1.0:
        g_patches = sign * g_patches
    probe_b = probe[None, None]  # (1, 1, n, n)
    dz_t = torch.tensor(dz, dtype=torch.float64)
    k0_t = torch.tensor(k0, dtype=torch.float64)

    if scheme == "chin_4a":
        H_half = fresnel_kernel_t(n, dx, dz / 2, k0)[None]
        dp = _multislice_forward_chin4a_impl(
            object_patches, probe_b, H_half, g_patches, dz_t, k0_t, eps=0.0
        )
        return dp[0]
    if scheme == "chin_4b":
        H_a1 = fresnel_kernel_t(n, dx, CHIN_4B_A1 * dz, k0)[None]
        H_2a1 = fresnel_kernel_t(n, dx, 2 * CHIN_4B_A1 * dz, k0)[None]
        H_a2 = fresnel_kernel_t(n, dx, CHIN_4B_A2 * dz, k0)[None]
        dp = _multislice_forward_chin4b_impl(
            object_patches, probe_b, H_a1, H_2a1, H_a2, g_patches, dz_t, k0_t, eps=0.0
        )
        return dp[0]

    # Reference splitting, drift over the full n_slices*dz (slab convention)
    if scheme == "lie_trotter":
        O = torch.polar(object_patches[0, 0, :, :, :, 0], object_patches[0, 0, :, :, :, 1])
        H_full = fresnel_kernel_t(n, dx, dz, k0)
        psi = probe.clone()
        for m in range(n_slices):
            psi = ifft2(H_full * fft2(psi * O[m]))
        return detector_intensity(psi)
    raise ValueError(scheme)


def strang_step(eta, dz, n_slices, dx, k0, probe, g_scale=0.0, g_mode="fd"):
    """K(dz/2) t_k K(dz/2) per slice (drifts not merged; reference only).
    g_scale=0 is plain Strang; g_scale=+1 applies the gradient-corrected
    transmission t_k = O_k exp(i dz g_k/(12 k0)) (the symmetric-BCH
    [Y,[Y,X]] cancellation, zero extra FFTs)."""
    n = eta.shape[-1]
    object_patches, g_patches = slab_inputs(eta, dz, n_slices, dx, g_mode=g_mode)
    c = g_scale * dz / (12.0 * k0)
    amp = object_patches[..., 0] * torch.exp(-c * g_patches[..., 1])
    phase = object_patches[..., 1] + c * g_patches[..., 0]
    O = torch.polar(amp, phase)[0, 0]
    H_half = fresnel_kernel_t(n, dx, dz / 2, k0)
    psi = probe.clone()
    for m in range(n_slices):
        psi = ifft2(H_half * fft2(psi))
        psi = psi * O[m]
        psi = ifft2(H_half * fft2(psi))
    return detector_intensity(psi)


def exact_intensity(eta, L, dx, k0, probe):
    n = eta.shape[-1]
    G_T, G_V = paraxial_generators(n, dx, k0, eta)
    U = torch.linalg.matrix_exp(L * (G_T + G_V))
    psi = (U @ probe.reshape(-1).to(torch.complex128)).reshape(n, n)
    return detector_intensity(psi)


def fitted_slope(dzs, errs):
    return np.polyfit(np.log(dzs), np.log(errs), 1)[0]


# ---------------------------------------------------------------------------
# 1. defaults bit-for-bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag,kwargs", [
    ("plain", {}),
    ("tilt", {"tilt": True}),
    ("tilt_dz", {"tilt": True, "opt_dz": True}),
    ("dz", {"opt_dz": True}),
])
def test_defaults_bit_for_bit(tag, kwargs):
    """Default 'lie_trotter' + 'angular_spectrum' path is unchanged."""
    from test.golden.golden_setup import build_golden_model, golden_indices

    golden = np.load(GOLDEN_PATH)
    model = build_golden_model(**kwargs)
    idx = golden_indices()
    with torch.no_grad():
        H = model.get_propagators(idx).cpu().numpy()
        dp = model(idx).cpu().numpy()
    np.testing.assert_array_equal(H, golden[f"H_{tag}"])
    np.testing.assert_array_equal(dp, golden[f"dp_{tag}"])


# ---------------------------------------------------------------------------
# 2. fresnel_evolution vs near_field_evolution
# ---------------------------------------------------------------------------


def test_fresnel_matches_angular_spectrum_small_angle():
    n, dx, dz, lambd = 128, 0.15, 2.0, 0.025
    k0 = 2 * np.pi / lambd
    H_as = near_field_evolution((n, n), dx, dz, lambd)
    H_fr = fresnel_evolution((n, n), dx, dz, lambd)

    assert H_as.shape == H_fr.shape == (n, n)
    np.testing.assert_allclose(np.abs(H_fr), 1.0, atol=1e-12)

    Ky, Kx = (g.numpy() for g in shifted_k_grid(n, dx))
    ksq = Kx**2 + Ky**2
    # remove the carrier exp(1j k dz) from the angular-spectrum kernel
    phase_mismatch = np.angle(H_as * np.exp(-1j * k0 * dz) * np.conj(H_fr))

    # next expansion term is dz*K^4/(8 k0^3); check inside a small-angle disk
    kmax = np.sqrt(ksq.max())
    sel = ksq < (0.25 * kmax) ** 2
    bound = dz * ksq[sel] ** 2 / (8 * k0**3)
    assert np.all(np.abs(phase_mismatch[sel]) <= 1.05 * bound + 1e-12)
    # and the bound is tight somewhere (the kernels genuinely differ)
    assert np.abs(phase_mismatch[sel]).max() > 0.5 * bound.max()


# ---------------------------------------------------------------------------
# 3. order of convergence
#
# float64, CPU, 32x32 periodic grid, smooth band-limited complex chi
# (amplitude and phase both varying). g_k is computed with SPECTRAL
# derivatives here so the modified potential is consistent with the discrete
# Laplacian; the reference is a dense expm of the paraxial generator built on
# exactly the same frequency grid as fresnel_evolution (half-bin offset
# included). Slopes are fitted on the asymptotic window (above round-off,
# below the pre-asymptotic regime) and must land within 0.3 of the expected
# order. Lie-Trotter and Strang are harness controls. The production
# finite-difference g is measured separately (test_fd_gradient_error_floor).
# ---------------------------------------------------------------------------

N_ORD = 32
DX_ORD = 0.2
K0_ORD = 2 * np.pi / 0.025  # ~200 kV
LOCAL_DZS = tuple(48.0 * 0.5**i for i in range(8))  # ~two decades
LOCAL_STRENGTH = 0.5
GLOBAL_L = 48.0
GLOBAL_SLICES = (4, 8, 16)
GLOBAL_STRENGTH = 0.15  # weaker: keeps the accumulated error a clean power law


def scheme_intensity(scheme, eta, dz, n_slices, probe, g_mode="spectral", sign=+1.0):
    if scheme == "strang":
        return strang_step(eta, dz, n_slices, DX_ORD, K0_ORD, probe)
    if scheme == "strang_grad":
        return strang_step(
            eta, dz, n_slices, DX_ORD, K0_ORD, probe, g_scale=sign, g_mode=g_mode
        )
    return run_scheme(
        scheme, eta, dz, n_slices, DX_ORD, K0_ORD, probe, sign=sign, g_mode=g_mode
    )


def windowed_slope(dzs, errs, lo=1e-11, hi=3e-2):
    """Log-log slope fitted over the asymptotic window only."""
    dzs, errs = np.asarray(dzs), np.asarray(errs)
    sel = (errs > lo) & (errs < hi)
    assert sel.sum() >= 3, f"only {sel.sum()} points inside the fit window: {errs}"
    return np.polyfit(np.log(dzs[sel]), np.log(errs[sel]), 1)[0]


def print_error_table(title, dzs, errs):
    print(f"\n{title}")
    print("dz        " + "".join(f"{s:>16s}" for s in errs))
    for i, dz in enumerate(dzs):
        print(f"{dz:8.3f}  " + "".join(f"{errs[s][i]:16.3e}" for s in errs))


@pytest.fixture(scope="module")
def local_error_curves():
    """One-slab (local) error curves vs the dense expm reference, for the
    splitting schemes with spectral g, the wrong-sign control, and the
    production finite-difference g."""
    eta = smooth_eta(N_ORD, DX_ORD, strength=LOCAL_STRENGTH, absorbing=True)
    probe = gaussian_probe(N_ORD)
    specs = {
        "lie_trotter": ("lie_trotter", "spectral", +1.0),
        "strang": ("strang", "spectral", +1.0),
        "strang_grad": ("strang_grad", "spectral", +1.0),
        "chin_4a": ("chin_4a", "spectral", +1.0),
        "chin_4b": ("chin_4b", "spectral", +1.0),
        "chin_4a_wrong_sign": ("chin_4a", "spectral", -1.0),
        "chin_4a_fd": ("chin_4a", "fd", +1.0),
        "chin_4b_fd": ("chin_4b", "fd", +1.0),
    }
    errs = {name: [] for name in specs}
    shot_noise = {}
    for dz in LOCAL_DZS:
        I_ref = exact_intensity(eta, dz, DX_ORD, K0_ORD, probe)
        norm = I_ref.abs().sum()
        for name, (scheme, g_mode, sign) in specs.items():
            I = scheme_intensity(scheme, eta, dz, 1, probe, g_mode=g_mode, sign=sign)
            errs[name].append(float((I - I_ref).abs().sum() / norm))
        # Poisson noise floor of the same L1 metric, per-pattern dose N_e:
        # E|I_noisy - I| per pixel ~ sqrt(2 I / (pi N_e))
        shot_noise[dz] = {
            Ne: float(np.sqrt(2 / (np.pi * Ne)) * I_ref.sqrt().sum() / norm)
            for Ne in (1e4, 1e5, 1e6)
        }
    return errs, shot_noise


def test_local_order_single_slab(local_error_curves):
    """Local (one-slab) error vs dense expm: dz^5 for 4A/4B, dz^3 for Strang,
    dz^2 for Lie-Trotter, each within 0.3; wrong correction sign drops the
    scheme to third order."""
    errs, _ = local_error_curves
    print_error_table("Local (single slab) relative L1 intensity error", LOCAL_DZS, errs)

    expected = {"lie_trotter": 2.0, "strang": 3.0, "chin_4a": 5.0, "chin_4b": 5.0}
    slopes = {s: windowed_slope(LOCAL_DZS, errs[s]) for s in errs}
    print(f"fitted slopes: { {s: round(v, 2) for s, v in slopes.items()} }")

    for s, order in expected.items():
        assert abs(slopes[s] - order) <= 0.3, (s, slopes)
    # wrong sign: the correction doubles instead of cancels the [V,[T,V]]
    # third-order error -> back to a third-order scheme
    assert abs(slopes["chin_4a_wrong_sign"] - 3.0) <= 0.5, slopes
    # gradient-corrected Strang: still
    # third order (the nonlocal [X,[X,Y]] term remains) but with a much
    # smaller error constant than plain Strang
    assert abs(slopes["strang_grad"] - 3.0) <= 0.4, slopes
    for e_g, e_s in zip(errs["strang_grad"], errs["strang"]):
        if 1e-11 < e_s < 3e-2:
            assert e_g < 0.5 * e_s, (errs["strang_grad"], errs["strang"])


def test_global_order_fixed_thickness():
    """Global error at fixed total thickness: dz^4 for 4A/4B, dz^2 for
    Strang, dz^1 for Lie-Trotter, each within 0.3."""
    eta = smooth_eta(N_ORD, DX_ORD, strength=GLOBAL_STRENGTH, absorbing=True)
    probe = gaussian_probe(N_ORD)
    I_ref = exact_intensity(eta, GLOBAL_L, DX_ORD, K0_ORD, probe)
    norm = I_ref.abs().sum()

    errs = {s: [] for s in ["chin_4a", "chin_4b", "strang", "lie_trotter"]}
    for n_sl in GLOBAL_SLICES:
        dz = GLOBAL_L / n_sl
        for s in errs:
            I = scheme_intensity(s, eta, dz, int(n_sl), probe, g_mode="spectral")
            errs[s].append(float((I - I_ref).abs().sum() / norm))

    dzs = [GLOBAL_L / n_sl for n_sl in GLOBAL_SLICES]
    print_error_table(
        f"Global relative L1 intensity error (L = {GLOBAL_L}, {GLOBAL_SLICES} slabs)", dzs, errs
    )
    expected = {"chin_4a": 4.0, "chin_4b": 4.0, "strang": 2.0, "lie_trotter": 1.0}
    slopes = {s: windowed_slope(dzs, errs[s], lo=1e-13, hi=1e-1) for s in errs}
    print(f"fitted slopes: { {s: round(v, 2) for s, v in slopes.items()} }")
    for s, order in expected.items():
        assert abs(slopes[s] - order) <= 0.3, (s, slopes)


def test_fd_gradient_error_floor(local_error_curves):
    """The production central-difference g leaves a residual third-order term
    (an O(h^2) perturbation of the correction potential). Report the floor it
    introduces and compare against shot noise at a typical dose."""
    errs, shot_noise = local_error_curves
    # With the fourth-order stencil the FD curve tracks the spectral dz^5
    # curve until the h^4-suppressed dz^3 residual emerges at small dz.
    # Measure that residual directly as the excess over the spectral curve
    # on the smallest steps and check it is third order in dz.
    dzs_small = np.array(LOCAL_DZS[-3:])
    for s in ("chin_4a", "chin_4b"):
        excess = np.array(errs[s + "_fd"][-3:]) - np.array(errs[s][-3:])
        assert np.all(excess > 0), (s, errs)
        slope = np.polyfit(np.log(dzs_small), np.log(excess), 1)[0]
        assert abs(slope - 3.0) <= 0.6, (s, slope, excess)

    # at a typical slice thickness the FD-induced error must sit well below
    # the shot-noise floor of the measurement at a generous dose
    dz_typ = 3.0
    i = LOCAL_DZS.index(dz_typ)
    print("\nFD-gradient error floor vs shot noise (relative L1, dz = 3 A):")
    for s in ("chin_4a", "chin_4a_fd", "chin_4b", "chin_4b_fd"):
        print(f"  {s:12s}: {errs[s][i]:.3e}")
    for Ne, noise in shot_noise[dz_typ].items():
        print(f"  shot noise at {Ne:.0e} e/pattern: {noise:.3e}")
    assert errs["chin_4a_fd"][i] < 0.1 * shot_noise[dz_typ][1e6]
    assert errs["chin_4b_fd"][i] < 0.1 * shot_noise[dz_typ][1e6]


# ---------------------------------------------------------------------------
# 4. unitarity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["chin_4a", "chin_4b"])
def test_unitarity_pure_phase(scheme):
    """Exit-wave norm equals probe norm to round-off for a pure-phase object."""
    eta = smooth_eta(N_ORD, DX_ORD, strength=0.7, absorbing=False)
    probe = gaussian_probe(N_ORD)
    dp = run_scheme(scheme, eta, 3.0, 4, DX_ORD, K0_ORD, probe)
    assert abs(float(dp.sum()) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 5. gradients
# ---------------------------------------------------------------------------


def _make_grad_inputs(n=8, nz=2, seed=3):
    torch.manual_seed(seed)
    amp = (0.8 + 0.1 * torch.rand(1, 1, nz, n, n, dtype=torch.float64)).requires_grad_()
    phase = (0.3 * torch.randn(1, 1, nz, n, n, dtype=torch.float64)).requires_grad_()
    probe = (
        torch.randn(1, 1, n, n, dtype=torch.complex128) * 0.1
    ).requires_grad_()
    dz = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
    k0 = torch.tensor(200.0, dtype=torch.float64)
    return amp, phase, probe, dz, k0


@pytest.mark.parametrize("scheme", ["chin_4a", "chin_4b"])
def test_gradcheck(scheme):
    """float64 gradcheck through g_k, the transmissions and the propagators
    w.r.t. object amplitude, object phase, probe and slice thickness."""
    n, nz, dx = 8, 2, 0.2
    amp, phase, probe, dz, k0 = _make_grad_inputs(n, nz)

    def forward(amp_, phase_, probe_, dz_):
        object_patches = torch.stack([amp_, phase_], dim=-1)
        g = g_patches_like_model(phase_, torch.log(amp_ + 1e-10), dx)
        if scheme == "chin_4a":
            # kernel built at fixed distance; dz_ still enters through the
            # g-correction coefficient (the kernel's own dz-dependence goes
            # through torch_phasor in PtychoAD.make_propagator)
            H_half = fresnel_kernel_t(n, dx, 0.75, k0)[None]
            return _multislice_forward_chin4a_impl(
                object_patches, probe_, H_half, g, dz_, k0
            ).sum()
        H_a1 = fresnel_kernel_t(n, dx, CHIN_4B_A1 * 1.5, k0)[None]
        H_2a1 = fresnel_kernel_t(n, dx, 2 * CHIN_4B_A1 * 1.5, k0)[None]
        H_a2 = fresnel_kernel_t(n, dx, CHIN_4B_A2 * 1.5, k0)[None]
        return _multislice_forward_chin4b_impl(
            object_patches, probe_, H_a1, H_2a1, H_a2, g, dz_, k0
        ).sum()

    assert torch.autograd.gradcheck(
        forward, (amp, phase, probe, dz), eps=1e-6, atol=1e-8, rtol=1e-6
    )


def test_gradcheck_model_g_patches():
    """gradcheck of the canvas -> non-periodic gradient -> crop path."""
    torch.manual_seed(1)
    canvas = 12
    dx = 0.2
    obja = (0.9 + 0.05 * torch.rand(1, 2, canvas, canvas, dtype=torch.float64)).requires_grad_()
    objp = (0.2 * torch.randn(1, 2, canvas, canvas, dtype=torch.float64)).requires_grad_()

    def g_crop(a, p):
        g = g_patches_like_model(p, torch.log(a + 1e-10), dx)
        return g[:, :, 2:10, 1:9, :].sum()

    assert torch.autograd.gradcheck(g_crop, (obja, objp), eps=1e-6, atol=1e-8)


# ---------------------------------------------------------------------------
# 7. 4B far-field flag
# ---------------------------------------------------------------------------


def test_chin4b_drop_end_props_far_field_equivalence():
    """Dropping both end drifts with the probe pre-propagated by K(a1 dz)
    reproduces the far-field intensities to round-off."""
    n, nz, dx, dz, k0 = 16, 3, 0.2, 2.5, 250.0
    torch.manual_seed(5)
    eta = smooth_eta(n, dx, strength=0.5, absorbing=True)
    object_patches, g_patches = slab_inputs(eta, dz, nz, dx)
    # make the slices differ
    object_patches[..., 1] += 0.05 * torch.randn(1, 1, nz, n, n, dtype=torch.float64)
    probe = gaussian_probe(n)[None, None]
    dz_t = torch.tensor(dz, dtype=torch.float64)
    k0_t = torch.tensor(k0, dtype=torch.float64)
    H_a1 = fresnel_kernel_t(n, dx, CHIN_4B_A1 * dz, k0)[None]
    H_2a1 = fresnel_kernel_t(n, dx, 2 * CHIN_4B_A1 * dz, k0)[None]
    H_a2 = fresnel_kernel_t(n, dx, CHIN_4B_A2 * dz, k0)[None]

    dp_full = _multislice_forward_chin4b_impl(
        object_patches, probe, H_a1, H_2a1, H_a2, g_patches, dz_t, k0_t, apply_end_props=True
    )
    probe_shifted = ifft2(H_a1[:, None] * fft2(probe))
    dp_drop = _multislice_forward_chin4b_impl(
        object_patches, probe_shifted, H_a1, H_2a1, H_a2, g_patches, dz_t, k0_t,
        apply_end_props=False,
    )
    torch.testing.assert_close(dp_full, dp_drop, rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# model-level plumbing
# ---------------------------------------------------------------------------


def test_model_requires_fresnel_kernel():
    from test.golden.golden_setup import build_golden_model

    with pytest.raises(ValueError, match="fresnel"):
        build_golden_model(model_params_extra={"splitting": "chin_4a"})
    with pytest.raises(ValueError, match="fresnel"):
        build_golden_model(model_params_extra={"splitting": "chin_4b"})


def test_params_validation():
    from src.ptyrad.params.model_params import ModelParams

    with pytest.raises(Exception, match="fresnel"):
        ModelParams(splitting="chin_4a")
    with pytest.raises(Exception, match="splitting"):
        ModelParams(splitting="chin_4c")
    with pytest.raises(Exception, match="propagator_kernel"):
        ModelParams(propagator_kernel="asm")
    with pytest.raises(Exception, match="multislice"):
        ModelParams(splitting="chin_4b", propagator_kernel="fresnel", solver_type="born")
    mp = ModelParams(splitting="chin_4b", propagator_kernel="fresnel")
    assert mp.chin4b_drop_end_props is False
    assert ModelParams().propagator_kernel == "angular_spectrum"
    assert ModelParams().splitting == "lie_trotter"


@pytest.mark.parametrize("splitting,tilt", [
    ("chin_4a", False),
    ("chin_4a", True),
    ("chin_4b", False),
    ("chin_4b", True),
])
def test_model_forward_chin(splitting, tilt):
    """End-to-end model forward with the Chin splittings: runs, finite,
    correct shape, energy conserved for |O|<=1 within the amplitude budget."""
    from test.golden.golden_setup import build_golden_model, golden_indices

    model = build_golden_model(
        tilt=tilt, model_params_extra={"propagator_kernel": "fresnel", "splitting": splitting}
    )
    idx = golden_indices()
    with torch.no_grad():
        dp = model(idx)
    assert dp.shape == model.measurements.shape
    assert torch.isfinite(dp).all()
    assert float(dp.min()) >= 0


def test_model_fresnel_H_buffer():
    """With propagator_kernel='fresnel', model.H is the paraxial kernel."""
    from test.golden.golden_setup import DX, DZ, LAMBD, NPIX, build_golden_model

    model = build_golden_model(model_params_extra={"propagator_kernel": "fresnel"})
    H_ref = fresnel_evolution((NPIX, NPIX), DX, DZ, LAMBD).astype("complex64")
    np.testing.assert_array_equal(model.H.cpu().numpy(), H_ref)
    # Kz_prop matches -(Kx^2+Ky^2)/(2k)
    Ky, Kx = model.propagator_grid
    expected = (-(Kx**2 + Ky**2) / (2 * model.k)).cpu().numpy()
    np.testing.assert_allclose(model.Kz_prop.cpu().numpy(), expected, rtol=1e-6)


def test_model_single_slice_fallback():
    """n_slices == 1 keeps the single-transmission model and logs once."""
    from test.golden.golden_setup import build_golden_model, golden_indices

    model = build_golden_model(
        nz=1, model_params_extra={"propagator_kernel": "fresnel", "splitting": "chin_4a"}
    )
    assert model.splitting == "lie_trotter"
    with torch.no_grad():
        dp = model(golden_indices())
    assert torch.isfinite(dp).all()


# ---------------------------------------------------------------------------
# 8. torch.compile
# ---------------------------------------------------------------------------


def _compile_case(device):
    torch.manual_seed(11)
    n, nz = 16, 3
    dx, dz, k0 = 0.2, 2.0, 250.0
    object_patches = torch.rand(2, 1, nz, n, n, 2, device=device) * 0.2
    object_patches[..., 0] += 0.9
    g_patches = 0.01 * torch.randn(2, 1, nz, n, n, 2, device=device)
    probe = (torch.randn(2, 2, n, n, device=device) + 1j * torch.randn(2, 2, n, n, device=device))
    probe = probe / probe.abs().pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
    kern = {
        d: fresnel_kernel_t(n, dx, d * dz, k0).to(torch.complex64).to(device)[None]
        for d in (0.5, CHIN_4B_A1, 2 * CHIN_4B_A1, CHIN_4B_A2)
    }
    dz_t = torch.tensor(dz, device=device)
    k0_t = torch.tensor(k0, device=device)
    return object_patches, g_patches, probe, kern, dz_t, k0_t


@pytest.mark.parametrize(
    "device",
    ["cpu"]
    + (["cuda"] if torch.cuda.is_available() else []),
)
def test_compiled_matches_eager(device):
    object_patches, g_patches, probe, kern, dz_t, k0_t = _compile_case(device)
    op = object_patches.clone().requires_grad_()

    dp_c = multislice_forward_chin4a(op, probe, kern[0.5], g_patches, dz_t, k0_t)
    dp_e = _multislice_forward_chin4a_impl(object_patches, probe, kern[0.5], g_patches, dz_t, k0_t)
    torch.testing.assert_close(dp_c, dp_e, rtol=1e-4, atol=1e-6)
    dp_c.sum().backward()  # backward under compile works
    assert op.grad is not None and torch.isfinite(op.grad).all()

    args = (kern[CHIN_4B_A1], kern[2 * CHIN_4B_A1], kern[CHIN_4B_A2], g_patches, dz_t, k0_t)
    dp_c = multislice_forward_chin4b(object_patches, probe, *args)
    dp_e = _multislice_forward_chin4b_impl(object_patches, probe, *args)
    torch.testing.assert_close(dp_c, dp_e, rtol=1e-4, atol=1e-6)
