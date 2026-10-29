"""
Analytic (no-autograd) adjoint gradients and the ePIE-preconditioned update
for the Chebyshev exact-generator multislice model (chebyshev.py).

The forward applies, per slab, U = sum_k c_k T_k(Btilde) with the three-term
recurrence t_{k+1} = M t_k - t_{k-1}, M = 2 Btilde. Autodiff through that
recurrence stores every t_k (degree M fields per slab). Here the backward
pass stores nothing: the slab operator is the exponential exp(i k0 dz
(sqrt(1+B)-1)) evaluated to the setup tolerance, so the exact reverse-mode
gradient is obtained by an adjoint solve of the same kind, using two facts.

1. The Chebyshev recurrence is time-reversible, t_{k-1} = M t_k - t_{k+1},
   so the forward states are reconstructed on the fly during the backward
   sweep from the last two, (t_{M-1}, t_M), checkpointed per slab.
2. The reverse-mode cotangents of the recurrence satisfy the SAME three-term
   recurrence run downward with the conjugate coefficients as sources,

       lam_M     = conj(c_M) lam,
       lam_{M-1} = conj(c_{M-1}) lam + M^H lam_M,
       lam_k     = conj(c_k) lam + M^H lam_{k+1} - lam_{k+2},   k = M-2 .. 1,
       lam_0     = conj(c_0) lam + 0.5 M^H lam_1 - lam_2,

   where M^H is M with eps conjugated (mu and the band mask are real), i.e.
   the adjoint solve. lam_0 is the exact input cotangent p(Btilde)^H lam, and
   the object gradient is the diagonal of the operator cotangent,

       dL/d(eps_t)(r) ~ sum_{k=2}^{M} conj[ifft2 lam_k](r) [ifft2 t_{k-1}](r)
                        + 0.5 conj[ifft2 lam_1](r) [ifft2 t_0](r),

   summed over probe modes (the Ny*Nx Parseval factor of the unnormalized
   transforms is applied by chebyshev_vjp; the ePIE update absorbs it into
   beta). Both sweeps run down in k together, so each degree costs two FFT
   pairs (one per sweep) whose real-space fields are exactly the factors the
   diagonal needs -- the backward is ~2x the forward cost and O(1) memory
   per slab (vs O(degree) for autodiff).

The reverse reconstruction excites the same |rho|^k growth modes as the
forward recurrence, so its roundoff is bounded by the same field-of-values
margin the setup's caps guarantee; test_chebyshev_adjoint.py checks the
gradients against autograd in float64 (machine precision) and float32.

cheb_epie_update is the preconditioned gradient-descent step: the detector
modulus projection seeds the adjoint solve (mixed-state ePIE ratio, shared
across modes), and each slab's phase/amplitude step is divided by the peak
incident intensity of that slab's entrance field, max_r sum_p |psi_in|^2, as
in ePIE/3PIE (pie.py); the probe step is divided by the peak occupancy-
weighted squared transmission amplitude (the slab operators are near-unitary,
so this is ~1).

Compilation follows the production forward's pattern: the pointwise work of
both sweeps is fused by torch.compile on real views (chebyshev._cheb_step
forward, _cheb_adj_step backward -- inductor has no complex codegen, and
whole-update torch.compile measured SLOWER than eager at batch 32 for that
reason), the FFTs stay eager, and cheb_epie_update wraps the whole update in
a CUDA graph (capture per shape, replay per batch). PTYRAD_PIE_EAGER=1 keeps
everything eager.
"""

import os

import torch
from torch.fft import fft2, ifft2

try:
    from torchvision.transforms.functional import gaussian_blur
except ImportError:  # torchvision unavailable: use the local separable blur
    from ptyrad.utils import gaussian_blur_2d as gaussian_blur

from ptyrad.utils import fftshift2

_TINY = 1e-12


def _apply_M(t, mu_t, band_mask, eps_t, conj_eps=False):
    """One application of M = 2*Btilde (or its adjoint M^H with conj_eps):
    mu_t and band_mask are real, so the adjoint only conjugates eps_t."""
    e = eps_t.conj() if conj_eps else eps_t
    return mu_t * t + band_mask * fft2(e * ifft2(t))


def _slab_eps(object_patches, s0, s1, k0_dz_slab, a, b):
    """Scaled slab multiplier eps_t = a*eps + b and nm1, as in the forward."""
    nm1 = (
        object_patches[:, :, s0:s1, ..., 1].sum(2)
        - 1j * object_patches[:, :, s0:s1, ..., 0].clamp_min(1e-10).log().sum(2)
    ) / k0_dz_slab
    return (a * nm1 * (nm1 + 2.0) + b)[:, None], nm1  # (N,1,omode,Ny,Nx), (N,omode,Ny,Nx)


def chebyshev_forward_ckpt(
    object_patches, probe, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab, groups
):
    """
    Chebyshev multislice forward (same arithmetic as
    multislice_forward_chebyshev up to the detector) that checkpoints, per
    slab, only the entrance field and the last two recurrence states.

    Returns:
        (T, eps_ts, nm1s, ckpts): T is the band-limited k-space exit wave
        (N, pmode, omode, Ky, Kx); ckpts[g] = (t_in, t_{M-1}, t_M).
    """
    from ptyrad.forward_models.chebyshev import _cheb_step

    object_patches = object_patches.contiguous()
    probe = probe.contiguous()
    a, b = 4.0 / (hi - lo), -2.0 * (hi + lo) / (hi - lo)
    mu_t = a * mu_mul
    mu_b, band_b = mu_t.unsqueeze(-1), band_mask.unsqueeze(-1)
    cre, cim = coeffs.real.contiguous(), coeffs.imag.contiguous()
    T = band_mask * fft2(probe[:, :, None])  # (N, pmode, 1, Ky, Kx)
    eps_ts, nm1s, ckpts = [], [], []
    for s0, s1 in groups:
        eps_t, nm1 = _slab_eps(object_patches, s0, s1, k0_dz_slab, a, b)
        t0, t1 = T, 0.5 * _apply_M(T, mu_t, band_mask, eps_t)
        acc = coeffs[0] * t0 + coeffs[1] * t1
        t0r, t1r = torch.view_as_real(t0.contiguous()), torch.view_as_real(t1.contiguous())
        accr = torch.view_as_real(acc.contiguous())
        for k in range(2, coeffs.shape[0]):
            convr = torch.view_as_real(fft2(eps_t * ifft2(torch.view_as_complex(t1r))))
            t2r, accr = _cheb_step(mu_b, t1r, band_b, convr, t0r, accr, cre[k], cim[k])
            t0r, t1r = t1r, t2r
        eps_ts.append(eps_t)
        nm1s.append(nm1)
        # entrance, t_{M-1}, t_M (t_0, t_1 when M = 1)
        ckpts.append((T, torch.view_as_complex(t0r), torch.view_as_complex(t1r)))
        T = torch.view_as_complex(accr)
    return T, eps_ts, nm1s, ckpts


# Fused pointwise combine of ONE reverse-sweep degree, on float real views
# (shape (..., 2)), mirroring chebyshev._cheb_step: inductor cannot fuse
# complex pointwise (it falls back to eager), so the adjoint recurrence, the
# reversed forward recurrence and the W accumulation run as real arithmetic
# in a single memory pass per degree. The FFTs stay outside; this coexists
# with CUDA-graph capture (cheb_epie_graphed).
@torch.compile(dynamic=False)
def _cheb_adj_step(
    mu_b, band_b, lam_sr, lam_kr, lam_kp1r, conv_lr, t_km1r, t_kr, conv_tr,
    lam_rr, t_rr, Wr, ccre, ccim,
):
    # lam_km1 = cc*lam_seed + mu lam_k + band conv_l - lam_kp1
    sre, sim = lam_sr[..., 0], lam_sr[..., 1]
    lam_km1_re = ccre * sre - ccim * sim + mu_b[..., 0] * lam_kr[..., 0] \
        + band_b[..., 0] * conv_lr[..., 0] - lam_kp1r[..., 0]
    lam_km1_im = ccre * sim + ccim * sre + mu_b[..., 0] * lam_kr[..., 1] \
        + band_b[..., 0] * conv_lr[..., 1] - lam_kp1r[..., 1]
    # t_km2 = mu t_km1 + band conv_t - t_k (reversed forward recurrence)
    t_km2r = mu_b * t_km1r + band_b * conv_tr - t_kr
    # W += conj(lam_r) * t_r, summed over pmode (dim 1)
    lre, lim = lam_rr[..., 0], lam_rr[..., 1]
    tre, tim = t_rr[..., 0], t_rr[..., 1]
    w_re = Wr[..., 0] + (lre * tre + lim * tim).sum(1)
    w_im = Wr[..., 1] + (lre * tim - lim * tre).sum(1)
    return (
        torch.stack((lam_km1_re, lam_km1_im), dim=-1),
        t_km2r,
        torch.stack((w_re, w_im), dim=-1),
    )


def chebyshev_adjoint(lam, eps_ts, ckpts, mu_mul, band_mask, coeffs, lo, hi):
    """
    Reverse sweep: adjoint solve + reversed forward recurrence per slab.

    Args:
        lam: cotangent of the k-space exit wave T (dL = 2 Re <lam, dT>).

    Returns:
        (lam_in, Ws): lam_in is the exact cotangent of the band-limited
        k-space entrance field band*fft2(probe); Ws[g] is the pmode-summed
        eps_t cotangent diagonal, (N, omode, Ny, Nx), WITHOUT the Ny*Nx
        Parseval factor of the unnormalized transforms (chebyshev_vjp applies
        it; the ePIE update absorbs it into beta).
    """
    a = 4.0 / (hi - lo)
    mu_t = a * mu_mul
    M = coeffs.shape[0] - 1
    # resolve_conj(): .imag of a lazy conj view is a neg-bit view, and
    # torch.compile DROPS the neg bit on such scalar inputs (sign flip)
    cc = coeffs.conj().resolve_conj()
    ccre, ccim = cc.real.contiguous(), cc.imag.contiguous()
    mu_b, band_b = mu_t.unsqueeze(-1), band_mask.unsqueeze(-1)  # (Ky, Kx, 1)
    # the forward masks after the eps convolution, so the true adjoint masks
    # BEFORE it; with an in-band seed the whole sweep stays in-band and the
    # trailing mask below is exact. Out-of-band seed content never pairs with
    # anything (dT is in-band), so project it out once here.
    lam = band_mask * lam
    Ws = [None] * len(ckpts)
    for g in range(len(ckpts) - 1, -1, -1):
        eps_t = eps_ts[g]
        eps_c = eps_t.conj()
        _, t_km1, t_k = ckpts[g]
        lam_k = cc[M] * lam
        t_k = t_k + torch.zeros_like(lam_k)  # omode-broadcast to match lam
        t_km1 = t_km1 + torch.zeros_like(lam_k)
        lam_sr = torch.view_as_real(lam.contiguous())
        lam_kr = torch.view_as_real(lam_k.contiguous())
        lam_kp1r = torch.zeros_like(lam_kr)
        t_kr = torch.view_as_real(t_k.contiguous())
        t_km1r = torch.view_as_real(t_km1.contiguous())
        W = torch.zeros(lam.shape[:1] + lam.shape[2:] + (2,),
                        dtype=lam_kr.dtype, device=lam.device)
        for j in range(M, 1, -1):
            lam_r = ifft2(torch.view_as_complex(lam_kr))  # lam_j, real space
            t_r = ifft2(torch.view_as_complex(t_km1r))  # t_{j-1}, real space
            conv_l = torch.view_as_real(fft2(eps_c * lam_r))
            conv_t = torch.view_as_real(fft2(eps_t * t_r))
            lam_km1r, t_km2r, W = _cheb_adj_step(
                mu_b, band_b, lam_sr, lam_kr, lam_kp1r, conv_l, t_km1r, t_kr,
                conv_t, torch.view_as_real(lam_r), torch.view_as_real(t_r),
                W, ccre[j - 1], ccim[j - 1],
            )
            lam_kp1r, lam_kr = lam_kr, lam_km1r
            t_kr, t_km1r = t_km1r, t_km2r
        lam_k = torch.view_as_complex(lam_kr)
        lam_r = ifft2(lam_k)  # lam_1 in real space
        t_r = ifft2(torch.view_as_complex(t_km1r))  # t_0 in real space
        W0 = 0.5 * (lam_r.conj() * t_r).sum(1)
        Ws[g] = torch.view_as_complex(W.contiguous()) + W0
        lam = cc[0] * lam + 0.5 * (mu_t * lam_k + band_mask * fft2(eps_c * lam_r)) \
            - torch.view_as_complex(lam_kp1r)
    return lam, Ws


def chebyshev_vjp(
    object_patches, probe, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab, groups, lam,
    forward_state=None,
):
    """
    Exact reverse-mode gradients of 2 Re <lam, T> w.r.t. the object
    amplitude/phase slices and the probe, with no autograd tape.

    Args:
        lam: cotangent of the k-space exit wave T (dL = 2 Re <lam, dT>).
        forward_state: optional (eps_ts, nm1s, ckpts) from
            chebyshev_forward_ckpt to skip re-running the forward.

    Returns:
        (grad_obja, grad_objp, grad_probe): real tensors matching
        object_patches[..., 0/1] (dL = sum(grad * d(param))), and the complex
        probe cotangent with dL = 2 Re <grad_probe, d(probe)> (i.e. the
        gradient on a view_as_real probe leaf is (2*Re, 2*Im) of it).
    """
    if forward_state is None:
        _, eps_ts, nm1s, ckpts = chebyshev_forward_ckpt(
            object_patches, probe, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab, groups
        )
    else:
        eps_ts, nm1s, ckpts = forward_state
    lam_in, Ws = chebyshev_adjoint(lam, eps_ts, ckpts, mu_mul, band_mask, coeffs, lo, hi)
    npix = lam.shape[-2] * lam.shape[-1]
    a = 4.0 / (hi - lo)
    grad_obja = torch.zeros_like(object_patches[..., 0])
    grad_objp = torch.zeros_like(object_patches[..., 1])
    for g, (s0, s1) in enumerate(groups):
        # chain eps_t = a*eps + b, eps = nm1*(nm1+2), nm1 = (sum phase
        # - i sum log amp)/theta through the Wirtinger pair dL = 2 Re[W deps_t]
        Q = (npix * Ws[g]) * (2.0 * a / k0_dz_slab) * (nm1s[g] + 1.0)
        grad_objp[:, :, s0:s1] = (2.0 * Q.real)[:, :, None].expand(-1, -1, s1 - s0, -1, -1)
        amp = object_patches[:, :, s0:s1, ..., 0]
        g_amp = (2.0 * Q.imag)[:, :, None] / amp.clamp_min(1e-10)
        grad_obja[:, :, s0:s1] = torch.where(amp >= 1e-10, g_amp, 0.0)
    grad_probe = npix * ifft2(lam_in).sum(2)  # probe is broadcast over omode
    if probe.shape[0] == 1 and grad_probe.shape[0] > 1:  # ... and over positions
        grad_probe = grad_probe.sum(0, keepdim=True)
    return grad_obja, grad_objp, grad_probe


def _cheb_epie(
    object_patches, probes, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab, groups,
    occu, meas, beta_obj, beta_probe, update_probe, eps, detector_blur_std=None,
):
    """ePIE-preconditioned analytic gradient step for the Chebyshev model.

    One forward (checkpointing slab boundaries), the mixed-state modulus
    projection at the detector, one adjoint solve, and the preconditioned
    writes. The seed lam = T*(ratio - 1) is the (negative) amplitude-loss
    gradient, so adding the preconditioned adjoint quantities is a descent
    step; the constant chain factors (2a/theta, Ny*Nx) are absorbed into the
    betas, while the ePIE normalisation max_r sum_p |psi_in|^2 carries the
    position-dependent conditioning."""
    object_patches = object_patches.contiguous()
    probes = probes.contiguous()
    T, eps_ts, nm1s, ckpts = chebyshev_forward_ckpt(
        object_patches, probes, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab, groups
    )
    scale = 1.0 / (T.shape[-2] * T.shape[-1])
    dp_model = (
        fftshift2(torch.sum(T.abs().square() * occu.view(1, 1, -1, 1, 1), dim=(1, 2)))
        * scale + eps
    )
    # The seed uses the UNBLURRED intensities even when the data term blurs
    # the model DP (models.get_forward_meas): the raw modulus seed is
    # self-bounded, |T * r| <= sqrt(meas * scale), because T is small exactly
    # where the ratio is huge, while a blur between the ratio and the T
    # multiplication smears dark-field ratios onto bright pixels and
    # diverges at batch 1 (measured). Ignoring the blur in the DIRECTION is
    # a second-order bias; the cap is residual insurance.
    ratio = torch.sqrt(meas.clamp_min(0.0) / dp_model).clamp(max=10.0)
    lam = T * (fftshift2(ratio)[:, None, None] - 1.0)
    if detector_blur_std:
        # returned blurred so the logged loss matches the autodiff path
        dp_model = gaussian_blur(dp_model, kernel_size=[5, 5], sigma=detector_blur_std)

    _lam_in, Ws = chebyshev_adjoint(lam, eps_ts, ckpts, mu_mul, band_mask, coeffs, lo, hi)

    a = 4.0 / (hi - lo)
    amp = object_patches[..., 0]
    phase_new = object_patches[..., 1].clone()
    amp_new = amp.clone()
    for g, (s0, s1) in enumerate(groups):
        psi_in = ifft2(ckpts[g][0])  # real-space incident field of the slab
        den = psi_in.abs().square().sum(1).amax(dim=(-2, -1), keepdim=True)
        Q = Ws[g] * (2.0 * a / k0_dz_slab) * (nm1s[g] + 1.0)
        dphase = beta_obj * 2.0 * Q.real / (den + _TINY)
        damp = beta_obj * 2.0 * Q.imag / (den + _TINY)
        phase_new[:, :, s0:s1] += dphase[:, :, None]
        amp_new[:, :, s0:s1] += damp[:, :, None] / amp[:, :, s0:s1].clamp_min(1e-10)
    # amp/phase stack, NOT a complex field: the chebyshev object is natively
    # amp/phase and the write-back must not wrap the phase through angle()
    obj_new = torch.stack([amp_new.clamp_min(_TINY), phase_new], dim=-1)

    dP = None
    if update_probe:
        A2 = amp.clamp_min(1e-10).square().prod(dim=2)  # squared transmission amplitude
        den_p = (
            (A2 * occu.view(1, -1, 1, 1)).sum(1).amax(dim=(-2, -1), keepdim=True)[:, None]
        )
        dP = beta_probe * ifft2((_lam_in * occu.view(1, 1, -1, 1, 1)).sum(2)) / (den_p + _TINY)

    return dp_model, obj_new, dP


def _cheb_grads(
    object_patches, probes, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab, groups,
    occu, meas, eps, detector_blur_std=None,
):
    """EXACT gradients of the (optionally detector-blurred) amplitude loss
    L = sum_r (sqrt(blur(dp)) - sqrt(meas))^2 w.r.t. the object amp/phase
    patches and the per-position probes -- the same loss the autodiff path
    differentiates, computed by the adjoint solve instead of a tape. For
    feeding a torch optimizer (update_rule='adjoint'): unlike the ePIE step,
    the seed keeps the full Wirtinger chain (1 - r at the detector, carried
    back through the symmetric blur; torchvision's reflect padding makes the
    blur adjoint approximate in a 2-pixel detector border).

    Returns (dp_model, grad_obja, grad_objp, grad_probe): dp_model blurred
    for loss logging; grad_probe complex per position with the convention
    that the gradient on a view_as_real probe leaf is view_as_real(2*it)."""
    object_patches = object_patches.contiguous()
    probes = probes.contiguous()
    T, eps_ts, nm1s, ckpts = chebyshev_forward_ckpt(
        object_patches, probes, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab, groups
    )
    scale = 1.0 / (T.shape[-2] * T.shape[-1])
    dp_model = (
        fftshift2(torch.sum(T.abs().square() * occu.view(1, 1, -1, 1, 1), dim=(1, 2)))
        * scale + eps
    )
    if detector_blur_std:
        dp_model = gaussian_blur(dp_model, kernel_size=[5, 5], sigma=detector_blur_std)
    g_I = 1.0 - torch.sqrt(meas.clamp_min(0.0) / dp_model)  # dL/d(blurred dp)
    if detector_blur_std:
        g_I = gaussian_blur(g_I, kernel_size=[5, 5], sigma=detector_blur_std)
    lam = fftshift2(g_I)[:, None, None] * occu.view(1, 1, -1, 1, 1) * scale * T
    grad_obja, grad_objp, grad_probe = chebyshev_vjp(
        object_patches, probes, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab,
        groups, lam, forward_state=(eps_ts, nm1s, ckpts),
    )
    return dp_model, grad_obja, grad_objp, grad_probe


class _CudaGraphedGrads:
    """CUDA-graph capture/replay of _cheb_grads, keyed on shapes (same
    pattern and output conventions as _CudaGraphedEpie)."""

    def __init__(self):
        self._cache = {}

    def __call__(self, object_patches, probes, mu_mul, band_mask, coeffs, lo, hi,
                 k0_dz_slab, groups, occu, meas, eps, detector_blur_std=None):
        if not object_patches.is_cuda:
            return _cheb_grads(object_patches, probes, mu_mul, band_mask, coeffs,
                               lo, hi, k0_dz_slab, groups, occu, meas, eps,
                               detector_blur_std)
        key = (tuple(object_patches.shape), tuple(probes.shape), detector_blur_std)
        entry = self._cache.get(key)
        if entry is None:
            s_obj = object_patches.detach().clone()
            s_pr = probes.detach().clone().contiguous()
            s_meas = meas.detach().clone()
            run = lambda: _cheb_grads(  # noqa: E731
                s_obj, s_pr, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab,
                groups, occu, s_meas, eps, detector_blur_std,
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                s_out = run()
            entry = self._cache[key] = (graph, s_obj, s_pr, s_meas, s_out)
        graph, s_obj, s_pr, s_meas, s_out = entry
        s_obj.copy_(object_patches)
        s_pr.copy_(probes)
        s_meas.copy_(meas)
        graph.replay()
        return s_out


class _CudaGraphedEpie:
    """CUDA-graph capture/replay of _cheb_epie, keyed on shapes and betas.

    The pointwise work is already fused by the torch.compile'd _cheb_step /
    _cheb_adj_step (whole-update torch.compile is NOT used: inductor has no
    complex codegen, so it falls back to eager complex ops and loses to this
    path -- measured slower than plain eager at batch 32). The capture
    removes the per-op launch overhead of the long recurrences, exactly like
    make_graphed_callables on the autodiff forward (models.py). Betas are
    baked into the capture; outputs are static buffers, consumed within the
    batch by pie.py (same convention as its compiled updates)."""

    def __init__(self):
        self._cache = {}

    def __call__(self, object_patches, probes, mu_mul, band_mask, coeffs, lo, hi,
                 k0_dz_slab, groups, occu, meas, beta_obj, beta_probe,
                 update_probe, eps, detector_blur_std=None):
        if not object_patches.is_cuda:
            return _cheb_epie(object_patches, probes, mu_mul, band_mask, coeffs,
                              lo, hi, k0_dz_slab, groups, occu, meas,
                              beta_obj, beta_probe, update_probe, eps,
                              detector_blur_std)
        key = (tuple(object_patches.shape), tuple(probes.shape),
               float(beta_obj), float(beta_probe), bool(update_probe),
               detector_blur_std)
        entry = self._cache.get(key)
        if entry is None:
            s_obj = object_patches.detach().clone()
            s_pr = probes.detach().clone().contiguous()
            s_meas = meas.detach().clone()
            run = lambda: _cheb_epie(  # noqa: E731
                s_obj, s_pr, mu_mul, band_mask, coeffs, lo, hi, k0_dz_slab,
                groups, occu, s_meas, beta_obj, beta_probe, update_probe, eps,
                detector_blur_std,
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):  # warmup (also triggers _cheb_*_step compiles)
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                s_out = run()
            entry = self._cache[key] = (graph, s_obj, s_pr, s_meas, s_out)
        graph, s_obj, s_pr, s_meas, s_out = entry
        s_obj.copy_(object_patches)
        s_pr.copy_(probes)
        s_meas.copy_(meas)
        graph.replay()
        return s_out


# Entry points: CUDA-graphed fused-step update / gradients (production
# pattern); PTYRAD_PIE_EAGER=1 keeps the plain eager arithmetic (identical
# numbers up to kernel fusion).
if os.environ.get("PTYRAD_PIE_EAGER", "0") == "1":
    cheb_epie_update = _cheb_epie
    cheb_adjoint_grads = _cheb_grads
else:
    cheb_epie_update = _CudaGraphedEpie()
    cheb_adjoint_grads = _CudaGraphedGrads()
