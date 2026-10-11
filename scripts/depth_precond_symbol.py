"""Verification and diagnostics for the depth-coupled ML-ISS object
preconditioner (ptyrad.mliss.depth_symbol_lags / depth_precond_*).

GATE (--gate): builds a small ISS problem from the project's own forward
code (project k-grid conventions copied verbatim from
models.PtychoAD.create_grids / init_propagator_vars, project probe via
utils.physics.make_stem_probe, field response via mliss.direction_response),
then

  1. checks <r, L x> == <L^dag r, x> for random complex x, r (adjoint),
  2. reads M(Q) off brute-force L^dag L applied to a random basis of
     slice stacks (dense scan on every pixel, periodic wrap) and compares,
     per propagator kernel ('fresnel' AND the default 'angular_spectrum'),
     against
       (a) the engine's exact finite-grid symbol (depth_symbol_lags), and
       (b) the continuum Fresnel closed form
               M_D(Q) = C exp(-i pi a_D |Q|^2) Gamma_P(a_D Q),
           with the quadratic AND linear coefficients of the per-slice
           propagator phase extracted from the project's H itself (the
           project's propagator k-grid carries a half-pixel offset, i.e. a
           linear phase term; see extract_quadratic_phase), Gamma_P
           evaluated both exactly (trigonometric polynomial) and by the
           padded-FFT sinc interpolation at padding factors PF and PF/2.

REAL (--real): tBL-WSe2 geometry (N = 12, dz = 1 A, 80 kV, 24.9 mrad,
the 6 reconstructed probe modes from the finished ML-ISS run): eigenvalues
of M(Q)/m0 and the effective depth-channel count vs |Q|.

Run with the repo venv from ~/ptyrad:
  TORCHDYNAMO_DISABLE=1 ../.venv/bin/python scripts/depth_precond_symbol.py --gate
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptyrad.mliss import depth_symbol_lags, direction_response, unscattered_illumination
from ptyrad.utils import torch_phasor
from ptyrad.utils.physics import make_stem_probe

OUTDIR = os.path.expanduser("~/ptyrad/demo/output/depth_precond/results/symbol_verification")


# --------------------------------------------------------------------------- #
# Small ISS problem with the project's conventions                             #
# --------------------------------------------------------------------------- #

def project_grids(Npix, dx, device, dtype=torch.float64):
    """Transverse k grid, copied verbatim (up to dtype) from
    models.PtychoAD.create_grids: NOTE the half-pixel offset (+0.5) — the
    project's propagator grid has no q = 0 sample."""
    ygrid = (torch.arange(-Npix // 2, Npix // 2, device=device, dtype=dtype) + 0.5) / Npix
    ky = torch.fft.ifftshift(2 * torch.pi * ygrid / dx)
    Ky, Kx = torch.meshgrid(ky, ky, indexing="ij")
    return Ky, Kx


def project_propagator(Npix, dx, dz, lambd, kernel, device):
    """Single-slice propagator h, same math as PtychoAD.init_propagator_vars
    + get_propagators (angular_spectrum: exp(i dz Kz), Kz = sqrt(k^2-Kr^2);
    fresnel: exp(-i dz Kr^2 / (2k)))."""
    Ky, Kx = project_grids(Npix, dx, device)
    k = 2 * torch.pi / lambd
    if kernel == "fresnel":
        Kz_prop = -(Kx**2 + Ky**2) / (2 * k)
    else:
        Kz_prop = torch.sqrt(torch.clamp(k**2 - Kx**2 - Ky**2, min=0.0))
    return torch_phasor(dz * Kz_prop)  # complex128 from float64 input


def small_problem(Npix=32, Nz=6, pmode=1, dz=1.0, dx=0.1532, kv=80.0, conv=24.9,
                  kernel="fresnel", device="cuda", seed=0):
    lambd = 12.398 / np.sqrt((2 * 511.0 + kv) * kv)  # project formula
    h = project_propagator(Npix, dx, dz, lambd, kernel, device)
    H = torch.stack([h**j for j in range(Nz)], dim=0).view(1, 1, 1, Nz, Npix, Npix)
    probes = []
    for m in range(pmode):
        p = make_stem_probe(
            {"kv": kv, "conv_angle": conv, "Npix": Npix, "dx": dx, "df": 20.0 * m},
            verbose=False,
        )
        probes.append(torch.as_tensor(np.asarray(p, dtype=np.complex128), device=device))
    probe = torch.stack(probes, 0).unsqueeze(0)  # (1, pmode, Ny, Nx)
    probe = probe / probe.abs().square().sum().sqrt()
    return dict(Npix=Npix, Nz=Nz, pmode=pmode, dz=dz, dx=dx, lambd=lambd, conv=conv,
                kernel=kernel, h=h, H=H, probe=probe, device=device)


def all_shifts(Npix, device):
    py, px = torch.meshgrid(torch.arange(Npix, device=device),
                            torch.arange(Npix, device=device), indexing="ij")
    return py.reshape(-1), px.reshape(-1)


def apply_L(prob, x, chunk=256):
    """L x: detector fields for the dense scan (every canvas pixel, wrap).
    x: (Nz, N, N) complex canvas stack -> (P, pmode, N, N). The per-view
    response is the PROJECT's own: mliss.direction_response (one window)."""
    Npix, Nz = prob["Npix"], prob["Nz"]
    py, px = all_shifts(Npix, x.device)
    out = []
    for i in range(0, py.numel(), chunk):
        sy, sx = py[i:i + chunk], px[i:i + chunk]
        # window at scan position p sees canvas pixel p + y at frame coord y
        wins = torch.stack(
            [torch.roll(x, shifts=(-int(a), -int(b)), dims=(-2, -1)) for a, b in zip(sy, sx)]
        ).unsqueeze(1)  # (B, omode=1, Nz, N, N)
        D = direction_response(None, wins, prob["probe"], prob["H"])  # (B,pmode,1,N,N)
        out.append(D.squeeze(2))
    return torch.cat(out, 0)


def apply_Ldag(prob, r, chunk=256):
    """L^dag r, hand-written adjoint of apply_L (unnormalised torch fft2:
    adjoint of fft2 is N^2 * ifft2; adjoint of *conj(H_j) is *H_j; adjoint
    of *phi_j is *conj(phi_j); adjoint of the window gather is scatter)."""
    Npix, Nz = prob["Npix"], prob["Nz"]
    phi = unscattered_illumination(prob["probe"], prob["H"])  # (1,pmode,1,Nz,N,N)
    py, px = all_shifts(Npix, r.device)
    acc = torch.zeros((Nz, Npix, Npix), dtype=r.dtype, device=r.device)
    n2 = Npix * Npix
    for i in range(0, py.numel(), chunk):
        sy, sx = py[i:i + chunk], px[i:i + chunk]
        rb = r[i:i + chunk]  # (B, pmode, N, N)
        t = prob["H"][0, 0, 0] * rb.unsqueeze(2)  # (B, pmode, Nz, N, N)
        s = n2 * torch.fft.ifft2(t)
        contrib = (phi[0, :, 0].conj() * s).sum(dim=1)  # sum pmode -> (B, Nz, N, N)
        for b, (a, c) in enumerate(zip(sy, sx)):
            acc += torch.roll(contrib[b], shifts=(int(a), int(c)), dims=(-2, -1))
    return acc


def brute_force_M(prob, seed=1):
    """Read the Fourier-diagonal blocks of L^dag L off a random basis."""
    Npix, Nz = prob["Npix"], prob["Nz"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    xs = [
        (torch.randn((Nz, Npix, Npix), generator=g, dtype=torch.float64)
         + 1j * torch.randn((Nz, Npix, Npix), generator=g, dtype=torch.float64)
         ).to(prob["device"])
        for _ in range(Nz)
    ]
    ys = [apply_Ldag(prob, apply_L(prob, x)) for x in xs]
    X = torch.stack([torch.fft.fft2(x) for x in xs], dim=-1)  # (Nz, N, N, Nz_in)
    Y = torch.stack([torch.fft.fft2(y) for y in ys], dim=-1)
    X = X.permute(1, 2, 0, 3)  # (N, N, Nz_slice, Nz_in)
    Y = Y.permute(1, 2, 0, 3)
    M = Y @ torch.linalg.inv(X)  # (N, N, Nz, Nz): M(Q)
    return M


# --------------------------------------------------------------------------- #
# Closed forms                                                                  #
# --------------------------------------------------------------------------- #

def lags_to_matrix(m):
    """(Nz, H, W) lag stack -> (H, W, Nz, Nz) Hermitian Toeplitz."""
    Nz = m.shape[0]
    i = torch.arange(Nz, device=m.device)
    lag = i[:, None] - i[None, :]
    mfull = torch.cat([m[1:].conj().flip(0), m], dim=0)
    return mfull[lag + Nz - 1].permute(2, 3, 0, 1)


def extract_quadratic_phase(h, Npix):
    """Fit phase(h)(q) = c - 2*pi*(b . q) - pi*a*|q|^2 on the fft q grid
    (cycles per pixel) from THREE propagator samples along each axis near
    q = 0 (exact for a quadratic phase; the residual over the full grid is
    reported by the caller). Returns (a, (by, bx), max_fit_residual_rad).

    The linear term b is NOT zero for this project: create_grids offsets the
    propagator k grid by half a pixel, so the per-slice phase is
    -pi*a*|q + delta|^2 + const with delta = (0.5/Npix, 0.5/Npix) in cycles
    per pixel — the fit absorbs it into b = a*delta (plus wrap irrelevant
    constants)."""
    q = torch.fft.fftfreq(Npix, dtype=torch.float64, device=h.device)
    step = float(q[1] - q[0])
    # wrap-safe finite differences: angles of RATIOS of unimodular samples
    # (the true differences are << pi near q = 0)
    d2y = float(torch.angle(h[1, 0] * h[-1, 0] * h[0, 0].conj() ** 2))
    d2x = float(torch.angle(h[0, 1] * h[0, -1] * h[0, 0].conj() ** 2))
    a_y = -d2y / (2 * np.pi * step**2)
    a_x = -d2x / (2 * np.pi * step**2)
    d1y = float(torch.angle(h[1, 0] * h[-1, 0].conj())) / 2
    d1x = float(torch.angle(h[0, 1] * h[0, -1].conj())) / 2
    by = -d1y / (2 * np.pi * step)
    bx = -d1x / (2 * np.pi * step)
    a = 0.5 * (a_y + a_x)
    # residual of the quadratic model over the whole grid (phase-wrapped)
    Qy, Qx = torch.meshgrid(q, q, indexing="ij")
    model = torch.exp(-1j * np.pi * a * (Qy**2 + Qx**2)
                      - 2j * np.pi * (by * Qy + bx * Qx))
    model = model * (h[0, 0] / model[0, 0])  # anchor the constant
    resid = (h / model).angle().abs().max().item()
    return a, (by, bx), resid


def fresnel_closed_form(prob, pad_factor=8, exact=True):
    """Task closed form M_D(Q) = C exp(-i pi a_D |Q|^2 - 2 pi i b_D.Q)
    * Gamma_P(a_D Q + shift), with a_D, b_D extracted from the project's own
    h^D phase. Gamma_P evaluated exactly (trig polynomial, exact=True) or by
    padded-FFT sinc interpolation with nearest-sample lookup (exact=False).
    Returns (Nz, N, N) lag stack on the fft Q grid (C NOT fixed)."""
    Npix, Nz = prob["Npix"], prob["Nz"]
    dev = prob["device"]
    W = torch.fft.fft2(prob["probe"].to(torch.complex128)).abs().square().sum(dim=(0, 1))
    q = torch.fft.fftfreq(Npix, dtype=torch.float64, device=dev)
    Qy, Qx = torch.meshgrid(q, q, indexing="ij")
    out = torch.zeros((Nz, Npix, Npix), dtype=torch.complex128, device=dev)
    a1, (by1, bx1), resid = extract_quadratic_phase(prob["h"], Npix)
    if not exact:
        pad = pad_factor * Npix
        Wpad = torch.zeros((pad, pad), dtype=torch.complex128, device=dev)
        Ws = torch.fft.fftshift(W)
        lo = pad // 2 - Npix // 2
        Wpad[lo:lo + Npix, lo:lo + Npix] = Ws
        gamma = torch.fft.ifft2(torch.fft.ifftshift(Wpad)) * pad * pad / (Npix * Npix)
        # gamma(r) on lag grid r = n / pad_factor pixels, fft-ordered, periodic
    for D in range(Nz):
        a, by, bx = a1 * D, by1 * D, bx1 * D
        vy, vx = a * Qy, a * Qx  # lag = a_D * Q, in pixels
        if exact:
            # Gamma_P(v) = sum_q W(q) exp(-2 pi i (q + delta) . v), evaluated
            # as the exact trigonometric polynomial (direct O(N^4) sum, fine
            # at N = 32); delta = b/a is the propagator grid's half-pixel
            # offset, read from the same phase fit
            dy = by1 / a1 if a1 != 0 else 0.0
            dx_ = bx1 / a1 if a1 != 0 else 0.0
            qy = q[:, None] + dy
            qx = q[None, :] + dx_
            G = torch.einsum(
                "ab,abyx->yx",
                W.to(torch.complex128),
                torch.exp(-2j * np.pi * (qy[:, :, None, None] * vy[None, None]
                                         + qx[:, :, None, None] * vx[None, None])),
            )
        else:
            # gamma[n] = const * Gamma(-n/PF) on the delta-free lag grid
            idxy = (-torch.round(vy * pad_factor).long()) % (pad_factor * Npix)
            idxx = (-torch.round(vx * pad_factor).long()) % (pad_factor * Npix)
            G = gamma[idxy, idxx]
            # sinc interpolation lives on the unshifted (delta-free) grid;
            # restore the half-pixel offset as an explicit phase
            dy = by1 / a1 if a1 != 0 else 0.0
            dx_ = bx1 / a1 if a1 != 0 else 0.0
            G = G * torch.exp(-2j * np.pi * (dy * vy + dx_ * vx))
        # chord identity: H_D(q) H_D(q-Q)^* = exp(-i pi a_D (2(q'+delta).Q
        # + |Q|^2)) with q' = q - Q; the (q'+delta).Q part IS Gamma's
        # exponent at lag a_D Q, so the chirp carries only the |Q|^2 term
        chirp = torch.exp(-1j * np.pi * a * (Qy**2 + Qx**2))
        out[D] = chirp * G
    return out, dict(a1=a1, b1=(by1, bx1), quad_fit_resid_rad=resid)


def ls_scale(ref, test):
    """Complex least-squares scalar c minimising ||ref - c*test||."""
    num = (test.conj() * ref).sum()
    den = test.abs().square().sum()
    return num / den


def rel_err(ref, test, mask=None):
    """Max relative deviation after fitting one complex scale (the positive
    constant C of the symbol). mask: optional (H, W) bool over Q."""
    if mask is not None:
        ref = ref[mask]
        test = test[mask]
    c = ls_scale(ref, test)
    e = (ref - c * test).abs().max() / ref.abs().max()
    return float(e), c


def wrapfree_mask(prob):
    """Q region where q + Q never crosses the Nyquist wrap for any q in the
    probe aperture (per axis, incl. the half-pixel grid offset) — the zone
    where the continuum Fresnel closed form can be exact on a cyclic grid."""
    Npix = prob["Npix"]
    k_cut = (prob.get("conv", 24.9) / 1e3) / prob["lambd"]  # 1/A
    r_ap = k_cut * prob["dx"]  # cycles/pixel
    q = torch.fft.fftfreq(Npix, dtype=torch.float64, device=prob["device"])
    Qy, Qx = torch.meshgrid(q, q, indexing="ij")
    lim = 0.5 - r_ap - 1.5 / Npix
    return (Qy.abs() <= lim) & (Qx.abs() <= lim)


# --------------------------------------------------------------------------- #
# Gate                                                                          #
# --------------------------------------------------------------------------- #

def run_gate(device):
    os.makedirs(OUTDIR, exist_ok=True)
    report = {}
    torch.manual_seed(0)
    for kernel in ("fresnel", "angular_spectrum"):
        for Nz, pmode in ((6, 1), (4, 2), (12, 1)):
            tag = f"{kernel}_Nz{Nz}_pmode{pmode}"
            prob = small_problem(Nz=Nz, pmode=pmode, kernel=kernel, device=device)
            # ---- adjoint check ------------------------------------------------
            g = torch.Generator(device="cpu").manual_seed(7)
            Npix = prob["Npix"]
            x = (torch.randn((Nz, Npix, Npix), generator=g, dtype=torch.float64)
                 + 1j * torch.randn((Nz, Npix, Npix), generator=g, dtype=torch.float64)).to(device)
            r = (torch.randn((Npix * Npix, pmode, Npix, Npix), generator=g, dtype=torch.float64)
                 + 1j * torch.randn((Npix * Npix, pmode, Npix, Npix), generator=g,
                                    dtype=torch.float64)).to(device)
            lhs = (r.conj() * apply_L(prob, x)).sum()
            rhs = (apply_Ldag(prob, r).conj() * x).sum()
            adj = float(abs(lhs - rhs) / abs(lhs))
            # ---- brute force vs closed forms ---------------------------------
            Mb = brute_force_M(prob)  # (N, N, Nz, Nz)
            herm = float((Mb - Mb.mH).abs().max() / Mb.abs().max())
            m_eng = depth_symbol_lags(prob["probe"], prob["h"], Nz, (Npix, Npix),
                                      dtype=torch.complex128)
            Me = lags_to_matrix(m_eng)
            e_eng, c_eng = rel_err(Mb, Me)
            entry = dict(adjoint_rel=adj, brute_hermiticity_rel=herm,
                         engine_symbol_rel=e_eng,
                         engine_scale=[float(c_eng.real), float(c_eng.imag)])
            if Nz == 6 and pmode == 1:
                mask = wrapfree_mask(prob)
                mf_exact, meta = fresnel_closed_form(prob, exact=True)
                Mf = lags_to_matrix(mf_exact)
                e_fre, c_fre = rel_err(Mb, Mf, mask=mask)
                e_fre_full, _ = rel_err(Mb, Mf)
                mf8, _ = fresnel_closed_form(prob, pad_factor=8, exact=False)
                mf4, _ = fresnel_closed_form(prob, pad_factor=4, exact=False)
                e8, _ = rel_err(Mb, lags_to_matrix(mf8), mask=mask)
                e4, _ = rel_err(Mb, lags_to_matrix(mf4), mask=mask)
                pad_change = float((mf8 - mf4).abs().max() / mf8.abs().max())
                entry.update(fresnel_form_rel_exactGamma_wrapfree=e_fre,
                             fresnel_form_rel_exactGamma_fullQ=e_fre_full,
                             wrapfree_fraction=float(mask.double().mean()),
                             fresnel_form_rel_padFFT8_wrapfree=e8,
                             fresnel_form_rel_padFFT4_wrapfree=e4,
                             padFFT_8_vs_4_rel=pad_change,
                             a_extracted=meta["a1"],
                             a_theory_lambda_dz_over_dx2=prob["lambd"] * prob["dz"] / prob["dx"]**2,
                             b_extracted=[float(meta["b1"][0]), float(meta["b1"][1])],
                             quad_fit_resid_rad=meta["quad_fit_resid_rad"])
            report[tag] = entry
            print(tag, json.dumps(entry, indent=1))
    with open(os.path.join(OUTDIR, "gate_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    # verdict
    bad = []
    for tag, e in report.items():
        if e["adjoint_rel"] > 1e-12 or e["engine_symbol_rel"] > 1e-10:
            bad.append(tag)
        if tag.startswith("fresnel_Nz6") and e.get("fresnel_form_rel_exactGamma_wrapfree", 0) > 1e-10:
            bad.append(tag + ":fresnel_form")
    print("GATE:", "FAILED " + str(bad) if bad else "PASSED")
    return not bad


# --------------------------------------------------------------------------- #
# Real-geometry diagnostic                                                      #
# --------------------------------------------------------------------------- #

def run_real(device, probe_npz, params_path):
    os.makedirs(OUTDIR, exist_ok=True)
    from ptyrad.load import load_params
    from ptyrad.models import PtychoAD
    from ptyrad.reconstruction import PtyRADSolver

    params = load_params(params_path)
    params["recon_params"]["SAVE_ITERS"] = None
    solver = PtyRADSolver(params, device=device, seed=42)
    model = PtychoAD(solver.init.init_variables, params["model_params"],
                     device=device, verbose=False)
    Nz = model.opt_obja.shape[-3]
    dx = float(model.dx)
    h = model.get_propagators_3d(model.get_propagators(torch.tensor([0], device=device)))
    h_unit = h[0, 0, 0, 1].to(torch.complex128)
    probe = torch.as_tensor(np.load(probe_npz)["probe"], device=device)
    print(f"model: Nz={Nz}, dx={dx:.6f} A, dz={float(model.opt_slice_thickness):.4f} A, "
          f"kernel={model.propagator_kernel}, probe modes={probe.shape}")
    Npix = probe.shape[-1]
    m = depth_symbol_lags(probe, h_unit, Nz, (Npix, Npix), dtype=torch.complex128)
    M = lags_to_matrix(m)  # (N, N, Nz, Nz)
    M = 0.5 * (M + M.mH)
    ev = torch.linalg.eigvalsh(M)  # ascending, (N, N, Nz)
    m0 = float(m[0].real.mean())
    ev = (ev / m0).cpu().numpy()
    q = np.fft.fftfreq(Npix)
    Qy, Qx = np.meshgrid(q, q, indexing="ij")
    Qabs = np.sqrt(Qy**2 + Qx**2) / dx  # 1/A
    eff = (ev.sum(-1) ** 2) / (ev**2).sum(-1)
    np.savez_compressed(
        os.path.join(OUTDIR, "real_geometry_symbol.npz"),
        eigenvalues_over_m0=ev, Q_abs_invA=Qabs, eff_channels=eff,
        m_lags=m.cpu().numpy(), dx=dx, Nz=Nz,
    )
    print("saved real_geometry_symbol.npz;",
          f"eig range [{ev.min():.3e}, {ev.max():.3e}] x m0,",
          f"eff channels at Q~0: {eff[0, 1]:.2f}, max {eff.max():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-npz", default=os.path.expanduser(
        "~/probe-grad-eval/results/tbl_gau_brent_post_s42/ckpt_final.npz"))
    ap.add_argument("--params", default=os.path.expanduser(
        "~/ptyrad/demo/params/tBL_WSe2_reconstruct_minimal_born.yml"))
    args = ap.parse_args()
    if args.gate:
        ok = run_gate(args.device)
        if not ok:
            sys.exit(1)
    if args.real:
        run_real(args.device, args.probe_npz, args.params)
