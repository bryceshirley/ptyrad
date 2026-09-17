"""Appendix-C analytic adjoints vs automatic differentiation, verbatim.

Forward: the paper's Eq. (model), entrance-referred, orthonormal transforms.
Gradients: Eqs. (grad-s)/(grad-p) exactly as printed. Autograd on the same
forward. Finite differences arbitrate. CPU, seconds.
"""
import glob, sys
import numpy as np, torch
sys.path.insert(0, "/home/dnz75396/ptyrad/demo")
sys.path.insert(0, "/home/dnz75396/ptyrad/src")
from ptyrad_diagnostics import Geom, electron_wavelength_A, gauge_fix, load_model
from ptyrad.load import load_raw

torch.manual_seed(0)
cdt = torch.complex128
m = load_model(glob.glob("/home/dnz75396/ptyrad/demo/output/tBL_WSe2_born/"
                         "2026*/model_iter0200.hdf5")[0], verbose=False)
obj, scale = gauge_fix(m["obj"][0]); probe = m["probe"] * scale
lam = electron_wavelength_A(m["kv"]); N = obj.shape[0]
ny, nx = probe.shape[-2:]
geom = Geom((ny, nx), m["dx"], lam, m["dz"], N, torch.device("cpu"), cdt,
            H0=m.get("H0"))
H = [geom.H(zj) for zj in geom.z]                      # H_{z_j}
pos = np.round(m["pos"]).astype(int)
raw = load_raw("/home/dnz75396/ptyrad/demo/data/tBL_WSe2/Panel_g-h_Themis/"
               "scan_x128_y128.raw", shape=(16384, 128, 128))
idx = [3000, 8200]
ob = torch.as_tensor(obj, dtype=cdt)
P0 = torch.as_tensor(np.ascontiguousarray(probe), dtype=cdt)
G0 = torch.stack([torch.stack([ob[j, p[0]:p[0]+ny, p[1]:p[1]+nx]
                               for p in pos[idx]]) for j in range(N)]) - 1.0
meas = torch.as_tensor(np.ascontiguousarray(
    np.clip(np.asarray(raw[idx], np.float64), 0, None).transpose(0, 2, 1)))
meas = torch.fft.ifftshift(meas, dim=(-2, -1))
w = 1.0 / (meas + 1.0)

F  = lambda x: torch.fft.fft2(x,  norm="ortho")
Fi = lambda x: torch.fft.ifft2(x, norm="ortho")

def forward(g, Pr):
    """psih = F[P] + sum_j H_{-z_j} F[dO_j psi0_j];  psi0_j = Fi[H_j F P]."""
    FP = F(Pr)                                          # (M,ny,nx)
    psih = FP[None].expand(len(idx), -1, -1, -1).clone()
    for j in range(N):
        psi0 = Fi(H[j] * FP)                            # (M,ny,nx)
        psih = psih + H[j].conj() * F(g[j][:, None] * psi0[None])
    return psih

def loss_of(g, Pr):
    psih = forward(g, Pr)
    I = (psih.abs() ** 2).sum(1)
    return (w * (I - meas) ** 2).sum(), psih

# autograd
gA = G0.clone().requires_grad_(True); PA = P0.clone().requires_grad_(True)
L, _ = loss_of(gA, PA); L.backward()

# analytic, Eqs. (grad-s)/(grad-p) with ascent residual dL/dpsih-bar
with torch.no_grad():
    _, psih = loss_of(G0, P0)
    I = (psih.abs() ** 2).sum(1)
    dpsih = (2.0 * w * (I - meas))[:, None] * psih      # dL/d psih-bar
    FP = F(P0)
    ana_g, accP = [], Fi(dpsih)                         # F^{-1} term of C-dagger
    for j in range(N):
        psi0 = Fi(H[j] * FP)
        r_j = Fi(H[j] * dpsih)                          # Fi[H_{z_j} dpsih]
        ana_g.append((psi0[None].conj() * r_j).sum(0))  # sum over positions? no:
    # careful: object patches are per-position; gradient per (j, b):
    ana_g = []
    for j in range(N):
        psi0 = Fi(H[j] * FP)                            # (M,ny,nx)
        r_j = Fi(H[j] * dpsih)                          # (B,M,ny,nx)
        ana_g.append((psi0[None].conj() * r_j).sum(1))  # sum over modes -> (B,ny,nx)
    ana_g = torch.stack(ana_g)
    accP = Fi(dpsih)
    for j in range(N):
        r_j = Fi(H[j] * dpsih)
        accP = accP + Fi(H[j].conj() * F(G0[j][:, None].conj() * r_j))
    ana_P = accP.sum(0)                                 # sum over positions -> (M,ny,nx)

def rel(a, b):
    return float((a - b).abs().pow(2).sum().sqrt() / b.abs().pow(2).sum().sqrt())

# resolve torch's convention empirically, then report
cands = {"raw": 1.0, "conj": "c"}
for name, auto, ana in (("object", gA.grad, ana_g), ("probe", PA.grad, ana_P)):
    r_raw, r_conj = rel(auto, ana), rel(auto.conj(), ana)
    best = min(r_raw, r_conj)
    print("%s gradient: analytic (App. C) vs autograd rel. diff = %.3e"
          % (name, best), "(torch conv: %s)" % ("raw" if r_raw < r_conj else "conj"))

# finite-difference arbitration (object, random direction)
d = torch.randn_like(G0) + 1j * torch.randn_like(G0)
d = d / d.abs().pow(2).sum().sqrt()
eps = 1e-5
Lp, _ = loss_of(G0 + eps * d, P0); Lm, _ = loss_of(G0 - eps * d, P0)
fd = float((Lp - Lm) / (2 * eps))
an = float(2 * torch.real((ana_g * d.conj()).sum()))
print("directional: FD %.8e | 2Re<ana,d> %.8e | rel %.1e"
      % (fd, an, abs(an - fd) / abs(fd)))

# factor-2 convention (torch grad over stacked real/imag = 2 * dL/dz-bar)
for name, auto, ana in (("object", gA.grad, ana_g), ("probe", PA.grad, ana_P)):
    print("%s gradient: rel diff (autograd/2 vs analytic) = %.3e"
          % (name, rel(auto / 2.0, ana)))
