"""2x2 cost figure (timing over memory, batch 1 | 32) with the PRODUCTION
adjoints — the autograd version:

  multislice        plain sequential multislice + torch.autograd
  ISS, parallel     forward_models.iss.iss_forward + torch.autograd
                    (materialises the O(batch x N) slice stacks)
  ISS, low memory   iss_forward_lowmem chunk 1 (ISSLowMemFunction's
                    slice-looped hand adjoint, as shipped)
  ML-ISS            one full ML-ISS Gaussian batch update: gradients by
                    autograd of mliss._fwd_lg, exact quartic steps

Outputs (this directory): cost_ptyrad_2x2_mliss_A100.{png,pdf,csv}
"""
import glob, os, time
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import h5py, numpy as np, torch
torch._dynamo.config.disable = True
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2, fftshift

import ptyrad.mliss as mlm
from ptyrad.forward_models import iss_forward
from ptyrad.forward_models.iss import iss_forward_lowmem

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cuda"
CKPT = sorted(glob.glob(
    "/home/dnz75396/ptyrad/demo/output/test_100/tBL_WSe2_born/2026*_*random32*/model_iter0100.hdf5"))[-1]
with h5py.File(CKPT) as f:
    obja = torch.tensor(f["optimizable_tensors/obja"][...], device=dev)
    objp = torch.tensor(f["optimizable_tensors/objp"][...], device=dev)
    probe0 = torch.tensor(f["optimizable_tensors/probe"][...], device=dev)
    crop_pos = torch.tensor(f["model_attributes/crop_pos"][...].astype(np.int64), device=dev)
    H1 = torch.tensor(f["model_attributes/H"][...], device=dev)
probe = (probe0 if probe0.ndim == 3 else torch.view_as_complex(probe0.contiguous()))[None]
Ny, Nx = probe.shape[-2:]
occu = torch.ones(1, device=dev)
BATCHES = (1, 32)
SLICES = (1, 2, 4, 8, 16, 32, 64)
REPS = 10
EPS = 1e-10


# ---------- line-search helpers (eager copies of ptyrad.linesearch) ----------
DN_EPS = float(torch.finfo(torch.float32).eps)


def _fields_from_complex(O, prb, H):
    """Detector-plane field from complex O; affine in (O-1), linear in probe."""
    probe_k = fft2(prb).view(-1, prb.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)
    g = (O - 1.0).unsqueeze(1)
    return probe_k.squeeze(3) + torch.sum(fft2(g * psi) * H.conj(), dim=3)


def response_terms(F, D, omode_occu):
    """Per-pixel linear (v) and quadratic (w) intensity coefficients."""
    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    v = fftshift(torch.sum((F.conj() * D).real * nw, dim=(1, 2)), dim=(-2, -1))
    w = fftshift(torch.sum(D.abs().square() * nw, dim=(1, 2)), dim=(-2, -1))
    return v, w


def direction_response(d, prb, H):
    """Per-slice object direction response D_j = FFT[d_j phi_j] H_j^*."""
    probe_k = fft2(prb).view(-1, prb.shape[1], 1, 1, Ny, Nx)
    psi = ifft2(H * probe_k)
    return fft2(d.unsqueeze(1) * psi) * H.conj()


def unscattered_illumination(prb, H):
    """phi_j = IFFT[H_j FFT(P)], shape (B, pmode, 1, Nz, Ny, Nx)."""
    probe_k = fft2(prb).view(-1, prb.shape[1], 1, 1, Ny, Nx)
    return ifft2(H * probe_k)


def object_denominator(phi):
    """K_j peak over space per slice ('max' recipe), floored."""
    K = phi.abs().square().sum(dim=(0, 1))
    K = K.sum(dim=0) if K.dim() == 4 else K
    return K.amax(dim=(-2, -1), keepdim=True).clamp_min(DN_EPS)


def probe_denominator(obj_complex):
    """K_P spatial peak at the pre-step object ('max' recipe), floored."""
    K = obj_complex.abs().square()
    return K.sum(dim=tuple(range(K.dim() - 2))).amax().clamp_min(DN_EPS)


def plain_multislice(object_patches, prb, H2, omode_occu, eps=EPS):
    """Standard 1st-order multislice: transmit each slice, propagate by H
    between slices (none after the last), detector intensity in PtyRAD units."""
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])
    n = O.shape[2]
    psi = prb[:, :, None]
    for j in range(n - 1):
        psi = ifft2(H2[:, None, None] * fft2(psi * O[:, None, :, j]))
    psi = psi * O[:, None, :, n - 1]
    nw = (omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    dp = torch.sum(fft2(psi).abs().square() * nw, dim=(1, 2)) + eps
    return fftshift(dp, dim=(-2, -1))


def make_fwd_adj(fn, patches, prb):
    def call():
        dp = fn()
        torch.autograd.grad(dp.sum(), (patches, prb))
    return call


def make_mliss_update(O0, probe_in, H3, I_dat, Nz, c=558.0):
    """One full ML-ISS (Gaussian, alternating) batch update at the tensor
    level: autograd gradients of L_G, K preconditioners, per-slice direction
    response, both exact quartic solves, probe response."""
    sigma2 = I_dat + 1.0 / c
    omega = 1.0 / sigma2

    def call():
        O = O0.detach().requires_grad_(True)
        P = probe_in.detach().requires_grad_(True)
        L, F, u = mlm._fwd_lg(O, P, H3, I_dat, None, sigma2, occu)
        gO, gP = torch.autograd.grad(L, (O, P))
        phi = unscattered_illumination(probe_in, H3)
        dn = object_denominator(phi)
        d = (-gO) / dn
        D = direction_response(d, probe_in, H3).sum(dim=3)
        F2 = F.detach()
        u2 = u.detach()
        v, w = response_terms(F2, D, occu)
        a, _ = mlm.ml_line_search(u2 - I_dat, v, w, omega, fallback=1.0 / Nz)
        F3 = F2 + a * D
        u3 = u2 + (2.0 * a) * v + (a * a) * w
        dn_p = probe_denominator(O.detach())
        q = (-gP) / dn_p
        D_P = _fields_from_complex(O.detach() + a * d, q, H3)
        v2, w2 = response_terms(F3, D_P, occu)
        mlm.ml_line_search(u3 - I_dat, v2, w2, omega, fallback=1.0 / Nz)

    return call


# ---------- benchmark ----------
def time_one(call, reps=REPS):
    call(); torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3, \
           (torch.cuda.max_memory_allocated() - base) / 1e9


def _win(B, Nz):
    wa = torch.stack([obja[:, :, y:y+Ny, x:x+Nx] for y, x in crop_pos[:B].tolist()])
    wp = torch.stack([objp[:, :, y:y+Ny, x:x+Nx] for y, x in crop_pos[:B].tolist()])
    idx = [j % wa.shape[2] for j in range(Nz)]
    return torch.stack([wa[:, :, idx], wp[:, :, idx]], dim=-1).contiguous()


rows = []
for Bsz in BATCHES:
    for Nz in SLICES:
        patches = _win(Bsz, Nz).requires_grad_(True)
        prb = probe.clone().requires_grad_(True)
        zj = torch.arange(Nz, device=dev).view(1, 1, 1, Nz, 1, 1)
        H3 = (H1 ** zj).contiguous()
        H2 = H1.unsqueeze(0).contiguous()
        with torch.no_grad():
            O0 = torch.polar(patches[..., 0], patches[..., 1]).contiguous()
            I_dat = 1.02 * iss_forward(patches.detach(), probe, H3, occu)
            torch.cuda.empty_cache()

        series = [
            ("multislice",
             make_fwd_adj(lambda: plain_multislice(patches, prb, H2, occu), patches, prb)),
            ("ISS, parallel",
             make_fwd_adj(lambda: iss_forward(patches, prb, H3, occu), patches, prb)),
            ("ISS, low memory (chunk 1)",
             make_fwd_adj(lambda: iss_forward_lowmem(patches, prb, H3, occu, EPS, False, 1),
                          patches, prb)),
            ("ISS + line search", make_mliss_update(O0, probe, H3, I_dat, Nz)),
        ]
        for name, call in series:
            try:
                ms, gb = time_one(call)
            except RuntimeError as e:
                print(f"  B={Bsz} N={Nz} {name}: skipped ({str(e)[:40]})")
                ms, gb = float("nan"), float("nan")
                torch.cuda.empty_cache()
            rows.append((name, Bsz, Nz, ms, gb))
            print(f"  B={Bsz:3d} N={Nz:3d} {name:26s} {ms:8.2f} ms  {gb:6.3f} GB")
        patches = prb = H3 = O0 = I_dat = None
        torch.cuda.empty_cache()

import csv as _csv
with open(f"{HERE}/cost_ptyrad_2x2_mliss_A100.csv", "w", newline="") as f:
    wcsv = _csv.writer(f)
    wcsv.writerow(["series", "batch", "slices", "fwd_adj_ms", "peak_GB"])
    wcsv.writerows(rows)

# ---------- plot (same layout as plot_cost_2x2.py) ----------
SERIES = [
    ("multislice", "#2a78d6", "o", "multislice"),
    ("ISS, parallel", "#e8590c", "s", "ISS, parallel"),
    ("ISS, low memory (chunk 1)", "#1baf7a", "^", "ISS, low memory"),
    ("ISS + line search", "#eda100", "D", "ML-ISS"),
]
fig, axes = plt.subplots(2, 2, figsize=(8.8, 8.2), dpi=200)
for col, Bsz in enumerate(BATCHES):
    for rown, (key, ylab) in enumerate([(3, "forward + adjoint (ms per batch)"),
                                        (4, "peak allocation (GB)")]):
        ax = axes[rown, col]
        for name, colr, mk, lab in SERIES:
            pts = sorted((r[2], r[key]) for r in rows if r[0] == name and r[1] == Bsz
                         and np.isfinite(r[key]))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=colr, marker=mk, ms=6, lw=2.0, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks(list(SLICES))
        ax.set_xticklabels([str(x) for x in SLICES]); ax.set_ylim(bottom=0)
        if rown == 0: ax.set_title(f"batch {Bsz}", fontsize=12)
        if rown == 1: ax.set_xlabel("slices $N$", fontsize=11)
        if col == 0: ax.set_ylabel(ylab, fontsize=11)
        ax.grid(True, color="#e8e7de", lw=0.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{HERE}/cost_ptyrad_2x2_mliss_A100.{ext}", facecolor="white")
print("saved cost_ptyrad_2x2_mliss_A100.png/.pdf/.csv")
