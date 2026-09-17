"""
First-Born vs multislice diagnostics for PtyRAD reconstructions.

The same measurements the ptypy engine makes on X-ray data, run instead on a
PtyRAD model checkpoint. This is the electron half of the comparison, and it is
the opposite corner of the parameter space from the X-ray case: geometrically
thin (a bilayer is a small fraction of the depth of field at 25 mrad) but
optically strong, where the X-ray specimen was 15 depths of field of separation
carrying only 0.37 rad. Those two axes are independent, and the pair of
datasets brackets them.

What it reports, for each reconstruction given:

  eps_M   Eq. (epsilon): the truncation error against exact multislice, for
          M = 0..max_order, with the orders obtained exactly from the
          roots-of-unity decomposition (N+1 cascade evaluations, no
          combinatorial sum). M = N must return to round-off; that is the
          end-to-end check on the whole pipeline.
  chord vs tangent: g = O - 1 against g = i s. The tangent hierarchy does not
          terminate in multislice, so its M = N member is a floor.
  blocked: Eq. (blocked), first Born within M contiguous blocks.
  Phi_tot, and the estimates it feeds.
  chi^2 against the measured frames, if the raw data is supplied.

and, when two models are given, compares the recovered objects with each other
after putting both in a common gauge -- the per-slice mean transmission is not
determined by the data, engines drift in it, and eps is measured from wherever
they drifted to.

    python ptyrad_diagnostics.py --inspect  born/model_iter0100.hdf5
    python ptyrad_diagnostics.py --model born/model_iter0100.hdf5 \
                                 --compare multislice/model_iter0100.hdf5 \
                                 --data .../scan_x128_y128.raw --plot

The loader reads geometry from the checkpoint itself rather than from a params
file, so it stays correct if the yml is edited afterwards. Run --inspect first:
PtyRAD's key names have moved between versions, and the loader prints what it
found and what it guessed.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

try:
    import torch
except ImportError:                                        # pragma: no cover
    torch = None
try:
    import h5py
except ImportError:                                        # pragma: no cover
    h5py = None


# ===========================================================================
#  Reading a PtyRAD checkpoint
# ===========================================================================

def inspect(path, max_depth=4):
    """Print the tree, so the key names can be checked rather than assumed."""
    with h5py.File(path, 'r') as f:
        def show(name, obj):
            if name.count('/') > max_depth:
                return
            if isinstance(obj, h5py.Dataset):
                v = ''
                if obj.size <= 8 and obj.dtype.kind in 'fiu':
                    v = '  = %s' % np.array(obj).ravel()[:8]
                print("  %-52s %-18s %s%s"
                      % (name, str(obj.shape), obj.dtype, v))
            else:
                print("  %s/" % name)
        f.visititems(show)


def _first(f, *candidates):
    for c in candidates:
        if c in f:
            return np.array(f[c])
    return None


def _search(f, *needles, ndim=None):
    """Find a dataset whose path contains all needles."""
    hits = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and all(n in name.lower() for n in needles):
            if ndim is None or obj.ndim == ndim:
                hits.append((name, obj.shape))
        return None

    f.visititems(visit)
    return hits


def load_model(path, verbose=True):
    """Object, probe and geometry from a PtyRAD model_iterXXXX.hdf5."""
    if h5py is None:
        raise ImportError("h5py is required")
    out = {}
    with h5py.File(path, 'r') as f:
        obja = _first(f, 'optimizable_tensors/obja', 'obja')
        objp = _first(f, 'optimizable_tensors/objp', 'objp')
        probe = _first(f, 'optimizable_tensors/probe', 'probe')
        if obja is None or objp is None or probe is None:
            raise KeyError("obja / objp / probe not found; run --inspect and "
                           "pass the right paths")
        out['obj'] = obja * np.exp(1j * objp)              # (omode, Nz, Ny, Nx)
        out['probe'] = probe                               # (pmode, Ny, Nx)

        def grab(*keys, default=None):
            for k in keys:
                v = _first(f, k)
                if v is not None:
                    return float(np.ravel(v)[0])
            return default

        out['dz'] = grab('optimizable_tensors/slice_thickness',
                         'model_attributes/slice_thickness',
                         'params/init_params/obj_slice_thickness')
        out['dx'] = grab('model_attributes/dx', 'params/exp/dx', 'params/dx')
        out['kv'] = grab('params/init_params/probe_kv', 'params/exp/kv',
                         'model_attributes/kv')
        out['conv_angle'] = grab('params/init_params/probe_conv_angle')
        out['dk'] = grab('model_attributes/dk')
        out['omode_occu'] = _first(f, 'model_attributes/omode_occu')

        # PtyRAD stores the single-step transfer function it actually used.
        # Taking it verbatim removes every convention that would otherwise have
        # to be reproduced: fft ordering, sign, bandwidth limit, and whether the
        # evanescent band is clamped. H_j is then H^j, exact because
        # H_a H_b = H_{a+b}.
        out['H0'] = _first(f, 'model_attributes/H')

        pos = _first(f, 'model_attributes/crop_pos', 'crop_pos',
                     'optimizable_tensors/probe_pos_shifts')
        out['pos'] = None if pos is None else np.asarray(pos)
        if verbose:
            print("[load] %s" % os.path.basename(path))
            print("       object %s, probe %s" % (out['obj'].shape, out['probe'].shape))
            print("       dx %.5f A, dz %.3f A, %.0f kV, conv. angle %.1f mrad"
                  % (out['dx'] or np.nan, out['dz'] or np.nan,
                     out['kv'] or np.nan, out['conv_angle'] or np.nan))
            if out['H0'] is not None:
                mod = np.abs(out['H0'])
                print("       using the stored propagator, |H| in [%.3f, %.3f]"
                      % (mod.min(), mod.max()))
            if out['pos'] is None:
                print("       no positions found; pass --positions or use "
                      "--inspect to locate them")
    return out


def electron_wavelength_A(kv):
    """Relativistic electron wavelength in angstrom."""
    h, m0, e, c = 6.62607015e-34, 9.1093837015e-31, 1.602176634e-19, 299792458.0
    v = float(kv) * 1e3
    lam_m = h / np.sqrt(2 * m0 * e * v * (1.0 + e * v / (2 * m0 * c ** 2)))
    return lam_m * 1e10


# ===========================================================================
#  Operators
# ===========================================================================

class Geom:
    """Angular-spectrum propagator on the reconstruction grid.

    Built the same way PtyRAD builds its H stack, from the reconstruction pixel
    size and the electron wavelength, with the evanescent band clamped so that
    |H| = 1 holds exactly.
    """

    def __init__(self, shape, dx_A, lam_A, dz_A, N, device, cdt, H0=None):
        ny, nx = int(shape[0]), int(shape[1])
        self.H0 = None if H0 is None else np.asarray(H0).astype(np.complex128)
        self.dz0 = float(dz_A)
        fy = np.fft.fftfreq(ny, d=float(dx_A))
        fx = np.fft.fftfreq(nx, d=float(dx_A))
        a2 = (lam_A * fy)[:, None] ** 2 + (lam_A * fx)[None, :] ** 2
        self.kz = np.sqrt(np.maximum(1.0 - a2, 0.0)) - 1.0
        self.evan = a2 > 1.0
        self.lam, self.device, self.cdt = float(lam_A), device, cdt
        self.z = np.arange(N, dtype=float) * float(dz_A)
        self.na = float(np.sqrt(np.clip(a2.max(), 0, 1)))
        self._cache = {}

    def H(self, dz):
        key = round(float(dz), 12)
        h = self._cache.get(key)
        if h is None:
            if self.H0 is not None:
                # integer multiples of the stored step; conj rather than a
                # negative power, so a band-limited H with exact zeros in it
                # stays finite
                n = int(round(key / self.dz0))
                hn = np.power(self.H0, abs(n))
                if n < 0:
                    hn = np.conj(hn)
            else:
                hn = np.exp(2j * np.pi * (key / self.lam) * self.kz)
                hn[self.evan] = 0.0
            h = torch.as_tensor(hn, dtype=self.cdt, device=self.device)
            self._cache[key] = h
        return h

    def prop(self, w, dz):
        if float(dz) == 0.0:
            return w
        return torch.fft.ifft2(self.H(dz) * torch.fft.fft2(w))


def multislice(O, P, geom):
    """Ordered product, Eq. (cascade). O (N,B,ny,nx), P (M,ny,nx)."""
    psi = O[0][:, None] * P[None]
    for j in range(1, len(geom.z)):
        psi = geom.prop(psi, geom.z[j] - geom.z[j - 1]) * O[j][:, None]
    return psi


def illumination(P, geom):
    return torch.stack([geom.prop(P, zj) for zj in geom.z])


def born(g, P, phi, geom):
    """Eq. (model), accumulated in the Fourier domain: N transforms, not 2N."""
    zN = geom.z[-1]
    acc = None
    for j in range(len(geom.z)):
        F = torch.fft.fft2(g[j][:, None] * phi[j][None]) * geom.H(zN - geom.z[j])
        acc = F if acc is None else acc + F
    return torch.fft.ifft2(acc) + geom.prop(P, zN)[None]


def blocked(g, P, geom, n_blocks):
    N = len(geom.z)
    nb = int(max(1, min(n_blocks, N)))
    blocks = np.array_split(np.arange(N), nb)
    psi, z_in = P[None], geom.z[0]
    for b, ids in enumerate(blocks):
        z_out = geom.z[blocks[b + 1][0]] if b + 1 < len(blocks) else geom.z[-1]
        out = geom.prop(psi, z_out - z_in)
        for j in ids:
            out = out + geom.prop(g[j][:, None] * geom.prop(psi, geom.z[j] - z_in),
                                  z_out - geom.z[j])
        psi, z_in = out, z_out
    return psi


def orders(O, P, geom, max_order):
    """Exact scattering orders by DFT over the (N+1)-th roots of unity."""
    N = len(geom.z)
    K, g = N + 1, O - 1.0
    mo = int(min(max_order, N))
    acc = psi_ms = None
    for k in range(K):
        t = complex(np.exp(2j * np.pi * k / K))
        tt = torch.as_tensor(t, dtype=O.dtype, device=O.device)
        psik = multislice(1.0 + tt * g, P, geom)
        if k == 0:
            psi_ms = psik.clone()
            acc = torch.zeros((mo + 1,) + tuple(psik.shape), dtype=psik.dtype,
                              device=psik.device)
        for m in range(mo + 1):
            acc[m] += torch.as_tensor(complex(np.conj(t) ** m) / K,
                                      dtype=psik.dtype, device=psik.device) * psik
        del psik
    return psi_ms, acc


# ===========================================================================
#  The measurement
# ===========================================================================

def gauge_fix(obj):
    """Divide each slice by its complex mean; the product goes on the probe.

    Multislice is pointwise invariant under this (Eq. null-const), the Born
    prediction is not, and two reconstructions only compare in a common gauge.
    """
    obj = obj.copy()
    scale = 1.0 + 0j
    for j in range(obj.shape[0]):
        m = obj[j].mean()
        if abs(m) > 1e-9:
            obj[j] = obj[j] / m
            scale *= m
    return obj, scale


def measure(model, positions, geom, device, cdt, batch=8, max_order=2,
            blocks=(1,), n_positions=256, data=None, gauge='mean',
            verbose=True):
    """eps_M for one reconstruction, over a subsample of scan positions."""
    obj = model['obj']
    if obj.ndim == 4:                    # (omode, Nz, Ny, Nx)
        if obj.shape[0] != 1:
            print("[warn] %d object modes; using the first" % obj.shape[0])
        obj = obj[0]
    N = obj.shape[0]
    probe = model['probe']
    if probe.ndim == 4:
        probe = probe[0]
    scale = 1.0 + 0j
    if gauge == 'mean':
        obj, scale = gauge_fix(obj)

    ob = torch.as_tensor(obj, dtype=cdt, device=device)
    pr = torch.as_tensor(np.ascontiguousarray(probe) * scale, dtype=cdt,
                         device=device)
    ny, nx = pr.shape[-2:]

    sel = np.arange(len(positions))
    if n_positions and n_positions < len(sel):
        sel = np.linspace(0, len(sel) - 1, int(n_positions)).astype(int)
    pos = np.asarray(positions)[sel].astype(int)

    mo = int(min(max_order, N))
    blocks = sorted({int(b) for b in blocks if 1 <= int(b) <= N})
    onum, oden = np.zeros(mo + 1), np.zeros(mo + 1)
    lnum = lden = 0.0
    tnum, tden = np.zeros(mo + 1), np.zeros(mo + 1)
    bnum = {b: 0.0 for b in blocks}
    bden = {b: 0.0 for b in blocks}
    chi = dict(ms=0.0, fb=0.0, n=0.0)
    phis, e1, e1t = [], [], []

    phi = illumination(pr, geom)
    pw = (pr.abs() ** 2).sum(0)

    for s in range(0, len(pos), batch):
        p = pos[s:s + batch]
        rows = torch.as_tensor(p[:, 0][:, None] + np.arange(ny), dtype=torch.long,
                               device=device)
        cols = torch.as_tensor(p[:, 1][:, None] + np.arange(nx), dtype=torch.long,
                               device=device)
        O = torch.stack([ob[j][rows[:, :, None], cols[:, None, :]]
                         for j in range(N)])
        g = O - 1.0

        psi_ms, ords = orders(O, pr, geom, mo)
        F_ms = torch.fft.fft2(psi_ms, norm='ortho')
        cum = torch.zeros_like(ords[0])
        F1 = None
        for m in range(mo + 1):
            cum = cum + ords[m]
            F = torch.fft.fft2(cum, norm='ortho')
            d = F_ms - F
            onum[m] += float((d.abs() ** 2).sum())
            # relative to the exact multislice field, not the truncated one
            oden[m] += float((F_ms.abs() ** 2).sum())
            if m == 1:
                F1 = F
                pn = (d.abs() ** 2).sum(dim=(1, 2, 3))
                pd = (F_ms.abs() ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-30)
                e1.append(np.sqrt((pn / pd).cpu().numpy()))
        del ords

        # tangent hierarchy: O -> 1 + i s = 1 + log O
        psi_lin, ords_l = orders(1.0 + torch.log(O), pr, geom, mo)
        cum = torch.zeros_like(ords_l[0])
        for m in range(mo + 1):
            cum = cum + ords_l[m]
            F = torch.fft.fft2(cum, norm='ortho')
            d = F_ms - F
            tnum[m] += float((d.abs() ** 2).sum())
            tden[m] += float((F_ms.abs() ** 2).sum())
            if m == 1:
                pn = (d.abs() ** 2).sum(dim=(1, 2, 3))
                pd = (F_ms.abs() ** 2).sum(dim=(1, 2, 3)).clamp_min(1e-30)
                e1t.append(np.sqrt((pn / pd).cpu().numpy()))
        F_lin = torch.fft.fft2(psi_lin, norm='ortho')
        lnum += float(((F_ms - F_lin).abs() ** 2).sum())
        lden += float((F_ms.abs() ** 2).sum())
        del ords_l

        for b in blocks:
            F_b = torch.fft.fft2(blocked(g, pr, geom, b), norm='ortho')
            bnum[b] += float(((F_ms - F_b).abs() ** 2).sum())
            bden[b] += float((F_ms.abs() ** 2).sum())

        ph = torch.stack([torch.angle(O[j]) for j in range(N)]).sum(0)
        phis.append(np.sqrt(((ph ** 2 * pw[None]).sum(dim=(1, 2))
                             / pw.sum().clamp_min(1e-30)).cpu().numpy()))

        if data is not None:
            idx = sel[s:s + batch]
            frames = data(idx) if callable(data) else np.asarray(data[idx])
            meas = torch.as_tensor(frames, dtype=torch.float64, device=device)
            # measured frames are centre-ordered; the model spectra come out
            # of fft2 corner-ordered, so shift the model to the data layout
            I_ms = torch.fft.fftshift((F_ms.abs() ** 2).sum(1).double(),
                                      dim=(-2, -1))
            I_fb = torch.fft.fftshift((F1.abs() ** 2).sum(1).double(),
                                      dim=(-2, -1))
            for I, key in ((I_ms, 'ms'), (I_fb, 'fb')):
                sc = meas.sum(dim=(1, 2)) / I.sum(dim=(1, 2)).clamp_min(1e-30)
                r = (I * sc[:, None, None] - meas) ** 2 / (meas + 1.0)
                chi[key] += float(r.sum())
            chi['n'] += float(meas.numel())

        del O, g, psi_ms, cum
        if verbose:
            print("\r  %d / %d positions" % (min(s + batch, len(pos)), len(pos)),
                  end='', flush=True)
    if verbose:
        print()

    e1 = np.concatenate(e1)
    e1t = np.concatenate(e1t)
    phis = np.concatenate(phis)
    phi_rms = float(np.sqrt((phis ** 2).mean()))
    res = dict(
        N=N, gauge=gauge, n_positions=int(len(pos)),
        eps_order={m: float(np.sqrt(onum[m] / max(oden[m], 1e-300)))
                   for m in range(mo + 1)},
        eps_order_tangent={m: float(np.sqrt(tnum[m] / max(tden[m], 1e-300)))
                           for m in range(mo + 1)},
        eps_block={b: float(np.sqrt(bnum[b] / max(bden[b], 1e-300)))
                   for b in blocks},
        eps_multislice_tangent=float(np.sqrt(lnum / max(lden, 1e-300))),
        phi_tot_rms=phi_rms,
        eps1_correlated=float(0.5 * phi_rms ** 2 * (1 - 1.0 / N)),
        eps1_decorrelated=float(phi_rms ** 2 / (np.sqrt(2) * N)),
        eps1_per_pos=e1, eps1_tangent_per_pos=e1t, phi_per_pos=phis)
    if data is not None and chi['n']:
        res['chi2_multislice'] = chi['ms'] / chi['n']
        res['chi2_born'] = chi['fb'] / chi['n']
    return res


def object_difference(a, b):
    """Two reconstructions against each other, in a common gauge."""
    a = a[0] if a.ndim == 4 else a
    b = b[0] if b.ndim == 4 else b
    if a.shape != b.shape:
        return None
    a, _ = gauge_fix(a)
    b, _ = gauge_fix(b)
    out = []
    for j in range(a.shape[0]):
        x, y = a[j].ravel(), b[j].ravel()
        al = np.vdot(y, x) / max(np.vdot(y, y).real, 1e-30)
        rel = float(np.linalg.norm(x - al * y) / max(np.linalg.norm(x), 1e-30))
        d = np.angle(x) - np.angle(al * y)
        d -= d.mean()
        out.append(dict(slice=int(j), relative=rel,
                        phase_rms_rad=float(np.sqrt((d ** 2).mean()))))
    return out


def report(name, r):
    print("\n" + "=" * 66)
    print("%s   [N = %d, gauge %s, %d positions]"
          % (name, r['N'], r['gauge'], r['n_positions']))
    print("=" * 66)
    for m in sorted(r['eps_order']):
        tag = ('unscattered' if m == 0 else 'first Born' if m == 1 else
               'exact multislice, round-off' if m == r['N'] else '')
        print("  M = %-3d chord %.4e   tangent %.4e   %s"
              % (m, r['eps_order'][m], r['eps_order_tangent'][m], tag))
    for b in sorted(r['eps_block']):
        print("  %d block(s)   %.4e" % (b, r['eps_block'][b]))
    print("  tangent floor at M = N: %.4e" % r['eps_multislice_tangent'])
    print("  Phi_tot %.4f rad -> correlated %.4e, decorrelated %.4e"
          % (r['phi_tot_rms'], r['eps1_correlated'], r['eps1_decorrelated']))
    e = r['eps1_per_pos']
    print("  eps_1 per position: median %.4e, p05 %.4e, p95 %.4e"
          % (np.median(e), np.percentile(e, 5), np.percentile(e, 95)))
    if 'chi2_multislice' in r:
        print("  chi^2/px vs data: multislice %.4g, first Born %.4g (+%.1f%%)"
              % (r['chi2_multislice'], r['chi2_born'],
                 100 * (r['chi2_born'] - r['chi2_multislice'])
                 / max(r['chi2_multislice'], 1e-30)))
    print("  eps_1 / eps_0 = %.3f: the first order captures %.0f%% of the "
          "object's effect"
          % (r['eps_order'][1] / max(r['eps_order'][0], 1e-30),
             100 * (1 - r['eps_order'][1] / max(r['eps_order'][0], 1e-30))))


def figures(results, outdir, diff=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter, LogLocator
    except Exception as e:                                  # noqa: BLE001
        print("[warn] plotting unavailable: %s" % e)
        return
    os.makedirs(outdir, exist_ok=True)

    # Figure labels name the reconstructed OBJECT the eps_M analysis is run
    # on, not a model: eps_M truncates the exact multislice expansion at
    # order M for each fitted object (first Born itself is the M=1 member).
    disp = {'first Born': 'Born-fitted object',
            'multislice': 'MS-fitted object'}

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for name, r in results.items():
        Ms = sorted(r['eps_order'])
        ax.semilogy(Ms, [r['eps_order'][m] for m in Ms], 'o-', ms=4,
                    label='%s, chord' % disp.get(name, name))
        ax.semilogy(Ms, [r['eps_order_tangent'][m] for m in Ms], 'X--', ms=4,
                    alpha=0.7, label='%s, tangent' % disp.get(name, name))
    ax.set_xlabel('truncation order $M$')
    ax.set_ylabel(r'$\epsilon_M$')
    ax.set_title('truncation error, electron dataset')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'e01_truncation_error.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for name, r in results.items():
        p = r['phi_per_pos']
        ax.loglog(p, r['eps1_per_pos'], '.', ms=3, alpha=0.4,
                  color='tab:blue', label=r'chord $\Delta O = O - 1$')
        if 'eps1_tangent_per_pos' in r:
            ax.loglog(p, r['eps1_tangent_per_pos'], '.', ms=3, alpha=0.4,
                      color='crimson', label=r'tangent $\Delta O = is$')
        break
    ax.xaxis.set_major_locator(LogLocator(subs=(1., 2., 3., 5., 8.)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel(r'$\Phi_{\rm tot}$ (rad)')
    ax.set_ylabel(r'$\epsilon_1$')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'e02_phi_scatter.png'), dpi=200)
    plt.close(fig)
    print("[save] figures in %s" % outdir)


# ===========================================================================
#  CLI
# ===========================================================================

EXAMPLE = """
Nothing to do: point it at a PtyRAD checkpoint.

Step 1 -- look inside, so the key names and geometry are read rather than
guessed. This prints the hdf5 tree and exits:

  python ptyrad_diagnostics.py --inspect \\
      output/test_100/tBL_WSe2_born/<run>/model_iter0100.hdf5

Step 2 -- measure one reconstruction. Add --dx/--dz/--kv only if step 1 shows
they are not stored in the file:

  python ptyrad_diagnostics.py \\
      --model output/test_100/tBL_WSe2_born/<run>/model_iter0100.hdf5 \\
      --n-positions 256 --plot

Step 3 -- compare the two reconstructions of the same data, and score both
against the raw frames:

  python ptyrad_diagnostics.py \\
      --model   output/test_100/tBL_WSe2_born/<run>/model_iter0100.hdf5 \\
      --compare output/test_100/tBL_WSe2_multislice/<run>/model_iter0100.hdf5 \\
      --data    data/tBL_WSe2/Panel_g-h_Themis/scan_x128_y128.raw \\
      --data-shape 16384,128,128 --plot --outdir diagnostics_wse2

--n-positions subsamples the 16384 frames; 256 is plenty for eps_M and takes
seconds. Drop --data if you only want eps_M and not chi^2.
"""


def main(argv=None):
    if argv is None and len(__import__('sys').argv) == 1:
        print(EXAMPLE)
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--model', help='model_iterXXXX.hdf5 to analyse')
    ap.add_argument('--compare', help='a second model of the same data')
    ap.add_argument('--inspect', nargs='?', const=True,
                    help='print the hdf5 tree and exit')
    ap.add_argument('--data', help='raw diffraction data, for chi^2')
    ap.add_argument('--data-shape', default='16384,128,128')
    ap.add_argument('--data-dtype', default='float32')
    ap.add_argument('--data-offset', type=int, default=0,
                    help='bytes before the first frame')
    ap.add_argument('--data-gap', type=int, default=0,
                    help='bytes between frames (EMPAD1 .raw: 1024)')
    ap.add_argument('--data-flipT', default='1,0,0',
                    help="meas_flipT from the reconstruction; the model was "
                         "fitted to the transformed frames")
    ap.add_argument('--rebuild-H', action='store_true',
                    help='rebuild the propagator from dx and lambda instead of '
                         "using PtyRAD's stored H (a check on both)")
    ap.add_argument('--positions', help='.npy of integer (row, col) frame origins')
    ap.add_argument('--dx', type=float, default=None, help='pixel size in A')
    ap.add_argument('--dz', type=float, default=None, help='slice spacing in A')
    ap.add_argument('--kv', type=float, default=80.0)
    ap.add_argument('--max-order', type=int, default=2)
    ap.add_argument('--blocks', default='1')
    ap.add_argument('--n-positions', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--gauge', default='mean', choices=('mean', 'none'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--precision', default='double', choices=('single', 'double'))
    ap.add_argument('--outdir', default='ptyrad_diagnostics')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args(argv)

    if a.inspect:
        inspect(a.inspect if isinstance(a.inspect, str) else a.model)
        return 0
    if not a.model:
        ap.error('--model is required (or --inspect)')
    if torch is None:
        raise ImportError('PyTorch is required')

    dev = torch.device(a.device if not (a.device.startswith('cuda')
                                        and not torch.cuda.is_available()) else 'cpu')
    cdt = torch.complex128 if a.precision == 'double' else torch.complex64

    m = load_model(a.model)
    dx = a.dx or m['dx']
    dz = a.dz or m['dz']
    kv = a.kv or m['kv']
    if dx is None or dz is None:
        ap.error('pixel size or slice spacing not found; pass --dx and --dz')
    lam = electron_wavelength_A(kv)
    N = m['obj'].shape[-3] if m['obj'].ndim == 4 else m['obj'].shape[0]
    ny, nx = m['probe'].shape[-2:]
    H0 = None if a.rebuild_H else m.get('H0')
    geom = Geom((ny, nx), dx, lam, dz, N, dev, cdt, H0=H0)
    na = (m.get('conv_angle') or 0.0) * 1e-3
    print("[geom] %.0f kV -> lambda %.4f A; dx %.5f A; dz %.3f A; N %d"
          % (kv, lam, dx, dz, N))
    if na > 0:
        dof = lam / na ** 2
        print("[geom] probe NA %.1f mrad -> depth of field %.1f A, over a "
              "%.1f A specimen = %.2f DOF"
              % (na * 1e3, dof, dz * (N - 1), dz * (N - 1) / dof))
        print("[geom] shear a_c = 2 NA dz = %.3f A = %.2f pixels; "
              "resolution lambda/2NA = %.2f A"
              % (2 * na * dz, 2 * na * dz / dx, lam / (2 * na)))
    print("[geom] propagator: %s"
          % ("PtyRAD's stored H, raised to powers" if H0 is not None
             else "rebuilt from dx and lambda"))

    if a.positions:
        pos = np.load(a.positions)
    elif m['pos'] is not None and m['pos'].ndim == 2 and m['pos'].shape[1] == 2:
        pos = np.round(m['pos']).astype(int)
    else:
        ap.error('scan positions not found; pass --positions (integer frame '
                 'origins, one row per frame) or locate them with --inspect')
    obs = m['obj'][0] if m['obj'].ndim == 4 else m['obj']
    ok = ((pos[:, 0] >= 0) & (pos[:, 1] >= 0)
          & (pos[:, 0] + ny <= obs.shape[-2]) & (pos[:, 1] + nx <= obs.shape[-1]))
    if not ok.all():
        print("[warn] dropping %d of %d positions outside the object"
              % (int((~ok).sum()), len(ok)))
        pos = pos[ok]

    data = None
    if a.data:
        shape = tuple(int(x) for x in a.data_shape.split(','))
        base = np.dtype(a.data_dtype)
        if a.data_gap:
            # EMPAD-style .raw: each frame is followed by a junk gap; a plain
            # memmap without it reads progressively misaligned frames
            rec = np.dtype([('data', base, shape[1:]),
                            ('gap', np.uint8, a.data_gap)])
            raw = np.memmap(a.data, dtype=rec, mode='r', offset=a.data_offset,
                            shape=(shape[0],))['data']
        else:
            raw = np.memmap(a.data, dtype=base, mode='r',
                            offset=a.data_offset, shape=shape)
        flipT = tuple(int(x) for x in a.data_flipT.split(','))
        print("[data] %s %s, flipT %s, offset %d, gap %d"
              % (a.data, shape, flipT, a.data_offset, a.data_gap))
        # PtyRAD applies meas_flipT on load; the model was fitted to the
        # transformed frames, so the same transform has to be applied here or
        # chi^2 compares a pattern with its own transpose. Negative detector
        # values are clipped, mirroring PtyRAD's clip_neg preprocessing.
        def prep(idx):
            d = np.clip(np.asarray(raw[idx], dtype=np.float64), 0.0, None)
            if flipT[0]:
                d = np.swapaxes(d, -2, -1)
            if len(flipT) > 1 and flipT[1]:
                d = d[..., ::-1, :]
            if len(flipT) > 2 and flipT[2]:
                d = d[..., :, ::-1]
            return np.ascontiguousarray(d)
        data = prep

    blocks = tuple(int(b) for b in a.blocks.split(',') if b.strip())
    results = {}
    r = measure(m, pos, geom, dev, cdt, batch=a.batch_size,
                max_order=a.max_order, blocks=blocks,
                n_positions=a.n_positions, data=data, gauge=a.gauge)
    results['first Born'] = r
    report('first-Born reconstruction', r)

    if a.compare:
        m2 = load_model(a.compare)
        r2 = measure(m2, pos, geom, dev, cdt, batch=a.batch_size,
                     max_order=a.max_order, blocks=blocks,
                     n_positions=a.n_positions, data=data, gauge=a.gauge)
        results['multislice'] = r2
        report('multislice reconstruction', r2)
        d = object_difference(m['obj'], m2['obj'])
        if d:
            print("\nRecovered objects against each other, common gauge:")
            for row in d:
                print("  slice %2d: %.4f relative, %.4f rad rms"
                      % (row['slice'], row['relative'], row['phase_rms_rad']))
            print("  mean over slices: %.4f relative, %.4f rad rms"
                  % (np.mean([x['relative'] for x in d]),
                     np.mean([x['phase_rms_rad'] for x in d])))
        e1, e2 = r['eps_order'][1], r2['eps_order'][1]
        print("\n  eps_1: %.4e (Born-fitted) vs %.4e (multislice-fitted), "
              "ratio %.2f" % (e1, e2, e2 / max(e1, 1e-30)))

    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, 'results.json'), 'w') as fh:
        json.dump({k: {kk: (vv if not isinstance(vv, np.ndarray) else
                            dict(median=float(np.median(vv)),
                                 p95=float(np.percentile(vv, 95))))
                       for kk, vv in v.items()} for k, v in results.items()},
                  fh, indent=2, default=str)
    if a.plot:
        figures(results, a.outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())