"""Closed-form PIE updates: ePIE for the inter-slice Born model, 3PIE for
multislice — no autograd, no optimizer.

Selected with model_params['update_rule'] = 'pie' (default 'autodiff' is the
usual loss.backward() + optimizer path). Each batch does one forward pass
under torch.no_grad() that keeps the per-slice intermediate fields, applies
the modulus (amplitude) projection at the detector, and writes the analytic
PIE updates straight into the parameter tensors:

- solver_type 'multislice': 3PIE (Maiden et al., multislice ePIE),
  following ptypy's ThreePIE engine (ptypy/custom/threepie_cupy.py): the
  updated exit wave is swept backwards slice by slice; each slice first
  takes the ePIE object update from its entrance wave (beta_obj), then the
  entrance wave itself is revised through the UPDATED object (beta_wave;
  ptypy's pr_update_local at every slice) and back-propagated one step as
  the updated exit of the slice above. At the entrance plane the revision
  is the probe update (beta_probe). beta_wave defaults to beta_probe
  (ptypy's coupling, which also damps the inter-slice transport by
  beta_probe per hop); beta_wave = 1.0 keeps the transport undamped, which
  deep stacks need. On tBL-WSe2 the engine locks into a limit cycle at
  beta = 1 (the per-view full-strength probe step); beta_obj = beta_probe
  = 0.1 with beta_wave = 1.0 converges below the batch-1 Adam baseline.
- solver_type 'born' (born_iterations = 1, the inter-slice first Born /
  ISS model): ePIE applied to every slice with its own defocused probe
  v0_j = P_{z_j} P (paper-1 Sec. Inversion). The detector correction is
  shared between the branches: chi_j = F^{-1}[H_j dPsi] returns it to each
  slice, the object step is conj(v0_j) chi_j / max |v0_j|^2 per slice, and
  the probe step is the exact adjoint through the direct term and all
  branches, evaluated with the same chi_j (two extra FFTs per slice).

Mixed-state probes follow mixed-state ePIE: one shared intensity ratio
sqrt(I_meas / I_model) scales every mode, object updates sum over probe
modes, the probe update is per mode. Object modes are carried through the
same broadcasting as the forward models (occupancy-weighted at the probe
update); betas come from model_params['pie_params'] = {'beta_obj', 'beta_probe'}.

The data loss is NOT part of the update (the modulus projection is the
update); loss_fn is evaluated on the model intensities afterwards for
logging only, exactly like mliss logs its diagnostics.

Conventions match the production forward models: internal propagation uses
unnormalized fft2/ifft2 round trips (unitary), the detector transform is
norm='ortho', intensities are fftshifted and compared against
model.get_measurements(). Batched write-back assumes the batch's patches do
not overlap (batch size 1 in practice); a guard refuses batches > 1 with
overlapping ROIs rather than silently racing.
"""

import torch
from torch.fft import fft2, ifft2

from ptyrad.utils import fftshift2, imshift_batch

_TINY = 1e-12


@torch.no_grad()
def pie_batch_update(model, indices, beta_obj=1.0, beta_probe=1.0, eps=1e-10,
                     probe_direct_only=False, beta_wave=None):
    """One PIE update from the batch `indices`. Returns (model_DP, measured_DP,
    object_patches) for loss logging; parameters are updated in place."""
    object_patches = model.get_obj_patches(indices)  # (N, omode, Nz, Ny, Nx, 2)
    probes = model.get_probes(indices)  # (N|1, pmode, Ny, Nx)
    H = model.get_propagators(indices)  # (N|1, Ky, Kx)
    meas = model.get_measurements(indices)  # (N, Ky, Kx), fftshifted
    occu = model.omode_occu

    N = object_patches.shape[0]
    if probes.shape[0] == 1 and N > 1:
        probes = probes.expand(N, -1, -1, -1)
    O = torch.polar(object_patches[..., 0], object_patches[..., 1])  # (N, omode, Nz, Ny, Nx)

    update_obj = model.opt_obja.requires_grad or model.opt_objp.requires_grad
    update_probe = model.opt_probe.requires_grad

    if model.solver_type == "born":
        if model.born_iterations != 1:
            raise ValueError(
                "update_rule 'pie' implements ePIE for the inter-slice FIRST Born "
                f"model only (born_iterations = 1), got {model.born_iterations}"
            )
        H3d = model.get_propagators_3d(H)  # (N|1,1,1,Nz,Ny,Nx)
        dp_model, O_new, dP = _epie_born_compiled(
            O, probes, H3d, occu, meas, beta_obj, beta_probe, update_probe, eps,
            probe_direct_only,
        )
    elif model.solver_type == "multislice" and model.splitting in (None, "", "lie_trotter"):
        dp_model, O_new, dP = _pie3_multislice_compiled(
            O, probes, H, occu, meas, beta_obj, beta_probe, update_probe, eps,
            beta_wave=beta_wave,
        )
    else:
        raise ValueError(
            f"update_rule 'pie' supports solver_type 'born' or plain 'multislice', "
            f"got '{model.solver_type}' (splitting='{getattr(model, 'splitting', None)}')"
        )

    if update_obj:
        _write_back_object(model, indices, O_new)
    if update_probe and dP is not None:
        _write_back_probe(model, indices, dP)

    return dp_model, meas, object_patches


def _modulus_ratio(Psi_hat, occu, meas, eps):
    """Shared mixed-state modulus projection: intensity in the fftshifted
    frame, ratio returned in the unshifted frame (even sizes: fftshift is an
    involution, as everywhere in this codebase)."""
    I_model = (
        fftshift2(
            torch.sum(Psi_hat.abs().square() * occu.view(1, 1, -1, 1, 1), dim=(1, 2))
        )
        + eps
    )
    ratio = torch.sqrt(meas.clamp_min(0.0) / I_model)
    return I_model, fftshift2(ratio)[:, None, None]  # broadcast over pmode, omode


def _epie_born(O, probes, H3d, occu, meas, beta_obj, beta_probe, update_probe, eps,
               probe_direct_only=False):
    """ePIE for the inter-slice first Born model (born_iterations = 1)."""
    P_k = fft2(probes, norm="ortho")[:, :, None, None]  # (N,pmode,1,1,Ny,Nx)
    v0 = ifft2(H3d * P_k, norm="ortho")  # (N,pmode,1,Nz,Ny,Nx)
    dO = (O - 1.0).unsqueeze(1)  # (N,1,omode,Nz,Ny,Nx)

    scat_k = fft2(dO * v0, norm="ortho") * H3d.conj()
    Psi_hat = P_k.squeeze(3) + scat_k.sum(dim=3)  # (N,pmode,omode,Ny,Nx)

    I_model, r = _modulus_ratio(Psi_hat, occu, meas, eps)
    dPsi = Psi_hat * (r - 1.0)  # (N,pmode,omode,Ny,Nx)

    # the correction returned to each slice: chi_j = F^{-1}[H_j dPsi]
    # (dPsi lives in k-space already — it is the detector-plane correction)
    chi = ifft2(H3d * dPsi[:, :, :, None], norm="ortho")

    # ePIE object step, each slice against its own probe v0_j. The
    # denominator is summed over the slices as well (sum-model ePIE): all
    # branches share ONE detector correction, so a per-slice denominator
    # would apply it at full strength N_z times — marginally stable at 12
    # slices, divergent at 48. The joint denominator shares the correction
    # between the branches and makes the step self-normalizing in N_z.
    num = (v0.conj() * chi).sum(dim=1)  # (N,omode,Nz,Ny,Nx)
    den = (
        (v0.abs().square())
        .sum(dim=1)
        .sum(dim=2, keepdim=True)
        .amax(dim=(-2, -1), keepdim=True)
    )
    O_new = O + beta_obj * num / (den + _TINY)

    dP = None
    if update_probe:
        # exact adjoint through the direct term and every branch
        if probe_direct_only:
            # 'free path' update: the probe sees only the direct unscattered
            # term of the readout -- the branch adjoints (which mix in the
            # noisy object) are dropped
            dP_modes = ifft2(dPsi, norm="ortho")
        else:
            back_k = (H3d.conj() * fft2(dO.conj() * chi, norm="ortho")).sum(dim=3)
            dP_modes = ifft2(dPsi + back_k, norm="ortho")
        dP = torch.sum(dP_modes * occu.view(1, 1, -1, 1, 1), dim=2)  # (N,pmode,Ny,Nx)
        den_p = (
            1.0
            + (dO.abs().square().sum(dim=3) * occu.view(1, 1, -1, 1, 1))
            .sum(dim=2)
            .amax(dim=(-2, -1), keepdim=True)
        )
        dP = beta_probe * dP / (den_p + _TINY)

    return I_model, O_new, dP


def _pie3_multislice(O, probes, H, occu, meas, beta_obj, beta_probe, update_probe, eps,
                     beta_wave=None):
    """3PIE: multislice ePIE with the sequential backward sweep, following
    ptypy's ThreePIE engine (ptypy/custom/threepie_cupy.py) step by step.

    Forward sweep: store the entrance wave of every slice and its exit
    O_s psi_s; propagate each exit one step to the next entrance. At the
    detector the modulus constraint on the last exit gives the UPDATED exit
    wave. Backward sweep, per slice (Gauss-Seidel, as in ptypy):

    1. object:   O_s   += beta_obj  * conj(psi_s)  * (back - exit_s) / max sum_p |psi_s|^2
    2. entrance: psi_s += beta_wave * conj(O_s^new) * (back - exit_s) / max |O_s^new|^2
       (ptypy's pr_update_local, applied at every slice with the probe
       coefficient; the object update is applied first and the REVISED
       object enters the wave revision)
    3. the revised entrance, back-propagated one step, is the updated exit
       of the slice above: back_{s-1} = P^{-1}[conj(H) F[psi_s^new]].

    At slice 0 the revision coefficient is beta_probe and the revision IS
    the probe update. beta_wave defaults to beta_probe, which is exactly
    ptypy's coupling (their single probe coefficient both revises the
    waves and damps the inter-slice transport); set beta_wave = 1.0 to
    keep the transport undamped while still damping the probe step, which
    matters for deep stacks, where a beta_wave < 1 suppresses the
    correction reaching the entrance slices geometrically."""
    if beta_wave is None:
        beta_wave = beta_probe
    Nz = O.shape[2]
    H1 = H[:, None, None]  # broadcast over pmode, omode

    entrance = [probes[:, :, None]]  # (N,pmode,omode|1,Ny,Nx)
    exits = []
    for s in range(Nz):
        exits.append(entrance[s] * O[:, None, :, s])
        if s < Nz - 1:
            entrance.append(ifft2(H1 * fft2(exits[s])))

    Psi_hat = fft2(exits[-1], norm="ortho")  # (N,pmode,omode,Ny,Nx)
    I_model, r = _modulus_ratio(Psi_hat, occu, meas, eps)
    back = ifft2(Psi_hat * r, norm="ortho")  # updated exit wave of the last slice

    O_new = O.clone()
    dP = None
    for s in range(Nz - 1, -1, -1):
        psi_s = entrance[s]
        diff = back - exits[s]  # updated minus recorded exit of slice s
        num = (psi_s.conj() * diff).sum(dim=1)  # (N,omode,Ny,Nx)
        den = (psi_s.abs().square()).sum(dim=1).amax(dim=(-2, -1), keepdim=True)
        O_new[:, :, s] = O_new[:, :, s] + beta_obj * num / (den + _TINY)

        On = O_new[:, None, :, s]  # revised object, as in ptypy
        coef = beta_wave if s > 0 else beta_probe
        dpsi = coef * On.conj() * diff / (
            On.abs().square().amax(dim=(-2, -1), keepdim=True) + _TINY
        )
        if s > 0:
            back = ifft2(H1.conj() * fft2(psi_s + dpsi))
        elif update_probe:
            dP = torch.sum(dpsi * occu.view(1, 1, -1, 1, 1), dim=2)

    return I_model, O_new, dP


# Compiled entry points, like the production forward models: at batch 1 the
# updates are kernel-launch bound in eager mode (the 3PIE sweep alone is
# ~200 launches), and max-autotune's CUDA graphs collapse that. The eager
# _epie_born/_pie3_multislice stay reachable for tests. Outputs are consumed
# within the same batch (write-back + loss logging), so CUDA-graph buffer
# reuse across calls is safe.
_epie_born_compiled = torch.compile(_epie_born, mode="max-autotune")
_pie3_multislice_compiled = torch.compile(_pie3_multislice, mode="max-autotune")


def _write_back_object(model, indices, O_new):
    """Scatter the updated complex patches back into the amp/phase canvases."""
    if len(indices) > 1:
        # batched PIE write-back requires disjoint ROIs; the production PIE
        # runs are batch 1, so refuse instead of racing
        raise ValueError("update_rule 'pie' supports BATCH_SIZE 1 only")
    grid_y = model.rpy_grid[None, :, :] + model.crop_pos[indices, None, None, 0]
    grid_x = model.rpx_grid[None, :, :] + model.crop_pos[indices, None, None, 1]
    amp = O_new.abs().clamp_min(_TINY)
    phase = O_new.angle()
    for n in range(O_new.shape[0]):
        model.opt_obja.data[:, :, grid_y[n], grid_x[n]] = amp[n]
        model.opt_objp.data[:, :, grid_y[n], grid_x[n]] = phase[n]


def _write_back_probe(model, indices, dP):
    """Average the per-view probe updates back at the unshifted probe."""
    if model.shift_probes:
        shifts = model.opt_probe_pos_shifts[indices]
        upd = torch.stack(
            [
                imshift_batch(dP[n], shifts=-shifts[n : n + 1], grid=model.shift_probes_grid)[0]
                for n in range(dP.shape[0])
            ]
        ).mean(dim=0)
    else:
        upd = dP.mean(dim=0)
    torch.view_as_complex(model.opt_probe.data).add_(upd)
