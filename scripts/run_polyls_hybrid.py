import os

# Change this to the ABSOLUTE PATH to the demo/ folder so you can correctly access data/ and params/
work_dir = "../"  # Leave as-is when running from ptyrad/demo/scripts/
os.chdir(work_dir)
print("Current working dir: ", os.getcwd())

import numpy as np
import torch

import ptyrad.linesearch as ls
from ptyrad.load import load_params
from ptyrad.models import PtychoAD
from ptyrad.reconstruction import (
    PtyRADSolver,
    create_optimizer,
    loss_logger,
    prepare_recon,
    refit_born_coeffs,
)
from ptyrad.save import save_results
from ptyrad.utils import CustomLogger, print_system_info, set_gpu_device, time_sync, vprint
from ptyrad.visualization import plot_summary

# Hybrid driver (docs/polyls_hybrid_order_plan.md §4.6): exact polynomial line
# search at the CURRENT Born order + the delta-driven detector-refit order
# schedule. Same layout as run_linesearch.py, with two additions:
#
#   1. refit_born_coeffs fires after the constraints each iteration (the
#      recon_step ordering, spec §4.4) — with born_coeffs_refit
#      {'method': 'detector', 'grow_tol': ...} it refits the coefficients AND
#      grows model.born_iterations whenever the detector-fit residual delta
#      (a direct, calibration-free detector-error estimate) exceeds grow_tol.
#      The line search re-reads M and the coefficients every view, so the
#      next sweep runs at the grown order; within a sweep both are constant.
#
#   2. Tolerance annealing (plan §4.5): grow_tol starts at the config value
#      (5e-2 — ends around M ~ 7 on PSO) and tightens to ANNEAL_TOL = 1e-2
#      (the bias-safe cutoff, ends at M ~ 10-11) when the total loss plateaus
#      (relative improvement < PLATEAU_RTOL over PLATEAU_SPAN iterations) or
#      at ANNEAL_FRAC of NITER, whichever comes first.

params_paths = [
    os.environ.get(
        "PTYRAD_LS_PARAMS",
        "/home/dnz75396/ptyrad/demo/params/pso_polyls_hybrid.yml",
    )
]
run_name = [os.environ.get("PTYRAD_LS_RUNNAME", "polyls_hybrid_PSO")]

ANNEAL_TOL = 1e-2  # bias-safe delta cutoff (frozen-run calibration, plan §6)
ANNEAL_FRAC = 0.6  # anneal at this fraction of NITER at the latest
PLATEAU_SPAN = 5  # iterations
PLATEAU_RTOL = 1e-3  # relative loss improvement below this = plateau

# §8 knobs. ls_damp = 0.5 is load-bearing with the default amplitude direction
# objective (spec §5) — identical policy at every Born order; do not "fix" it.
ls_config = ls.LineSearchConfig(
    alpha=1.0,
    beta=1.0,
    ls_damp=0.5,
    max_step=0.0,
    object_denom="max",
    probe_denom="max",
    direction_objective="amplitude",
)

WATCHDOG_EVERY = 512  # views between in-sweep §6 watchdog lines

for i, params_path in enumerate(params_paths):
    print(f"Running polyls-hybrid reconstruction with params file: {params_path}")
    logger = CustomLogger(
        log_file=f"ptyrad_log_{run_name[i]}.txt",
        log_dir="auto",
        prefix_time="datetime",
        show_timestamp=True,
    )
    print_system_info()

    params = load_params(params_path, validate=True)

    batch_size = params["recon_params"]["BATCH_SIZE"]["size"]
    params["recon_params"]["BATCH_SIZE"]["grad_accumulation"] = 1
    if batch_size == 1:
        params["recon_params"]["GROUP_MODE"] = "random"

    device = set_gpu_device(gpuid=0)  # pin the PHYSICAL device via CUDA_VISIBLE_DEVICES=<UUID>

    solver = PtyRADSolver(params, device=device, logger=logger)
    model = PtychoAD(
        solver.init.init_variables, params["model_params"], device=device, verbose=True
    )
    if model.solver_type != "born":
        raise ValueError(f"{params_path} has solver_type={model.solver_type!r}; need 'born'")
    refit_cfg = getattr(model, "born_refit", None)
    if not (refit_cfg and refit_cfg.get("method") == "detector" and refit_cfg.get("grow_tol")):
        raise ValueError(
            "The hybrid driver expects born_coeffs_refit with method: 'detector' and "
            "grow_tol set — that is the delta-driven order schedule (plan §1). For a "
            "fixed-order line-search run use run_linesearch.py."
        )
    grow_tol_0 = float(refit_cfg["grow_tol"])

    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
    indices, batches, output_path = prepare_recon(model, solver.init, params)
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    ls_state = ls.LineSearchState()
    NITER = params["recon_params"]["NITER"]
    SAVE_ITERS = params["recon_params"]["SAVE_ITERS"]
    selected_figs = params["recon_params"]["selected_figs"]
    probe_on = model.lr_params.get("probe", 0) != 0
    probe_start = model.start_iter.get("probe") or 1
    rng = np.random.default_rng(model.random_seed)
    loss_names = list(solver.loss_fn.loss_params.keys())
    watchdog_batches = max(WATCHDOG_EVERY // batch_size, 1)
    anneal_iter_cap = max(int(ANNEAL_FRAC * NITER), 1)
    annealed = False
    loss_history = []

    vprint(
        f"### Start the PtyRAD polyls-hybrid reconstruction (batch size {batch_size}, "
        f"start order n={model.born_iterations}, grow_tol {grow_tol_0:g} -> "
        f"{ANNEAL_TOL:g}) ###"
    )
    for niter in range(1, NITER + 1):
        start_iter_t = time_sync()
        update_probe = probe_on and niter >= probe_start
        M_iter = int(model.born_iterations)  # constant within the sweep (refit fires after)
        fallback_o = ls_config.ls_damp * ls_config.alpha / model.n_slice
        n_views = 0
        view_losses = []
        iter_losses = []
        n0 = len(ls_state.steps_o)

        order = rng.permutation(len(batches))
        t_block, v_block = time_sync(), 0
        for nbatch, batch_i in enumerate(order, start=1):
            batch = np.atleast_1d(np.asarray(batches[batch_i]))
            if nbatch == 1:
                H_iter = model.get_propagators_3d(
                    model.get_propagators(torch.as_tensor(batch, device=device))
                ).detach()
            if len(batch) == 1:
                diag = ls.linesearch_model_update(
                    model,
                    int(batch[0]),
                    config=ls_config,
                    state=ls_state,
                    update_probe=update_probe,
                    loss_fn=solver.loss_fn,
                    H=H_iter,
                )
            else:
                diag = ls.linesearch_model_update_batched(
                    model,
                    batch,
                    config=ls_config,
                    state=ls_state,
                    update_probe=update_probe,
                    loss_fn=solver.loss_fn,
                    H=H_iter,
                )
            n_views += len(batch)
            v_block += len(batch)
            view_losses.append(diag["loss"])
            iter_losses.append(torch.stack([lv.detach() for lv in diag["losses"]]))

            if nbatch % watchdog_batches == 0:
                t_now = time_sync()
                e_dir = float(torch.stack(view_losses[-watchdog_batches:]).mean())
                vprint(
                    f"  iter {niter} view {n_views}/{len(indices)} | n={M_iter} | "
                    f"E_dir {e_dir:.4e} | "
                    f"probe peak/mean {diag['probe_peak'].item():.3e}"
                    f"/{diag['probe_mean'].item():.3e} | "
                    f"{(t_now - t_block) / v_block * 1e3:.2f} ms/view"
                )
                t_block, v_block = t_now, 0
                if not np.isfinite(e_dir):
                    raise FloatingPointError(
                        f"non-finite direction objective at iter {niter}, view {n_views}"
                    )

        # Constraints then coefficient refit/order growth, once per iteration,
        # after the sweep — the recon_step ordering (spec §4.4), so c and M
        # are constant within a sweep and the exact in-batch updates stay valid.
        solver.constraint_fn(model, niter)
        refit_born_coeffs(model, niter, verbose=True)

        iter_t = time_sync() - start_iter_t
        err = float(torch.stack(view_losses).mean())
        losses_np = torch.stack(iter_losses).cpu().numpy()
        batch_losses = {name: list(losses_np[:, k]) for k, name in enumerate(loss_names)}
        a_iter = np.asarray(ls_state.steps_o[n0:])
        b_iter = np.asarray(ls_state.steps_p[n0:] if update_probe else [np.nan])

        frac_fallback = float(np.mean(np.isclose(a_iter, fallback_o, rtol=1e-12)))
        pint = torch.view_as_complex(model.opt_probe.data).abs().square()
        grew_str = "" if model.born_iterations == M_iter else f" -> n={model.born_iterations}"
        vprint(
            f"Iter {niter:4d} | n={M_iter}{grew_str} | E_dir {err:.5e} | "
            f"a med {np.median(a_iter):.3e} (iqr {np.percentile(a_iter, 25):.2e}"
            f"..{np.percentile(a_iter, 75):.2e}, fallback frac {frac_fallback:.2f}) | "
            f"b med {np.median(b_iter):.3e} | "
            f"probe peak/mean {float(pint.max()):.3e}/{float(pint.mean()):.3e} | "
            f"{iter_t:.2f} s ({iter_t / max(n_views, 1) * 1e3:.2f} ms/view)"
        )
        if frac_fallback > 0.9:
            vprint(
                "WARNING: >90% of object steps at exactly ls_damp*alpha/N — "
                "the polynomial solve is degenerating (see spec §4.3). Check the "
                "coefficient magnitudes before trusting this run."
            )
        loss_iter = loss_logger(batch_losses, niter, iter_t, verbose=True)
        loss_history.append(loss_iter)

        # ---- tolerance annealing (plan §4.5): 5e-2 -> 1e-2 -------------------
        if not annealed:
            plateau = (
                len(loss_history) > PLATEAU_SPAN
                and abs(loss_history[-1 - PLATEAU_SPAN]) > 0
                and (loss_history[-1 - PLATEAU_SPAN] - loss_history[-1])
                / abs(loss_history[-1 - PLATEAU_SPAN])
                < PLATEAU_RTOL
            )
            if plateau or niter >= anneal_iter_cap:
                model.born_refit["grow_tol"] = ANNEAL_TOL
                annealed = True
                vprint(
                    f"### Annealed grow_tol {grow_tol_0:g} -> {ANNEAL_TOL:g} at iter "
                    f"{niter} ({'loss plateau' if plateau else 'iteration cap'}) ###"
                )

        model.loss_iters.append((niter, loss_iter))
        model.iter_times.append(iter_t)
        model.dz_iters.append((niter, model.opt_slice_thickness.detach().cpu().numpy()))
        model.avg_tilt_iters.append((niter, model.opt_obj_tilts.detach().mean(0).cpu().numpy()))

        if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
            with torch.no_grad():
                save_results(output_path, model, params, optimizer, niter, indices, batch_losses)
                plot_summary(
                    output_path,
                    model,
                    niter,
                    indices,
                    solver.init.init_variables,
                    selected_figs=selected_figs,
                    show_fig=False,
                    save_fig=True,
                    verbose=True,
                )

    vprint(
        f"### Finished {NITER} polyls-hybrid iterations at final order "
        f"n={model.born_iterations}, averaged iter_t = {np.mean(model.iter_times):.5g} s ###"
    )
