import os

# Change this to the ABSOLUTE PATH to the demo/ folder so you can correctly access data/ and params/
work_dir = "../"  # Leave this as-is if you're running from the `ptyrad/demo/scripts/` folder, this will change it back to demo/

os.chdir(work_dir)
print("Current working dir: ", os.getcwd())
# The printed working dir should be ".../ptyrad/demo" to locate the demo params files easily
# Note that the output/ directory will be automatically generated under your working directory

import numpy as np
import torch

import ptyrad.linesearch as ls
from ptyrad.load import load_params
from ptyrad.models import PtychoAD
from ptyrad.reconstruction import PtyRADSolver, create_optimizer, loss_logger, prepare_recon
from ptyrad.save import save_results
from ptyrad.utils import CustomLogger, print_system_info, set_gpu_device, time_sync, vprint
from ptyrad.visualization import plot_summary

# Same layout as run_basic.py, but the iteration loop drives the exact quartic
# line search (src/ptyrad/linesearch.py, LINESEARCH_BORN_SPEC.md) instead of the
# gradient-descent optimizer. Requires a params file with solver_type: born and
# born_iterations: 1 — the far field is affine in the object only for single
# scattering, which is what makes the step exact.
#
# BATCH_SIZE (from the params file) selects the update mode:
#   size = 1  : per-view updates, probe stepped every view (the spec §3 design
#               point; converges fastest per iteration)
#   size > 1  : ptypy-style joint batch update — object gradient
#               scatter-accumulated on the canvas, ONE scalar step per batch,
#               probe step averaged over the batch. Use this for batch-parity
#               comparisons against optimizer runs.
params_paths = [
    "/home/dnz75396/ptyrad/demo/params/PSO_reconstruct_born_paper.yml"
]

run_name = ["linesearch_born_PSO"]  # Used to name the log file. Change to any string you like

# None = honor the params file's BATCH_SIZE.
FORCE_BATCH_SIZE = None

# §8 knobs. ls_damp = 0.5 is load-bearing with the default amplitude direction
# objective — it is not a stability fudge factor, do not "clean it up" to 1.0.
ls_config = ls.LineSearchConfig(
    alpha=1.0,  # object fallback step alpha/N, taken only on cubic degeneracy
    beta=1.0,  # probe fallback step beta/N
    ls_damp=0.5,
    max_step=0.0,  # 0 = no clip
    object_denom="max",
    probe_denom="max",
    direction_objective="amplitude",  # §5 option (a), parity with the ptypy engine
)

WATCHDOG_EVERY = 512  # views between in-sweep §6 watchdog lines

for i, params_path in enumerate(params_paths):
    print(f"Running line-search reconstruction with params file: {params_path}")
    logger = CustomLogger(
        log_file=f"ptyrad_log_{run_name[i]}.txt",
        log_dir="auto",
        prefix_time="datetime",
        show_timestamp=True,
    )
    print_system_info()

    params = load_params(params_path, validate=True)

    batch_size = params["recon_params"]["BATCH_SIZE"]["size"]
    if FORCE_BATCH_SIZE is not None:
        batch_size = FORCE_BATCH_SIZE
        params["recon_params"]["BATCH_SIZE"]["size"] = batch_size
    params["recon_params"]["BATCH_SIZE"]["grad_accumulation"] = 1
    if batch_size == 1:
        # grouping is meaningless for single-view batches (and 'sparse' kmeans
        # breaks at n_clusters == n_views)
        params["recon_params"]["GROUP_MODE"] = "random"

    device = set_gpu_device(gpuid=0)  # Pass gpuid=None to run on CPU (much slower)

    # Reuse PtyRADSolver only for initialization (data, loss, constraints);
    # the optimizer below is created for checkpoint compatibility and never stepped.
    solver = PtyRADSolver(params, device=device, logger=logger)
    model = PtychoAD(
        solver.init.init_variables, params["model_params"], device=device, verbose=True
    )
    if model.solver_type != "born" or model.born_iterations != 1:
        raise ValueError(
            f"{params_path} has solver_type={model.solver_type!r}, "
            f"born_iterations={model.born_iterations} — the exact line search needs "
            "solver_type: born with born_iterations: 1"
        )

    optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)
    indices, batches, output_path = prepare_recon(model, solver.init, params)
    if logger is not None and logger.flush_file:
        logger.flush_to_file(log_dir=output_path)

    ls_state = ls.LineSearchState()
    NITER = params["recon_params"]["NITER"]
    SAVE_ITERS = params["recon_params"]["SAVE_ITERS"]
    selected_figs = params["recon_params"]["selected_figs"]
    fallback_o = ls_config.ls_damp * ls_config.alpha / model.n_slice
    # honor the model's own probe schedule: lr acts as an on/off flag here
    # (the exact step sets its own magnitude), start_iter gates when it begins
    probe_on = model.lr_params.get("probe", 0) != 0
    probe_start = model.start_iter.get("probe") or 1
    rng = np.random.default_rng(model.random_seed)
    loss_names = list(solver.loss_fn.loss_params.keys())
    watchdog_batches = max(WATCHDOG_EVERY // batch_size, 1)

    vprint(
        f"### Start the PtyRAD exact-line-search reconstruction "
        f"(batch size {batch_size}, {'joint-batch' if batch_size > 1 else 'per-view'} mode) ###"
    )
    for niter in range(1, NITER + 1):
        start_iter_t = time_sync()
        update_probe = probe_on and niter >= probe_start
        n_views = 0
        view_losses = []  # E_dir per update, kept on-device until the iter ends
        iter_losses = []  # standard PtyRAD losses per update, on-device
        n0 = len(ls_state.steps_o)

        order = rng.permutation(len(batches))
        t_block, v_block = time_sync(), 0
        for nbatch, batch_i in enumerate(order, start=1):
            batch = np.atleast_1d(np.asarray(batches[batch_i]))
            # tilt/thickness are static within an iteration here, so the 3D
            # propagator stack is hoisted out of the per-batch work
            if nbatch == 1:
                H_iter = model.get_propagators_3d(
                    model.get_propagators(torch.as_tensor(batch, device=device))
                ).detach()
            if len(batch) == 1:
                diag = ls.linesearch_model_update(
                    model, int(batch[0]), config=ls_config, state=ls_state,
                    update_probe=update_probe, loss_fn=solver.loss_fn, H=H_iter,
                )
            else:
                diag = ls.linesearch_model_update_batched(
                    model, batch, config=ls_config, state=ls_state,
                    update_probe=update_probe, loss_fn=solver.loss_fn, H=H_iter,
                )
            n_views += len(batch)
            v_block += len(batch)
            view_losses.append(diag["loss"])
            iter_losses.append(torch.stack([lv.detach() for lv in diag["losses"]]))

            # §6 in-sweep watchdog: probe runaway must be visible before the
            # per-iteration constraints (e.g. fix_probe_int) get to fire.
            # This is the only host sync inside the sweep.
            if nbatch % watchdog_batches == 0:
                t_now = time_sync()
                e_dir = float(torch.stack(view_losses[-watchdog_batches:]).mean())
                vprint(
                    f"  iter {niter} view {n_views}/{len(indices)} | "
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

        # Constraints fire once per iteration, after the sweep — same call
        # site as recon_loop, so the in-batch exact field update stays valid.
        solver.constraint_fn(model, niter)

        iter_t = time_sync() - start_iter_t
        err = float(torch.stack(view_losses).mean())
        losses_np = torch.stack(iter_losses).cpu().numpy()  # (n_updates, n_losses)
        batch_losses = {name: list(losses_np[:, k]) for k, name in enumerate(loss_names)}
        a_iter = np.asarray(ls_state.steps_o[n0:])
        b_iter = np.asarray(ls_state.steps_p[n0:] if update_probe else [np.nan])

        # §6: the step log is the only cheap detector of the silent-fallback
        # failure — a healthy run shows live spread in the accepted steps.
        frac_fallback = float(np.mean(np.isclose(a_iter, fallback_o, rtol=1e-12)))
        pint = torch.view_as_complex(model.opt_probe.data).abs().square()
        vprint(
            f"Iter {niter:4d} | E_dir {err:.5e} | "
            f"a med {np.median(a_iter):.3e} (iqr {np.percentile(a_iter, 25):.2e}"
            f"..{np.percentile(a_iter, 75):.2e}, fallback frac {frac_fallback:.2f}) | "
            f"b med {np.median(b_iter):.3e} | "
            f"probe peak/mean {float(pint.max()):.3e}/{float(pint.mean()):.3e} | "
            f"{iter_t:.2f} s ({iter_t / max(n_views, 1) * 1e3:.2f} ms/view)"
        )
        if frac_fallback > 0.9:
            vprint(
                "WARNING: >90% of object steps at exactly ls_damp*alpha/N — "
                "the cubic solve is degenerating (see spec §4.3). Check the "
                "coefficient magnitudes before trusting this run."
            )
        # standard PtyRAD loss line — identical format to the optimizer runs,
        # so old and line-search logs can be compared directly
        loss_iter = loss_logger(batch_losses, niter, iter_t, verbose=True)

        # bookkeeping mirrors recon_loop so save_results/plot_summary work unchanged
        model.loss_iters.append((niter, loss_iter))
        model.iter_times.append(iter_t)
        model.dz_iters.append((niter, model.opt_slice_thickness.detach().cpu().numpy()))
        model.avg_tilt_iters.append(
            (niter, model.opt_obj_tilts.detach().mean(0).cpu().numpy())
        )

        if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
            with torch.no_grad():
                save_results(
                    output_path, model, params, optimizer, niter, indices, batch_losses
                )
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
        f"### Finished {NITER} line-search iterations, "
        f"averaged iter_t = {np.mean(model.iter_times):.5g} s ###"
    )
