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
from ptyrad.reconstruction import PtyRADSolver, create_optimizer, prepare_recon
from ptyrad.save import save_results
from ptyrad.utils import CustomLogger, print_system_info, set_gpu_device, time_sync, vprint
from ptyrad.visualization import plot_summary

# Same layout as run_basic.py, but the iteration loop drives the exact quartic
# line search (src/ptyrad/linesearch.py, LINESEARCH_BORN_SPEC.md) instead of the
# gradient-descent optimizer. Requires a params file with solver_type: born and
# born_iterations: 1 — the far field is affine in the object only for single
# scattering, which is what makes the step exact.
params_paths = [
    "/home/dnz75396/ptyrad/demo/params/PSO_reconstruct_born_paper.yml"
]

run_name = ["linesearch_born"]  # Used to name the log file. Change to any string you like

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

    # Line-search design point (spec §3): batch size 1, probe updated every view.
    # Larger batches average the probe update and stall in a spiky-probe minimum.
    # Grouping is meaningless for single-view batches (and 'sparse' kmeans breaks
    # at n_clusters == n_views), so use 'random' — order is reshuffled per iter.
    params["recon_params"]["BATCH_SIZE"]["size"] = 1
    params["recon_params"]["BATCH_SIZE"]["grad_accumulation"] = 1
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

    vprint("### Start the PtyRAD exact-line-search reconstruction ###")
    for niter in range(1, NITER + 1):
        start_iter_t = time_sync()
        update_probe = probe_on and niter >= probe_start
        view_losses = []
        n0 = len(ls_state.steps_o)

        order = rng.permutation(len(batches))
        for nview, batch_i in enumerate(order, start=1):
            for index in np.atleast_1d(np.asarray(batches[batch_i])):
                diag = ls.linesearch_model_update(
                    model, int(index), config=ls_config, state=ls_state,
                    update_probe=update_probe,
                )
                view_losses.append(diag["loss"])
            # §6 in-sweep watchdog: probe runaway must be visible before the
            # per-iteration constraints (e.g. fix_probe_int) get to fire
            if nview % 512 == 0:
                vprint(
                    f"  iter {niter} view {nview}/{len(order)} | "
                    f"E_dir {np.mean(view_losses[-512:]):.4e} | "
                    f"probe peak/mean {diag['probe_peak']:.3e}/{diag['probe_mean']:.3e}"
                )
                if not np.isfinite(diag["loss"]):
                    raise FloatingPointError(
                        f"non-finite direction objective at iter {niter}, view {nview}"
                    )

        # Constraints fire once per iteration, after the view sweep — same call
        # site as recon_loop, so the in-batch exact field update stays valid.
        solver.constraint_fn(model, niter)

        iter_t = time_sync() - start_iter_t
        err = float(np.mean(view_losses))
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
            f"{iter_t:.2f} s"
        )
        if frac_fallback > 0.9:
            vprint(
                "WARNING: >90% of object steps at exactly ls_damp*alpha/N — "
                "the cubic solve is degenerating (see spec §4.3). Check the "
                "coefficient magnitudes before trusting this run."
            )

        # bookkeeping mirrors recon_loop so save_results/plot_summary work unchanged
        model.loss_iters.append((niter, err))
        model.iter_times.append(iter_t)
        model.dz_iters.append((niter, model.opt_slice_thickness.detach().cpu().numpy()))
        model.avg_tilt_iters.append(
            (niter, model.opt_obj_tilts.detach().mean(0).cpu().numpy())
        )

        if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
            with torch.no_grad():
                batch_losses = {"loss_e_dir": [np.float32(v) for v in view_losses]}
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
