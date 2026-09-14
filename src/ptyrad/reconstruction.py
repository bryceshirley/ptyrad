"""
Reconstruction and hypertune workflows for ptychographic reconstructions
"""

import concurrent.futures
import logging
import warnings
from copy import deepcopy
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.utils.data import Dataset

from ptyrad.constraints import CombinedConstraint
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss, get_objp_contrast, get_objp_frc_auc
from ptyrad.models import PtychoAD
from ptyrad.save import copy_params_to_dir, make_output_folder, save_results
from ptyrad.utils import (
    fftshift2,
    get_blob_size,
    get_time,
    ifftshift2,
    ndarrays_to_tensors,
    parse_hypertune_params_to_str,
    parse_sec_to_time_str,
    safe_filename,
    set_random_seed,
    time_sync,
    vprint,
)
from ptyrad.visualization import plot_pos_grouping, plot_summary

torch.set_float32_matmul_precision("high")

warnings.filterwarnings(
    "ignore",
    message="Torchinductor does not support code generation for complex operators. Performance may be worse than eager.",
)
warnings.filterwarnings("ignore", message=".*Profiler function.*will be ignored.*")
warnings.filterwarnings("ignore", message=".*No device id is provided.*")
warnings.filterwarnings("ignore", message=".*Dynamo does not know how to trace.*")


class PtyRADSolver:
    """
    A wrapper class to perform ptychographic reconstruction or hyperparameter tuning.
    """

    def __init__(self, params, device=None, seed=None, acc=None, logger=None):
        self.params = deepcopy(params)
        self.if_hypertune = self.params.get("hypertune_params", {}).get("if_hypertune", False)
        self.verbose = not self.params["recon_params"]["if_quiet"]
        self.accelerator = acc
        self.use_acc_device = device is None and acc is not None
        self.device = self.accelerator.device if self.use_acc_device else device
        self.random_seed = seed
        self.logger = logger

        self.init_initializer()
        self.init_loss()
        self.init_constraint()
        vprint("### Done initializing PtyRADSolver ###")
        vprint(" ")

    def init_initializer(self):
        vprint("### Initializing Initializer ###")
        self.init = Initializer(self.params["init_params"], seed=self.random_seed).init_all()
        vprint(" ")

    def init_loss(self):
        vprint("### Initializing loss function ###")
        loss_params = self.params["loss_params"]

        vprint("Active loss types:")
        for key, value in loss_params.items():
            if value.get("state", False):
                vprint(f"  {key.ljust(12)}: {value}")

        self.loss_fn = CombinedLoss(loss_params, device=self.device)
        vprint(" ")

    def init_constraint(self):
        vprint("### Initializing constraint function ###")
        constraint_params = self.params["constraint_params"]

        vprint("Active constraint types:")
        for key, value in constraint_params.items():
            if value.get("start_iter", None) is not None:
                vprint(f"  {key.ljust(14)}: {value}")

        self.constraint_fn = CombinedConstraint(
            constraint_params, device=self.device, verbose=self.verbose
        )
        vprint(" ")

    def reconstruct(self):
        params = self.params
        device = self.device
        logger = self.logger

        model = PtychoAD(
            self.init.init_variables, params["model_params"], device=device, verbose=self.verbose
        )
        optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)

        if not self.use_acc_device:
            indices, batches, output_path = prepare_recon(model, self.init, params)
        else:
            if (
                params["model_params"]["optimizer_params"]["name"] == "LBFGS"
                and self.accelerator.num_processes > 1
            ):
                vprint(
                    f"WARNING: Optimizer 'LBFGS' is not supported for multiGPU mode (accelerator.num_processes = {self.accelerator.num_processes}), switch to default optimizer 'Adam'"
                )
                params["model_params"]["optimizer_params"]["name"] = "Adam"
                model.optimizer_params["name"] = "Adam"
                optimizer = create_optimizer(model.optimizer_params, model.optimizable_params)

            vprint(
                f"params['recon_params']['GROUP_MODE'] is set to 'random' because `use_acc_device` = {self.use_acc_device}",
                verbose=self.verbose,
            )
            params["recon_params"]["GROUP_MODE"] = "random"
            indices, batches, output_path = prepare_recon(model, self.init, params)
            ds = IndicesDataset(indices)
            dl = torch.utils.data.DataLoader(
                ds, batch_size=params["recon_params"]["BATCH_SIZE"]["size"], shuffle=True
            )
            batches = self.accelerator.prepare(dl)
            model, optimizer = self.accelerator.prepare(model, optimizer)

            vprint(
                f"len(DataLoader) = num_batches = {len(dl)}, DataLoader.batch_size = {len(indices) // len(dl)}",
                verbose=self.verbose,
            )
            vprint(
                "Note that the DataLoader will be duplicated for each process, while DataLoader.batch_size is the effective batch size (batch_size_per_process * num_process)",
                verbose=self.verbose,
            )
            vprint(
                "The actual batch_size_per_process will be printed below for the reported batches from the main process",
                verbose=self.verbose,
            )
            vprint(
                "For example, batch size = 512 with 2 GPUs (2 processes), the reported/observed batch size per GPU will be 512/2=256.",
                verbose=self.verbose,
            )

        if logger is not None and logger.flush_file:
            logger.flush_to_file(log_dir=output_path)

        recon_loop(
            model,
            self.init,
            params,
            optimizer,
            self.loss_fn,
            self.constraint_fn,
            indices,
            batches,
            output_path,
            acc=self.accelerator,
        )
        self.reconstruct_results = model
        self.optimizer = optimizer

    def hypertune(self):
        import optuna

        torch._inductor.config.triton.cudagraphs = False

        hypertune_params = self.params["hypertune_params"]
        params_path = self.params.get("params_path")
        n_trials = hypertune_params.get("n_trials")
        timeout = hypertune_params.get("timeout")
        study_name = hypertune_params.get("study_name")
        storage_path = hypertune_params.get("storage_path")
        sampler_params = hypertune_params["sampler_params"]
        pruner_params = hypertune_params["pruner_params"]
        error_metric = hypertune_params["error_metric"]
        sampler = create_optuna_sampler(sampler_params)
        pruner = create_optuna_pruner(pruner_params)
        logger = self.logger

        vprint("### Hypertune params ###")
        for key, value in hypertune_params.items():
            if key == "tune_params":
                vprint("Active tune_params:")
                for param, param_config in value.items():
                    if param_config.get("state", False):
                        vprint(f"    {param.ljust(12)}: {param_config}")
            else:
                vprint(f"{key.ljust(16)}: {value}")
        vprint(" ")

        valid_metrics = {"contrast", "loss", "frc"}
        if error_metric not in valid_metrics:
            raise ValueError(
                f"Invalid error metric: '{error_metric}'. Expected one of {valid_metrics}."
            )

        copy_params = self.params["recon_params"]["copy_params"]
        output_dir = self.params["recon_params"]["output_dir"]
        prefix_time = self.params["recon_params"]["prefix_time"]
        prefix = self.params["recon_params"]["prefix"]
        postfix = self.params["recon_params"]["postfix"]

        optuna_logger = logging.getLogger("optuna")
        optuna_logger.setLevel(logging.INFO)
        for handler in optuna_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                optuna_logger.removeHandler(handler)
        optuna_logger.addHandler(logger.buffer_handler)
        optuna_logger.addHandler(logger.console_handler)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage_path,
            study_name=study_name,
            load_if_exists=True,
        )

        prefix = prefix + "_" if prefix != "" else ""
        postfix = "_" + postfix if postfix != "" else ""

        if prefix_time is True or (isinstance(prefix_time, str) and prefix_time):
            time_str = get_time(prefix_time)
            prefix = f"{time_str}_{prefix}"
        sampler_str = sampler_params["name"]
        pruner_str = "_" + pruner_params["name"] if pruner_params is not None else ""

        output_dir += f"/{prefix}hypertune_{sampler_str}{pruner_str}_{error_metric}{postfix}"
        self.params["recon_params"]["output_dir"] = output_dir
        self.params["recon_params"]["prefix_time"] = ""
        self.params["recon_params"]["prefix"] = ""
        self.params["recon_params"]["postfix"] = ""

        if copy_params:
            copy_params_to_dir(params_path, output_dir, self.params)

        if (
            not copy_params
            and self.params["recon_params"]["SAVE_ITERS"] is None
            and not hypertune_params["collate_results"]
        ):
            output_dir = None

        if logger is not None and logger.flush_file:
            logger.flush_to_file(log_dir=output_dir)
            optuna_logger.addHandler(logger.file_handler)

        if error_metric == "frc":
            from ptyrad.split import generate_frc_splits

            path_A, path_B = generate_frc_splits(
                self.params, verbose=self.verbose, plot=False, device=self.device
            )

            self.params["hypertune_params"]["split_A_path"] = path_A
            self.params["hypertune_params"]["split_B_path"] = path_B

        study.optimize(
            lambda trial: optuna_objective(
                trial,
                self.params,
                self.init,
                self.loss_fn,
                self.constraint_fn,
                self.device,
                self.verbose,
            ),
            n_trials=n_trials,
            timeout=timeout,
        )
        vprint(
            f"Hypertune study is finished due to either (1) n_trials = {n_trials} or (2) study timeout = {timeout} sec has reached"
        )
        vprint("Best hypertune params:")
        for key, value in study.best_params.items():
            vprint(f"\t{key}: {value}")

    def run(self):
        start_t = time_sync()
        solver_mode = "hypertune" if self.if_hypertune else "reconstruct"

        vprint(f"### Starting the PtyRADSolver in {solver_mode} mode ###")
        vprint(" ")

        if self.if_hypertune:
            self.hypertune()
        else:
            self.reconstruct()
        end_t = time_sync()
        solver_t = end_t - start_t
        time_str = "" if solver_t < 60 else f", or {parse_sec_to_time_str(solver_t)}"

        vprint(f"### The PtyRADSolver is finished in {solver_t:.3f} sec{time_str} ###")
        vprint(" ")
        if self.logger is not None and self.logger.flush_file:
            self.logger.close()

        if dist.is_initialized():
            dist.destroy_process_group()


class IndicesDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]


def create_optimizer(optimizer_params, optimizable_params, verbose=True):
    def _fix_optimizer_state_dict_format(optim_state_dict: dict) -> dict:
        fixed = {}

        for key, val in optim_state_dict.items():
            if isinstance(val, dict):
                fixed_val = {}
                for nested_key, nested_val in val.items():
                    try:
                        fixed_nested_key = int(nested_key)
                    except (ValueError, TypeError):
                        fixed_nested_key = nested_key
                    fixed_val[fixed_nested_key] = nested_val
                fixed[key] = fixed_val
            else:
                fixed[key] = val

        if isinstance(fixed.get("param_groups"), dict):
            param_groups_dict = fixed["param_groups"]
            fixed["param_groups"] = [
                param_groups_dict[k] for k in sorted(param_groups_dict, key=lambda x: int(x))
            ]

        for group in fixed.get("param_groups", []):
            if isinstance(group.get("params"), torch.Tensor):
                group["params"] = group["params"].tolist()
            elif isinstance(group.get("params"), np.ndarray):
                group["params"] = group["params"].tolist()

        return fixed

    optimizer_name = optimizer_params["name"]
    optimizer_configs = optimizer_params.get("configs") or {}
    ptyrad_path = optimizer_params.get("load_state")

    vprint(
        f"### Creating PyTorch '{optimizer_name}' optimizer with configs = {optimizer_configs} ###",
        verbose=verbose,
    )

    optimizer_class = getattr(torch.optim, optimizer_name, None)

    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")
    if optimizer_name == "LBFGS":
        vprint(
            "Note: LBFGS optimizer is a quasi-Newton 2nd order optimizer that will run multiple forward passes (default: 20) for 1 update step"
        )
        vprint(
            "Note: LBFGS usually converges faster for convex problem with full-batch non-noisy gradients, but each update step is computationally slower"
        )
        non_zero_lr = [p["lr"] for p in optimizable_params if p["lr"] != 0]
        optimizer_configs["lr"] = min(non_zero_lr)
        vprint(
            f"Note: LBFGS optimizer does not support per parameter learning rate so it'll be set to the minimal non-zero learning rate = {min(non_zero_lr)}"
        )
        optimizable_params = [
            p["params"][0] for p in optimizable_params if p["params"][0].requires_grad
        ]

    optimizer = optimizer_class(optimizable_params, **optimizer_configs)
    device = optimizer.param_groups[0]["params"][0].device

    if ptyrad_path is not None and isinstance(ptyrad_path, str):
        try:
            from ptyrad.load import load_ptyrad

            optim_state_dict = load_ptyrad(ptyrad_path)["optim_state_dict"]
            optim_state_dict = _fix_optimizer_state_dict_format(optim_state_dict)
            optim_state_dict["state"] = ndarrays_to_tensors(
                optim_state_dict["state"], device=device
            )
            optimizer.load_state_dict(optim_state_dict)
            vprint(f"Loaded optimizer state from '{ptyrad_path}'", verbose=verbose)
        except Exception as e:
            vprint(
                f"Failed to load optimizer state from '{ptyrad_path}': {e}. Using fresh optimizer.",
                verbose=verbose,
            )
    vprint(" ", verbose=verbose)
    return optimizer


def prepare_recon(model, init, params):
    verbose = not params["recon_params"]["if_quiet"]
    vprint("### Generating indices, batches, and output_path ###", verbose=verbose)

    init_variables = init.init_variables
    init_params = init.init_params
    params_path = params.get("params_path")
    loss_params = params.get("loss_params")
    constraint_params = params.get("constraint_params")
    recon_params = params.get("recon_params")
    INDICES_MODE = recon_params["INDICES_MODE"].get("mode")
    subscan_slow = recon_params["INDICES_MODE"].get("subscan_slow")
    subscan_fast = recon_params["INDICES_MODE"].get("subscan_fast")
    GROUP_MODE = recon_params["GROUP_MODE"]
    SAVE_ITERS = recon_params["SAVE_ITERS"]
    batch_size = recon_params["BATCH_SIZE"].get("size")
    grad_accumulation = recon_params["BATCH_SIZE"].get("grad_accumulation")
    output_dir = recon_params["output_dir"]
    recon_dir_affixes = recon_params["recon_dir_affixes"]
    copy_params = recon_params["copy_params"]
    if_hypertune = params.get("hypertune_params", {}).get("if_hypertune", False)

    pos = (model.crop_pos + model.opt_probe_pos_shifts).detach().cpu().numpy()

    probe_view = model.get_complex_probe_view()
    probe_int = probe_view.abs().pow(2).sum(0).detach().cpu().numpy()

    dx = init_variables["dx"]
    d_out = get_blob_size(dx, probe_int, output="d90", verbose=verbose)
    indices = select_scan_indices(
        init_variables["N_scan_slow"],
        init_variables["N_scan_fast"],
        subscan_slow=subscan_slow,
        subscan_fast=subscan_fast,
        mode=INDICES_MODE,
        verbose=verbose,
    )
    batches = make_batches(
        indices,
        pos,
        batch_size,
        mode=GROUP_MODE,
        seed=init_variables["random_seed"],
        verbose=verbose,
    )
    fig_grouping = plot_pos_grouping(
        pos,
        batches,
        circle_diameter=d_out / dx,
        diameter_type="90%",
        dot_scale=1,
        show_fig=False,
        pass_fig=True,
    )
    vprint(
        f"The effective batch size is batch_size * grad_accumulation = {batch_size} * {grad_accumulation} = {batch_size * grad_accumulation}",
        verbose=verbose,
    )

    if SAVE_ITERS is not None:
        output_path = make_output_folder(
            output_dir,
            indices,
            init_params,
            recon_params,
            model,
            constraint_params,
            loss_params,
            recon_dir_affixes,
            verbose=verbose,
        )
        fig_grouping.savefig(safe_filename(output_path + "/summary_pos_grouping.png"))
        if copy_params and not if_hypertune:
            copy_params_to_dir(params_path, output_path, params, verbose=verbose)
    else:
        output_path = None

    plt.close(fig_grouping)
    vprint(" ", verbose=verbose)
    return indices, batches, output_path


def select_scan_indices(
    N_scan_slow, N_scan_fast, subscan_slow=None, subscan_fast=None, mode="full", verbose=True
):
    N_scans = N_scan_slow * N_scan_fast
    vprint(f"Selecting indices with the '{mode}' mode ", verbose=verbose)

    if mode == "full":
        return np.arange(N_scans)

    if subscan_slow is None and subscan_fast is None:
        vprint(
            "Subscan params are not provided, setting subscans to default as half of the total scan for both directions",
            verbose=verbose,
        )
        subscan_slow = N_scan_slow // 2
        subscan_fast = N_scan_fast // 2

    if mode == "center":
        vprint(f"Choosing subscan with {(subscan_slow, subscan_fast)}", verbose=verbose)
        start_row = (N_scan_slow - subscan_slow) // 2
        end_row = start_row + subscan_slow
        start_col = (N_scan_fast - subscan_fast) // 2
        end_col = start_col + subscan_fast
        indices = np.array(
            [
                row * N_scan_fast + col
                for row in range(start_row, end_row)
                for col in range(start_col, end_col)
            ]
        )

    elif mode == "sub":
        vprint(f"Choosing subscan with {(subscan_slow, subscan_fast)}", verbose=verbose)
        full_indices = np.arange(N_scans).reshape(N_scan_slow, N_scan_fast)
        subscan_slow_id = np.linspace(0, N_scan_slow - 1, num=subscan_slow, dtype=int)
        subscan_fast_id = np.linspace(0, N_scan_fast - 1, num=subscan_fast, dtype=int)
        slow_grid, fast_grid = np.meshgrid(subscan_slow_id, subscan_fast_id, indexing="ij")
        indices = full_indices[slow_grid, fast_grid].reshape(-1)

    else:
        raise ValueError(
            f"Indices selection mode {mode} not implemented, please use either 'full', 'center', or 'sub'"
        )

    return indices


def make_batches(indices, pos, batch_size, mode="random", seed=None, verbose=True):
    from time import time

    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        missing_package = str(e).split()[-1]
        vprint(
            f"### {missing_package} is not available, group mode set to 'random'. 'scikit-learn' is needed for 'sparse' and 'compact' ###"
        )
        mode = "random"

    if len(indices) > len(pos):
        raise ValueError(
            f"len(indices) = '{len(indices)}' is larger than total number of probe positions ({len(pos)}), check your indices generation params"
        )

    if indices.max() > len(pos):
        raise ValueError(
            f"Maximum index '{indices.max()}' is larger than total number of probe positions ({len(pos)}), check your indices generation params"
        )

    num_batch = len(indices) // batch_size
    t_start = time()
    if mode == "random":
        rng = np.random.default_rng(seed=seed)
        shuffled_indices = rng.permutation(indices)
        random_batches = np.array_split(shuffled_indices, num_batch)
        vprint(
            f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec",
            verbose=verbose,
        )
        return random_batches

    else:
        pos_s = pos[indices]
        kmeans = MiniBatchKMeans(
            init="k-means++",
            n_init=10,
            n_clusters=num_batch,
            max_iter=10,
            batch_size=3072,
            random_state=seed,
        )
        kmeans.fit(pos_s)
        labels = kmeans.labels_

        compact_batches = []
        for batch_idx in range(num_batch):
            batch_indices_s = np.where(labels == batch_idx)[0]
            compact_batches.append(indices[batch_indices_s])

        if mode == "compact":
            vprint(
                f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec",
                verbose=verbose,
            )
            return compact_batches

        else:
            from scipy.spatial.distance import cdist

            sparse_indices = indices.copy()
            sparse_batches = []

            centroids = np.array([np.mean(pos[cbatch], axis=0) for cbatch in compact_batches])
            pairwise_distances = cdist(pos, pos)

            used_indices = []
            for batch_idx in range(num_batch):
                distances = np.linalg.norm(pos_s - centroids[batch_idx], axis=1)
                closest_idx_s = np.argmin(distances)
                closest_idx = indices[closest_idx_s]
                sparse_batches.append([closest_idx])
                used_indices.append(closest_idx_s)
            sparse_indices = np.delete(sparse_indices, used_indices)

            for idx in sparse_indices:
                min_distances = []
                for batch_idx in range(num_batch):
                    distances = pairwise_distances[sparse_batches[batch_idx], idx]
                    min_distances.append(np.min(distances))

                max_group_index = np.argmax(min_distances)
                sparse_batches[max_group_index].append(idx)

            flatten_indices = np.concatenate(sparse_batches)
            flatten_indices.sort()
            indices.sort()
            assert all(flatten_indices == indices), (
                "Sorry, something went wrong with the sparse grouping, please try 'random' for now"
            )
            vprint(
                f"Generated {num_batch} '{mode}' groups of ~{batch_size} scan positions in {time() - t_start:.3f} sec",
                verbose=verbose,
            )

            sparse_batches = [np.array(batch) for batch in sparse_batches]
            return sparse_batches


def parse_torch_compile_configs(configs):
    if "enable" in configs:
        configs["disable"] = not configs.pop("enable")
    return configs


def recon_loop(
    model,
    init,
    params,
    optimizer,
    loss_fn,
    constraint_fn,
    indices,
    batches,
    output_path,
    acc=None,
):
    init_variables = init.init_variables
    recon_params = params.get("recon_params")
    NITER = recon_params["NITER"]
    SAVE_ITERS = recon_params["SAVE_ITERS"]
    grad_accumulation = recon_params["BATCH_SIZE"].get("grad_accumulation", 1)
    selected_figs = recon_params["selected_figs"]
    compiler_configs = parse_torch_compile_configs(recon_params["compiler_configs"])
    verbose = not recon_params["if_quiet"]

    model_instance = model.module if hasattr(model, "module") else model


    vprint("### Start the PtyRAD iterative ptycho reconstruction ###", verbose=verbose)

    recon_step_compiled = recon_step

    for niter in range(1, NITER + 1):
        toggle_grad_requires(model_instance, niter, verbose)

        if niter in model_instance.compilation_iters:
            vprint(
                f"Setting up PyTorch compiler with {compiler_configs}",
                verbose=verbose,
            )
            torch._dynamo.reset()
            recon_step_compiled = torch.compile(recon_step, **compiler_configs)


        batch_losses = recon_step_compiled(
            batches,
            grad_accumulation,
            model,
            optimizer,
            loss_fn,
            constraint_fn,
            niter,
            verbose=verbose,
            acc=acc,
        )

        if acc is None or acc.is_main_process:
            if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
                with torch.no_grad():
                    save_results(
                        output_path,
                        model_instance,
                        params,
                        optimizer,
                        niter,
                        indices,
                        batch_losses,
                    )

                    plot_summary(
                        output_path,
                        model_instance,
                        niter,
                        indices,
                        init_variables,
                        selected_figs=selected_figs,
                        show_fig=False,
                        save_fig=True,
                        verbose=verbose,
                    )

    vprint(
        f"### Finished {NITER} iterations, averaged iter_t = {np.mean(model_instance.iter_times):.5g} with std = {np.std(model_instance.iter_times):.3f} ###",
        verbose=verbose,
    )
    vprint(" ", verbose=verbose)


def recon_step(
    batches,
    grad_accumulation,
    model,
    optimizer,
    loss_fn,
    constraint_fn,
    niter,
    verbose=True,
    acc=None,
    **kwargs,
):
    batch_losses = {name: [] for name in loss_fn.loss_params.keys()}
    start_iter_t = time_sync()

    # Safely unwrap DDP and TorchDynamo wrappers to access custom methods
    model_instance = model
    while hasattr(model_instance, "module") or hasattr(model_instance, "_orig_mod"):
        if hasattr(model_instance, "module"):
            model_instance = model_instance.module
        if hasattr(model_instance, "_orig_mod"):
            model_instance = model_instance._orig_mod


    if isinstance(optimizer, torch.optim.LBFGS):
        num_batch = len(batches)
        batch_indices = np.arange(num_batch)
        if model.random_seed is not None:
            set_random_seed(seed=model.random_seed + niter)
        np.random.shuffle(batch_indices)
        accu_batch_indices = np.array_split(batch_indices, num_batch // grad_accumulation)

        def closure():
            optimizer.zero_grad()
            total_loss = 0
            for batch_idx in accu_batch_idx:
                batch = batches[batch_idx]
                model_DP = model(batch)
                measured_DP = model_instance.get_measurements(batch)
                object_patches = model_instance._current_object_patches
                loss_batch, losses = loss_fn(
                    model_DP, measured_DP, object_patches, model_instance.omode_occu
                )
                total_loss += loss_batch
            total_loss = total_loss / len(accu_batch_idx)
            acc.backward(total_loss) if acc is not None else total_loss.backward()
            return total_loss, losses

        for accu_batch_idx in accu_batch_indices:
            optimizer.step(lambda: closure()[0])

        _, losses = closure()
        optimizer.zero_grad()
        model_instance.clear_cache()

        if acc is not None:
            acc.wait_for_everyone()
        for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses, strict=False):
            batch_losses[loss_name].append(loss_value.detach().cpu().numpy())

    else:
        optimizer.zero_grad(set_to_none=True)
        
        # 🌟 INITIALIZE PRECONDITIONER CANVAS
        precond_canvas = (
            torch.zeros_like(model_instance.opt_obja) 
            if model_instance.solver_type == "born" else None
        )

        for batch_idx, batch in enumerate(batches):
            start_batch_t = time_sync()

            loss_batch, losses = compute_loss(batch, model, model_instance, loss_fn, acc)
            loss_batch = loss_batch / grad_accumulation

            acc.backward(loss_batch) if acc is not None else loss_batch.backward()
            
            # ACCUMULATE BATCH ILLUMINATION
            if precond_canvas is not None:
                model_instance.accumulate_iss_preconditioner(batch, precond_canvas)

            if (batch_idx + 1) % grad_accumulation == 0 or (batch_idx + 1) == len(batches):
                if acc is not None:
                    acc.wait_for_everyone()
                    
                # APPLY PRECONDITIONER TO GRADIENTS BEFORE OPTIMIZER STEP
                if precond_canvas is not None:
                    with torch.no_grad():
                        # Normalize to 1 so the learning rate defined in config remains valid
                        max_val = precond_canvas.amax(dim=(-2, -1), keepdim=True)
                        epsilon = 1e-4 * max_val.clamp(min=1e-8) # Tikhonov regularization
                        
                        precond = (precond_canvas + epsilon) / (max_val + epsilon)
                        
                        if model_instance.opt_obja.grad is not None:
                            model_instance.opt_obja.grad /= precond
                        if model_instance.opt_objp.grad is not None:
                            model_instance.opt_objp.grad /= precond
                        
                        # Reset canvas for the next accumulation cycle
                        precond_canvas.zero_()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_t = time_sync() - start_batch_t
            model_instance.clear_cache()

            if acc is not None:
                acc.wait_for_everyone()
            for loss_name, loss_value in zip(loss_fn.loss_params.keys(), losses, strict=False):
                batch_losses[loss_name].append(loss_value.detach().cpu().numpy())

            if batch_idx in np.linspace(0, len(batches) - 1, num=6, dtype=int):
                vprint(
                    f"Done batch {batch_idx + 1} with {len(batch)} indices ({batch[:5].tolist()}...) in {batch_t:.3f} sec",
                    verbose=verbose,
                )

    constraint_fn(model_instance, niter)

    iter_t = time_sync() - start_iter_t
    model_instance.loss_iters.append(
        (niter, loss_logger(batch_losses, niter, iter_t, verbose=verbose))
    )
    model_instance.iter_times.append(iter_t)
    model_instance.dz_iters.append(
        (niter, model_instance.opt_slice_thickness.detach().cpu().numpy())
    )
    model_instance.avg_tilt_iters.append(
        (niter, model_instance.opt_obj_tilts.detach().mean(0).cpu().numpy())
    )

    return batch_losses


def toggle_grad_requires(model, niter, verbose=True):
    vprint(" ", verbose=verbose)

    optimizable_tensors = model.optimizable_tensors
    for param_name in model.optimizable_tensors.keys():
        start_iter = model.start_iter.get(param_name)
        end_iter = model.end_iter.get(param_name)

        grad_started = start_iter is not None and niter >= start_iter
        grad_ended = end_iter is not None and niter + 1 > end_iter
        requires_grad = grad_started and not grad_ended

        optimizable_tensors[param_name].requires_grad = requires_grad
        vprint(f"Iter: {niter}, {param_name}.requires_grad = {requires_grad}", verbose=verbose)


def compute_loss(batch, model, model_instance, loss_fn, acc=None):
    if acc is not None:
        with acc.autocast():
            model_DP = model(batch)
            measured_DP = model_instance.get_measurements(batch)
            object_patches = model_instance._current_object_patches
            loss_batch, losses = loss_fn(
                model_DP, measured_DP, object_patches, model_instance.omode_occu
            )
    else:
        model_DP = model(batch)
        measured_DP = model_instance.get_measurements(batch)
        object_patches = model_instance._current_object_patches
        loss_batch, losses = loss_fn(
            model_DP, measured_DP, object_patches, model_instance.omode_occu
        )

    return loss_batch, losses


@torch.compiler.disable
def loss_logger(batch_losses, niter, iter_t, verbose=True):
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    loss_str = ", ".join([f"{name}: {value:.4f}" for name, value in avg_losses.items()])
    vprint(
        f"Iter: {niter}, Total Loss: {sum(avg_losses.values()):.4f}, {loss_str}, in {parse_sec_to_time_str(iter_t)}",
        verbose=verbose,
    )
    loss_iter = sum(avg_losses.values())
    return loss_iter


def create_optuna_sampler(sampler_params, verbose=True):
    import optuna

    sampler_name = sampler_params["name"]
    sampler_configs = sampler_params.get("configs") or {}

    vprint(
        f"### Creating Optuna '{sampler_name}' sampler with configs = {sampler_configs} ###",
        verbose=verbose,
    )

    sampler_class = getattr(optuna.samplers, sampler_name, None)

    if sampler_class is None or sampler_name == "ParitalFixedSampler":
        raise ValueError(f"Optuna sampler '{sampler_name}' is not supported.")

    sampler = sampler_class(**sampler_configs)

    vprint(" ", verbose=verbose)
    return sampler


def create_optuna_pruner(pruner_params, verbose=True):
    import optuna

    if pruner_params is None:
        return None
    else:
        pruner_name = pruner_params["name"]
        pruner_configs = pruner_params.get("configs") or {}

        vprint(
            f"### Creating Optuna '{pruner_name}' pruner with configs = {pruner_configs} ###",
            verbose=verbose,
        )

        pruner_class = getattr(optuna.pruners, pruner_name, None)

        if pruner_class is None or pruner_name == "WilcoxonPruner":
            raise ValueError(f"Optuna pruner '{pruner_name}' is not supported.")
        elif pruner_name == "NopPruner":
            raise ValueError(
                "Optuna NopPruner is an empty pruner, please set pruner_params = None if you don't want to prune."
            )
        elif pruner_name == "PatientPruner":
            wrapped_pruner = create_optuna_pruner(
                pruner_configs["wrapped_pruner_configs"], verbose=verbose
            )
            pruner_configs.pop("wrapped_pruner_configs", None)
            pruner = pruner_class(wrapped_pruner, **pruner_configs)
        else:
            pruner = pruner_class(**pruner_configs)

        vprint(" ", verbose=verbose)
        return pruner


def optuna_objective(trial, params, init, loss_fn, constraint_fn, device="cuda", verbose=False):
    import optuna

    init.verbose = verbose
    params = deepcopy(params)

    recon_params = params.get("recon_params")
    NITER = recon_params["NITER"]
    SAVE_ITERS = recon_params["SAVE_ITERS"]
    grad_accumulation = recon_params["BATCH_SIZE"].get("grad_accumulation", 1)
    output_dir = recon_params["output_dir"]
    selected_figs = recon_params["selected_figs"]
    compiler_configs = parse_torch_compile_configs(recon_params["compiler_configs"])

    hypertune_params = params["hypertune_params"]
    collate_results = hypertune_params["collate_results"]
    append_params = hypertune_params["append_params"]
    error_metric = hypertune_params["error_metric"]
    tune_params = hypertune_params["tune_params"]
    trial_id = "t" + str(trial.number).zfill(4)
    params["recon_params"]["prefix"] += trial_id

    if tune_params["batch_size"]["state"]:
        vname = "batch_size"
        vparams = tune_params[vname]
        params["recon_params"]["BATCH_SIZE"]["size"] = get_optuna_suggest(
            trial, vparams["suggest"], vname, vparams["kwargs"]
        )

    if tune_params["optimizer"]["state"]:
        vname = "optimizer"
        vparams = tune_params[vname]
        optim_name = get_optuna_suggest(trial, vparams["suggest"], vname, vparams["kwargs"])
        params["model_params"]["optimizer_params"]["name"] = optim_name
        params["model_params"]["optimizer_params"]["configs"] = vparams["kwargs"][
            "optim_configs"
        ].get(optim_name, {})

    lr_to_tensor = {
        "plr": "probe",
        "oalr": "obja",
        "oplr": "objp",
        "slr": "probe_pos_shifts",
        "tlr": "obj_tilts",
        "dzlr": "slice_thickness",
    }
    for vname in ["plr", "oalr", "oplr", "slr", "tlr", "dzlr"]:
        if tune_params[vname]["state"]:
            vparams = tune_params[vname]
            params["model_params"]["update_params"][lr_to_tensor[vname]]["lr"] = get_optuna_suggest(
                trial, vparams["suggest"], vname, vparams["kwargs"]
            )

    if tune_params["dx"]["state"]:
        vname = "dx"
        vparams = tune_params[vname]
        init.init_params["meas_calib"] = {
            "mode": vname,
            "value": get_optuna_suggest(trial, vparams["suggest"], vname, vparams["kwargs"]),
        }
        init.init_calibration()
        init.set_variables_dict()
        init.init_probe()
        init.init_pos()
        init.init_obj()
        init.init_H()

    remake_probe = False
    for vname in ["pmode_max", "conv_angle", "defocus", "z_shift", "c3", "c5"]:
        if tune_params[vname]["state"]:
            vparams = tune_params[vname]
            init.init_params["probe_" + vname] = get_optuna_suggest(
                trial, vparams["suggest"], vname, vparams["kwargs"]
            )
            remake_probe = True
    if remake_probe:
        init.init_probe()

    if tune_params["Nlayer"]["state"]:
        vname = "Nlayer"
        vparams = tune_params[vname]
        init.init_params["obj_Nlayer"] = get_optuna_suggest(
            trial, vparams["suggest"], vname, vparams["kwargs"]
        )
        init.init_obj()

    if tune_params["dz"]["state"]:
        vname = "dz"
        vparams = tune_params[vname]
        init.init_params["obj_slice_thickness"] = get_optuna_suggest(
            trial, vparams["suggest"], vname, vparams["kwargs"]
        )
        init.set_variables_dict()
        init.init_obj()
        init.init_H()

    scan_affine = []
    scan_affine_init = params["init_params"]["pos_scan_affine"]
    if scan_affine_init is not None:
        default_affine = {
            "scale": scan_affine_init[0],
            "asymmetry": scan_affine_init[1],
            "rotation": scan_affine_init[2],
            "shear": scan_affine_init[3],
        }
    else:
        default_affine = {"scale": 1, "asymmetry": 0, "rotation": 0, "shear": 0}
    for vname in ["scale", "asymmetry", "rotation", "shear"]:
        if tune_params[vname]["state"]:
            vparams = tune_params[vname]
            scan_affine.append(
                get_optuna_suggest(trial, vparams["suggest"], vname, vparams["kwargs"])
            )
        else:
            scan_affine.append(default_affine[vname])
    if scan_affine != [1, 0, 0, 0]:
        init.init_params["pos_scan_affine"] = scan_affine
        init.init_pos()
        init.init_obj()

    obj_tilts = []
    for vname in ["tilt_y", "tilt_x"]:
        if tune_params[vname]["state"]:
            vparams = tune_params[vname]
            obj_tilts.append(
                get_optuna_suggest(trial, vparams["suggest"], vname, vparams["kwargs"])
            )
        else:
            obj_tilts.append(0)
    obj_tilts = [obj_tilts]
    if obj_tilts != [[0, 0]]:
        init.init_variables["obj_tilts"] = obj_tilts

    if error_metric == "frc":
        path_A = hypertune_params.get("split_A_path")
        path_B = hypertune_params.get("split_B_path")

        if not path_A or not path_B:
            raise ValueError(
                "FRC metric requires 'split_A_path' and 'split_B_path' in 'hypertune_params'."
            )

        def run_split(split_path, target_device):
            from copy import deepcopy

            from ptyrad.constraints import CombinedConstraint
            from ptyrad.initialization import Initializer
            from ptyrad.losses import CombinedLoss
            from ptyrad.models import PtychoAD

            split_params = deepcopy(params)
            split_params["init_params"]["meas_params"]["path"] = split_path
            split_params["init_params"]["meas_params"]["key"] = "data"
            split_params["init_params"]["meas_params"].pop("gap", None)
            split_params["init_params"]["meas_params"].pop("offset", None)

            split_params["recon_params"]["compiler_configs"] = {"enable": False}
            split_params["recon_params"]["if_quiet"] = True

            split_init = Initializer(
                split_params["init_params"],
                seed=split_params["init_params"].get("random_seed"),
                verbose=False,
            ).init_all()

            split_model = PtychoAD(
                split_init.init_variables,
                split_params["model_params"],
                device=target_device,
                verbose=False,
            )

            split_optimizer = create_optimizer(
                split_model.optimizer_params, split_model.optimizable_params, verbose=False
            )
            split_loss_fn = CombinedLoss(split_params["loss_params"], device=target_device)
            split_constraint_fn = CombinedConstraint(
                split_params["constraint_params"], device=target_device, verbose=False
            )

            split_indices, split_batches, _ = prepare_recon(split_model, split_init, split_params)
            split_grad_acc = split_params["recon_params"]["BATCH_SIZE"].get("grad_accumulation", 1)

            step_fn = recon_step

            last_losses = None
            for niter in range(1, NITER + 1):
                toggle_grad_requires(split_model, niter, verbose=False)
                if split_model.random_seed is not None:
                    set_random_seed(seed=split_model.random_seed + niter)
                shuffle(split_batches)

                last_losses = step_fn(
                    split_batches,
                    split_grad_acc,
                    split_model,
                    split_optimizer,
                    split_loss_fn,
                    split_constraint_fn,
                    niter,
                    verbose=False,
                )

            return split_model, last_losses, split_indices

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_A = executor.submit(run_split, path_A, "cuda:0")
            future_B = executor.submit(run_split, path_B, "cuda:1")

            model_A, losses_A, indices_A = future_A.result()
            model_B, losses_B, indices_B = future_B.result()

        frc_error = compute_optuna_error(
            model_A,
            indices=None,
            metric="frc",
            model_B=model_B,
            output_dir=output_dir,
            trial_id=trial_id,
        )
        contrast_error = compute_optuna_error(model_A, indices_A, "contrast")

        frc_auc = max(0.001, -1.0 * frc_error)
        contrast_val = max(0.001, -1.0 * contrast_error)

        optuna_error = -1.0 * (contrast_val * frc_auc)

        vprint(
            f"Trial {trial_id} metrics -> FRC: {frc_auc:.4f}, Contrast: {contrast_val:.4f} | Combined Score: {optuna_error:.4f}",
            verbose=verbose,
        )

        if collate_results:
            params_str = parse_hypertune_params_to_str(trial.params) if append_params else ""
            collate_str = f"_SCORE_{optuna_error:.5f}_frc_{frc_auc:.2f}_con_{contrast_val:.2f}_{trial_id}{params_str}"

            save_results(
                output_dir,
                model_A,
                params,
                optimizer=None,
                niter=NITER,
                indices=indices_A,
                batch_losses=losses_A,
                collate_str=collate_str,
            )
            plot_summary(
                output_dir,
                model_A,
                NITER,
                indices=indices_A,
                init_variables=init.init_variables,
                selected_figs=selected_figs,
                collate_str=collate_str,
                show_fig=False,
                save_fig=True,
                verbose=verbose,
            )

        return optuna_error

    elif error_metric in ["loss", "contrast"]:
        model = PtychoAD(
            init.init_variables, params["model_params"], device=device, verbose=verbose
        )
        optimizer = create_optimizer(
            model.optimizer_params, model.optimizable_params, verbose=verbose
        )
        indices, batches, output_path = prepare_recon(model, init, params)

        recon_step_compiled = recon_step

        for niter in range(1, NITER + 1):
            toggle_grad_requires(model, niter, verbose)

            if niter in model.compilation_iters:
                vprint(f"Setting up PyTorch compiler with {compiler_configs}", verbose=verbose)
                torch._dynamo.reset()
                recon_step_compiled = torch.compile(recon_step, **compiler_configs)

            if model.random_seed is not None:
                set_random_seed(seed=model.random_seed + niter)
            shuffle(batches)
            batch_losses = recon_step_compiled(
                batches,
                grad_accumulation,
                model,
                optimizer,
                loss_fn,
                constraint_fn,
                niter,
                verbose=verbose,
            )

            if SAVE_ITERS is not None and niter % SAVE_ITERS == 0:
                save_results(
                    output_path,
                    model,
                    params,
                    optimizer,
                    niter,
                    indices,
                    batch_losses,
                    collate_str="",
                )
                plot_summary(
                    output_path,
                    model,
                    niter,
                    indices,
                    init.init_variables,
                    selected_figs=selected_figs,
                    collate_str="",
                    show_fig=False,
                    save_fig=True,
                    verbose=verbose,
                )

            if hypertune_params["pruner_params"] is not None:
                optuna_error = compute_optuna_error(
                    model, indices, error_metric, output_dir=output_dir, trial_id=trial_id
                )
                trial.report(optuna_error, niter)

                if trial.should_prune():
                    params_str = (
                        parse_hypertune_params_to_str(trial.params) if append_params else ""
                    )
                    collate_str = f"_error_{optuna_error:.5f}_{trial_id}{params_str}"
                    if collate_results:
                        save_results(
                            output_dir,
                            model,
                            params,
                            optimizer,
                            niter,
                            indices,
                            batch_losses,
                            collate_str=collate_str,
                        )
                        plot_summary(
                            output_dir,
                            model,
                            niter,
                            indices,
                            init.init_variables,
                            selected_figs=selected_figs,
                            collate_str=collate_str,
                            show_fig=False,
                            save_fig=True,
                            verbose=verbose,
                        )
                    raise optuna.exceptions.TrialPruned()

        if hypertune_params["pruner_params"] is None:
            optuna_error = compute_optuna_error(
                model, indices, error_metric, output_dir=output_dir, trial_id=trial_id
            )

        params_str = parse_hypertune_params_to_str(trial.params) if append_params else ""
        collate_str = f"_error_{optuna_error:.5f}_{trial_id}{params_str}"
        if collate_results:
            save_results(
                output_dir,
                model,
                params,
                optimizer,
                niter,
                indices,
                batch_losses,
                collate_str=collate_str,
            )
            plot_summary(
                output_dir,
                model,
                niter,
                indices,
                init.init_variables,
                selected_figs=selected_figs,
                collate_str=collate_str,
                show_fig=False,
                save_fig=True,
                verbose=verbose,
            )

        vprint(
            f"### Finished {NITER} iterations, averaged iter_t = {np.mean(model.iter_times):.3g} sec ###",
            verbose=verbose,
        )
        vprint(" ", verbose=verbose)

        return optuna_error


def get_optuna_suggest(trial, suggest, name, kwargs):
    if suggest == "cat":
        return trial.suggest_categorical(name, **kwargs)
    elif suggest == "int":
        return trial.suggest_int(name, **kwargs)
    elif suggest == "float":
        return trial.suggest_float(name, **kwargs)
    else:
        raise ValueError(f"Optuna trial.suggest method '{suggest}' is not supported.")


def compute_optuna_error(model, indices, metric, model_B=None, output_dir=None, trial_id=""):
    if metric == "contrast":
        return -1 * get_objp_contrast(model, indices)
    elif metric == "loss":
        return model.loss_iters[-1][-1]
    elif metric == "frc":
        if model_B is None:
            raise ValueError("FRC metric requires a second model (model_B) to compute correlation.")

        return get_objp_frc_auc(
            model, model_B, margin=200, apod_width=20, output_dir=output_dir, trial_id=trial_id
        )
    else:
        raise ValueError(
            f"Unsupported hypertune error metric: '{metric}'. Expected 'contrast', 'loss', or 'frc'."
        )
