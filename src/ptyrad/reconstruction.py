"""
Reconstruction and hypertune workflows for ptychographic reconstructions
"""

import concurrent.futures
import logging
import os
import warnings
from copy import deepcopy
from random import shuffle
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

try:
    from torchvision.transforms.functional import gaussian_blur
except ImportError:  # torchvision unavailable: use the local separable blur
    from ptyrad.utils import gaussian_blur_2d as gaussian_blur

from ptyrad.constraints import CombinedConstraint
from ptyrad.initialization import Initializer
from ptyrad.losses import CombinedLoss, get_objp_contrast, get_objp_frc_auc
from ptyrad.models import PtychoAD
from ptyrad.save import copy_params_to_dir, make_output_folder, save_results
from ptyrad.utils import (
    get_blob_size,
    get_time,
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


class EPIE(torch.optim.Optimizer):
    """ePIE-style preconditioned gradient descent (Maiden & Rodenburg 2009).

    Plain SGD with each group's gradient divided by the peak intensity of the
    field that multiplies its unknown in the forward model -- the Lipschitz
    normalisation of the ISS paper's preconditioned update: max_r sum_p |P|^2
    for the object groups (obja/objp), max_r sum_z |O|^2 for the probe group.
    Group lrs act as the ePIE step sizes alpha. Any other group falls back to
    plain SGD at its lr. Stateless (no moments), so its step is far cheaper
    than Adam's and there is no optimizer state to save or load.
    """

    def __init__(self, params, **kwargs):
        super().__init__(params, dict(lr=0.0))

    @torch.no_grad()
    def step(self, closure=None):
        by_name = {g.get("name"): g for g in self.param_groups}
        probe_g = by_name.get("probe")
        obja_g = by_name.get("obja")
        K_obj = K_prb = None
        if probe_g is not None:
            pr = probe_g["params"][0]
            K_obj = pr.abs().square().sum(0).amax().clamp_min(1e-12)
        if obja_g is not None:
            oa = obja_g["params"][0]
            K_prb = oa.square().sum(1).amax().clamp_min(1e-12)
        for g in self.param_groups:
            name, lr = g.get("name"), g["lr"]
            for p in g["params"]:
                if p.grad is None or lr == 0:
                    continue
                if name in ("obja", "objp") and K_obj is not None:
                    p.sub_(p.grad * (lr / K_obj))
                elif name == "probe" and K_prb is not None:
                    p.sub_(p.grad * (lr / K_prb))
                else:
                    p.add_(p.grad, alpha=-lr)


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

    optimizer_class = EPIE if optimizer_name == "EPIE" else getattr(torch.optim, optimizer_name, None)

    if optimizer_class is None:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")
    if optimizer_name == "LBFGS":
        raise ValueError(
            "LBFGS support was removed (closure-based stepping, no per-parameter "
            "lr, incompatible with multi-device param groups); use a first-order "
            "optimizer like 'Adam'."
        )

    # Param groups spread across multiple CUDA devices (e.g. a depth-
    # partitioned object with per-GPU block leaves) break torch's multi-
    # tensor foreach/fused step paths ("Tensors of the same index must be on
    # the same device"); fall back to the single-tensor path, which updates
    # each leaf on its own device. Explicit user configs take precedence.
    param_devices = {
        p.device
        for g in optimizable_params
        for p in (g["params"] if isinstance(g, dict) else [g])
    }
    if len(param_devices) > 1:
        import inspect

        accepted = inspect.signature(optimizer_class.__init__).parameters
        for key in ("foreach", "fused"):
            if key in accepted and key not in optimizer_configs:
                optimizer_configs[key] = False
        vprint(
            f"Note: optimizable params span {len(param_devices)} devices -> "
            "forcing single-tensor optimizer path (foreach=False, fused=False)",
            verbose=verbose,
        )

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

    # PTYRAD_HOLDOUT_PATH: npy file of scan indices excluded from every
    # update (validation positions). Their data loss is evaluated under
    # no_grad each iteration and appended to <output>/holdout_loss.csv, so
    # arms with different forward models can be compared on unfitted data.
    holdout_path = os.environ.get("PTYRAD_HOLDOUT_PATH")
    if holdout_path:
        holdout = np.load(holdout_path).astype(indices.dtype)
        n_before = len(indices)
        indices = np.setdiff1d(indices, holdout)
        vprint(
            f"PTYRAD_HOLDOUT_PATH: {n_before - len(indices)} of {n_before} positions "
            f"held out from updates ({holdout_path})",
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

    holdout_path = os.environ.get("PTYRAD_HOLDOUT_PATH")
    holdout_idx = np.load(holdout_path) if holdout_path else None

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

        if holdout_idx is not None and (acc is None or acc.is_main_process):
            # data loss on the held-out positions (never updated), same
            # loss_fn under no_grad, chunked to bound memory
            with torch.no_grad():
                names = list(loss_fn.loss_params.keys())
                sums = np.zeros(len(names))
                chunks = np.array_split(holdout_idx, max(1, len(holdout_idx) // 32))
                for chunk in chunks:
                    model_DP = model(chunk)
                    measured_DP = model_instance.get_measurements(chunk)
                    _, losses = loss_fn(
                        model_DP,
                        measured_DP,
                        model_instance._current_object_patches,
                        model_instance.omode_occu,
                    )
                    sums += np.array([float(v) for v in losses]) * len(chunk)
                sums /= len(holdout_idx)
                model_instance.clear_cache()
            if output_path is not None:
                csv_path = os.path.join(output_path, "holdout_loss.csv")
                is_new = not os.path.exists(csv_path)
                with open(csv_path, "a") as f:
                    if is_new:
                        f.write("iter," + ",".join(names) + "\n")
                    f.write(f"{niter}," + ",".join(f"{v:.8g}" for v in sums) + "\n")

        if acc is None or acc.is_main_process:
            # append the Born-refit trajectory row (built by the refit, which
            # doesn't know the output folder) with this iteration's loss
            row = getattr(model_instance, "born_refit_csv_row", None)
            if row is not None:
                model_instance.born_refit_csv_row = None
                if output_path is not None:
                    row["values"][1] = (
                        model_instance.loss_iters[-1][1]
                        if model_instance.loss_iters
                        else float("nan")
                    )
                    csv_path = os.path.join(output_path, "born_refit_trajectory.csv")
                    is_new = not os.path.exists(csv_path)
                    with open(csv_path, "a") as f:
                        if is_new:
                            f.write(",".join(row["header"]) + "\n")
                        f.write(
                            ",".join(
                                str(v) if isinstance(v, int) else f"{v:.8g}"
                                for v in row["values"]
                            )
                            + "\n"
                        )

            # same for the GMRES forward-model stats
            gmres_rows = getattr(model_instance, "born_gmres_rows", None)
            if gmres_rows:
                if output_path is not None:
                    csv_path = os.path.join(output_path, "born_gmres_stats.csv")
                    is_new = not os.path.exists(csv_path)
                    with open(csv_path, "a") as f:
                        if is_new:
                            f.write("iter,step,fallback,k,rel_res,beta\n")
                        for step, (fb, k, rel_res, beta) in enumerate(gmres_rows):
                            f.write(f"{niter},{step},{fb},{k},{rel_res:.8g},{beta:.8g}\n")
                model_instance.born_gmres_rows = []

            # per-position Born-residual forward-model stats (born_residual): one
            # pooled row per coefficient-table refresh (or per batch when
            # solving on the fly without a table)
            residual_rows = getattr(model_instance, "born_residual_rows", None)
            if residual_rows:
                if output_path is not None:
                    csv_path = os.path.join(output_path, "born_residual_stats.csv")
                    is_new = not os.path.exists(csv_path)
                    with open(csv_path, "a") as f:
                        if is_new:
                            f.write("iter,step,fallback,k,rel_res,beta\n")
                        for step, (fb, k, rel_res, beta) in enumerate(residual_rows):
                            f.write(f"{niter},{step},{fb},{k},{rel_res:.8g},{beta:.8g}\n")
                model_instance.born_residual_rows = []

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

    optimizer.zero_grad(set_to_none=True)

    # closed-form PIE updates (ePIE / 3PIE): no backward, no optimizer,
    # no illumination preconditioner — the modulus projection IS the update
    use_pie = getattr(model_instance, "update_rule", "autodiff") == "pie"

    # 🌟 INITIALIZE PRECONDITIONER CANVAS
    # PTYRAD_DISABLE_ISS_PRECOND=1 skips the illumination preconditioner:
    # at batch 1 the canvas holds a single position, so the focused
    # entrance slice divides its patch-periphery gradients by ~1e-4
    # (measured to destroy the entrance slice of a 2-slab test), and the
    # canvas costs one sequential vacuum multislice pass per position.
    precond_canvas = (
        torch.zeros_like(model_instance.opt_obja)
        if model_instance.solver_type == "born"
        and not use_pie
        and not os.environ.get("PTYRAD_DISABLE_ISS_PRECOND")
        else None
    )

    if use_pie:
        from ptyrad.pie import pie_batch_update

        for batch_idx, batch in enumerate(batches):
            start_batch_t = time_sync()
            model_DP, measured_DP, object_patches = pie_batch_update(
                model_instance, batch, **model_instance.pie_params
            )
            with torch.no_grad():
                _, losses = loss_fn(
                    model_DP, measured_DP, object_patches, model_instance.omode_occu
                )
            batch_t = time_sync() - start_batch_t
            model_instance.clear_cache()
            for loss_name, loss_value in zip(
                loss_fn.loss_params.keys(), losses, strict=False
            ):
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

    # update_rule 'adjoint': analytic adjoint-solve gradients fill .grad in
    # place of compute_loss + backward; the optimizer path below is shared
    # with autodiff unchanged.
    use_adjoint = getattr(model_instance, "update_rule", "autodiff") == "adjoint"
    if use_adjoint and acc is not None and getattr(acc, "num_processes", 1) > 1:
        raise ValueError("update_rule 'adjoint' does not support multi-process accelerate")

    for batch_idx, batch in enumerate(batches):
        start_batch_t = time_sync()

        if use_adjoint:
            from ptyrad.pie import adjoint_grad_batch

            model_DP, measured_DP, obj_patches_b = adjoint_grad_batch(model_instance, batch)
            with torch.no_grad():
                _, losses = loss_fn(
                    model_DP, measured_DP, obj_patches_b, model_instance.omode_occu
                )
        else:
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
                    epsilon = 1e-4 * max_val.clamp(min=1e-8)  # Tikhonov regularization

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
    refit_born_coeffs(model_instance, niter, verbose=verbose)

    # per-iteration coefficient trajectory, whatever updates the
    # coefficients (AD or refit) — consumed by plot_summary and the
    # checkpoint dict, like loss_iters/dz_iters
    if getattr(model_instance, "use_born_coeffs", False):
        model_instance.born_coeffs_iters.append(
            (niter, model_instance.opt_born_coeffs.detach().cpu().numpy().copy())
        )

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


@torch._dynamo.disable
def refit_born_coeffs(model, niter, verbose=True):
    """Schedule gate for the Born-coefficient refit
    (model_params['born_coeffs_refit']); delegates to
    _refit_born_coeffs_fourier — the least squares of the retained orders
    onto the exact multislice-sweep target on the FOURIER basis, over all
    N+1 planes plus the exit block. THE coefficient algorithm, and now the
    only one: that is the norm the per-position born_residual forward
    minimises, and the extra plane rows pin down directions the detector
    rows alone cannot (measured on real bases at Nz = 21, M = 8: |c|max
    3.6 vs 71, volume error 0.40 vs 20, coefficient swing under a 1%
    object perturbation 3.9e-4 vs 2.8e-3).

    Earlier variants were removed: method='detector' (the same remainder
    fit restricted to the detector rows — 2-8x better in the DETECTOR norm
    in the mid-range, which is the one thing given up here, but with
    coefficients an order of magnitude larger and a wavefield up to 50x
    further from multislice inside the sample; grow_tol and 'hybrid_amp'
    were carried over to the Fourier rows before it went), method='data'
    (multi-start L-BFGS
    against measured intensities — data-fitted coefficients absorb object
    error, the gauge leak) and method='gmres' (equation-residual fit on the
    Krylov Gram — 4-135x oblique to the true detector error and the most
    memory-hungry variant at depth; git history, commit 18a2d35, if ever
    needed). Runs after the constraints each iteration and writes
    opt_born_coeffs.data directly — no autograd involved.
    """
    cfg = getattr(model, "born_refit", None)
    if not cfg or cfg.get("start_iter") is None or model.solver_type != "born":
        return
    start, step, end = cfg["start_iter"], cfg["step"], cfg.get("end_iter")
    if niter < start or (end is not None and niter >= end) or (niter - start) % step != 0:
        return
    _refit_born_coeffs_fourier(model, niter, cfg, verbose=verbose)


def _born_refit_dp(model, field):
    """Mode-weighted, fftshifted (and detector-blurred) intensity of a
    detector field — the same quantity the loss compares, used both for the
    hybrid target's Upsilon and for the refit's logged diagnostics."""
    Ny, Nx = field.shape[-2:]
    norm_weight = (model.omode_occu / (Nx * Ny)).view(1, 1, -1, 1, 1)
    dp = torch.fft.fftshift(
        torch.sum(field.abs().square() * norm_weight, dim=(1, 2)), dim=(-2, -1)
    )
    if model.detector_blur_std:
        dp = gaussian_blur(dp, kernel_size=[5, 5], sigma=model.detector_blur_std)
    return dp


@torch._dynamo.disable
def _refit_born_coeffs_fourier(model, niter, cfg, verbose=True):
    """FOURIER-VIEW (volume + detector) Born-coefficient refit against the
    exact multislice target: the remainder fit of born_remainder_coeffs extended
    from the detector rows to the full wavefield, in the SAME norm the
    per-position born_residual forward minimises (every block of Psi_hat at
    unit weight — in the Fourier view the plane and exit rows already carry
    the same fft2 factor, so no rescaling enters;
    born_remainder.born_volume_remainder_rows
    builds the rows). Shared coefficients over the calibration views,
    solved for c DIRECTLY (zero prior, so an unresolved direction drops
    its order rather than falling back on the plain series) by an
    equilibrated complex128 Gram solve with split-k
    moments — complex128 is mandatory and the ridge must act on the
    unit-diagonal Gram (see born_residual for both convictions).
    Comparison arm for the Fourier-view GMRES forward: same norm, same
    multislice physics, but a SHARED fitted polynomial instead of a
    per-position Krylov solve.

    cfg['grow_tol'] promotes born_iterations by 1 (up to n_limit, capped
    at Nz) whenever the fit residual at the current order exceeds the
    tolerance, exactly as the removed detector refit did. Growth is free
    of a second pass here: the moments are accumulated ONCE at the
    lookahead order n_keep, and the fit at any order n <= n_keep is the
    leading n-by-n block of the same Gram, so promotion is a resolve of an
    (n, n) system with no new basis, target or reduction.

    cfg['target'] = 'hybrid_amp' replaces the DETECTOR rows' target with
    the hybrid field Upsilon * psi_MS (amplitude from the measured data,
    phase and mode structure from multislice, Upsilon = sqrt(I_data/I_MS)
    per calibration view against the blurred model intensity the loss
    compares). The plane rows keep the data-free multislice target — the
    data only ever constrains the exit plane — so the data's pull on the
    fit is diluted by the Nz plane rows relative to the old detector-only
    hybrid fit; it is a weaker hybrid, not the same one. The pure
    multislice coefficients c_MS come back from the same factorisation as
    a second right-hand side and are logged as a diagnostic.
    WARNING: 'hybrid_amp' feeds measured intensities into the coefficient
    update, deliberately reopening the gauge-leak channel the data-free
    fit closed — an experiment to be judged on object metrics, not loss."""
    from ptyrad.forward_models.born import born_detector_basis, born_volume_basis
    from ptyrad.forward_models.born_coeff_fit import (
        _born_residual_orders,
        born_gram_moments,
        born_gram_solve,
        born_multislice_target,
        born_multislice_volume_target,
        born_residual_order_gram,
        born_residual_solve_gram,
        born_volume_remainder_rows,
    )

    n = model.born_iterations
    idx = model.born_refit_views
    dev = model.opt_born_coeffs.device
    target = cfg.get("target", "multislice")
    hybrid = target == "hybrid_amp"
    residual = target == "residual"
    grow_tol = cfg.get("grow_tol")
    n_limit = min(int(cfg.get("n_limit") or model.n_slice), model.n_slice)
    # lookahead order: the moments are built once at n_keep and every
    # promotion reads a leading submatrix of them. grow_fast goes straight
    # to n_limit so one refit can promote repeatedly.
    if grow_tol is None:
        n_keep = n
    elif cfg.get("grow_fast"):
        n_keep = n_limit
    else:
        n_keep = min(n + 1, n_limit)
    grew = False
    t_refit = perf_counter()
    with torch.no_grad():
        K = n_keep
        G = torch.zeros(K, K, dtype=torch.complex128, device=dev)
        # row 0: the fitted target; row 1: (multislice - fitted) at the
        # detector, which recovers c_MS by linearity when hybrid is on
        gb = torch.zeros(2, K, dtype=torch.complex128, device=dev)
        b2 = torch.zeros((), dtype=torch.float64, device=dev)
        # residual target: the accumulable order Gram, summed over views
        n_res = min(K + 1, model.n_slice)
        P_res = torch.zeros(n_res, n_res, dtype=torch.complex128, device=dev)
        D0_parts, D_parts, Tdet_parts, shift2 = [], [], [], 0.0
        for sl in idx.split(2 if not residual else 4):  # volume targets are memory-heavy
            patches = model.get_obj_patches(sl)
            probes = model.get_probes(sl)
            H3 = model.get_propagators_3d(model.get_propagators(sl))
            if residual:
                # target-free: the fixed-point residual needs ONE order
                # beyond those retained and NO multislice pass, so the
                # whole sequential sweep is skipped here
                Vw, dts = _born_residual_orders(patches, probes, H3,
                                                min(K + 1, model.n_slice))
                P_res += born_residual_order_gram(
                    Vw, dts, min(K + 1, model.n_slice), model.omode_occu).sum(0)
                D0c, Dc = born_detector_basis(patches, probes, H3, K)
                T_det = born_multislice_target(patches, probes, H3)
                D0_parts.append(D0c)
                D_parts.append(Dc)
                Tdet_parts.append(T_det)
                del Vw, dts, D0c, Dc, T_det
                continue
            V, D0c, Dc = born_volume_basis(patches, probes, H3, K)
            T_vol = born_multislice_volume_target(patches, probes, H3)
            T_det = born_multislice_target(patches, probes, H3)
            if hybrid:
                # Upsilon = sqrt(I_data / I_MS) per view, shared across
                # modes; |T_hybrid| <= sqrt(I_data) per pixel regardless of
                # the ratio, so the guards are numerical only.
                dp_ms = _born_refit_dp(model, T_det)
                floor = 1e-12 * dp_ms.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-30)
                ups = (
                    model.get_measurements(sl).clamp(min=0)
                    / dp_ms.clamp(min=0).maximum(floor)
                ).sqrt()
                ups = torch.fft.ifftshift(ups.clamp(max=1e3), dim=(-2, -1))
                T_fit_det = ups[:, None, None] * T_det
                shift2 += float((T_fit_det - T_det).abs().square().sum())
                A, b, b_ms = born_volume_remainder_rows(
                    V, Dc, T_vol, T_fit_det, D0c, K,
                    omode_occu=model.omode_occu, extra_det=T_det - T_fit_det,
                )
            else:
                A, b = born_volume_remainder_rows(
                    V, Dc, T_vol, T_det, D0c, K, omode_occu=model.omode_occu
                )
                b_ms = torch.zeros_like(b)
            # shared split-k complex128 kernel; the moments of each chunk
            # simply add (occupancy weights are already in the rows)
            Gc, gc = born_gram_moments(
                A.T, torch.stack([b, b_ms]), k_block=D0c.shape[-2] * D0c.shape[-1]
            )
            G += Gc
            gb += gc
            b2 += b.to(torch.complex128).abs().square().sum().real
            # small detector-space pieces for the standard diagnostics
            D0_parts.append(D0c)
            D_parts.append(Dc)
            Tdet_parts.append(T_det)
            del V, T_vol, A, b, b_ms, Gc, gc
        D0 = torch.cat(D0_parts, dim=0)
        Dn = torch.cat(D_parts, dim=1)
        T = torch.cat(Tdet_parts, dim=0) - D0

        # direct solve for c (zero prior, as in born_residual and
        # born_remainder_coeffs): min_c ||A c - b||, on the leading n
        # columns. Promotion re-reads a leading submatrix of the SAME
        # moments — no second basis pass, no second reduction.
        def _solve(n_cur):
            if residual:  # target-free: solve from the summed order Gram
                c_r, _beta, rel = born_residual_solve_gram(P_res, n_cur)
                return c_r, torch.zeros_like(c_r), rel.item()
            sol, rel = born_gram_solve(G[:n_cur, :n_cur], gb[:, :n_cur], b2)
            return sol[0], sol[1], rel.item()

        cc, x_ms, fit_res = _solve(n)
        # promotion loop: the lookahead moments already cover n_keep
        while grow_tol is not None and fit_res > float(grow_tol) and n < n_keep:
            n += 1
            model.born_iterations = n
            cc, x_ms, fit_res = _solve(n)
            grew = True
        Dn = Dn[:n]
        c = cc.to(torch.complex64)
        c_ms = (cc + x_ms).to(torch.complex64) if hybrid else c

        model.opt_born_coeffs.data = torch.stack([c.real, c.imag], -1).to(
            dtype=model.opt_born_coeffs.dtype, device=model.opt_born_coeffs.device
        )

        # detector-space diagnostics for logging only — none of these feed
        # back into the fit. delta_model: model error against the EXACT
        # multislice field at the coefficients actually used, on fields.
        # eps_int: its intensity counterpart. target_shift: how far the
        # data amplitude moved the target; dc_rel: how far it moved the
        # coefficients off the pure multislice fit.
        meas = model.get_measurements(idx)
        wf = model.omode_occu.to(torch.float32).clamp(min=0).view(1, 1, -1, 1, 1)

        def _fnorm(field):
            return (field.abs().square().to(torch.float64) * wf).sum().sqrt().item()

        dp_exact = _born_refit_dp(model, D0 + T)
        Fs = (c.view(-1, 1, 1, 1, 1, 1) * Dn).sum(dim=0)
        psi_norm = _fnorm(D0 + T)
        delta_model = _fnorm(Fs - T) / psi_norm
        target_shift = (shift2**0.5) / psi_norm if hybrid else 0.0
        dc_rel = (
            (torch.linalg.vector_norm(c - c_ms) / torch.linalg.vector_norm(c_ms)).item()
            if hybrid
            else 0.0
        )
        dp = _born_refit_dp(model, D0 + Fs)
        eps_int = ((dp - dp_exact).norm() / dp_exact.norm()).item()
        res = ((dp - meas).norm() / meas.norm()).item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_refit = perf_counter() - t_refit
    coeff_str = " ".join(f"{a:+.3f}{b:+.3f}j" for a, b in model.opt_born_coeffs.tolist())
    tag = "fourier" if grow_tol is None else f"fourier, n={n}{' grew' if grew else ''}"
    if hybrid:
        tag += ", hybrid_amp"
    vprint(
        f"Refit born_coeffs at iter {niter} ({len(idx)} views, {tag}): "
        f"rel fourier residual {fit_res:.3e}, rel int error {eps_int:.3e}, "
        f"rel data residual {res:.3e}, c = [{coeff_str}] ({t_refit:.2f} s)",
        verbose=verbose,
    )
    if hybrid:
        vprint(
            f"  hybrid target at iter {niter}: rel model error {delta_model:.3e}, "
            f"target shift {target_shift:.3e}, ||c - c_MS||/||c_MS|| {dc_rel:.3e}",
            verbose=verbose,
        )

    # per-iteration trajectory row; recon_loop fills the loss and appends it
    # to <run_dir>/born_refit_trajectory.csv
    ncols = n_limit if grow_tol is not None else n
    header = (
        ["iter", "loss", "n", "hybrid", "det_res", "delta_model", "eps_int",
         "data_res", "target_shift", "dc_rel"]
        + [f"cabs{m}" for m in range(1, ncols + 1)]
        + [s2 for m in range(1, ncols + 1) for s2 in (f"c{m}_re", f"c{m}_im")]
        + [s2 for m in range(1, ncols + 1) for s2 in (f"cms{m}_re", f"cms{m}_im")]
    )
    nan = float("nan")
    pad = [nan] * (ncols - n)
    c_cpu, cms_cpu = c.cpu(), c_ms.cpu()
    values = (
        [niter, None, n, int(hybrid), fit_res, delta_model, eps_int, res,
         target_shift, dc_rel]
        + c_cpu.abs().tolist() + pad
        + [v for z in c_cpu.tolist() for v in (z.real, z.imag)] + pad + pad
        + [v for z in cms_cpu.tolist() for v in (z.real, z.imag)] + pad + pad
    )
    model.born_refit_csv_row = {"header": header, "values": values}
    model.born_refit_iters.append(
        (niter, fit_res, delta_model, eps_int, res, target_shift, dc_rel)
    )


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
        "bclr": "born_coeffs",
    }
    for vname in ["plr", "oalr", "oplr", "slr", "tlr", "dzlr", "bclr"]:
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
