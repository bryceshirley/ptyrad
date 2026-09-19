import pathlib
from typing import Any

import torch.optim
from pydantic import BaseModel, Field, FilePath, field_validator, model_serializer, model_validator


class OptimizerParams(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = Field(default="Adam", description="Optimizer name")
    configs: dict[str, Any] = Field(default_factory=dict, description="Optimizer configurations")
    load_state: FilePath | None = Field(
        default=None, description="Path str of a PtyRAD model file to load previous optimizer state"
    )

    @field_validator("name")
    @classmethod
    def validate_optimizer_name(cls, v: str) -> str:
        """Ensure optimizer name is a valid PyTorch optimizer."""
        if not hasattr(torch.optim, v) or not callable(getattr(torch.optim, v)):
            raise ValueError(f"Optimizer name '{v}' is not a valid PyTorch optimizer")
        return v

    @model_serializer
    def serialize_model(self):
        """Custom serializer to convert pathlib.Path back to str."""
        data = self.__dict__.copy()
        if data.get("load_state") is not None and isinstance(data["load_state"], pathlib.Path):
            data["load_state"] = str(data["load_state"])
        return data


class UpdateParams(BaseModel):
    model_config = {"extra": "forbid"}

    obja: dict[str, int | float | None] = Field(
        default={"start_iter": 1, "lr": 5.0e-4}, description="Object amplitude update params"
    )
    objp: dict[str, int | float | None] = Field(
        default={"start_iter": 1, "lr": 5.0e-4}, description="Object phase update params"
    )
    obj_tilts: dict[str, int | float | None] = Field(
        default={"start_iter": None, "lr": 0.0}, description="Object tilts update params"
    )
    slice_thickness: dict[str, int | float | None] = Field(
        default={"start_iter": None, "lr": 0.0}, description="Slice thickness update params"
    )
    probe: dict[str, int | float | None] = Field(
        default={"start_iter": 1, "lr": 1.0e-4}, description="Probe update params"
    )
    probe_pos_shifts: dict[str, int | float | None] = Field(
        default={"start_iter": 1, "lr": 5.0e-4},
        description="Sub-pixel probe position shifts update params",
    )
    born_coeffs: dict[str, int | float | None] = Field(
        default={"start_iter": None, "lr": 0.0},
        description="Born-series scattering-order coefficients update params (solver_type='born'). "
        "One pseudo-complex coefficient per order, initialized to 1; tuning them lets the "
        "truncated series absorb the multiple-scattering tail instead of biasing the object.",
    )

    @field_validator(
        "obja",
        "objp",
        "obj_tilts",
        "slice_thickness",
        "probe",
        "probe_pos_shifts",
        "born_coeffs",
        mode="after",
    )
    @classmethod
    def validate_update_params(cls, v: dict[str, Any], field) -> dict[str, Any]:
        """Validate start_iter and lr for update parameters."""
        start_iter = v.get("start_iter")
        lr = v.get("lr", 0.0)

        # start_iter must be None or >= 1
        if not (start_iter is None or (isinstance(start_iter, int) and start_iter >= 1)):
            raise ValueError(f"{field.field_name}.start_iter must be None or an integer >= 1")

        # If start_iter is not None, lr must be non-zero
        if start_iter is not None and lr == 0.0:
            raise ValueError(f"{field.field_name}.lr must be non-zero when start_iter is not None")

        # lr must be >= 0
        if not (isinstance(lr, (int, float)) and lr >= 0.0):
            raise ValueError(f"{field.field_name}.lr must be a non-negative number")

        return v

    @model_validator(mode="after")
    def validate_all_start_iter(self):
        """Ensure not all start_iter are None or all > 1."""
        fields = [
            "obja",
            "objp",
            "obj_tilts",
            "slice_thickness",
            "probe",
            "probe_pos_shifts",
            "born_coeffs",
        ]
        start_iters = [self.__dict__[field].get("start_iter") for field in fields]

        # start_iter can not be all None or all > 1
        if all(si is None for si in start_iters):
            raise ValueError("start_iter values can not be all None")
        non_none_iters = [si for si in start_iters if si is not None]
        if non_none_iters and all(si > 1 for si in non_none_iters):
            raise ValueError(
                "Non-None start_iter values can not be all > 1"
            )  # Early iterations would have no gradients to work with

        return self


class ModelParams(BaseModel):
    """
    "model_params" determines the forward model behavior, the optimizer configuration, and the learning of the PyTorch model (PtychoAD)

    optimizer configurations are specified in 'optimizer_params', see https://pytorch.org/docs/stable/optim.html for detailed information of available optimizers and configs.
    update behaviors of optimizable variables (tensors) are specified in 'update_params'.
    'start_iter' specifies the iteration at which the variables (tensors) can start being updated by automatic differentiation (AD)
    'lr' specifies the learning rate for the variables (tensors)
    Usually slower learning rate leads to better convergence/results, but is also updating slower.
    The variable optimization has 2 steps, (1) calculate gradient and (2) apply update based on learning rate * gradient
    'start_iter: null' will disable grad calculation and would not update the variable regardless the learning rate through out the whole reconstruction
    'start_iter: N(int)' would only calculate the grad when iteration >= N, so no grad will be calculated when iteration < N
    Therefore, only the variable with non-zero learning rate would be optimized when iteration > start_iter.
    If you don't want/need to optimize certain parameters, set their start_iter to null AND learning rate to 0 for faster computation.
    Typical learning rate is 1e-3 to 1e-4.
    """

    model_config = {"extra": "forbid"}

    solver_type: str = Field(
        default="multislice",
        description="Type of solver to use for forward model: 'multislice' or 'born'",
    )
    """
    The solver type determines which forward model implementation to use:
    - 'multislice': Standard multislice algorithm
    - 'born': Born approximation with configurable iterations
    """

    born_iterations: int = Field(
        default=1,
        ge=1,
        description="Number of Born iterations to use when solver_type='born'",
    )
    """
    When using the Born approximation solver, this controls the number of iterations:
    - 1: First-order Born approximation (single scattering)
    - >1: Higher-order Born approximation (multiple scattering)
    - =Nz: Full multislice equivalent
    """

    linduda_order: int = Field(
        default=1,
        ge=1,
        description="Order of Linduda approximation when solver_type='linduda'",
    )
    """
    When using the Linduda approximation solver, this controls the order of the approximation.
    """

    born_coeffs_refit: dict[str, int | float | bool | str | None] | None = Field(
        default=None,
        description="In-reconstruction refit of the Born coefficients on a fixed "
        "calibration view set (solver_type='born'). Dict with 'start_iter' (null "
        "disables), 'step', 'end_iter', 'n_views', 'pin_first', 'ridge', 'method'. "
        "method='detector' (recommended): closed-form least-squares fit of the "
        "detector field against the exact detector field of the current object "
        "— direct minimization of the detector error, still data-free (no "
        "measured intensities, no reconstruction oracle). 'target' picks how "
        "that exact field is built on the calibration views: 'born' (default) "
        "runs the Born recursion to its nilpotent cutoff (~Nz^2 slice-FFTs per "
        "view; one Gram prices every order up to Nz), 'multislice' runs one "
        "sequential sweep (~2*n*(Nz-n/2)+2*Nz slice-FFTs, linear in depth, O(1) "
        "transient frames — the option for deep stacks; identical objective and "
        "coefficients). "
        "method='gmres': closed-form residual-minimizing fit over the Krylov "
        "Gram matrix (equation residual in the 3D volume norm); cheaper recursion "
        "(stops at n+1 orders vs full depth), the option for deep stacks Nz >> n. "
        "Either way the measured data never enters, so the coefficients track "
        "truncation error only and cannot absorb object error. "
        "'grow_tol' > 0 enables adaptive order growth: each "
        "refit promotes born_iterations by 1 (up to 'n_limit', capped at Nz) "
        "whenever the fit residual at the current order exceeds grow_tol — start "
        "small (e.g. n=3) and let the solver track the object's scattering "
        "strength. Keep update_params['born_coeffs'] lr at 0 when this is enabled "
        "— the refit replaces gradient updates.",
    )
    """
    Defaults when enabled: {'start_iter': 1, 'step': 1, 'end_iter': null,
    'n_views': 64, 'pin_first': true, 'ridge': 1e-3, 'method': 'gmres',
    'target': 'born'}.
    'pin_first' fixes c1 = 1 — optional with GMRES (the data-driven gauge leak
    is closed by construction) and ignored by method='detector' (no pin needed;
    c0 = 1 is implicit); 'ridge' is the Tikhonov pull toward the plain series
    inside the convex solve (scaled by ||psi_0||^2 for gmres, ||D_0||^2 for
    detector). The former method='data' (multi-start L-BFGS against measured
    intensities) was removed: data-fitted coefficients absorb object error and
    degrade the reconstruction.
    """

    born_coeffs_init: list[list[float]] | str | None = Field(
        default=None,
        description="Warm start for the Born-series coefficients (solver_type='born'): "
        "an (n, 2) nested list of (real, imag) pairs per scattering order, or a path to "
        "a PtyRAD model .hdf5 whose 'optimizable_tensors/born_coeffs' is loaded. Orders "
        "beyond the provided values start at (1, 0). Default null starts all at (1, 0).",
    )
    """
    Use this to continue a previous run's coefficients (pass the model_iterXXXX.hdf5 path)
    or to seed with externally fitted values (e.g. an oracle fit or the Shanks estimate).
    Warm starting matters most when the coefficients are far from 1 (strong scattering):
    it spares the early iterations where jointly cold-started coefficients drift.
    """

    @field_validator("born_coeffs_refit", mode="after")
    @classmethod
    def validate_born_coeffs_refit(cls, v):
        """Fill refit defaults and sanity-check the gating fields."""
        if v is None:
            return v
        merged = {
            "start_iter": 1,
            "step": 1,
            "end_iter": None,
            "n_views": 64,
            "pin_first": True,
            "ridge": 1e-3,
            "method": "gmres",
            "target": "born",
            "grow_tol": None,
            "n_limit": None,
        }
        unknown = set(v) - set(merged)
        if unknown:
            raise ValueError(f"born_coeffs_refit has unknown keys: {sorted(unknown)}")
        merged.update(v)
        if merged["start_iter"] is not None and merged["start_iter"] < 1:
            raise ValueError("born_coeffs_refit.start_iter must be None or >= 1")
        if not isinstance(merged["step"], int) or merged["step"] < 1:
            raise ValueError("born_coeffs_refit.step must be an integer >= 1")
        if not isinstance(merged["n_views"], int) or merged["n_views"] < 1:
            raise ValueError("born_coeffs_refit.n_views must be an integer >= 1")
        if merged["method"] not in ("gmres", "detector"):
            raise ValueError(
                "born_coeffs_refit.method must be 'gmres' or 'detector' — the "
                "former method='data' (L-BFGS fit against measured intensities) "
                "was removed: data-fitted coefficients absorb object error"
            )
        if merged["target"] not in ("born", "multislice"):
            raise ValueError(
                "born_coeffs_refit.target must be 'born' (full-depth Born "
                "recursion) or 'multislice' (sequential sweep); only used by "
                "method='detector'"
            )
        if merged["grow_tol"] is not None:
            if not (isinstance(merged["grow_tol"], (int, float)) and merged["grow_tol"] > 0):
                raise ValueError("born_coeffs_refit.grow_tol must be a positive number")
        if merged["n_limit"] is not None and (
            not isinstance(merged["n_limit"], int) or merged["n_limit"] < 1
        ):
            raise ValueError("born_coeffs_refit.n_limit must be None or an integer >= 1")
        return merged

    obj_preblur_std: float | None = Field(
        default=None,
        ge=0.0,
        description="Gaussian blur std for object before forward pass. unit: px (real space)",
    )
    """
    This applies Gaussian blur to the object before simulating diffraction patterns.
    Since the gradient would flow to the original "object" before blurring, it's essentially deconvolving the object with a Gaussian kernel of specified std.
    This sort of deconvolution can generate sharp features, but the usage is not easily justifiable so treat it carefully as a visualization exploration
    """

    detector_blur_std: float | None = Field(
        default=None,
        ge=0.0,
        description="Gaussian blur std for simulated diffraction patterns. unit: px (k-space)",
    )
    """
    This applies Gaussian blur to the forward model simulated diffraction patterns to emulate the PSF of high-energy electrons on detector for experimental data.
    Typical value is 0-1 px (std) based on the acceleration voltage
    """

    optimizer_params: OptimizerParams = Field(
        default_factory=OptimizerParams, description="Optimizer configuration"
    )
    """
    Support all PyTorch optimizer.
    The suggested optimizer is 'Adam' with default configs (null).
    You can load the previous optimizer state by passing the path of `model.hdf5` to `load_state`, this way you can continue previous reconstruciton smoothly without abrupt gradients.
    (Because lots of the optimizers are adaptive and have history-dependent learning rate manipulation, so loading the optimizer state is necessary if you want to continue the previous optimization trajectory).
    However, the optimizer state must be coming from previous reconstructions with the same set of optimization variables with identical size of the dimensions otherwise it won't run.
    """

    update_params: UpdateParams = Field(
        default_factory=UpdateParams, description="Update parameters for optimizable tensors"
    )
