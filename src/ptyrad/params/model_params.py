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

    propagator_kernel: str = Field(
        default="angular_spectrum",
        description="Dispersion relation of the inter-slice propagator H: "
        "'angular_spectrum' (default; exact exp(i*dz*sqrt(k^2-Kx^2-Ky^2)), carrier "
        "included) or 'fresnel' (paraxial exp(-i*dz*(Kx^2+Ky^2)/(2k)), carrier "
        "removed). Any per-position tilt terms are applied identically to both. "
        "The Chin 4A/4B splittings require 'fresnel'.",
    )

    splitting: str = Field(
        default="lie_trotter",
        description="Operator splitting of the multislice forward model "
        "(solver_type='multislice' only): 'lie_trotter' (default; first-order "
        "transmit-then-propagate) or 'chin_4a' / 'chin_4b' (Chin fourth-order "
        "gradient splittings; two FFT pairs per slice — the cost of Lie-Trotter "
        "with twice the slices). The fourth-order schemes require "
        "propagator_kernel='fresnel'. With n_slices=1 they fall back to the "
        "single-transmission model (logged once). Removed 2026-09-30 (fe_v3 "
        "verdicts, git history): 'lt_x2', 'saba3', kick_mode='untied', and "
        "transmission_correction='gradient'.",
    )

    splitting_gradient_term: bool = Field(
        default=True,
        description="Apply the gradient (g = grad(chi).grad(chi)) term inside "
        "the chin_4a/chin_4b kicks (default True). False drops the term "
        "(g = 0) and skips the canvas-g gather entirely, saving its overhead; "
        "on depth-varying potentials the term is measured neutral-to-harmful "
        "(fe_v3 items 1.5/1.7), so g = 0 is the recommended production setting.",
    )

    chin4b_drop_end_props: bool = Field(
        default=False,
        description="For splitting='chin_4b' with far-field data only: drop "
        "the entrance/exit end drifts K(a1*dz). The exit drift is unit-modulus "
        "in k-space (no far-field intensity change); the entrance drift is "
        "absorbed into the optimized probe, whose plane thereby shifts by "
        "a1*dz ~ 0.2113*dz into the first slab. Saves one FFT pair per "
        "forward pass. Default False (end drifts applied explicitly).",
    )

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
        "disables), 'step', 'end_iter', 'n_views', 'pin_first', 'method'. "
        "method='detector' (recommended): closed-form least-squares fit of the "
        "detector field against the exact detector field of the current object "
        "— direct minimization of the detector error, still data-free (no "
        "measured intensities, no reconstruction oracle). The exact field is "
        "built by one sequential multislice sweep per calibration view "
        "(~2*n*(Nz-n/2)+2*Nz slice-FFTs with the basis pass, linear in depth, "
        "one rolling wavefield) and the coefficients solved by thin QR plus "
        "truncated SVD at the numerical rank (born_qr_coeffs) — this IS the "
        "algorithm, not an option; the TSVD cutoff is the only regularisation. "
        "The measured data never enters, so the coefficients track "
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
    'n_views': 64, 'pin_first': true, 'method': 'detector',
    'target': 'multislice'}.
    'target': 'multislice' (default) fits against
    the exact multislice-sweep field of the current object (data-free);
    'hybrid_amp' fits against the hybrid target T_hybrid = Upsilon * psi_MS,
    Upsilon = sqrt(I_data / I_MS) — amplitude from the measured data, phase
    from multislice. The hybrid target is an EXPERIMENT that reopens the
    data channel into the coefficients (judge on object metrics, not loss).
    'pin_first' is accepted but ignored (no pin
    needed; c0 = 1 is implicit). Removed
    methods: 'data' (multi-start L-BFGS against measured intensities —
    data-fitted coefficients absorb object error, the gauge leak),
    'gmres' (Krylov-Gram equation-residual fit — 4-135x oblique to the true
    detector error; git history, commit 18a2d35), and the ridge/'lcurve'
    regularisation policies (benchmarked irrelevant at M<=8 and worse than
    the unregularised solve at M=16; only the TSVD cutoff remains).
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

    @field_validator("propagator_kernel", mode="after")
    @classmethod
    def validate_propagator_kernel(cls, v: str) -> str:
        if v not in ("angular_spectrum", "fresnel"):
            raise ValueError(
                f"propagator_kernel must be 'angular_spectrum' or 'fresnel', got '{v}'"
            )
        return v

    @field_validator("splitting", mode="after")
    @classmethod
    def validate_splitting(cls, v: str) -> str:
        if v not in ("lie_trotter", "chin_4a", "chin_4b"):
            raise ValueError(
                f"splitting must be 'lie_trotter', 'chin_4a', or 'chin_4b', got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_splitting_requirements(self):
        """Chin 4A/4B need the paraxial kernel and the multislice solver."""
        if self.splitting in ("chin_4a", "chin_4b"):
            if self.solver_type != "multislice":
                raise ValueError(
                    f"splitting='{self.splitting}' requires solver_type='multislice', "
                    f"got '{self.solver_type}'"
                )
        if self.splitting in ("chin_4a", "chin_4b"):
            if self.propagator_kernel != "fresnel":
                raise ValueError(
                    f"splitting='{self.splitting}' requires propagator_kernel='fresnel': "
                    "with the angular-spectrum kernel the double commutator [V,[T,V]] is "
                    "not a local multiplication and the fourth-order schemes are not "
                    "fourth order"
                )
        return self

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
            "method": "detector",
            "grow_tol": None,
            "n_limit": None,
            "target": "multislice",
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
        if merged["method"] != "detector":
            raise ValueError(
                "born_coeffs_refit.method must be 'detector' (the only "
                "method). Removed: 'data' (L-BFGS against measured "
                "intensities — coefficients absorb object error) and 'gmres' "
                "(Krylov-Gram equation-residual fit, superseded by the "
                "detector fit; git history, commit 18a2d35)"
            )
        if merged["grow_tol"] is not None:
            if not (isinstance(merged["grow_tol"], (int, float)) and merged["grow_tol"] > 0):
                raise ValueError("born_coeffs_refit.grow_tol must be a positive number")
        if merged["n_limit"] is not None and (
            not isinstance(merged["n_limit"], int) or merged["n_limit"] < 1
        ):
            raise ValueError("born_coeffs_refit.n_limit must be None or an integer >= 1")
        if merged["target"] is None:
            merged["target"] = "multislice"
        if merged["target"] not in ("multislice", "hybrid_amp"):
            raise ValueError(
                "born_coeffs_refit.target must be 'multislice' (default; the "
                "exact multislice-sweep target, data-free) or 'hybrid_amp' "
                "(amplitude from the measured data, phase from multislice — "
                "an experiment that feeds data into the coefficients)"
            )
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
