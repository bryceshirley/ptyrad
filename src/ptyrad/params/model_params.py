"""
Defines available options and validation rules for the ``model_params`` dictionary.

"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, FilePath, field_validator, model_serializer, model_validator


class BornConfig(BaseModel):
    """'model_config' of the inter-slice Born series."""

    model_config = {"extra": "forbid"}

    order: int = Field(default=1, ge=1, description="Truncation order M of the series")
    """
    1 is the inter-slice first Born model, which is the cheapest and models single scattering
    only. Each extra order costs one more scatter-propagate pass over the slices.
    """

    background: Literal["vacuum", "mean"] = Field(
        default="mean", description="Background the series expands about"
    )
    """
    'vacuum' is a_j = 1, the plain series. 'mean' is a_j = <O_j>, the mean transmission of slice
    j over the patch; the factorisation O_j = a_j (1 + dO_j) is exact, so the model is unchanged
    once the series terminates, but centring lowers the overshoot of the truncated series about
    three-fold.
    """

    refit_views: Optional[int] = Field(
        default=None, ge=1, description="Calibration views for the per-iteration coefficient refit"
    )
    """
    null runs the plain series. Otherwise the retained orders are least-squares projected onto
    the exact multislice field of the current object at this many evenly spaced scan positions,
    once per iteration, giving one shared complex weight per order. The fit is data-free, so the
    coefficients cannot absorb object error. 32 is a usual value.
    """


class OptimizerParams(BaseModel):
    model_config = {"extra": "forbid"}
    
    name: Literal["Adadelta", "Adafactor", "Adagrad", "Adam",
                  "AdamW", "SparseAdam", "Adamax", "ASGD", 
                  "LBFGS", "Muon", "Nadam", "RAdam",
                  "RMSprop", "Rprop", "SGD", "pie"] = Field(default="Adam", description="Optimizer name")
    configs: Dict[str, Any] = Field(default_factory=dict, description="Optimizer configurations")
    load_state: Optional[FilePath] = Field(
        default=None, description="Path str of a PtyRAD model file to load previous optimizer state"
    )
    
    @model_serializer
    def serialize_model(self):
        """Custom serializer to convert pathlib.Path back to str."""
        data = self.__dict__.copy()
        if data.get('load_state') is not None and isinstance(data['load_state'], pathlib.Path):
            data['load_state'] = str(data['load_state'])
        return data


class SchedulerParams(BaseModel):
    model_config = {"extra": "forbid"}

    name: Literal["LambdaLR", "MultiplicativeLR", "StepLR", "MultiStepLR",
                  "ConstantLR", "LinearLR", "ExponentialLR", "PolynomialLR",
                  "CosineAnnealingLR", "ReduceLROnPlateau",
                  "CyclicLR", "OneCycleLR", "CosineAnnealingWarmRestarts"] = Field(description="Scheduler class name from torch.optim.lr_scheduler")
    configs: Dict[str, Any] = Field(default_factory=dict, description="Scheduler configurations")
    load_state: Optional[FilePath] = Field(
        default=None, description="Path str of a PtyRAD model file to load previous scheduler state"
    )
    step_unit: Literal["iter", "batch"] = Field(
        default="iter",
        description=(
            "When to call scheduler.step(): 'iter' (once per outer iteration, default) or "
            "'batch' (once per optimizer.step() call, i.e. per grad-accumulation boundary). "
            "Use 'batch' for CyclicLR and OneCycleLR, which are designed to step every optimizer "
            "update. ReduceLROnPlateau ignores this setting and always steps per iteration."
        )
    )

    @model_serializer
    def serialize_model(self):
        """Custom serializer to convert pathlib.Path back to str."""
        data = self.__dict__.copy()
        if data.get('load_state') is not None and isinstance(data['load_state'], pathlib.Path):
            data['load_state'] = str(data['load_state'])
        return data


class UpdateParams(BaseModel):
    model_config = {"extra": "forbid"}
    
    obja: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": 1, "lr": 5.0e-4}, description="Object amplitude update params"
    )
    objp: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": 1, "lr": 5.0e-4}, description="Object phase update params"
    )
    obj_tilts: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": None, "lr": 0.0}, description="Object tilts update params"
    )
    slice_thickness: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": None, "lr": 0.0}, description="Slice thickness update params"
    )
    probe: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": 1, "lr": 1.0e-4}, description="Probe update params"
    )
    probe_pos_shifts: Dict[str, Union[int, float, None]] = Field(
        default={"start_iter": 1, "lr": 5.0e-4},
        description="Sub-pixel probe position shifts update params",
    )

    @field_validator(
        "obja", "objp", "obj_tilts", "slice_thickness", "probe", "probe_pos_shifts", mode="after"
    )
    @classmethod
    def validate_update_params(cls, v: Dict[str, Any], field) -> Dict[str, Any]:
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
        fields = ["obja", "objp", "obj_tilts", "slice_thickness", "probe", "probe_pos_shifts"]
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

    # 'model_config' is pydantic's own attribute, so the field carrying the params key of
    # that name reaches it by alias; populate_by_name lets a dumped params dict load again
    model_config = {"extra": "forbid", "populate_by_name": True}
    
    
    detector_blur_std: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Gaussian blur std for simulated diffraction patterns. unit: px (k-space)",
    )
    """
    This applies Gaussian blur to the forward model simulated diffraction patterns to emulate the PSF of high-energy electrons on detector for experimental data. 
    Typical value is 0-1 px (std) based on the acceleration voltage
    """

    model: Literal["multislice", "Born"] = Field(
        default="multislice",
        description="Forward model: sequential multislice, or the inter-slice Born series",
    )
    """
    'multislice' is the Lie-Trotter split step, sequential in depth, and takes no 'model_config'.
    'Born' expands it into a Born series in the Fourier view, where the only operation coupling
    the slices is a cumulative sum, so the sequential depth is the truncation order rather than
    the number of slices. The series is nilpotent: an order equal to the slice count reproduces
    multislice exactly, and anything beyond adds nothing.
    Requires the optional `thickptycho` dependency, which supplies the propagation.
    """

    forward_config: Optional[BornConfig] = Field(
        default=None, alias="model_config", description="Settings of the chosen 'model'"
    )
    """
    null for multislice, which has nothing to configure. For 'Born' it is
    {'order': 1, 'refit_views': null, 'background': 'mean'}; null takes those defaults.
    """

    preload_data: Optional[bool] = Field(
        default=True,
        description="Boolean flag for either to preload data into device memory or not",
    )
    """
    type: bool. 
    This flag determines how the measurements data is stored and transferred to device during reconstruciton. 
    If true, measurement data will be fully loaded into device memory during model initialization for best performance. 
    However, dataset larger than device memory (i.e., GPU VRAM) would throw Out-Of-Memory error. 
    If 'preload_data': false, measurement data is kept on host memory (CPU RAM) and only the mini-batch is transferred to device memory in a streaming way. 
    This would enable reconstruction of large dataset that doesn't fit into GPU VRAM with a little data transfer overhead. 
    The default is 'true' for performance although the difference is negligible on demo datasets.
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
    'pie' is not a gradient optimizer: it replaces the loss, the backward pass and the step with
    the closed-form projection at the detector, which is ePIE for 'Born' and 3PIE for
    'multislice'. Its 'configs' are the step sizes {'beta_obj': 1.0, 'beta_probe': 1.0,
    'beta_wave': 1.0}; 'beta_wave' is 3PIE only and defaults to 'beta_probe', which is the usual
    coupling - one coefficient both revises the waves and damps the inter-slice transport by a
    factor per hop, and setting it to 1.0 keeps the transport undamped, which deep stacks need.
    The projection is Gauss-Seidel over positions, so it requires BATCH_SIZE 1 and will refuse
    anything larger rather than racing its own write-back, and the learning rates play no part.
    Requires the optional `thickptycho` dependency.
    """

    scheduler_params: Optional[SchedulerParams] = Field(
        default=None, description="LR scheduler configuration"
    )
    """
    Optional learning rate scheduler. 
    Set to e.g. {name: CosineAnnealingLR, configs: {T_max: 500}} to decay the learning rate during reconstruction. 
    Supports any class from torch.optim.lr_scheduler (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
    except ChainedScheduler and SequentialLR because both of them require other scheduler objects as constructor arguments, 
    so you will have to specify them after param loading if you want to use them. 
    ReduceLROnPlateau is handled automatically (mean loss is passed). 
    Set load_state to a model.hdf5 path to resume mid-schedule. 
    Note: not compatible with LBFGS optimizer (scheduler will be ignored with a warning). 
    """

    update_params: UpdateParams = Field(
        default_factory=UpdateParams, description="Update parameters for optimizable tensors"
    )

    @model_validator(mode="after")
    def validate_forward_config(self):
        if self.model == "multislice" and self.forward_config is not None:
            raise ValueError("'multislice' takes no 'model_config'; it must be null")
        return self

# Make explicit list so autodoc_pydantic can sort by this when go by `autodoc_pydantic_model_member_order = 'bysource'` in conf.py
__all__ = [
    "BornConfig",
    "ModelParams",
    "OptimizerParams",
    "SchedulerParams",
    "UpdateParams"
]