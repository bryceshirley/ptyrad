"""
Optimizable model of the ptychographic reconstruction using automatic differentiation (AD)

This is the PyTorch model that holds optimizable tensors and interacts with loss and constraints.
Updated for high-speed computation and direct probe optimization support via SBCD.
"""

from math import prod

import torch
import torch.nn as nn
from torch.fft import fft2, ifft2
from torchvision.transforms.functional import gaussian_blur

from ptyrad.utils import imshift_batch, torch_phasor, vprint


class PtychoAD(torch.nn.Module):
    """
    Main optimization class for ptychographic reconstruction using automatic differentiation (AD).
    """

    def __init__(self, init_variables, model_params, device="cuda", verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            vprint("### Initializing PtychoAD model ###", verbose=verbose)

            # Setup model behaviors
            self.device = device
            self.verbose = verbose
            self.detector_blur_std = model_params["detector_blur_std"]
            self.obj_preblur_std = model_params["obj_preblur_std"]
            self.solver_type = model_params.get("solver_type", "multislice")
            self.born_iterations = model_params.get("born_iterations", 1)
            self.linduda_order = model_params.get("linduda_order", 2)

            if init_variables.get("on_the_fly_meas_padded", None) is not None:
                self.meas_padded = torch.tensor(
                    init_variables["on_the_fly_meas_padded"], dtype=torch.float32, device=device
                )
                self.meas_padded_idx = torch.tensor(
                    init_variables["on_the_fly_meas_padded_idx"], dtype=torch.int32, device=device
                )
            else:
                self.meas_padded = None
            self.meas_scale_factors = init_variables.get("on_the_fly_meas_scale_factors", None)

            # Parse the learning rate and start iter for optimizable tensors
            start_iter_dict = {}
            end_iter_dict = {}
            lr_dict = {}
            for key, params in model_params["update_params"].items():
                start_iter_dict[key] = params.get("start_iter")
                end_iter_dict[key] = params.get("end_iter")
                lr_dict[key] = params["lr"]
            self.optimizer_params = model_params["optimizer_params"]
            self.start_iter = start_iter_dict
            self.end_iter = end_iter_dict
            self.lr_params = lr_dict

            # Standard Optimizable parameters registered as explicit Leaf Parameters
            self.opt_obja = nn.Parameter(
                torch.abs(torch.tensor(init_variables["obj"], device=device)).to(torch.float32)
            )
            self.opt_objp = nn.Parameter(
                torch.angle(torch.tensor(init_variables["obj"], device=device)).to(torch.float32)
            )
            self.opt_obj_tilts = nn.Parameter(
                torch.tensor(init_variables["obj_tilts"], dtype=torch.float32, device=device)
            )
            self.opt_slice_thickness = nn.Parameter(
                torch.tensor(init_variables["slice_thickness"], dtype=torch.float32, device=device)
            )

            # Explicit real representation for autograd-compatible probe updates
            self.opt_probe = nn.Parameter(
                torch.view_as_real(
                    torch.tensor(init_variables["probe"], dtype=torch.complex64, device=device)
                ).contiguous()
            )
            self.opt_probe_pos_shifts = nn.Parameter(
                torch.tensor(init_variables["probe_pos_shifts"], dtype=torch.float32, device=device)
            )

            # Buffers used during forward pass
            self.register_buffer(
                "omode_occu",
                torch.tensor(init_variables["omode_occu"], dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "H", torch.tensor(init_variables["H"], dtype=torch.complex64, device=device)
            )
            self.register_buffer(
                "measurements",
                torch.tensor(init_variables["measurements"], dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "N_scan_slow",
                torch.tensor(init_variables["N_scan_slow"], dtype=torch.int32, device=device),
            )
            self.register_buffer(
                "N_scan_fast",
                torch.tensor(init_variables["N_scan_fast"], dtype=torch.int32, device=device),
            )
            self.register_buffer(
                "crop_pos",
                torch.tensor(init_variables["crop_pos"], dtype=torch.int32, device=device),
            )
            self.register_buffer(
                "slice_thickness",
                torch.tensor(init_variables["slice_thickness"], dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "dx", torch.tensor(init_variables["dx"], dtype=torch.float32, device=device)
            )
            self.register_buffer(
                "dk", torch.tensor(init_variables["dk"], dtype=torch.float32, device=device)
            )
            self.register_buffer(
                "lambd", torch.tensor(init_variables["lambd"], dtype=torch.float32, device=device)
            )

            # 3D Propagator cache
            self.H_3d = None

            self.n_slice = self.opt_objp.shape[1]
            self.Nx, self.Ny = self.opt_probe.shape[-1], self.opt_probe.shape[-2]

            self.random_seed = init_variables["random_seed"]
            self.length_unit = init_variables["length_unit"]
            self.scan_affine = init_variables["scan_affine"]
            self.tilt_obj = bool(
                self.lr_params.get("obj_tilts", 0) != 0 or torch.any(self.opt_obj_tilts)
            )
            self.shift_probes = bool(self.lr_params.get("probe_pos_shifts", 0) != 0)
            self.change_thickness = bool(self.lr_params.get("slice_thickness", 0) != 0)
            self.probe_int_sum = self.get_complex_probe_view().abs().pow(2).sum()

            self.loss_iters = []
            self.iter_times = []
            self.dz_iters = []
            self.avg_tilt_iters = []

            # Create grids for shifting
            self.create_grids()

            # Dictionary mapping to optimizable parameters
            self.optimizable_tensors = {
                "obja": self.opt_obja,
                "objp": self.opt_objp,
                "obj_tilts": self.opt_obj_tilts,
                "slice_thickness": self.opt_slice_thickness,
                "probe": self.opt_probe,
                "probe_pos_shifts": self.opt_probe_pos_shifts,
            }

            self.create_optimizable_params_dict(self.lr_params, self.verbose)
            self.init_propagator_vars()
            self.init_compilation_iters()

            vprint("### Done initializing PtychoAD model ###", verbose=verbose)

    def get_complex_probe_view(self):
        """
        Retrieves a complex view of the probe while preserving the Autograd computation graph.
        """
        return torch.view_as_complex(self.opt_probe)

    def create_grids(self):
        """Create the coordinate grids for spatial/fourier operations."""
        device = self.device
        probe = self.get_complex_probe_view()
        Npy, Npx = probe.shape[-2:]
        Noy, Nox = self.opt_objp.shape[-2:]

        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=device, dtype=torch.float32) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=device, dtype=torch.float32) + 0.5) / Npx
        ky = torch.fft.ifftshift(2 * torch.pi * ygrid / self.dx)
        kx = torch.fft.ifftshift(2 * torch.pi * xgrid / self.dx)
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.register_buffer("propagator_grid", torch.stack([Ky, Kx], dim=0), persistent=False)

        rpy, rpx = torch.meshgrid(
            torch.arange(Npy, dtype=torch.int32, device=device),
            torch.arange(Npx, dtype=torch.int32, device=device),
            indexing="ij",
        )
        self.register_buffer("rpy_grid", rpy, persistent=False)
        self.register_buffer("rpx_grid", rpx, persistent=False)

        kpy, kpx = torch.meshgrid(
            torch.fft.fftfreq(Npy, dtype=torch.float32, device=device),
            torch.fft.fftfreq(Npx, dtype=torch.float32, device=device),
            indexing="ij",
        )
        koy, kox = torch.meshgrid(
            torch.fft.fftfreq(Noy, dtype=torch.float32, device=device),
            torch.fft.fftfreq(Nox, dtype=torch.float32, device=device),
            indexing="ij",
        )
        self.register_buffer("shift_probes_grid", torch.stack([kpy, kpx], dim=0), persistent=False)
        self.register_buffer("shift_object_grid", torch.stack([koy, kox], dim=0), persistent=False)

    def create_optimizable_params_dict(self, lr_params, verbose=True):
        """Sets up and connects parameter gradients for optimizers."""
        self.lr_params = lr_params
        self.optimizable_params = []
        for param_name, lr in lr_params.items():
            if param_name not in self.optimizable_tensors:
                raise ValueError(f"Invalid parameter name: '{param_name}'")

            tensor = self.optimizable_tensors[param_name]
            is_active = (lr != 0) and (self.start_iter.get(param_name, 1) == 1)
            tensor.requires_grad = is_active

            if is_active:
                self.optimizable_params.append({"params": [tensor], "lr": lr})

        if verbose:
            self.print_model_summary()

    def init_propagator_vars(self):
        """Pre-compute propagator state vectors to avoid execution overhead."""
        dz = self.opt_slice_thickness.detach()
        Ky, Kx = self.propagator_grid
        tilts_y_full = self.opt_obj_tilts[:, 0, None, None] / 1e3
        tilts_x_full = self.opt_obj_tilts[:, 1, None, None] / 1e3

        self.H_fixed_tilts_full = self.H * torch_phasor(
            dz * (Ky * torch.tan(tilts_y_full) + Kx * torch.tan(tilts_x_full))
        )

        self.k = 2 * torch.pi / self.lambd
        self.Kz = torch.sqrt(torch.clamp(self.k**2 - Kx**2 - Ky**2, min=0.0))

    def init_compilation_iters(self):
        """Determine iteration bounds requiring dynamic graph compilation."""
        compilation_iters = {1}
        for param_name in self.optimizable_tensors.keys():
            start = self.start_iter.get(param_name)
            end = self.end_iter.get(param_name)
            if start is not None and start >= 1:
                compilation_iters.add(start)
            if end is not None and end >= 1:
                compilation_iters.add(end)
        self.compilation_iters = sorted(compilation_iters)

    def print_model_summary(self):
        """Prints variable statistics."""
        vprint("### PtychoAD optimizable variables ###")
        for name, tensor in self.optimizable_tensors.items():
            vprint(
                f"{name.ljust(16)}: {str(tensor.shape).ljust(32)}, {str(tensor.dtype).ljust(16)}, device:{tensor.device}, grad:{str(tensor.requires_grad).ljust(5)}, lr:{self.lr_params[name]:.0e}"
            )
        total_var = sum(
            tensor.numel() for tensor in self.optimizable_tensors.values() if tensor.requires_grad
        )
        vprint(" ")
        vprint(f"Total measurement values  : {self.measurements.numel():,d}")
        vprint(f"Total optimizing variables: {total_var:,d}")
        vprint(f"Solver type               : {self.solver_type}")

    def get_obj_ROI(self, indices):
        """Vectorized extraction of region-of-interest object patches."""
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None, :, :] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None, :, :] + self.crop_pos[indices, None, None, 1]

        return opt_obj[:, :, obj_ROI_grid_y, obj_ROI_grid_x, :].permute(2, 0, 1, 3, 4, 5)

    def get_obj_patches(self, indices):
        """Extract object patches, with optional Gaussian pre-blur filter."""
        object_patches = self.get_obj_ROI(indices)

        if self.obj_preblur_std is None or self.obj_preblur_std == 0:
            return object_patches

        obj = object_patches.permute(5, 0, 1, 2, 3, 4)
        obj_shape = obj.shape
        obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
        return (
            gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std)
            .reshape(obj_shape)
            .permute(1, 2, 3, 4, 5, 0)
        )

    def get_probes(self, indices):
        """Batch-generate complex probes with shift offsets."""
        probe = self.get_complex_probe_view()

        if self.shift_probes:
            return imshift_batch(
                probe, shifts=self.opt_probe_pos_shifts[indices], grid=self.shift_probes_grid
            )
        return probe.unsqueeze(0)

    def get_propagators(self, indices):
        """Retrieves Fresnel propagators for the current batch state."""
        tilt_obj = self.tilt_obj
        global_tilt = self.opt_obj_tilts.shape[0] == 1
        change_tilt = self.lr_params.get("obj_tilts", 0) != 0
        change_thickness = self.change_thickness

        dz = self.opt_slice_thickness
        Kz = self.Kz
        Ky, Kx = self.propagator_grid

        tilts = self.opt_obj_tilts if global_tilt else self.opt_obj_tilts[indices]
        tilts_y = tilts[:, 0, None, None] / 1e3
        tilts_x = tilts[:, 1, None, None] / 1e3

        if tilt_obj and change_thickness:
            H_opt_dz = torch_phasor(dz * Kz)
            return H_opt_dz * torch_phasor(dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x)))
        elif tilt_obj and not change_thickness:
            if change_tilt:
                return self.H * torch_phasor(
                    dz * (Ky * torch.tan(tilts_y) + Kx * torch.tan(tilts_x))
                )
            return self.H_fixed_tilts_full if global_tilt else self.H_fixed_tilts_full[indices]
        elif not tilt_obj and change_thickness:
            return torch_phasor(dz * Kz).unsqueeze(0)

        return self.H.unsqueeze(0)

    def get_propagators_3d(self, H_2d):
        """Precomputes/calculates 3D propagation matrices for Born approximation."""
        if not torch.allclose(H_2d, self.H.unsqueeze(0)) or self.H_3d is None:
            z_idx = torch.arange(self.n_slice, device=H_2d.device).view(1, 1, 1, self.n_slice, 1, 1)
            # H_view = H_2d.view(H_2d.shape[0], 1, 1, 1, self.Ny, self.Nx)
            self.H_3d = H_2d.pow(z_idx)
        return self.H_3d

    def get_forward_meas(self, object_patches, probes, propagators):
        """Dispatches forward model evaluation to specialized math engines."""
        if self.solver_type in ["born", "stochastic_born"]:
            if self.born_iterations == 1 or self.solver_type == "stochastic_born":
                from ptyrad.forward_models import firstborn_forward

                dp_fwd = firstborn_forward(object_patches, probes, propagators, self.omode_occu)
            else:
                from ptyrad.forward_models import born_forward

                dp_fwd = born_forward(
                    object_patches,
                    probes,
                    propagators,
                    omode_occu=self.omode_occu,
                    n_max=self.born_iterations,
                )
        elif self.solver_type == "suzuki_trotter":
            from ptyrad.forward_models import suzukitrotter_forward

            dp_fwd = suzukitrotter_forward(
                object_patches, probes, propagators, omode_occu=self.omode_occu
            )
        elif self.solver_type == "strang":
            from ptyrad.forward_models import strang_forward

            dp_fwd = strang_forward(object_patches, probes, propagators, omode_occu=self.omode_occu)
        elif self.solver_type == "linduda":
            from ptyrad.forward_models import linduda_forward

            dp_fwd = linduda_forward(
                object_patches,
                probes,
                propagators,
                omode_occu=self.omode_occu,
                delta=self.opt_slice_thickness.detach(),
                M=self.linduda_order,
            )
        elif self.solver_type == "multislice":
            from ptyrad.forward_models import multislice_forward

            dp_fwd = multislice_forward(
                object_patches, probes, propagators, omode_occu=self.omode_occu
            )
        else:
            raise ValueError(f"Invalid solver_type: {self.solver_type}")

        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)

        return dp_fwd

    def get_measurements(self, indices=None):
        """Retrieve slice or batch measurement arrays."""
        if indices is None:
            return self.measurements

        measurements = self.measurements[indices]
        if self.meas_padded is not None:
            pad_h1, pad_h2, pad_w1, pad_w2 = self.meas_padded_idx
            canvas = torch.zeros(
                (measurements.shape[0], *self.meas_padded.shape[-2:]),
                dtype=measurements.dtype,
                device=self.device,
            )
            canvas += self.meas_padded
            canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements
            measurements = canvas

        if self.meas_scale_factors is not None and any(f != 1 for f in self.meas_scale_factors):
            scale_factor = tuple(self.meas_scale_factors)
            measurements = torch.nn.functional.interpolate(
                measurements.unsqueeze(0), scale_factor=scale_factor, mode="bilinear"
            )[0]
            measurements = measurements / prod(scale_factor)

        return measurements

    def get_propagated_probe(self, index):
        probe = self.get_probes(index)[
            0
        ].detach()  # (pmode, Ny, Nx), just grab the probe at 1st index
        H = self.get_propagators(
            index
        )[
            [0]
        ].detach()  # (1, Ny, Nx) or (N, Ny, Nx) depends on tilt_type ('all' or 'each'), so we need to grab the 1st index without reducing dimension

        probe_prop = torch.zeros(
            (self.n_slice, *probe.shape), dtype=probe.dtype, device=probe.device
        )

        psi = probe  # (z, pmode, Ny, Nx)
        for n in range(self.n_slice):
            probe_prop[n] = psi
            psi = ifft2(H[None,] * fft2(psi))

        return probe_prop

    def clear_cache(self):
        """Resets dynamic memory references."""
        self._current_object_patches = None

    def forward(self, indices):
        """Standard Forward Pass."""
        object_patches = self.get_obj_patches(indices)
        probes = self.get_probes(indices)
        propagators = self.get_propagators(indices)

        if self.solver_type == "born":
            propagators = self.get_propagators_3d(propagators)

        elif self.solver_type == "strang":
            dz_half = self.opt_slice_thickness / 2.0
            H_half_tensor = torch_phasor(dz_half * self.Kz)
            propagators = (propagators, H_half_tensor.unsqueeze(0))

        dp_fwd = self.get_forward_meas(object_patches, probes, propagators)

        self._current_object_patches = object_patches
        return dp_fwd

    # ==========================================================
    # HIGH-SPEED STOCHASTIC BORN HELPER METHODS (SBCD Protocol)
    # ==========================================================

    def forward_probe_update(self, indices):
        """
        Phase 1: Collapsed 2D Probe Forward Pass.
        """
        from ptyrad.forward_models import stochastic_born_probe_forward

        object_patches = self.get_obj_patches(
            indices
        ).detach()  # Frozen object [B, omode, Nz, Ny, Nx, 2]
        probes = self.get_probes(indices)  # Active probe autograd leaf [B, pmode, Ny, Nx]
        H = self.get_propagators(indices)
        H_3d = self.get_propagators_3d(H)

        dp_fwd = stochastic_born_probe_forward(
            object_patches, probes, H_3d, omode_occu=self.omode_occu, eps=1e-10, linearise_obj=True
        )

        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)

        return dp_fwd

    def get_born_components(self, indices):
        """
        Phase 2: Calculates background wavefields for Stochastic Block Coordinate Descent (SBCD).

        Evaluates Psi_state and the detached 3D wavefield cache (u_stack) using the current probe.
        """
        from ptyrad.forward_models import stochastic_born_components

        object_patches = self.get_obj_patches(indices).detach()
        probes = self.get_probes(indices)
        H = self.get_propagators(indices)
        H_3d = self.get_propagators_3d(H)

        return stochastic_born_components(object_patches, probes, H_3d, linearise_obj=True)

    # ==========================================================
    # HIGH-SPEED STOCHASTIC BORN HELPER METHODS (Block SBCD)
    # ==========================================================

    def compute_block_u(self, frame_indices, slice_indices, Psi_state_active):
        """
        Evaluate scattered wavefields (u_block) for a block of slice indices.

        Parameters:
            frame_indices: Batch patch indices for the current ptychographic scan positions.
            slice_indices: Indices of the Z-slices in the current active block (e.g. [2, 3] or range).
            Psi_state_active: Full 3D pre-computed probe illumination stack [B, pmode, 1, Nz, Ny, Nx].
        """
        from ptyrad.forward_models import stochastic_born_single_block_u

        object_patches = self.get_obj_patches(frame_indices)
        H = self.get_propagators(frame_indices)
        H_3d = self.get_propagators_3d(H)

        obj_block = object_patches[:, :, slice_indices, :, :, :]
        H_3d_block = H_3d[:, :, :, slice_indices, :, :]
        Psi_block = Psi_state_active[:, :, :, slice_indices, :, :]
        return stochastic_born_single_block_u(obj_block, Psi_block, H_3d_block, linearise_obj=True)

    def forward_stochastic_block(
        self,
        indices: torch.Tensor,
        slice_indices,
        cache_k: torch.Tensor,
        Psi_state_active: torch.Tensor,
    ):
        """
        Joint mini-block forward pass evaluating cross-slice interaction within slice_indices.

        Args:
            indices: Batch patch indices for the current ptychographic scan positions.
            slice_indices: Indices of the Z-slices in the current active block (e.g. [2, 3] or range).
            cache_k: Pre-computed k-space background wavefield sum of all inactive slices + probe spectrum.
            Psi_state_active: Full 3D pre-computed probe illumination stack [B, pmode, 1, Nz, Ny, Nx].
        """
        from ptyrad.forward_models import stochastic_born_forward_block

        # 1. Fetch current object patches & propagators
        object_patches = self.get_obj_patches(indices)

        H = self.get_propagators(indices)

        # 2. Extract active object block: [B, omode, N_block, Ny, Nx, 2]
        obj_block = object_patches[:, :, slice_indices, :, :, :]

        # 4. Slice pre-computed 3D illumination for active block slices: [B, pmode, 1, N_block, Ny, Nx]
        Psi_state_block = Psi_state_active[:, :, :, slice_indices, :, :]
        H_3d_block = self.get_propagators_3d(H)[:, :, :, slice_indices, :, :]

        # 5. Execute stochastic block forward pass with exact physical depth propagation
        dp_fwd = stochastic_born_forward_block(
            obj_block,
            Psi_state_block,
            H_3d_block,
            cache_k,
            omode_occu=self.omode_occu,
            eps=1e-10,
            linearise_obj=True,
        )

        # 6. Optional detector blurring
        if getattr(self, "detector_blur_std", None) is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)

        return dp_fwd

    @torch.compiler.disable
    def accumulate_block_gradients(self, batch_indices, slice_indices, grad_obja_block, grad_objp_block):
        """
        Scatter/accumulate local block patch gradients back onto full global object gradients.
        """
        # Ensure full gradient tensors exist
        if self.opt_obja.grad is None:
            self.opt_obja.grad = torch.zeros_like(self.opt_obja)
            self.opt_objp.grad = torch.zeros_like(self.opt_objp)

        # 1. Compute global 2D grid Y, X coordinates for the batch patches
        obj_ROI_grid_y = self.rpy_grid[None, :, :] + self.crop_pos[batch_indices, None, None, 0] # [B, Y, X]
        obj_ROI_grid_x = self.rpx_grid[None, :, :] + self.crop_pos[batch_indices, None, None, 1] # [B, Y, X]

        # 2. Reshape coordinates and block gradients for scattering
        # grad_obja_block shape: [B, omode, N_block, Y, X]
        B, omode, N_block, Y, X = grad_obja_block.shape

        # Flatten batch and ROI spatial dimensions
        flat_y = obj_ROI_grid_y.view(-1)  # [B * Y * X]
        flat_x = obj_ROI_grid_x.view(-1)  # [B * Y * X]

        # Convert 2D spatial indices to 1D linear index for full object dimensions (NY, NX)
        NY, NX = self.opt_obja.shape[-2], self.opt_obja.shape[-1]
        flat_linear_idx = flat_y * NX + flat_x  # [B * Y * X]

        # 3. Accumulate gradients across batch overlaps for active Z-slices
        for s_idx_rel, s_idx_abs in enumerate(slice_indices):
            for m in range(omode):
                # Flatten patch gradient spatial dimensions
                g_amp = grad_obja_block[:, m, s_idx_rel, :, :].reshape(-1) # [B * Y * X]
                g_phs = grad_objp_block[:, m, s_idx_rel, :, :].reshape(-1) # [B * Y * X]

                # Target 2D slice view flattened
                target_a_slice = self.opt_obja.grad[m, s_idx_abs].view(-1)
                target_p_slice = self.opt_objp.grad[m, s_idx_abs].view(-1)

                # Atomic addition for overlapping scan positions
                target_a_slice.index_add_(0, flat_linear_idx, g_amp)
                target_p_slice.index_add_(0, flat_linear_idx, g_phs)

    def stochastic_born_analytical_block_grad(self,
                                        indices: torch.Tensor,
                                        residual: torch.Tensor,
                                        Psi_hat_k: torch.Tensor,
                                        Psi_state_active: torch.Tensor,
                                        slice_indices: torch.Tensor,
                                        linearise_obj: bool = True,
                                    ):
        """
        Phase 3: Computes analytical gradients for the current active block of slices.

        Returns:
            grad_obja_block: Gradient w.r.t object amplitude [B, omode, N_block, Y, X].
            grad_objp_block: Gradient w.r.t object phase [B, omode, N_block, Y, X].
        """
        from ptyrad.forward_models import stochastic_born_analytical_block_grad

        object_patches = self.get_obj_patches(indices)
        object_block = object_patches[:, :, slice_indices, :, :, :]
        H = self.get_propagators(indices)  # Ensure propagators are up-to-date for the current block
        Psi_state_block = Psi_state_active[:, :, :, slice_indices, :, :]
        return stochastic_born_analytical_block_grad(
            residual = residual,
            Psi_hat_k = Psi_hat_k,
            Psi_block = Psi_state_block,
            H_3d_block = self.get_propagators_3d(H)[:, :, :, slice_indices, :, :],
            object_block = object_block,
            omode_occu=self.omode_occu,
            linearise_obj=linearise_obj,
        )

    def stochastic_born_analytical_probe_grad(
        self,
        indices: torch.Tensor,
        residual: torch.Tensor,
        Psi_hat_k: torch.Tensor,
        linearise_obj: bool = True
    ) -> torch.Tensor:
        """
        Computes analytical gradients for the probe across the full 3D stack.
        """
        from ptyrad.forward_models import stochastic_born_analytical_probe_grad
        object_patches = self.get_obj_patches(indices)
        H_3d = self.get_propagators_3d(self.get_propagators(indices))
        return stochastic_born_analytical_probe_grad(
            residual=residual,
            Psi_hat_k=Psi_hat_k,
            H_3d=H_3d,
            object_patches=object_patches,
            omode_occu=self.omode_occu,
            linearise_obj=linearise_obj,
        )
