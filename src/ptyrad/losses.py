"""
Loss functions and soft regularizations calculated using forward simulations against experimental measurements

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms.functional import gaussian_blur

from ptyrad.utils import normalize_from_zero_to_one


# The CombinedLoss takes a user-defined dict of loss_params, which specifies the state, weight, and param of each loss term
# The DP related loss takes a parameter of dp_pow which raise the DP with certain power,
# usually 0.5 for loss_single and 0.2 for loss_pacbed to emphasize the diffuse background
# The obj-dependent regularization loss_sparse is using the objp_patches as input
# In this way it'll only calculate values within the ROI, so the edges of the object would not be included
class CombinedLoss(torch.nn.Module):
    """
    Computes the combined loss for ptychographic reconstruction, incorporating multiple loss components.

    This class implements various loss functions that are combined to optimize the reconstruction
    in ptychography. The loss components include losses based on Gaussian and Poisson statistics,
    PACBED loss, sparsity regularization, and similarity between different object modes.

    Args:
        loss_params (dict): A dictionary containing the configuration and weights for each of the loss components.
        device (str, optional): The device on which the computations will be performed, e.g., 'cuda'. Defaults to 'cuda'.

    """

    def __init__(self, loss_params, device="cuda"):
        super(CombinedLoss, self).__init__()
        self.device = device
        self.loss_params = loss_params
        self.mse = torch.nn.MSELoss(reduction="mean")

    def get_loss_single(self, model_DP, measured_DP):
        """Computes the loss based on Gaussian statistics of the diffraction patterns."""
        # Calculate loss_single
        # This loss function emulates the likelihood function of diffraction patterns with Gaussian statistics (higher dose)
        # For exact Gaussian statistics, the dp_pow should be 0.5

        single_params = self.loss_params["loss_single"]
        if single_params["state"]:
            dp_pow = single_params.get("dp_pow", 0.5)
            data_mean = measured_DP.pow(dp_pow).mean()
            loss_single = (
                self.mse(model_DP.pow(dp_pow), measured_DP.pow(dp_pow)) ** 0.5 / data_mean
            )  # Doing Normalized RMSE makes the value quite consistent between dp_pow 0.2-0.5.
            loss_single *= single_params["weight"]
        else:
            loss_single = torch.tensor(
                0, dtype=torch.float32, device=self.device
            )  # Return a scalar 0 tensor so that the append/sum would work normally without NaN
        return loss_single

    def get_loss_poissn(self, model_DP, measured_DP):
        """Computes the loss based on Poisson statistics of the diffraction patterns."""
        # Calculate loss_poissn
        # This loss function emulates the likelihood function of diffraction patterns with Poisson statistics (low dose)
        # For exact Poisson statistics, the dp_pow should be 1
        # No need to worry about the DP having most pixel value smaller than 1, DP int scaling has no effect to the reconstruction
        # The eps in log is needed for numerical stability during optimization and to avoid negative infinite when the DP intensity is approaching 0
        # Typical eps is within 1e-3 to 1e-9

        # function L = get_loglik(modF, aPsi)
        # modF2 = modF.^2; # exp
        # aPsi2 = aPsi.^2; # model
        # L = -(modF2 .* log(aPsi2+1e-6) - aPsi2) ;
        poissn_params = self.loss_params["loss_poissn"]

        if poissn_params["state"]:
            dp_pow = poissn_params.get("dp_pow", 1)
            eps = poissn_params.get("eps", 1e-6)
            data_mean = measured_DP.pow(dp_pow).mean()
            loss_poissn = (
                -torch.mean(
                    measured_DP.pow(dp_pow) * torch.log(model_DP.pow(dp_pow) + eps)
                    - model_DP.pow(dp_pow)
                )
                / data_mean
            )  # Doing Normalized RMSE makes the value quite consistent between dp_pow 0.2-0.5.
            loss_poissn *= poissn_params["weight"]
        else:
            loss_poissn = torch.tensor(
                0, dtype=torch.float32, device=self.device
            )  # Return a scalar 0 tensor so that the append/sum would work normally without NaN
        return loss_poissn

    def get_loss_pacbed(self, model_DP, measured_DP):
        """Computes the PACBED loss by comparing averaged diffraction patterns."""

        # Calculate loss_pacbed
        pacbed_params = self.loss_params["loss_pacbed"]
        if pacbed_params["state"]:
            dp_pow = pacbed_params.get("dp_pow", 0.2)
            data_mean = measured_DP.pow(dp_pow).mean()
            loss_pacbed = (
                self.mse(model_DP.mean(0).pow(dp_pow), measured_DP.mean(0).pow(dp_pow)) ** 0.5
                / data_mean
            )  # Doing Normalized RMSE makes the value quite consistent between dp_pow 0.2-0.5.
            loss_pacbed *= pacbed_params["weight"]
        else:
            loss_pacbed = torch.tensor(0, dtype=torch.float32, device=self.device)
        return loss_pacbed

    def get_loss_sparse(self, objp_patches, omode_occu):
        """Computes the sparsity regularization loss on object phase patches."""
        # Calculate loss_sparse by considering the ln norm
        # For obj-dependent regularization terms, the omode contribution should be weighting the individual loss for each omode.
        # Scaling the obj value by its omode_occu would make non-linear loss like l2 dependent on # of omode.
        # Therefore, the proper way is to get a loss tensor L(obj) shaped (N, omode, Nz, Ny, Nx) and then do the voxel-wise mean across (N,:,Nz,Ny,Nx)
        # and lastly we do the weighted sum with omode_occu so that the loss value is not batch, object size, or omode dependent.
        sparse_params = self.loss_params["loss_sparse"]
        if sparse_params["state"]:
            ln_order = sparse_params["ln_order"]
            loss_sparse = (
                sparse_params["weight"]
                * (
                    torch.mean(objp_patches.abs().pow(ln_order), dim=(0, 2, 3, 4)).pow(1 / ln_order)
                    * omode_occu
                ).sum()
            )
        else:
            loss_sparse = torch.tensor(0, dtype=torch.float32, device=self.device)
        return loss_sparse

    def get_loss_simlar(self, object_patches, omode_occu):
        """Computes the similarity loss between different object modes."""

        # Calculate loss_simlar by calculating the similarity between different omodes
        # This loss term is specifically designed for regularizing omode by reducing the std of Gaussian_blurred / downsampled obj along the omode dimension
        # obja/p_patches = (N,omode,Nz,Ny,Nx)
        simlar_params = self.loss_params["loss_simlar"]
        if simlar_params["state"]:
            obj_type = simlar_params["obj_type"]
            obj_blur_std = simlar_params["blur_std"]
            scale_factor = simlar_params["scale_factor"]
            obja_patches = object_patches[..., 0]
            objp_patches = object_patches[..., 1]
            temp_loss = torch.tensor(0, dtype=torch.float32, device=self.device)

            if obj_type in ["amplitude", "both"]:
                if obj_blur_std is not None and obj_blur_std != 0:
                    obja_shape = obja_patches.shape
                    obja = obja_patches.reshape(-1, obja_shape[-2], obja_shape[-1])
                    obja_patches = gaussian_blur(obja, kernel_size=5, sigma=obj_blur_std).reshape(
                        obja_shape
                    )
                if scale_factor is not None and any(scale != 1 for scale in scale_factor):
                    obja_patches = interpolate(obja_patches, scale_factor=scale_factor, mode="area")
                temp_loss += (obja_patches * omode_occu[:, None, None, None]).std(1).mean()

            if obj_type in ["phase", "both"]:
                if obj_blur_std is not None and obj_blur_std != 0:
                    objp_shape = objp_patches.shape
                    objp = objp_patches.reshape(-1, objp_shape[-2], objp_shape[-1])
                    objp_patches = gaussian_blur(objp, kernel_size=5, sigma=obj_blur_std).reshape(
                        objp_shape
                    )
                if scale_factor is not None and any(scale != 1 for scale in scale_factor):
                    objp_patches = interpolate(objp_patches, scale_factor=scale_factor, mode="area")
                temp_loss += (objp_patches * omode_occu[:, None, None, None]).std(1).mean()
            loss_simlar = simlar_params["weight"] * temp_loss
        else:
            loss_simlar = torch.tensor(0, dtype=torch.float32, device=self.device)
        return loss_simlar

    def forward(self, model_DP, measured_DP, object_patches, omode_occu):
        """
        Combines all the loss components and returns the total loss and individual losses.

        """
        losses = []
        losses.append(self.get_loss_single(model_DP, measured_DP))
        losses.append(self.get_loss_poissn(model_DP, measured_DP))
        losses.append(self.get_loss_pacbed(model_DP, measured_DP))
        losses.append(self.get_loss_sparse(object_patches[..., 1], omode_occu))
        losses.append(self.get_loss_simlar(object_patches, omode_occu))
        total_loss = sum(losses)
        return total_loss, losses


# This constrast function is currently only used for Hypertune objective
def get_objp_contrast(model, indices):
    """Calculate the contrast from objp zsum imgage for Hypertune purpose"""
    with torch.no_grad():
        probe = model.get_complex_probe_view()
        objp = (
            model.opt_objp.detach().sum(1).squeeze()
        )  # Sum along z and squeeze the omode dimension

        # Get crop positions and compute bounds
        crop_pos = (
            model.crop_pos[indices].detach()
            + torch.tensor(probe.shape[-2:], device=model.crop_pos.device) // 2
        )
        y_min, y_max = crop_pos[:, 0].min().item(), crop_pos[:, 0].max().item()
        x_min, x_max = crop_pos[:, 1].min().item(), crop_pos[:, 1].max().item()

        # Crop object phase tensor
        objp_crop = objp[y_min - 1 : y_max, x_min - 1 : x_max]

        objp_crop = normalize_from_zero_to_one(
            objp_crop
        )  # In case the background is very negative for reconstructions without positivity constraint. Normalization doesn't change the contrast.

        contrast = torch.std(objp_crop) / (torch.mean(objp_crop) + 1e-8)  # Avoid division by zero

    return contrast


"""
Metric utility functions.

This file has been adapted from the PTYPY package to torch.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""


def remove_margins_torch(img, margin):
    if margin == 0:
        return img
    if img.ndim == 2:
        return img[margin:-margin, margin:-margin]
    elif img.ndim == 3:
        return img[:, margin:-margin, margin:-margin]
    else:
        raise ValueError("Input image must be 2D or 3D.")


def apodization_torch(img, apod_width):
    if apod_width == 0:
        return torch.ones_like(img)
    nr, nc = img.shape[-2:]
    device = img.device

    Nr = torch.fft.fftshift(torch.arange(nr, device=device, dtype=torch.float32))
    Nc = torch.fft.fftshift(torch.arange(nc, device=device, dtype=torch.float32))

    window1D1 = (
        1.0
        + torch.cos(2 * torch.pi * (Nr - ((nr - 2 * apod_width - 1) // 2)) / (1 + 2 * apod_width))
    ) / 2.0
    window1D2 = (
        1.0
        + torch.cos(2 * torch.pi * (Nc - ((nc - 2 * apod_width - 1) // 2)) / (1 + 2 * apod_width))
    ) / 2.0

    window1D1[apod_width:-apod_width] = 1.0
    window1D2[apod_width:-apod_width] = 1.0

    return torch.outer(window1D1, window1D2)


def ringthickness_torch(nr, nc, device):
    nmax = max(nr, nc)
    x = (
        torch.arange(-np.fix(nc / 2.0), np.ceil(nc / 2.0), device=device)
        * np.floor(nmax / 2.0)
        / np.floor(nc / 2.0)
    )
    y = (
        torch.arange(-np.fix(nr / 2.0), np.ceil(nr / 2.0), device=device)
        * np.floor(nmax / 2.0)
        / np.floor(nr / 2.0)
    )

    x = torch.fft.ifftshift(x)
    y = torch.fft.ifftshift(y)

    Y, X = torch.meshgrid(y, x, indexing="ij")
    sumsquares = X**2 + Y**2
    index = torch.round(torch.sqrt(sumsquares)).to(torch.int64)
    return index


def fourierringcorrelation_torch(input1, input2, apod_width=0):
    nr, nc = input1.shape

    window = apodization_torch(input1, apod_width)
    img1_apod = input1 * window
    img2_apod = input2 * window

    F1 = torch.fft.fft2(torch.fft.ifftshift(img1_apod))
    F2 = torch.fft.fft2(torch.fft.ifftshift(img2_apod))

    index = ringthickness_torch(nr, nc, input1.device)
    index_flat = index.flatten()

    num = (F1 * F2.conj()).real.flatten()
    C1_raw = (torch.abs(F1) ** 2).flatten()
    C2_raw = (torch.abs(F2) ** 2).flatten()

    # Matches the len(rfftfreq(nmax)) logic in PTYPY
    nmax = max(nr, nc)
    num_bins = nmax // 2 + 1

    C = torch.bincount(index_flat, weights=num)[:num_bins]
    C1 = torch.bincount(index_flat, weights=C1_raw)[:num_bins]
    C2 = torch.bincount(index_flat, weights=C2_raw)[:num_bins]
    npts = torch.bincount(index_flat)[:num_bins]

    FRC = C / torch.sqrt(C1 * C2 + 1e-12)
    fn = torch.linspace(0, 1.0, num_bins, device=input1.device)

    # 1-bit threshold
    snrt = 1.0
    npts_f = npts.float() + 1e-12
    Tnum = snrt + (2 * np.sqrt(snrt) / torch.sqrt(npts_f)) + 1 / torch.sqrt(npts_f)
    Tden = snrt + (2 * np.sqrt(snrt) / torch.sqrt(npts_f)) + 1
    T = Tnum / Tden

    return FRC, T, fn


def get_objp_frc_auc(model_A, model_B, margin=0, apod_width=0, output_dir=None, trial_id=""):
    """
    Calculate FRC AUC between two models, generate a visual summary subplot,
    and save the figure to the trial's result directory without interpolation.
    """
    with torch.no_grad():
        # 1. Extract and project the object phase
        # Grab the first layer [0], then sum across the modes (which is now dim=0)
        img1 = model_A.opt_objp.detach()[0].sum(dim=0)
        img2 = model_B.opt_objp.detach()[0].sum(dim=0).to(img1.device)

        # 2. Fix canvas size mismatches
        if img1.shape != img2.shape:
            ny = min(img1.shape[0], img2.shape[0])
            nx = min(img1.shape[1], img2.shape[1])
            img1 = img1[:ny, :nx]
            img2 = img2[:ny, :nx]

        # 3. Remove margins
        if margin > 0:
            from ptyrad.losses import remove_margins_torch

            img1 = remove_margins_torch(img1, margin)
            img2 = remove_margins_torch(img2, margin)

        # 4. Compute FRC
        from ptyrad.losses import fourierringcorrelation_torch

        FRC_curve, T, fn = fourierringcorrelation_torch(img1, img2, apod_width=apod_width)

        # 5. Integrate AUC
        auc = torch.trapz(FRC_curve, fn)
        if torch.isnan(auc):
            auc = torch.tensor(0.0, device=img1.device)

        optuna_score = -1.0 * auc.item()

    # =========================================================
    # COMPULSORY PLOTTING & SAVING
    # =========================================================
    if output_dir is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Get phase maps
        ph1 = img1.cpu()
        ph2 = img2.cpu()

        # Robust contrast: Ensure vmin and vmax are not identical
        vmin = min(ph1.min().item(), ph2.min().item())
        vmax = max(ph1.max().item(), ph2.max().item())
        if vmin == vmax:  # Handle cases with zero signal
            vmin, vmax = -0.1, 0.1

        # Plot Phase A
        im1 = axes[0].imshow(ph1, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[0].set_title(f"Phase A ({trial_id})")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot Phase B
        im2 = axes[1].imshow(ph2, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[1].set_title(f"Phase B ({trial_id})")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # --- NEW: Call the FRC plotting function for the 3rd axis ---
        frc_plot(FRC_curve.cpu().numpy(), T.cpu().numpy(), fn.cpu().numpy(), ax=axes[2])
        axes[2].set_title(f"FRC (AUC: {auc.item():.4f})")

        plt.tight_layout()

        # Save to the results directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"frc_summary_{trial_id}.png")
        plt.savefig(save_path, dpi=120)
        plt.close(fig)

    return optuna_score


def frc_plot(FRC, T, fn, ax=None):
    """
    Plot raw FRC
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Plot raw discrete data
    ax.plot(fn, FRC.real, "-b", linewidth=1.5, label="FRC")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Spatial freq (Nyquist normalized)")
    ax.set_ylabel("FRC Magnitude")
    ax.legend(loc="upper right", fontsize="x-small")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
