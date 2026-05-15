"""
Dataset splitting routines for Fourier Ring Correlation (FRC) evaluations.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch

from ptyrad.load import load_array_from_file
from ptyrad.utils import vprint


def binomial_split_stack_torch(stack, p=0.5, seed=42, device="cuda", chunk_size=256):
    """
    Binomial split of a frame or stack using GPU-accelerated PyTorch.
    Processes the stack in chunks to prevent VRAM Out-Of-Memory (OOM) errors.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1).")

    # Ensure input is a CPU tensor first (to avoid dumping massive arrays directly to VRAM)
    if not isinstance(stack, torch.Tensor):
        stack = torch.from_numpy(stack)

    if torch.any(stack < 0):
        min_val = stack.min().item()
        vprint(
            f"WARNING: Negative values found (min = {min_val:.4f}). Clipping to 0 to simulate physical electron counts."
        )
        stack = torch.clamp(stack, min=0.0)

    if torch.any(stack < 0):
        raise ValueError("Counts must be nonnegative for binomial splitting.")

    # Set up PyTorch Generator for reproducibility on the target device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    prob_tensor = torch.tensor(p, device=device, dtype=torch.float32)

    A_chunks = []
    B_chunks = []

    # Process the stack in chunks to protect VRAM
    for i in range(0, stack.shape[0], chunk_size):
        # Move chunk to GPU
        chunk = stack[i : i + chunk_size].to(device)

        # Torch.binomial requires float32 count tensors
        # Rounding handles fractional floats from normalized/pre-processed datasets
        chunk_float = torch.round(chunk).to(torch.float32)

        # Extremely fast GPU-accelerated binomial sampling
        A_chunk = torch.binomial(chunk_float, prob_tensor, generator=gen)
        B_chunk = chunk_float - A_chunk

        # Move back to CPU and cast to original dtype
        A_chunks.append(A_chunk.to(dtype=stack.dtype, device="cpu"))
        B_chunks.append(B_chunk.to(dtype=stack.dtype, device="cpu"))

    # Reassemble the full stacks on CPU memory
    A = torch.cat(A_chunks, dim=0)
    B = torch.cat(B_chunks, dim=0)

    if torch.equal(A, B):
        raise ValueError("Binomial split failed: split A and B datasets are identical!")

    return A, B


def odd_even_split_stack_torch(stack):
    """
    Odd/Even splitting of a dataset across the scan positions (0th dimension).
    """
    if not isinstance(stack, torch.Tensor):
        stack = torch.from_numpy(stack)

    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (N_scans, Ny, Nx), got {stack.ndim}D.")

    # Slicing is virtually instantaneous and requires zero GPU computation
    A = stack[1::2, :, :]
    B = stack[0::2, :, :]

    return A, B


def _save_split_data(data_np, filepath, ext, key="data"):
    """
    Helper function to save numpy arrays into various file formats.
    """
    # Electron counts never need float64 (double precision).
    # Downcasting to float32 instantly cuts disk footprint in half without losing data,
    # which is also consistent with PtyRAD's internal measurement casting.
    if data_np.dtype == np.float64:
        data_np = data_np.astype(np.float32)

    ext = ext.lower()
    if ext in [".h5", ".hdf5"]:
        with h5py.File(filepath, "w") as f:
            f.create_dataset(key, data=data_np, compression="gzip", chunks=True)
    elif ext == ".mat":
        # Save as standard .mat file with compression enabled to prevent massive file bloat
        sio.savemat(filepath, {key: data_np}, do_compression=True)
    elif ext == ".npy":
        np.save(filepath, data_np)
    elif ext in [".tif", ".tiff"]:
        import tifffile

        # Apply zlib compression to TIFFs as well
        tifffile.imwrite(filepath, data_np, compression="zlib")
    elif ext == ".raw":
        data_np.tofile(filepath)
    else:
        raise ValueError(f"Unsupported save format for splitting: {ext}")


def generate_frc_splits(params, verbose=True, plot=False, device="cuda"):
    """
    Loads the original dataset via PtyRAD, splits it using PyTorch,
    and saves the halves as HDF5 files in the output directory.
    Returns the paths to the newly created split files.
    """
    split_method = params["hypertune_params"].get("frc_split_method", "binomial")

    # Get Save dir from full params path
    full_data_path = params["init_params"]["meas_params"]["path"]
    save_dir = os.path.dirname(full_data_path)

    vprint(f"### Preparing {split_method} data splits for FRC evaluation ###", verbose=verbose)

    meas_params = params["init_params"]["meas_params"]

    if not isinstance(meas_params, dict) or "path" not in meas_params:
        raise TypeError(
            "meas_params must be a dictionary containing a 'path' to the original file."
        )

    original_path = meas_params["path"]

    # Extract base name and ORIGINAL extension
    original_filename = os.path.basename(original_path)
    base_name, orig_ext = os.path.splitext(original_filename)

    # =========================================================================
    # Dynamically inject shape for .raw files (Using orig_ext)
    # =========================================================================
    if orig_ext.lower() == ".raw" and meas_params.get("shape") is None:
        N_scans = params["init_params"].get("pos_N_scans")
        Npix = params["init_params"].get("meas_Npix")
        if N_scans is not None and Npix is not None:
            meas_params["shape"] = (N_scans, Npix, Npix)
            vprint(f"Injected shape {meas_params['shape']} for .raw file loading.", verbose=verbose)
        else:
            raise ValueError(
                "Could not infer shape for .raw file. Ensure 'pos_N_scans' and 'meas_Npix' are in init_params."
            )
    # =========================================================================

    # FORCE the output extension to be HDF5 to avoid EMPAD .raw gap logic issues
    save_ext = ".hdf5"
    save_key = "data"  # Force a clean key name for the new HDF5 files

    path_A = os.path.join(save_dir, f"{base_name}_split1{save_ext}")
    path_B = os.path.join(save_dir, f"{base_name}_split2{save_ext}")

    if os.path.exists(path_A) and os.path.exists(path_B):
        vprint(f"Found existing split files in {save_dir}. Skipping generation.", verbose=verbose)
        return path_A, path_B

    # 1. Load data
    vprint(f"Loading original data from {original_path}...", verbose=verbose)
    data = load_array_from_file(**meas_params)

    # 2. Split data
    vprint(f"Executing GPU-accelerated {split_method} split...", verbose=verbose)
    if split_method == "binomial":
        seed = params["init_params"].get("random_seed", 42)
        if seed is None:
            seed = 42
        A, B = binomial_split_stack_torch(data, p=0.5, seed=seed, device=device)
    elif split_method == "odd_even":
        A, B = odd_even_split_stack_torch(data)
        params["init_params"]["pos_N_scans"] = A.shape[0]
        params["init_params"]["pos_N_scan_fast"] = (
            A.shape[0] // params["init_params"]["pos_N_scan_slow"]
        )
    else:
        raise ValueError(f"Unknown frc_split_method: {split_method}.")

    if plot:
        plot_split_diffraction_patterns(data, A, B, index=0, log_scale=True)

    # 3. Save as clean HDF5
    vprint(f"Saving splits to {save_dir} with extension {save_ext}...", verbose=verbose)
    A_np = A.cpu().numpy() if isinstance(A, torch.Tensor) else A
    B_np = B.cpu().numpy() if isinstance(B, torch.Tensor) else B

    _save_split_data(A_np, path_A, save_ext, key=save_key)
    _save_split_data(B_np, path_B, save_ext, key=save_key)

    vprint("Splits successfully generated and saved.", verbose=verbose)
    return path_A, path_B


def plot_split_diffraction_patterns(
    full_data, split_A, split_B, index=0, log_scale=True, cmap="viridis"
):
    """
    Plots a side-by-side comparison of a diffraction pattern from the full dataset
    and its corresponding binomial/odd-even splits, using a shared color scale.

    Args:
        full_data: Original stack of diffraction patterns (N, Ny, Nx)
        split_A: First split stack (N, Ny, Nx)
        split_B: Second split stack (N, Ny, Nx)
        index (int): Which diffraction pattern to plot from the stack
        log_scale (bool): Whether to apply log10 scaling for visualization (recommended for DPs)
        cmap (str): Matplotlib colormap to use
    """

    # Helper to safely move PyTorch GPU tensors to CPU numpy arrays
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    # Extract the single diffraction pattern
    full_img = to_numpy(full_data[index])
    img_A = to_numpy(split_A[index])
    img_B = to_numpy(split_B[index])

    # Optional log scaling for better visibility of the dynamic range
    if log_scale:
        # Add a small epsilon to avoid log(0) warnings
        eps = 1e-6
        # Clip to ensure no negative values accidentally crept in
        full_img = np.log10(np.clip(full_img, a_min=0, a_max=None) + eps)
        img_A = np.log10(np.clip(img_A, a_min=0, a_max=None) + eps)
        img_B = np.log10(np.clip(img_B, a_min=0, a_max=None) + eps)

    # Determine the global minimum and maximum for the shared colorbar
    vmin = min(full_img.min(), img_A.min(), img_B.min())
    vmax = max(full_img.max(), img_A.max(), img_B.max())

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Full Data
    im0 = axes[0].imshow(full_img, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Full Dataset (Index {index})")
    axes[0].axis("off")

    # Plot Split A
    axes[1].imshow(img_A, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Split A")
    axes[1].axis("off")

    # Plot Split B
    axes[2].imshow(img_B, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title("Split B")
    axes[2].axis("off")

    # Add a single shared colorbar at the bottom
    cbar = fig.colorbar(im0, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, aspect=40)
    cbar.set_label("Log10(Intensity)" if log_scale else "Intensity")

    plt.suptitle("Diffraction Pattern Splitting Comparison", fontsize=16, y=1.05)
    plt.show()

    return fig
