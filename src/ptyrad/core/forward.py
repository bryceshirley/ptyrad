"""
Physical forwad model that generates diffraction patterns from mixed-state probe/object in a fully vectorized way

"""

import torch
from torch.fft import fft2, ifft2

from ptyrad.core.functional import fftshift2

# The forward model takes a batch of object patches and probes with their mixed states
# By introducing and aligning the singleton dimensions carefully,
# we can vectorize all the operations except the serial z-dimension propagation
# For 3D object with n_slices, the for loop would go through n-1 loops and multiply the last slice without further Fresnel propagaiton
# This way we can skip the if statement and make it slightly faster
# For 2D object (n_slices = 1), the entire for loop is skipped
# Note that element-wise multiplication of tensor (*) is defaulted as out-of-place operation
# So new tensor is being created and referenced to the old graph to keep the gradient flowing

def multislice_forward(obja_patches, objp_patches, probe, H, omode_occu=None, eps=1e-10):
    """
    Computes the multislice electron diffraction pattern with multiple incoherent probe
    and object modes using a vectorized forward model.

    Args:
        obja_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx), representing
            object amplitude patches with float32.
            N is the number of samples in a batch, omode is the number of object modes,
            Nz, Ny, Nx are the dimensions of the object patches.
        objp_patches (torch.Tensor): Tensor of shape (N, omode, Nz, Ny, Nx), representing
            object phase patches with float32.
            N is the number of samples in a batch, omode is the number of object modes,
            Nz, Ny, Nx are the dimensions of the object patches.
        omode_occu (torch.Tensor): Tensor of shape (omode,) with float32 values, representing
            the occupancy/expectation for each object mode. The sum of all elements should be 1.
        probe (torch.Tensor): Tensor of shape (N, pmode, Ny, Nx) with complex64 values,
            representing the probe(s). N is the number of samples in the batch, pmode is the
            number of probe modes. By default, N is 1, assuming the same probe for all samples.
        H (torch.Tensor): Tensor of shape (N, Ky, Kx) with complex64 values, representing the Fresnel
            propagator that propagates the wave by a slice thickness.
        eps (float, optional): A small value added for numerical stability. Defaults to 1e-10.

    Returns:
        torch.Tensor: Tensor of shape (N, Ky, Kx) with float32 positive values, representing the
        forward diffraction pattern for each sample in the batch.
    """

    assert obja_patches.shape == objp_patches.shape

    # Initialize omode_occu if it's not specified
    if omode_occu is None:
        device = objp_patches.device
        dtype = objp_patches.dtype
        omode = objp_patches.size(1)
        omode_occu = torch.ones(omode, dtype=dtype, device=device) / omode

    # Unbind the Z-dimension (dim=2) BEFORE the loop
    # This returns a tuple of n_slices independent tensors of shape (N, omode, Ny, Nx)
    # This is critical for efficient torch.compile triton code generation during .backward(), especially for pytorch >= 2.8.0
    obja_slices = torch.unbind(obja_patches, dim=2)
    objp_slices = torch.unbind(objp_patches, dim=2)
    n_slices = len(obja_slices)

    # Expand psi to include omode dimension
    psi = probe[:, :, None, :, :] # (N, pmode, Ny, Nx) -> (N, pmode, omode, Ny, Nx)

    # Propagating each object layer using broadcasting
    for n in range(n_slices - 1):
        object_slice = torch.polar(obja_slices[n], objp_slices[n]) # object_slice -> (N, omode, Ny, Nx)
        psi = (psi * object_slice[:, None, :, :]) # psi -> (N, pmode, omode, Ny, Nx). Note that psi is always centered in real space
        psi = ifft2(H[:, None, None] * fft2(psi)) # Note that fft2 and ifft2 are applying to the last 2 axes. Although preshift psi before fft2 would seem more natural, it's nearly 50% slower to do it as fftshift2(ifft2(fft2(ifftshift2(psi))))

    # Interacting with the last layer, and no propagation is needed afterward
    object_slice = torch.polar(obja_slices[-1], objp_slices[-1])
    psi = psi * object_slice[:, None, :, :]

    # Propagate the object-modified exit wave psi(r) to detector plane into psi(k)
    # The contribution from probe / object modes are incoherently summed together
    # Chained all operations for lower peak memory consumption
    # Doing fftshift2 last reduces the needed memory moves
    # Note that norm = 'ortho' is needed to ensure that for each sample, sum(|psi|^2) and sum(dp) has the same scale (should be 1)

    dp_fwd = (
        fftshift2(
            torch.sum(
                fft2(psi, norm="ortho").abs().square() * omode_occu[:, None, None],
                dim=(1, 2),
            ),
        )
        + eps
    )  # Add eps for numerical stability
    return dp_fwd

def born_gauge(object_cplx, H1, background):
    """
    The Born background gauge of these object patches.

    H1 is the single-step propagator as the models take it,
    (N, 1, 1, Ky, Kx). The gauge carries the cumulative powers H^{z_j}
    rather than the step, the Fourier view referring every slice block
    to the entrance plane.
    """
    from ptychomodels import centre, cumulative_propagators

    return centre(object_cplx, cumulative_propagators(H1, object_cplx.shape[2]), background)

def born_forward(obja_patches, objp_patches, probe, H, n_max, background="mean",
                 coeffs=None, omode_occu=None, eps=1e-10):
    """
    Inter-slice Born series diffraction pattern, truncated at order ``n_max``.

    The sequential depth loop of ``multislice_forward`` is replaced by a Born
    expansion in the Fourier view, where the only operation coupling the slices
    is a cumulative sum, so the sequential depth becomes ``n_max`` instead of
    ``n_slices``. The series is nilpotent, so ``n_max = n_slices`` reproduces
    ``multislice_forward`` exactly. The propagation itself lives in the optional
    ``ptychomodels`` dependency; this function is the change of layout.

    Args:
        obja_patches, objp_patches, probe, H, omode_occu, eps: as in
            ``multislice_forward``.
        n_max (int): truncation order M of the series. 1 is the inter-slice
            first Born model.
        background (str): the background a_j the series expands about,
            'vacuum' (a_j = 1) or 'mean' (a_j = <O_j> over the patch).
        coeffs (torch.Tensor, optional): complex weight per retained order,
            reweighting only the exit-block sum, from a coefficient refit.

    Returns:
        torch.Tensor: Tensor of shape (N, Ky, Kx), as ``multislice_forward``.
    """
    from ptychomodels import born_forward as born_series
    from ptychomodels import intensity

    object_cplx = torch.polar(obja_patches, objp_patches)
    gauge = born_gauge(object_cplx, H[:, None, None], background)
    return intensity(born_series(gauge, probe, n_max, coeffs), omode_occu, eps)
