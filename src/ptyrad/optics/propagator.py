"""
Numpy-based propagator functions

"""

import numpy as np
from numpy.fft import ifftshift


# Propagator function used in init/initializer > init_H
def near_field_evolution(Npix_shape, dx, dz, lambd):
    r"""Generates the free-space propagation transfer function using the Angular Spectrum Method (ASM).

    This function calculates the exact wave propagator in Fourier space, often 
    referred to in literature as the Angular Spectrum Method, rather than the 
    paraxial approximation. The transfer function is defined as:

    .. math::

        H(k_x, k_y) = \exp\left(i \Delta z \sqrt{k^2 - k_x^2 - k_y^2}\right)

    The output array is shifted via ``ifftshift`` so that the zero-frequency 
    component is located at the corners (index ``[0, 0]``). This allows it to be 
    directly multiplied with the output of standard unshifted FFT routines (e.g., ``fft2``).

    Note:
        Translated and simplified from Yi's fold_slice MATLAB implementation 
        into NumPy by Chia-Hao Lee.

    Args:
        Npix_shape (tuple of int): The dimensions of the 2D grid in pixels, 
            typically given as :math:`(N_y, N_x)`.
        dx (float): The real-space pixel size (assumed isotropic in :math:`x` and :math:`y`).
        dz (float, list, or numpy.ndarray): The propagation distance(s) along the 
            optical axis. Can be a single scalar or a 1D array of distances.
        lambd (float): The wavelength of the electron or illumination wave.

    Returns:
        numpy.ndarray: A complex array representing the propagation transfer function 
        in :math:`k`-space. If `dz` is a scalar, returns a 2D array of shape :math:`(N_y, N_x)`. 
        If `dz` is an array of length :math:`N_z`, returns a 3D array of shape :math:`(N_z, N_y, N_x)`.
    """

    ygrid = (np.arange(-Npix_shape[0] // 2, Npix_shape[0] // 2) + 0.5) / Npix_shape[0]
    xgrid = (np.arange(-Npix_shape[1] // 2, Npix_shape[1] // 2) + 0.5) / Npix_shape[1]

    # Standard ASM
    k  = 2 * np.pi / lambd
    ky = 2 * np.pi * ygrid / dx
    kx = 2 * np.pi * xgrid / dx
    Ky, Kx = np.meshgrid(ky, kx, indexing="ij")
    
    # Add +0j to safely handle evanescent waves (where kx^2 + ky^2 > k^2) for defensive programming (nearly impossible in electron microscopy)
    kz = np.sqrt(k ** 2 - Kx ** 2 - Ky ** 2 + 0j) 

    # Handle scalar vs array for dz
    is_scalar = np.isscalar(dz) or (isinstance(dz, np.ndarray) and dz.ndim == 0)
    dz_arr = np.atleast_1d(dz)
    
    # Reshape dz for broadcasting: turns (N_z,) into (N_z, 1, 1)
    dz_arr = dz_arr[:, None, None]
    
    # Compute H: broadcasts (N_z, 1, 1) * (N_y, N_x) -> (N_z, N_y, N_x)
    H = np.exp(1j * dz_arr * kz)
    
    # Final H has zero frequency at the corner in k-space
    H = ifftshift(H, axes=(-2, -1)) 

    # Return a 2D array if the input was a scalar, otherwise return the 3D stack
    if is_scalar:
        return H[0]
    
    return H