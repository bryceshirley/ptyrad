# -*- coding: utf-8 -*-
"""
Metric utility functions.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fftshift, ifftshift
from scipy.interpolate import interp1d

__all__ = [
    "nyquist",
    "ringthickness",
    "apodization",
    "fourierringcorrelation",
    "frc_plot",
]


def nyquist(arraysize):
    """
    Evaluate the Nyquist Frequency

    Parameters
    ----------
    arraysize :  int
        input array length

    Returns
    -------
    f : array-like
        Array containing the frequencies

    fnyquist : array-like
        The Nyquist-frequency
    """
    nmax = np.max(arraysize)
    f = np.fft.rfftfreq(nmax)
    fnyquist = np.max(f)
    return f, fnyquist


def ringthickness(inputarray):
    """
    Defines indexes for ring thickness

    Parameters
    ----------
    inputarray :  array-like
        input array, must be at least two-dimensional

    Returns
    -------
    index : array-like
        Indexes for the rings
    """
    nr, nc = inputarray.shape
    nmax = np.max((nr, nc)).astype(np.int16)
    x = np.arange(-np.fix(nc / 2.0), np.ceil(nc / 2.0)) * np.floor(nmax / 2.0) / np.floor(nc / 2.0)
    y = np.arange(-np.fix(nr / 2.0), np.ceil(nr / 2.0)) * np.floor(nmax / 2.0) / np.floor(nr / 2.0)
    # bring the central pixel to the corners (important for odd array dimensions)
    x = ifftshift(x)
    y = ifftshift(y)
    # meshgriding
    X = np.meshgrid(x, y)
    # sum of the squares
    sumsquares = X[0] ** 2 + X[1] ** 2
    index = np.round(np.sqrt(sumsquares)).astype(np.int16)
    return index


def apodization(inputarray, apod_width=1):
    """
    Compute a tapered Hanning-like window of the size of the data
    for the apodization

    Parameters
    ----------
    inputarray :  array-like
        input array, must be two-dimensional

    apod_width : array-like
        width of the apodization margin

    Returns
    -------
    out : array-like
        2D array containing the apodization mask

    """
    # print("Calculating the transverse apodization")
    nr, nc = inputarray.shape
    Nr = fftshift(np.arange(nr))
    Nc = fftshift(np.arange(nc))
    window1D1 = (
        1.0
        + np.cos(2 * np.pi * (Nr - np.floor((nr - 2 * apod_width - 1) / 2)) / (1 + 2 * apod_width))
    ) / 2.0
    window1D2 = (
        1.0
        + np.cos(2 * np.pi * (Nc - np.floor((nc - 2 * apod_width - 1) / 2)) / (1 + 2 * apod_width))
    ) / 2.0
    window1D1[apod_width:-apod_width] = 1
    window1D2[apod_width:-apod_width] = 1

    return np.outer(window1D1, window1D2)


def fourierringcorrelation(
    input1,
    input2,
    apod_width=0,
    ringthick=1,
):
    """
    Routine to compute the FRC

    Parameters
    ----------
    input1 :  array-like
        array containing the first image, must be two-dimensional

    input2 : array-like
        array containing the second image, must be two-dimensional

    apod_width : array-like
        width of the apodization margin

    ringthick : int
        thickness of the ring for averaging the correlation

    Returns
    -------
    FRC : array-like
        1D array containing the FRC values

    T : array-like
        1D array containing the 1-bit threshold

    fn : array-like
        1D array containing the normalized frequencies

    """
    # Check if the arrays have 2 dimensions
    if input1.ndim == 2 and input2.ndim == 2:
        nr, nc = input1.shape
    else:
        raise ValueError("The arrays must have 2 dimensions")
    # Check if the arrays have the same size
    if input1.shape != input2.shape:
        raise ValueError("The arrays must have the same size")

    # Forcing to using 1 bit threshold because it is ring correlation
    # 1/2 bit threshold must only be used for tomography
    snrt = 1.0

    # Apodization of the borders
    window = apodization(input1, apod_width)
    img1_apod = input1 * window
    img2_apod = input2 * window

    # Computation of the FFTs
    F1 = np.fft.fft2(np.fft.ifftshift(img1_apod))  # FFT of input1
    F2 = np.fft.fft2(np.fft.ifftshift(img2_apod))  # FFT of input2

    # normalized frequencies
    f, fnyquist = nyquist((nr, nc))  # Frequencies and Nyquist frequency
    fn = f / fnyquist

    # initializing variables
    C = np.empty_like(f)
    C1 = np.empty_like(f)
    C2 = np.empty_like(f)
    npts = np.zeros_like(f)

    # print("Calculating the correlation...")
    index = ringthickness(F1)  # indexes for the ring thickness
    for ii in range(len(f)):
        # for ii in (range(len(f))):
        if ringthick == 0 or ringthick == 1:
            auxF1 = F1[np.where(index == ii)]
            auxF2 = F2[np.where(index == ii)]
        else:
            auxF1 = F1[
                (np.where((index >= (ii - ringthick / 2)) & (index <= (ii + ringthick / 2))))
            ]
            auxF2 = F2[
                (np.where((index >= (ii - ringthick / 2)) & (index <= (ii + ringthick / 2))))
            ]
        C[ii] = np.abs((auxF1 * np.conj(auxF2)).sum())  # Cross-correlation
        C1[ii] = np.abs((auxF1 * np.conj(auxF1)).sum())  # auto-correlation
        C2[ii] = np.abs((auxF2 * np.conj(auxF2)).sum())  # auto-correlation
        npts[ii] = auxF1.shape[0]

    # The correlation
    FRC = C / (np.sqrt(C1 * C2))

    # The computation of the threshold
    Tnum = snrt + (2 * np.sqrt(snrt) / np.sqrt(npts)) + 1 / np.sqrt(npts)
    Tden = snrt + (2 * np.sqrt(snrt) / np.sqrt(npts)) + 1
    # The threshold
    T = Tnum / Tden

    return FRC, T, fn


def frc_plot(FRC, T, fn, ax=None):
    """
    Plot FRC and thresholds onto the provided axis (or a new one if None).
    """
    FRC = np.array(FRC)
    T = np.array(T)
    fn = np.array(fn)

    # Interp both curves to find intersection with T
    f1 = interp1d(fn, FRC, kind="linear")
    x1_interp = np.linspace(min(fn), max(fn), len(fn) * 100)
    y1_interp = f1(x1_interp)

    f2 = interp1d(fn, T, kind="linear")
    x2_interp = np.linspace(min(fn), max(fn), len(fn) * 100)
    y2_interp = f2(x2_interp)

    # Use provided axis or create one
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.plot(x1_interp, y1_interp.real, "-b")
    ax.plot(x2_interp, y2_interp, "--r")
    ax.axhline(0.143, color="g", linestyle=":")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Spatial frequency/Nyquist [normalized units]")
    ax.set_ylabel("Magnitude [normalized units]")
    # If we created the axis here, show; otherwise caller manages drawing
    if hasattr(ax.figure.canvas, "draw_idle"):
        ax.figure.canvas.draw_idle()


"""
Non Pty-py
"""


def two_image_frc(
    image1, image2, margin=0, plot=False, apod_width=0, plot_images=False, ax_frc=None
):
    """
    Calculate the Fourier Ring Correlation (FRC) between two images.

    Parameters
    ----------
    image1 : np.ndarray
        First 2D image array.
    image2 : np.ndarray
        Second 2D image array.
    margin : int, optional
        Number of pixels to remove from each edge.
    apod_with : int, optional
        Number of pixels to apodize from each edge after margin removed.

    Returns
    -------
    auc : float (negative)
        The area under the FRC curves.
    """
    # Remove margins from the images
    image1 = remove_margins(image1, margin=margin)
    image2 = remove_margins(image2, margin=margin)

    FRC_curve, T, fn = fourierringcorrelation(
        image1, image2, apod_width=apod_width, plot_images=plot_images
    )

    # Calculate the error metric
    auc = -np.trapz(FRC_curve, fn)

    if auc is None or np.isnan(auc):
        auc = 0.0

    if plot:
        frc_plot(FRC_curve, T, fn, ax=ax_frc)

    return auc


def remove_margins(img, margin):
    """
    Remove margins from all sides of the image.
    Parameters
    ----------
    img : np.ndarray
        Input 2D image array.
    margin : int
        Number of pixels to remove from each edge.
    Returns
    -------
    cropped_img : np.ndarray
        Cropped image with margins removed.
    """
    if margin == 0:
        return img

    if img.ndim == 2:
        return img[margin:-margin, margin:-margin]
    elif img.ndim == 3:
        return img[:, margin:-margin, margin:-margin]
    else:
        raise ValueError("Input image must be 2D or 3D.")
