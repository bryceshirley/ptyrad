"""
Plotting functions for basic outputs including probe modes, loss curves, positions, etc.
"""
# This module only plots numpy arrays

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from numpy.fft import fft2, fftshift, ifftshift
from itertools import cycle


def plot_sigmoid_mask(Npix, relative_radius, relative_width, img=None, show_circles=False):
    """ Plot a sigmoid mask overlay on img with a line profile """
    # Note that relative_radius ranges from 0 - 1 for center -> edge. radius = 1 corresponds to a inscribed circle
    # While relative_width also ranges from 0 - 1 for Npix * relative_width. width = 0.05 corresponds to a width of 5% of the image width would have sigmoid value change from 0 - 1
    from ptyrad.core.functional import make_sigmoid_mask # This is used in constraints so will pull torch during execution
    
    mask = make_sigmoid_mask(Npix, relative_radius, relative_width).detach().cpu().numpy()
    img = np.ones((Npix,Npix)) if img is None else img/img.max()
    masked_img = mask * img
    fig, axs = plt.subplots(1,2, figsize=(13,6))
    fig.suptitle(f"Sigmoid mask with radius = {relative_radius}, width = {relative_width}", fontsize=18)
    im = axs[0].imshow(masked_img)
    axs[0].axhline(y=Npix//2, xmin=0.5, c='r', linestyle='--')
    axs[1].plot(mask[Npix//2, Npix//2:], c='r', label='mask')
    if img is not None:
        axs[1].plot(img[Npix//2, Npix//2:], label='image')
        axs[1].plot(masked_img[Npix//2, Npix//2:], label='masked_img')
    
    # Draw circles on the imshow
    if show_circles:
        circle1 = plt.Circle((Npix // 2, Npix // 2), (relative_radius-relative_width) * Npix/2, color='k', fill=False, linestyle='--')
        circle2 = plt.Circle((Npix // 2, Npix // 2), (relative_radius+relative_width) * Npix/2, color='k', fill=False, linestyle='--')
        axs[0].add_artist(circle1)
        axs[0].add_artist(circle2)
        axs[1].axvline(x=(relative_radius-relative_width) * Npix/2, color='k', linestyle='--')
        axs[1].axvline(x=(relative_radius+relative_width) * Npix/2, color='k', linestyle='--')
    
    fig.colorbar(im, shrink=0.7)
    plt.legend()
    plt.show()

def plot_obj_tilts_avg(avg_tilt_iters, last_n_iters=2, show_fig=True, pass_fig=False):
    last_n_iters = int(last_n_iters)

    iters = np.array(avg_tilt_iters['niter'])
    tilts = np.column_stack([avg_tilt_iters['tilt_y'], avg_tilt_iters['tilt_x']])

    plt.ioff()  # Temporarily disable interactive mode
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), sharex=True)

    # Plot first component (tilt_y)
    axes[0].plot(iters, tilts[:, 0], marker='o', color='C0')
    axes[0].set_ylabel('Avg Obj tilt_y (mrad)', fontsize=16)
    axes[0].set_title(f'Avg Obj tilt_y (mrad): {tilts[-1,0]:.3f} at iter {iters[-1]}', fontsize=16)
    axes[0].grid(True)

    # Plot second component (tilt_x)
    axes[1].plot(iters, tilts[:, 1], marker='o', color='C1')
    axes[1].set_xlabel('Iterations', fontsize=16)
    axes[1].set_ylabel('Avg Obj tilt_x (mrad)', fontsize=16)
    axes[1].set_title(f'Avg Obj tilt_x (mrad): {tilts[-1,1]:.3f} at iter {iters[-1]}', fontsize=16)
    axes[1].grid(True)

    for i, ax in enumerate(axes):
        # Plot the last n iters as an inset
        if len(iters) > 20 and last_n_iters is not None:
            axins = ax.inset_axes([0.45, 0.3, 0.4, 0.5])

            # Correctly match inset plots to main plots
            axins.plot(iters[-last_n_iters:], tilts[-last_n_iters:, i], marker='o', color = f'{"C0" if i == 0 else "C1"}')

            axins.set_xlabel('Iterations', fontsize=12)
            axins.set_ylabel(f'Avg Obj tilt_{"y" if i == 0 else "x"} (mrad)', fontsize=12)
            axins.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))
            ax.indicate_inset_zoom(axins, edgecolor="gray")
            axins.set_title(f'Last {last_n_iters} iterations', fontsize=12, pad=10)
            axins.grid(True)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_obj_tilts(pos, tilts, figsize=(16,16), show_fig=True, pass_fig=False):
    """ Plot the obj tilts given the probe position and pos-dependent tilts """
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig = plt.figure(figsize = figsize)
    ax = plt.gca() # There's only 1 ax for plt.figure(), and plt.title is an Axes-level attribute so I need to pass the Axes out because I like plt.title layout better
    plt.title("Object tilts", fontsize=16)
    
    tilts = np.broadcast_to(tilts, shape=(len(pos),2))
    if np.allclose(tilts[:,0], 0, atol=1e-3):
        # All tilts are effectively zero; skip quiver plot and annotate
        ax.text(
            0.5, 0.5, "All tilts are effectively zero (<1e-3) mrad, no quiver plot",
            ha="center", va="center", fontsize=18, color="gray", transform=ax.transAxes
        )
    else:
        M = np.hypot(tilts[:,0], tilts[:,1])
        q = ax.quiver(pos[:,1], pos[:,0], tilts[:,1], tilts[:,0], M, pivot='mid', angles='xy', scale_units='xy', label='Obj tilts')
        cbar = fig.colorbar(q, shrink=0.75)
        cbar.ax.set_ylabel('mrad')
        cbar.ax.get_yaxis().labelpad = 15
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
    plt.xlabel('X (obj coord, px)')
    plt.ylabel('Y (obj coord, px)')
    
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig, ax

def plot_scan_positions(pos, init_pos=None, img=None, offset=None, figsize=(16,16), dot_scale=0.001, show_arrow=True, show_fig=True, pass_fig=False):
    """ Plot the scan positions given an array of (N,2) """
    # The array is expected to have shape (N,2)
    # Each row is rendered as (y, x), or equivalently (height, width)
    # The dots are plotted with asending size and color changes to represent the relative order
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig = plt.figure(figsize = figsize)
    ax = plt.gca() # There's only 1 ax for plt.figure(), and plt.title is an Axes-level attribute so I need to pass the Axes out because I like plt.title layout better
    plt.title("Scan positions", fontsize=16)
    
    if img is not None:
        plt.imshow(img)
        pos = np.array(pos) + np.array(offset)
        plt.gca().invert_yaxis()  # Pre-flip y-axis so the y-axis is image-like no matter what
    
    if init_pos is None:
        plt.scatter(x=pos[:,1], y=pos[:,0], c=np.arange(len(pos)), s=dot_scale*np.arange(len(pos)), label='Scan positions')
    else:
        plt.scatter(x=init_pos[:,1], y=init_pos[:,0], c='C0', s=dot_scale, label='Init scan positions')
        plt.scatter(x=pos[:,1],      y=pos[:,0],      c='C1', s=dot_scale, label='Opt scan positions')
        plt.ylim(init_pos[:,0].min()-10, init_pos[:,0].max()+10)
        plt.xlim(init_pos[:,1].min()-10, init_pos[:,1].max()+10)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
    plt.xlabel('X (obj coord, px)')
    plt.ylabel('Y (obj coord, px)')
    
    # Draw arrow from 1st position to 10th position
    if show_arrow:
        plt.arrow(pos[0, 1], pos[0, 0], pos[9, 1] - pos[0, 1], pos[9, 0] - pos[0, 0],
                color='red', head_width=2.5, head_length=5)
    plt.legend()
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig, ax
    
def plot_affine_transformation(scale, asymmetry, rotation, shear):
    from ptyrad.utils.affine import compose_affine_matrix
    # Example
    # plot_affine_transformation(2,0,45,0)
    A = np.eye(2)
    Af = compose_affine_matrix(scale, asymmetry, rotation, shear)
    
    plt.figure()
    plt.title(f"Visualize affine transformation \n (scale, asym, rot, shear) = {scale, asymmetry, rotation, shear}", fontsize=14)

    # Add origin and scatter points
    plt.scatter(0, 0, color='gray', marker='o', s=3)
    plt.scatter(A[:,1], A[:,0], label='Original')
    plt.scatter(Af[:,1], Af[:,0], label='Transformed')

    # Adding arrows
    plt.quiver(A[0,1], A[0,0], angles='xy', scale_units='xy', scale=1, color='C0', alpha=0.5)
    plt.quiver(A[1,1], A[1,0], angles='xy', scale_units='xy', scale=1, color='C0', alpha=0.5)
    plt.quiver(Af[0,1], Af[0,0], angles='xy', scale_units='xy', scale=1, color='C1', alpha=0.5)
    plt.quiver(Af[1,1], Af[1,0], angles='xy', scale_units='xy', scale=1, color='C1', alpha=0.5)

    # Adding grid lines
    plt.grid(True, linestyle='--', color='gray', linewidth=0.5)

    plt.ylim(-2,2)
    plt.xlim(-2,2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()  # Flipped y-axis if there's only scatter plot
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.legend()
    plt.show()

def plot_pos_grouping(pos, batches, circle_diameter=False, diameter_type='90%', figsize=(16,8), dot_scale=1, show_fig=True, pass_fig=False):
    
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(1,2, figsize = figsize)
    
    for i, ax in enumerate(axs):
        if i == 0:
            axs[0].set_title(f"Scan positions for all {len(batches)} groups", fontsize=18)
            for batch in batches:
                ax.scatter(x=pos[batch, 1], y=pos[batch, 0], s=dot_scale)
        else:
            axs[1].set_title("Scan positions from group 0", fontsize=18)
            ax.scatter(x=pos[batches[0], 1], y=pos[batches[0], 0], s=dot_scale)
            
            # Draw a circle at the first point with the given diameter
            if circle_diameter:
                first_point = pos[batches[0][0]]
                circle = plt.Circle((first_point[1], first_point[0]), circle_diameter / 2, fill=False, color='r', linestyle='--')
                ax.scatter(x=first_point[1], y=first_point[0], s=dot_scale, color='r')
                ax.add_artist(circle)
                
                # Add annotation for "90% probe intensity"
                annotation_text = f"{diameter_type} probe intensity"
                annotation_x = first_point[1]
                annotation_y = first_point[0] #+ circle_diameter / 2 + 10  # Adjust the vertical offset as needed
                ax.annotate(annotation_text, xy=(annotation_x-circle_diameter/2, annotation_y-circle_diameter/2-3))
            
        ax.set_xlabel('X (obj coord, px)')
        ax.set_ylabel('Y (obj coord, px)')
        ax.set_xlim(pos[:,1].min()-10, pos[:,1].max()+10) # Show the full range to better visualize if a sub-group (like 'center') is selected
        ax.set_ylim(pos[:,0].min()-10, pos[:,0].max()+10)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if show_fig:
        plt.show(block=False)
    if pass_fig:
        return fig
    
def plot_loss_curves(loss_iters, last_n_iters=10, show_fig=True, pass_fig=False):
    last_n_iters = int(last_n_iters)
    data = np.array(loss_iters)

    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Plot all loss values
    axs.plot(data[:,0], data[:,1], marker='o')

    # Plot the last n iters as an inset
    if len(data) > 20 and last_n_iters is not None:
        # Create inset subplot for zoomed-in plot
        axins = axs.inset_axes([0.45, 0.3, 0.4, 0.5])
        axins.plot(data[-last_n_iters:,0], data[-last_n_iters:,1], marker='o')
        axins.set_xlabel('Iterations', fontsize=12)
        axins.set_ylabel('Loss value', fontsize=12)
        axins.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.5f}'))
        axs.indicate_inset_zoom(axins, edgecolor="gray")
        axins.set_title(f'Last {last_n_iters} iterations', fontsize=12, pad=10)

    # Set labels and title for the main plot
    axs.set_xlabel('Iterations', fontsize=16)
    axs.set_ylabel('Loss value', fontsize=16)
    axs.set_title(f'Loss value: {data[-1,1]:.5f} at iter {int(data[-1,0])}', fontsize=16)
    axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_learning_rates_schedule(lr_iters, log=True, show_fig=True, pass_fig=False):
    """Plots the learning rate schedule for each optimizable parameter group over iterations."""

    styles = cycle(['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10))])
    colors = cycle(plt.cm.tab10.colors) # High-contrast palette

    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    for name in [k for k in lr_iters.keys() if k != 'niter']:
        axs.plot(
            lr_iters['niter'],
            lr_iters[name],
            label=name,
            linestyle=next(styles),
            color=next(colors),
            linewidth=2,
            alpha=0.8,
        )

    # Set labels and title for the main plot
    axs.set_xlabel('Iterations', fontsize=16)
    axs.set_ylabel('Learning rates', fontsize=16)
    axs.set_title(f'Learning rates schedule up to iter {lr_iters['niter'][-1]}', fontsize=16)
    if log:
        axs.set_yscale('log')
    axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axs.tick_params(axis='both', which='major', labelsize=14)
    axs.grid(True, linestyle=':', alpha=0.6)
    axs.legend(frameon=True, loc='best')

    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig
    else:
        plt.close(fig)

def plot_slice_thickness(dz_iters, last_n_iters=10, show_fig=True, pass_fig=False):
    last_n_iters = int(last_n_iters)
    data = np.array(dz_iters)

    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    # Plot all loss values
    axs.plot(data[:,0], data[:,1], marker='o')
    axs.grid(True)

    # Plot the last n iters as an inset
    if len(data) > 20 and last_n_iters is not None:
        # Create inset subplot for zoomed-in plot
        axins = axs.inset_axes([0.45, 0.3, 0.4, 0.5])
        axins.plot(data[-last_n_iters:,0], data[-last_n_iters:,1], marker='o')
        axins.set_xlabel('Iterations', fontsize=12)
        axins.set_ylabel('Slice thickness (Ang)', fontsize=12)
        axins.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.5f}'))
        axs.indicate_inset_zoom(axins, edgecolor="gray")
        axins.set_title(f'Last {last_n_iters} iterations', fontsize=12, pad=10)

    # Set labels and title for the main plot
    axs.set_xlabel('Iterations', fontsize=16)
    axs.set_ylabel('Slice thickness (Ang)', fontsize=16)
    axs.set_title(f'Slice thickness (Ang): {data[-1,1]:.5f} at iter {int(data[-1,0])}', fontsize=16)
    axs.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig

def plot_probe_modes(init_probe=None, opt_probe=None, amp_or_phase='amplitude', real_or_fourier='real', phase_cmap=None, amplitude_cmap=None, dpi=200, show_fig=True, pass_fig=False):
    # The input probes are expected to be numpy array
    # This is for visualization so each mode has its own colorbar.
    # See the actual probe amplitude output for absolute scale visualizaiton
    
    # Initial checks
    if init_probe is None and opt_probe is None:
        raise ValueError("At least one of init_probe or opt_probe must be provided.")
    if all(p is not None for p in (init_probe, opt_probe)) and init_probe.shape[0] != opt_probe.shape[0]:
        raise ValueError(f"All provided probes must have the same number of probe modes (axis 0), got {init_probe.shape} and {opt_probe.shape}.")
    
    # Initialize
    probes = [init_probe, opt_probe]
    labels = ["Init pmode", "Opt pmode"]   # row titles
    processed_probes = []
    probes_pow = []
    
    # Loop through possible input probes
    for probe in probes:
        if probe is None:
            processed_probes.append(None)
            probes_pow.append(None)
            continue
        
        # Power distribution
        probe_int = np.abs(probe)**2 
        probe_pow = np.sum(probe_int, axis=(-2,-1))/np.sum(probe_int)
        probes_pow.append(probe_pow)
    
        # Fourier or real
        # While it might seem redundant, the sandwitch fftshift(fft(ifftshift(probe)))) is needed for the following reason:
        # Although probe_fourier = fft2(ifftshift(probe)) and probe_fourier = fft2(probe) gives the same abs(probe_fourier),
        # pre-fftshifting the probe back to corner gives more accurate phase angle while plotting the angle(probe_fourier)
        # On the other hand, fft2(probe) would generate additional phase shifts that looks like checkerboard artifact in angle(probe_fourier)
        if real_or_fourier == 'fourier':
            probe  = fftshift(fft2(ifftshift(probe,  axes=(-2,-1)), norm='ortho'), axes=(-2,-1))
        elif real_or_fourier =='real':
            pass
        else:
            raise ValueError("Please use 'real' or 'fourier' for probe mode visualization!")
        
        # Amplitude or phase
        # Negative sign for consistency with chi(k), because psi = exp(-i*chi(k)). 
        # Overfocus (negative df = positive C1) should give positive phase shift near the edge of aperture
        # Scale the plotted phase by the amplitude so we can focus more on the relevant phases
        # Although note that noisy amplitude will also make the phase appears noisy
        if amp_or_phase == 'phase':
            probe = -np.angle(probe)*np.abs(probe)
            cmap = phase_cmap if phase_cmap else 'twilight'
        elif amp_or_phase in ('amplitude', 'amp'):
            probe = np.abs(probe)
            cmap = amplitude_cmap if amplitude_cmap else 'viridis'
        else:
            raise ValueError("Please use 'amplitude' or 'phase' for probe mode visualization!")

        processed_probes.append(probe)

    # Parse variables
    non_none = [(label, probe, probe_pow) for label, probe, probe_pow in zip(labels, processed_probes, probes_pow) if probe is not None]
    n_modes = non_none[0][1].shape[0] # non_none[0][1] would be probe, probe = (pmode, Ny, Nx)
    rows = len(non_none)

    # Actual plotting
    plt.ioff() # Temporaily disable the interactive plotting mode
    fig, axs = plt.subplots(rows, n_modes, figsize=(n_modes*2.5, rows*3), dpi=dpi)

    # Normalize axs shapes
    axs = np.asarray(axs)
    if axs.ndim == 0:
        axs = axs.reshape(1, 1)
    elif axs.ndim == 1:
        if rows == 1:
            axs = axs.reshape(1, n_modes)
        else:
            axs = axs.reshape(rows, 1)

    for row_idx, (label, probe, probe_pow) in enumerate(non_none):
        for i in range(n_modes):
            ax = axs[row_idx, i]
            ax.set_title(f"{label} {i}: {probe_pow[i]:.1%}")
            im = ax.imshow(probe[i], cmap=cmap)
            ax.axis('off')
            fig.colorbar(im, ax=ax, shrink=0.6)

    plt.suptitle(f"Probe modes {amp_or_phase} in {real_or_fourier} space", fontsize=18)
    plt.tight_layout()
    if show_fig:
        plt.show()
    if pass_fig:
        return fig


# ---------------------------------------------------------------------------
# Dashboard helpers — one per panel type, each draws into a provided Axes
# ---------------------------------------------------------------------------

# Human-readable title for each convergence tensor key
_CONVERGENCE_PANEL_TITLES = {
    "obja":             "Object amplitude",
    "objp":             "Object phase",
    "probe":            "Probe intensity",
    "probe_pos_shifts": "Probe position shifts",
}


def _is_empty(x):
    """Return True for None or zero-length lists/ndarrays (avoids ambiguous ndarray truth test)."""
    return x is None or len(x) == 0


def _panel_no_data(ax, label):
    ax.text(0.5, 0.5, f'{label}\n(no data)', transform=ax.transAxes,
            ha='center', va='center', fontsize=12, color='lightgray')
    ax.set_xticks([])
    ax.set_yticks([])


def _iter_to_idx(niters, iter_offset):
    """Return the array index of the first entry at or after ``iter_offset``.

    Uses binary search on the iteration column so panels with different logging
    strides (e.g. every iter vs every 50 iters) each resolve the correct index.
    """
    return int(np.searchsorted(niters, iter_offset, side='left'))


def _standard_kneedle(y, offset=0):
    """Core Kneedle: max perpendicular distance from the first-to-last chord.

    Returns the index into the *original* array (offset + local argmax).
    Assumes y is a flat float ndarray with at least 3 elements and non-zero range.
    """
    n = len(y)
    if n < 3:
        return offset
    y_range = y.max() - y.min()
    if y_range == 0:
        return offset
    x_norm = np.arange(n, dtype=float) / (n - 1)
    y_norm = (y - y.min()) / y_range
    chord = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    chord_len = np.linalg.norm(chord)
    if chord_len == 0:
        return offset
    chord_unit = chord / chord_len
    vecs = np.column_stack((x_norm - x_norm[0], y_norm - y_norm[0]))
    distances = np.abs(vecs[:, 0] * chord_unit[1] - vecs[:, 1] * chord_unit[0])
    return offset + int(np.argmax(distances))


def _double_kneedle(y):
    """Run Kneedle twice: find the first knee, then find the knee in its tail."""
    first_knee = _standard_kneedle(y)
    return _standard_kneedle(y[first_knee:], offset=first_knee)


def _kneedle_start_idx(values):
    """Smart Kneedle router — picks the best strategy based on dynamic range.

    * Small dynamic range (ratio < 2, e.g. loss on a large baseline): log-space
      compression is useless, so double Kneedle zooms into the converging tail.
    * Large dynamic range with all-positive values: log-space linearises the
      exponential decay; falls back to double Kneedle if the log knee is still
      stuck in the first 5% of the series.
    * Non-positive values: plain Kneedle in linear space.
    """
    y = np.asarray(values, dtype=float)
    n = len(y)
    if n < 3 or y.max() == y.min():
        return 0

    dynamic_ratio = y.max() / (y.min() + 1e-9)

    if dynamic_ratio < 2.0:
        return _double_kneedle(y)

    if np.all(y > 0):
        log_knee = _standard_kneedle(np.log(y))
        if log_knee < n * 0.05:
            return _double_kneedle(y)
        return log_knee

    return _standard_kneedle(y)


def _draw_loss_panel(ax, loss_iters, iter_offset=None):
    if _is_empty(loss_iters):
        _panel_no_data(ax, 'Loss')
        return
    data      = np.array(loss_iters)
    niters    = data[:, 0]
    vals      = data[:, 1]
    start     = _iter_to_idx(niters, iter_offset) if iter_offset is not None else _kneedle_start_idx(vals)
    ax.plot(niters[start:], vals[start:], marker='o', linewidth=1.5, markersize=3,
            label=f'Loss: {vals[-1]:.5f}')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Loss', fontsize=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.tick_params(labelsize=10)


def _draw_lr_panel(ax, lr_iters, iter_offset=None):
    if not lr_iters or _is_empty(lr_iters.get('niter')):
        _panel_no_data(ax, 'Learning rates')
        return
    niters = np.asarray(lr_iters['niter'])
    start  = _iter_to_idx(niters, iter_offset) if iter_offset is not None else 0
    styles = cycle(['-', '--', '-.', ':'])
    colors = cycle(plt.cm.tab10.colors)
    for name in [k for k in lr_iters if k != 'niter']:
        vals = np.asarray(lr_iters[name])
        ax.plot(niters[start:], vals[start:], label=f'{name}: {vals[-1]:.3e}',
                linestyle=next(styles), color=next(colors), linewidth=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('LR', fontsize=12)
    ax.set_title('Learning rates', fontsize=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.tick_params(labelsize=10)


def _draw_dz_panel(ax, dz_iters, iter_offset=None):
    if _is_empty(dz_iters):
        _panel_no_data(ax, 'Slice thickness')
        return
    data   = np.array(dz_iters)
    niters = data[:, 0]
    vals   = data[:, 1]
    start  = _iter_to_idx(niters, iter_offset) if iter_offset is not None else 0
    ax.plot(niters[start:], vals[start:], marker='o', linewidth=1.5, markersize=3,
            label=f'{vals[-1]:.4f} Å')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('dz (Å)', fontsize=12)
    ax.set_title('Slice thickness', fontsize=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.tick_params(labelsize=10)


def _draw_tilt_panel(ax, avg_tilt_iters, iter_offset=None):
    if not avg_tilt_iters or _is_empty(avg_tilt_iters.get('niter')):
        _panel_no_data(ax, 'Object tilts')
        return
    iters  = np.array(avg_tilt_iters['niter'])
    tilt_y = np.array(avg_tilt_iters['tilt_y'])
    tilt_x = np.array(avg_tilt_iters['tilt_x'])
    start  = _iter_to_idx(iters, iter_offset) if iter_offset is not None else 0
    ax.plot(iters[start:], tilt_y[start:], linewidth=1.5, marker='o', markersize=3,
            label=f'tilt_y: {tilt_y[-1]:.3f} mrad')
    ax.plot(iters[start:], tilt_x[start:], linewidth=1.5, marker='o', markersize=3,
            label=f'tilt_x: {tilt_x[-1]:.3f} mrad')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Object tilts (mrad)', fontsize=12)
    ax.set_title('Object tilts', fontsize=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.tick_params(labelsize=10)


_CONVERGENCE_YLABELS = {
    "probe":            "Fractional intensity change",
    "probe_pos_shifts": "RMS shift change (Å)",
}


def _draw_convergence_panel(ax, tensor_name, convergence_iters, iter_offset=None):
    """Draw a convergence panel for ``tensor_name``.

    For 'obja'/'objp', draws bg (C0) and fg (C1) lines from the ``_bg``/``_fg`` sub-keys.
    For other tensors, draws a single line from the matching key.
    If ``iter_offset`` is given, both lines are clipped to that iteration number;
    otherwise the smart Kneedle router determines the start index.
    ``convergence_iters`` is the full ci dict (not a single history list).
    """
    title = _CONVERGENCE_PANEL_TITLES.get(tensor_name, tensor_name)

    if tensor_name in ("obja", "objp"):
        bg_hist = convergence_iters.get(f"{tensor_name}_bg", [])
        fg_hist = convergence_iters.get(f"{tensor_name}_fg", [])
        if _is_empty(bg_hist) and _is_empty(fg_hist):
            _panel_no_data(ax, title)
            return
        ylabel  = "Mean |Δ| (abs change)"
        bg_data = np.array(bg_hist) if not _is_empty(bg_hist) else None
        fg_data = np.array(fg_hist) if not _is_empty(fg_hist) else None
        if iter_offset is not None:
            ref = next(d for d in [bg_data, fg_data] if d is not None)
            start = _iter_to_idx(ref[:, 0], iter_offset)
        else:
            start = max(_kneedle_start_idx(d[:, 1]) for d in [bg_data, fg_data] if d is not None)
        for data_arr, suffix, color in [(bg_data, "bg", "C0"), (fg_data, "fg", "C1")]:
            if data_arr is None:
                continue
            sl = data_arr[start:]
            ax.plot(sl[:, 0], sl[:, 1], marker='o', linewidth=1.5, markersize=3, color=color,
                    label=f'{suffix}: {sl[-1, 1]:.3e}')
    else:
        hist = convergence_iters.get(tensor_name, [])
        ylabel = _CONVERGENCE_YLABELS.get(tensor_name, "Change")
        if _is_empty(hist):
            _panel_no_data(ax, title)
            return
        data  = np.array(hist)
        start = _iter_to_idx(data[:, 0], iter_offset) if iter_offset is not None else _kneedle_start_idx(data[:, 1])
        ax.plot(data[start:, 0], data[start:, 1], marker='o', linewidth=1.5, markersize=3,
                label=f'{tensor_name}: {data[-1, 1]:.3e}')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.tick_params(labelsize=10)


def _get_last_iter(loss_iters, convergence_iters):
    if not _is_empty(loss_iters):
        return int(np.array(loss_iters)[-1, 0])
    for hist in (convergence_iters or {}).values():
        if not _is_empty(hist):
            return int(np.array(hist)[-1, 0])
    return None


def plot_convergence_dashboard(loss_iters, lr_iters, dz_iters, avg_tilt_iters,
                               convergence_iters,
                               iter_offset=None, show_fig=True, pass_fig=False):
    """Unified dashboard of all scalar time-series in a fixed 2x4 grid.

    Layout::

        Row 0: Loss | obja | objp | Object tilts
        Row 1: LR   | Probe amplitude | Probe position shifts | Slice thickness

    Panels with no data show a blank placeholder so the layout stays fixed across save cycles.
    ``iter_offset`` sets the starting iteration for all panels; if None, each panel
    determines its own start via the smart Kneedle router. Because panels have different
    logging strides (e.g. loss every iter, convergence metrics every 50 iters), each
    panel independently converts the iteration number to its own array index.
    """
    ci        = convergence_iters or {}
    last_iter = _get_last_iter(loss_iters, ci)

    plt.ioff()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), squeeze=False)

    _draw_loss_panel        (axes[0, 0], loss_iters,                    iter_offset)
    _draw_convergence_panel (axes[0, 1], 'obja',             ci,        iter_offset)
    _draw_convergence_panel (axes[0, 2], 'objp',             ci,        iter_offset)
    _draw_tilt_panel        (axes[0, 3], avg_tilt_iters,                iter_offset)
    _draw_lr_panel          (axes[1, 0], lr_iters,                      iter_offset)
    _draw_convergence_panel (axes[1, 1], 'probe',            ci,        iter_offset)
    _draw_convergence_panel (axes[1, 2], 'probe_pos_shifts', ci,        iter_offset)
    _draw_dz_panel          (axes[1, 3], dz_iters,                      iter_offset)

    if last_iter is not None:
        fig.suptitle(f"Convergence Dashboard at Iter {last_iter}", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show_fig:
        plt.show()
    if pass_fig:
        return fig
    else:
        plt.close(fig)