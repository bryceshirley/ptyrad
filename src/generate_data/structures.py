"""ASE structure builders for the simulated 4D-STEM samples."""

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.build import bulk

A_SI = 5.43  # Si conventional lattice constant (A)
A_GE = 5.658  # Ge conventional lattice constant (A)


@dataclass
class StructureInfo:
    atoms: Atoms
    lateral_extent_A: float
    thickness_A: float
    # z ranges (start, stop) of the marker/slab regions, beam enters at z=0
    slab_z_ranges: dict = field(default_factory=dict)
    # slice thicknesses for the ground-truth potential (sums to thickness_A)
    gt_slice_thicknesses: tuple = ()
    # slice thickness for the PtyRAD reconstruction (scalar; transmit planes
    # sit at multiples of this, entrance convention)
    recon_dz_A: float = 0.0


def build_smoke_sample() -> StructureInfo:
    """Milestone-0 sample: two thin slabs separated by vacuum.

    Slab A: 2 unit cells of Si(001) at the top (beam entrance, z=0).
    Slab B: 2 unit cells of laterally strained Ge at the bottom.  The Ge is
    strained to 10 cells per 11 Si cells (a=5.973 A) so the two slabs tile
    the same lateral cell but have DIFFERENT lateral periodicities - this
    makes the depth-separation correlation matrix discriminating (identical
    lattices would correlate equally with both recon slices).

    PtyRAD transmits at slice entrances (transmit -> propagate dz), so the
    reconstruction slice thickness is chosen to put transmit plane 2 at the
    Ge slab entrance, not at thickness/2.
    """
    n_si = 11
    lateral = n_si * A_SI  # 59.73 A
    thickness = 230.0

    si = bulk("Si", "diamond", a=A_SI, cubic=True) * (n_si, n_si, 2)

    a_ge = lateral / 10  # 5.973 A, ~5.6% tensile strain
    ge = bulk("Ge", "diamond", a=a_ge, cubic=True) * (10, 10, 2)
    ge_thickness = 2 * a_ge  # 11.946 A
    ge_z0 = thickness - ge_thickness  # 218.054 A
    ge.positions[:, 2] += ge_z0

    atoms = si + ge
    atoms.set_cell([lateral, lateral, thickness])
    atoms.set_pbc(True)

    return StructureInfo(
        atoms=atoms,
        lateral_extent_A=lateral,
        thickness_A=thickness,
        slab_z_ranges={"Si": (0.0, 2 * A_SI), "Ge": (ge_z0, thickness)},
        gt_slice_thicknesses=(ge_z0, ge_thickness),
        recon_dz_A=ge_z0,
    )


def build_sample_a_diluted(
    marker_depths_A=(60.0, 170.0, 290.0),
    si_slab_depths_A=(15.0, 115.0, 225.0, 330.0),
    slab_cells: int = 2,
    marker_cells: int | None = 3,
) -> StructureInfo:
    """Depth-diluted Sample A: the Step-3 truncation gate measured the SOLID
    35 nm Si matrix at M*(1e-2) = 11 (too strong for the n=4 story), so the
    matrix is thinned to sparse Si-lattice slabs in vacuum while keeping the
    stack depth (347.5 A, 100 recon slices) and the three full-lateral Ge
    marker slabs at ~60/170/290 A (separations >> the 5.6 nm sectioning
    limit).  All slabs are `slab_cells` unit cells thick, cut cell-aligned
    from the same 64-cell Si(001) crystal, so lattice registration across
    depth is exact; markers are Si->Ge substitution as in the prompt.
    """
    n_lat, n_z = 17, 64
    atoms = bulk("Si", "diamond", a=A_SI, cubic=True) * (n_lat, n_lat, n_z)
    lateral = n_lat * A_SI
    thickness = n_z * A_SI  # 347.52 A

    marker_cells = marker_cells or slab_cells

    def cell_range(center_A: float, cells: int) -> tuple[float, float]:
        c0 = round(center_A / A_SI - cells / 2)
        c0 = max(0, min(n_z - cells, c0))
        return c0 * A_SI, (c0 + cells) * A_SI

    z = atoms.positions[:, 2]
    keep = np.zeros(len(atoms), dtype=bool)
    slab_ranges = {}
    symbols = np.array(atoms.get_chemical_symbols())
    for i, d in enumerate(marker_depths_A):
        lo, hi = cell_range(d, marker_cells)
        mask = (z >= lo - 1e-6) & (z < hi - 1e-6)
        keep |= mask
        symbols[mask] = "Ge"
        slab_ranges[f"Ge{i}"] = (lo, hi)
    for i, d in enumerate(si_slab_depths_A):
        lo, hi = cell_range(d, slab_cells)
        keep |= (z >= lo - 1e-6) & (z < hi - 1e-6)
        slab_ranges[f"Si{i}"] = (lo, hi)
    atoms.set_chemical_symbols(symbols.tolist())
    atoms = atoms[keep]

    n_slices = 100
    dz = thickness / n_slices
    return StructureInfo(
        atoms=atoms,
        lateral_extent_A=lateral,
        thickness_A=thickness,
        slab_z_ranges=slab_ranges,
        gt_slice_thicknesses=(dz,) * n_slices,
        recon_dz_A=dz,
    )


def build_sample_a(
    marker_depths_A=(60.0, 170.0, 290.0), marker_cells: int = 2
) -> StructureInfo:
    """Sample A: Si(001) 17x17x64 cells (~92.3 x 92.3 x 347.5 A) with three
    buried Ge delta-layers (full lateral extent, `marker_cells` unit cells of
    depth, centered at `marker_depths_A`)."""
    n_lat, n_z = 17, 64
    atoms = bulk("Si", "diamond", a=A_SI, cubic=True) * (n_lat, n_lat, n_z)
    lateral = n_lat * A_SI  # 92.31 A
    thickness = n_z * A_SI  # 347.52 A

    z = atoms.positions[:, 2]
    half = marker_cells * A_SI / 2
    slab_ranges = {}
    for i, depth in enumerate(marker_depths_A):
        lo, hi = depth - half, depth + half
        mask = (z >= lo) & (z < hi)
        symbols = np.array(atoms.get_chemical_symbols())
        symbols[mask] = "Ge"
        atoms.set_chemical_symbols(symbols.tolist())
        slab_ranges[f"Ge{i}"] = (lo, hi)

    n_slices = 100
    dz = thickness / n_slices  # 3.4752 A
    return StructureInfo(
        atoms=atoms,
        lateral_extent_A=lateral,
        thickness_A=thickness,
        slab_z_ranges=slab_ranges,
        gt_slice_thicknesses=(dz,) * n_slices,
        recon_dz_A=dz,
    )


def build_multi_element_stack() -> StructureInfo:
    """Five chemically and crystallographically DISTINCT slabs at five depths
    (user request: complicated multi-element structure; also makes every
    depth laterally unique, so depth assignment cannot be confused between
    slices): diamond-C, Si, SrTiO3, Ge, Au in vacuum, each commensurately
    strained (<=2%) onto the 92.31 A cell.  Same stack depth / slicing as
    Sample A (347.52 A, 100 x 3.4752 A slices)."""
    lateral = 17 * A_SI  # 92.31 A
    thickness = 347.52
    n_slices = 100

    def diamond(sym, n_cells, uc):
        a = lateral / n_cells
        return bulk(sym, "diamond", a=a, cubic=True) * (n_cells, n_cells, uc), a

    def fcc(sym, n_cells, uc):
        a = lateral / n_cells
        return bulk(sym, "fcc", a=a, cubic=True) * (n_cells, n_cells, uc), a

    def perovskite(n_cells, uc):
        a = lateral / n_cells
        cell = Atoms(
            symbols="SrTiO3",
            scaled_positions=[
                (0, 0, 0),
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0),
                (0.5, 0, 0.5),
                (0, 0.5, 0.5),
            ],
            cell=(a, a, a),
            pbc=True,
        )
        return cell * (n_cells, n_cells, uc), a

    slabs = [
        ("C", *diamond("C", 26, 2), 30.0),
        ("Si", *diamond("Si", 17, 2), 100.0),
        ("SrTiO3", *perovskite(24, 3), 170.0),
        ("Ge", *diamond("Ge", 16, 2), 240.0),
        ("Au", *fcc("Au", 23, 1), 310.0),
    ]
    stack = Atoms(cell=(lateral, lateral, thickness))
    ranges = {}
    for name, atoms, a, depth in slabs:
        t = atoms.cell[2, 2]
        z0 = depth - t / 2
        atoms = atoms.copy()
        atoms.positions[:, 2] += z0
        ranges[name] = (z0, z0 + t)
        stack = stack + atoms
    stack.set_cell([lateral, lateral, thickness])
    stack.set_pbc(True)

    dz = thickness / n_slices
    return StructureInfo(
        atoms=stack,
        lateral_extent_A=lateral,
        thickness_A=thickness,
        slab_z_ranges=ranges,
        gt_slice_thicknesses=(dz,) * n_slices,
        recon_dz_A=dz,
    )


def build_twisted_stack(
    dz_A: float = 3.0,
    n_slices: int = 128,
    plane_every: int = 4,
    first_plane_slice: int = 2,
    ge_plane_indices: tuple = (8, 16, 24),
    plane_layers: int = 4,
    twist_total_deg: float = 60.0,
    ramp_slabs: tuple = (10, 21),
) -> StructureInfo:
    """dense128t (user request): the dense128b skeleton — `plane_layers`
    consecutive Si(001) atomic planes per slab, one slab every `plane_every`
    slices — with a twist about the cell-center z-axis CONFINED TO THE
    CENTRAL SLABS: slabs before `ramp_slabs[0]` sit at 0 deg, slabs after
    `ramp_slabs[1]` at `twist_total_deg`, and the angle ramps linearly in
    between (defaults: 10 untwisted end slabs on each side, 60 deg across
    the 12 central slabs = 5 deg/slab). Rationale (user): the end blocks
    are internally UNIFORM in orientation, so a reconstruction that smears
    a terminal slice into its neighbors still shows the block's angle —
    mis-sectioning is detected by orientation, not hidden by it — while
    the two ends differ strongly from each other (0 vs 60 deg) and the
    central ramp gives every mid-stack slab a unique orientation
    (adjacent-slab moire ~31 A at 5 deg, inside the 54 A scan window).
    Ge markers land at 0 deg (slab 8), mid-ramp (16), and 60 deg (24).

    Each slab is cut from an OVERSIZED parent plane (covering the cell
    diagonal), rotated with ASE, then cropped to [0, L)^2 — the cell stays
    fully covered at every angle (no corner voids). The twist necessarily
    breaks lateral periodicity at the cell boundary (only special
    coincidence angles are commensurate on a square cell); the rim
    mismatch sits >= 19 A outside the centered 54 A scan window, and the
    exported ground truth contains the same rim, so all comparisons remain
    self-consistent.
    """
    n_lat = 17
    lateral = n_lat * A_SI
    thickness = n_slices * dz_A

    # oversized parent: covers the rotated cell diagonal (L*sqrt(2) = 130.5 A)
    n_big = 25  # 135.75 A
    base = bulk("Si", "diamond", a=A_SI, cubic=True) * (n_big, n_big, 1)
    zmax = (plane_layers - 1) * A_SI / 4 + 1e-6
    plane = base[base.positions[:, 2] < zmax]
    plane.positions[:, 2] -= plane.positions[:, 2].mean()
    # center the parent plane on the cell center
    plane.positions[:, :2] += lateral / 2 - n_big * A_SI / 2

    stack = Atoms(cell=(lateral, lateral, thickness))
    ranges = {}
    twists = {}
    plane_slices = list(range(first_plane_slice, n_slices, plane_every))
    k0, k1 = ramp_slabs
    for k, j in enumerate(plane_slices):
        p = plane.copy()
        frac = min(max((k - k0) / (k1 - k0), 0.0), 1.0)
        angle = twist_total_deg * frac
        p.rotate(angle, "z", center=(lateral / 2, lateral / 2, 0.0))
        xy = p.positions[:, :2]
        p = p[
            (xy[:, 0] >= 0)
            & (xy[:, 0] < lateral - 1e-6)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < lateral - 1e-6)
        ]
        z = (j + 0.5) * dz_A
        p.positions[:, 2] += z
        name = f"{'Ge' if k in ge_plane_indices else 'Si'}_slab{k}"
        if k in ge_plane_indices:
            p.set_chemical_symbols(["Ge"] * len(p))
        ranges[name] = (z - 0.1, z + 0.1)
        twists[name] = angle
        stack = stack + p
    stack.set_cell([lateral, lateral, thickness])
    stack.set_pbc(True)

    info = StructureInfo(
        atoms=stack,
        lateral_extent_A=lateral,
        thickness_A=thickness,
        slab_z_ranges=ranges,
        gt_slice_thicknesses=(dz_A,) * n_slices,
        recon_dz_A=dz_A,
    )
    info.slab_twists_deg = twists  # depth -> twist angle ground truth
    return info


def build_plane_stack(
    dz_A: float = 3.0,
    n_slices: int = 128,
    plane_every: int = 8,
    first_plane_slice: int = 4,
    ge_plane_indices: tuple = (4, 8, 12),
    plane_layers: int = 1,
) -> StructureInfo:
    """128-slice plane stack (user request): single atomic (001) planes with
    large vacuum gaps, each plane at the CENTER of its own model slice so
    slice assignment is unambiguous (planes every `plane_every` slices ->
    24 A gaps at the defaults; no slice holds more than one plane).

    Planes are the z=0 sublattice of the Si diamond cell (square net,
    a/2 = 2.715 A pitch, 578 atoms/plane); three planes are Ge markers.
    """
    n_lat = 17
    lateral = n_lat * A_SI
    thickness = n_slices * dz_A  # 384 A

    base = bulk("Si", "diamond", a=A_SI, cubic=True) * (n_lat, n_lat, 1)
    # plane_layers=1: single atomic plane (z=0 sublattice); >1: that many
    # consecutive (004)-type atomic planes (spacing a/4), centered on z below
    zmax = (plane_layers - 1) * A_SI / 4 + 1e-6
    plane = base[base.positions[:, 2] < zmax]
    plane.positions[:, 2] -= plane.positions[:, 2].mean()

    stack = Atoms(cell=(lateral, lateral, thickness))
    ranges = {}
    plane_slices = list(range(first_plane_slice, n_slices, plane_every))
    for k, j in enumerate(plane_slices):
        p = plane.copy()
        z = (j + 0.5) * dz_A
        p.positions[:, 2] += z
        if k in ge_plane_indices:
            p.set_chemical_symbols(["Ge"] * len(p))
            ranges[f"Ge_plane{k}"] = (z - 0.1, z + 0.1)
        else:
            ranges[f"Si_plane{k}"] = (z - 0.1, z + 0.1)
        stack = stack + p
    stack.set_cell([lateral, lateral, thickness])
    stack.set_pbc(True)

    return StructureInfo(
        atoms=stack,
        lateral_extent_A=lateral,
        thickness_A=thickness,
        slab_z_ranges=ranges,
        gt_slice_thicknesses=(dz_A,) * n_slices,
        recon_dz_A=dz_A,
    )
