"""Tile a raster-scan 4D-STEM measurement stack to a larger scan grid.

Purpose: reproduce a paper-scale workload (e.g. the PrScO3 Fig-S5 cost
benchmark: 128 x 128 scan positions) when only a sub-scan was distributed
(the public PSO demo file carries the 64 x 64 scan01 subset). The frame at
tiled scan position (SY, SX) is the source frame at (SY % sy, SX % sx), so
the implied object is the source field-of-view repeated t_slow x t_fast
times -- physically consistent for a periodic crystal up to the FOV edges,
and workload-exact for timing (frame count, batch count, object extent all
match the full scan geometry).

The same (slow, fast) raster convention is assumed for source and output;
if the source ordering were transposed, the output is the tiling of the
transposed field -- still a self-consistent dataset for benchmarking.

Output is a single-dataset HDF5 file (key ``dp``, float32) that
``ptyrad.load.load_array_from_file`` reads directly.
"""

import argparse

import numpy as np

__all__ = ["tile_scan_indices", "tile_scan_file"]


def tile_scan_indices(n_slow: int, n_fast: int, t_slow: int, t_fast: int) -> np.ndarray:
    """Source-frame index for each tiled scan position, in raster order.

    Returns an int64 array of length (n_slow * t_slow) * (n_fast * t_fast)
    mapping tiled raster index -> source raster index (slow-major).
    """
    if min(n_slow, n_fast, t_slow, t_fast) < 1:
        raise ValueError("scan dimensions and tile factors must be >= 1")
    sy = np.arange(n_slow * t_slow) % n_slow
    sx = np.arange(n_fast * t_fast) % n_fast
    return (sy[:, None] * n_fast + sx[None, :]).ravel().astype(np.int64)


def tile_scan_file(
    src_path: str,
    dst_path: str,
    t_slow: int = 2,
    t_fast: int = 2,
    key: str = "dp",
    n_slow: int | None = None,
    n_fast: int | None = None,
    chunk_rows: int = 256,
) -> tuple[int, int, int]:
    """Write a tiled copy of the ``key`` stack in ``src_path`` to ``dst_path``.

    The source stack must be (N, ky, kx) with N = n_slow * n_fast frames in
    raster order; square scans may omit n_slow/n_fast. Frames are copied in
    ``chunk_rows``-sized slabs (float32), so memory stays bounded. Returns
    (n_frames_out, n_slow_out, n_fast_out).
    """
    import h5py

    with h5py.File(src_path, "r") as fsrc:
        dp = fsrc[key]
        if not isinstance(dp, h5py.Dataset) or dp.ndim != 3:
            raise ValueError(f"'{key}' in {src_path} is not a (N, ky, kx) dataset")
        n, ky, kx = dp.shape
        if n_slow is None or n_fast is None:
            side = int(round(float(np.sqrt(n))))
            if side * side != n:
                raise ValueError(f"N={n} is not square; pass n_slow/n_fast explicitly")
            n_slow = n_fast = side
        if n_slow * n_fast != n:
            raise ValueError(f"n_slow*n_fast={n_slow * n_fast} != N={n}")
        idx = tile_scan_indices(n_slow, n_fast, t_slow, t_fast)
        n_out = idx.size
        with h5py.File(dst_path, "w") as fdst:
            out = fdst.create_dataset(
                key, shape=(n_out, ky, kx), dtype=np.float32, chunks=(1, ky, kx)
            )
            # copy scalar metadata datasets (ADU, voltage, ...) verbatim
            for name, obj in fsrc.items():
                if isinstance(obj, h5py.Dataset) and name != key and obj.size <= 16:
                    fdst.create_dataset(name, data=obj[()])
            cache: dict[int, np.ndarray] = {}
            for lo in range(0, n_out, chunk_rows):
                hi = min(lo + chunk_rows, n_out)
                src_rows = idx[lo:hi]
                if len(cache) > 4 * chunk_rows:
                    cache.clear()
                for u in np.unique(src_rows):
                    if int(u) not in cache:
                        cache[int(u)] = np.asarray(dp[int(u)], dtype=np.float32)
                out[lo:hi] = np.stack([cache[int(u)] for u in src_rows])
    return n_out, n_slow * t_slow, n_fast * t_fast


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", help="source HDF5/MAT-7.3 file with a (N, ky, kx) stack")
    p.add_argument("dst", help="output HDF5 file (key 'dp', float32)")
    p.add_argument("--tile", type=int, nargs=2, default=(2, 2), metavar=("T_SLOW", "T_FAST"))
    p.add_argument("--key", default="dp")
    p.add_argument("--scan", type=int, nargs=2, default=(None, None), metavar=("N_SLOW", "N_FAST"))
    args = p.parse_args(argv)
    n_out, so, fo = tile_scan_file(
        args.src,
        args.dst,
        t_slow=args.tile[0],
        t_fast=args.tile[1],
        key=args.key,
        n_slow=args.scan[0],
        n_fast=args.scan[1],
    )
    print(f"wrote {args.dst}: {n_out} frames ({so} slow x {fo} fast)")


if __name__ == "__main__":
    main()
