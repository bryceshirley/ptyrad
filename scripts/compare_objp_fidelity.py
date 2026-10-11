"""Object-fidelity comparison of PtyRAD checkpoints against a reference.

Loads objp from each checkpoint, z-sums over slices, translation-registers to the
reference (FFT cross-correlation, sub-pixel), and reports the Pearson correlation
and std/mean contrast on the shared central crop. The operative criterion for
"born iterations needed": object agreement with the multislice reference, NOT the
detector refit residual (see li_cathode/RESULTS.md).

Usage:
  python scripts/compare_objp_fidelity.py REF.hdf5 LABEL1=CK1.hdf5 LABEL2=CK2.hdf5 ...
"""
import sys
import numpy as np
import h5py


def load_zsum(path):
    with h5py.File(path, "r") as f:
        objp = f["optimizable_tensors/objp"][()]  # (omode, Nz, Ny, Nx)
    return objp[0].sum(axis=0).astype(np.float64)


def register_shift(ref, mov):
    F = np.fft.fft2(ref) * np.conj(np.fft.fft2(mov))
    cc = np.fft.ifft2(F / (np.abs(F) + 1e-12))
    pk = np.unravel_index(np.argmax(np.abs(cc)), cc.shape)
    sh = [p - (s if p > s // 2 else 0) for p, s in zip(pk, cc.shape)]
    return np.roll(mov, (-sh[0], -sh[1]), axis=(0, 1))


def crop_center(a, frac=0.7):
    ny, nx = a.shape
    cy, cx = int(ny * (1 - frac) / 2), int(nx * (1 - frac) / 2)
    return a[cy:ny - cy, cx:nx - cx]


def main():
    ref_path = sys.argv[1]
    ref = load_zsum(ref_path)
    print(f"REF {ref_path.split('/')[-2][:50]}  shape={ref.shape}")
    rc = crop_center(ref)
    print(f"{'label':<14}{'corr':>8}{'contrast':>10}{'ref_contr':>10}")
    print(f"{'MS(ref)':<14}{1.0:>8.4f}{rc.std()/abs(rc.mean()):>10.4f}{rc.std()/abs(rc.mean()):>10.4f}")
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        mov = load_zsum(path)
        mov = register_shift(ref, mov)
        mc = crop_center(mov)
        a, b = rc.ravel() - rc.mean(), mc.ravel() - mc.mean()
        corr = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"{label:<14}{corr:>8.4f}{mc.std()/abs(mc.mean()):>10.4f}{rc.std()/abs(rc.mean()):>10.4f}")


if __name__ == "__main__":
    main()
