"""Multislice (Adam) reconstruction warm-started from a line-search Born
checkpoint. Usage:

    python run_ms_warm.py <path/to/model_iterNNNN.hdf5> [NITER]

Loads the tuned PSO multislice params, replaces the object, probe, AND
positions init with the given checkpoint (positions must come from the same
checkpoint — each fresh init re-jitters the scan positions, which would
mismatch the warm-started object), then runs the standard PtyRADSolver
optimizer loop. Everything else (Adam, lrs, constraints, batch 32, sparse
grouping, losses) is untouched, so the run is directly comparable to the
cold-start multislice reference log."""

import os
import sys

work_dir = "../"
os.chdir(work_dir)
print("Current working dir: ", os.getcwd())

from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import CustomLogger, print_system_info, set_gpu_device

ckpt = os.path.abspath(sys.argv[1])
niter = int(sys.argv[2]) if len(sys.argv) > 2 else 87  # match the cold reference log
assert os.path.exists(ckpt), ckpt

params = load_params(
    "/home/dnz75396/ptyrad/demo/params/PSO_reconstruct_ms_paper.yml", validate=True
)
ip = params["init_params"]
ip["obj_source"], ip["obj_params"] = "PtyRAD", ckpt
ip["probe_source"], ip["probe_params"] = "PtyRAD", ckpt
ip["pos_source"], ip["pos_params"] = "PtyRAD", ckpt
params["recon_params"]["NITER"] = niter
params["recon_params"]["output_dir"] = "output/PSO_warmstart/"

logger = CustomLogger(
    log_file="ptyrad_log_ms_warmstart.txt",
    log_dir="auto",
    prefix_time="datetime",
    show_timestamp=True,
)
print_system_info()
print(f"Warm-starting multislice from: {ckpt} for {niter} iterations")

device = set_gpu_device(gpuid=0)
solver = PtyRADSolver(params, device=device, logger=logger)
solver.run()
