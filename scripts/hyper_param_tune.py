import os

from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import CustomLogger, print_system_info, set_gpu_device

# Change this to the ABSOLUTE PATH to the demo/ folder so you can correctly access data/ and params/
work_dir = "../"  # Leave this as-is if you're running the notebook from the `ptyrad/demo/scripts/` folder, this will change it back to demo/

os.chdir(work_dir)
print("Current working dir: ", os.getcwd())
# The printed working dir should be ".../ptyrad/demo" to locate the demo params files easily
# Note that the output/ directory will be automatically generated under your working directory

logger = CustomLogger(
    log_file="ptyrad_log.txt", log_dir="auto", prefix_time="datetime", show_timestamp=True
)

# All the following params files are provided in demo/params/ and we're using relative path here
# So if you change the working directory, or have moved params files around, you'll have to provide absolute path to the params file

# params_path = "params/tBL_WSe2_reconstruct_minimal.yml"
# params_path = "params/tBL_WSe2_reconstruct.yml"
# params_path = "params/PSO_reconstruct.yml"
params_path = "params/PSO_hypertune.yml"  # This will run PtyRAD with the hyperparameter tuning mode

print_system_info()

# We enable validation to auto-fill defaults and check parameter consistency since PtyRAD 0.1.0b8
# If you run into issues with validation (e.g., false positives or unexpected errors),
# you can temporarily disable it by setting `validate=False` and prepare a fully complete params file yourself.
# If this happens, please report the bug so we can improve the validation logic.
params = load_params(params_path, validate=True)

print("Loaded params: ", params)
device = set_gpu_device(
    gpuid=0
)  # Pass in `gpuid = None` if you don't have access to a CUDA-compatible GPU. Note that running PtyRAD with CPU would be much slower than on GPU.

ptycho_solver = PtyRADSolver(params, device=device, logger=logger)

ptycho_solver.run()

# Only `reconstruct` mode will return the final reconstructed model, because it's infeasible to store all models in `hypertune` mode and we don't know which model to return in `hypertune` mode
if not ptycho_solver.if_hypertune:
    model = ptycho_solver.reconstruct_results
