import os

# Change this to the ABSOLUTE PATH to the demo/ folder so you can correctly access data/ and params/
work_dir = "../"  # Leave this as-is if you're running the notebook from the `ptyrad/demo/scripts/` folder, this will change it back to demo/

os.chdir(work_dir)
print("Current working dir: ", os.getcwd())
# The printed working dir should be ".../ptyrad/demo" to locate the demo params files easily
# Note that the output/ directory will be automatically generated under your working directory

from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import CustomLogger, print_system_info, set_gpu_device

logger = CustomLogger(
    log_file="ptyrad_log.txt", log_dir="auto", prefix_time="datetime", show_timestamp=True
)


params_path = ["params/PSO_reconstruct_linduda.yml","params/PSO_reconstruct_ms.yml"]

print_system_info()


for param_path in params_path:
    params = load_params(param_path, validate=True)

    print("Loaded params: ", params)
    device = set_gpu_device(
        gpuid=0
    )  # Pass in `gpuid = None` if you don't have access to a CUDA-compatible GPU. Note that running PtyRAD with CPU would be much slower than on GPU.

    ptycho_solver = PtyRADSolver(params, device=device, logger=logger)

    ptycho_solver.run()