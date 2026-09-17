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

# All the following params files are provided in demo/params/ and we're using relative path here
# So if you change the working directory, or have moved params files around, you'll have to provide absolute path to the params file
# "params/tBL_WSe2_reconstruct_minimal_born.yml", "params/tBL_WSe2_reconstruct_minimal_multislice.yml",
params_paths = [
    "/home/dnz75396/ptyrad/demo/params/PSO_reconstruct_ms.yml"
]


run_name = [ "multislice_subslices7" ]  # This is used to name the log file and output folder. You can change it to any string you like

for i, params_path in enumerate(params_paths):
    print(f"Running reconstruction with params file: {params_path}")
    logger = CustomLogger(
        log_file=f"ptyrad_log_{run_name[i]}.txt",
        log_dir="auto",
        prefix_time="datetime",
        show_timestamp=True,
    )
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
