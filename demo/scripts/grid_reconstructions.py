"""
The Purpose of this script is to compare reconstructions for different solvers
at different slice thicknesses across multiple GPUs, and benchmark their execution times.
"""

import concurrent.futures
import copy
import csv
import multiprocessing as mp
import os
import time
from datetime import datetime

import torch
from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import CustomLogger, print_system_info, set_gpu_device

# For a clean progress bar
from tqdm import tqdm


def run_reconstruction(solver, slice_thickness, base_params, start_time, gpu_id):
    """
    Worker function to run a single reconstruction on a specific GPU and measure time.
    """
    params = copy.deepcopy(base_params)

    # Update parameters
    params["model_params"]["solver_type"] = solver

    # These belong in 'init_params', not 'model_params'
    params["init_params"]["obj_slice_thickness"] = slice_thickness
    total_thickness = 240
    layers = int(total_thickness / slice_thickness)
    params["init_params"]["obj_Nlayer"] = layers
    params["recon_params"]["NITER"] = 50
    params["recon_params"]["SAVE_ITERS"] = 25

    if solver == "strang":
        params["init_params"]["probe_z_shift"] = -slice_thickness / 2.0
    else:
        params["init_params"]["probe_z_shift"] = 0.0

    # 1. Generate an ABSOLUTE path for the logs directory
    base_work_dir = os.path.abspath(os.getcwd())
    log_dir = os.path.join(base_work_dir, "output", start_time, "logs")
    os.makedirs(log_dir, exist_ok=True)

    params["recon_params"]["output_dir"] = f"output/{start_time}/{solver}_slice{slice_thickness}A"

    device = set_gpu_device(gpuid=gpu_id)

    # 2. Pass the absolute log directory to CustomLogger and turn off log_dir auto-generation
    local_logger = CustomLogger(
        log_file=f"GPU{gpu_id}_{solver}_{slice_thickness}A.txt",
        log_dir=log_dir,  # Force absolute path here
        prefix_time="datetime",
        show_timestamp=True,
    )

    status = "Failed"
    error_msg = ""
    elapsed_time = 0.0

    t0 = time.time()

    try:
        ptycho_solver = PtyRADSolver(params, device=device, logger=local_logger)
        ptycho_solver.run()
        status = "Success"
    except torch.cuda.OutOfMemoryError:
        status = "OOM_Error"
        error_msg = "CUDA Out of Memory"
    except Exception as e:
        status = "Error"
        error_msg = str(e)
    finally:
        t1 = time.time()
        elapsed_time = t1 - t0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "solver": solver,
        "slice_thickness_A": slice_thickness,
        "n_layers": layers,
        "gpu_id": gpu_id,
        "elapsed_time_sec": round(elapsed_time, 2),
        "status": status,
        "error": error_msg,
    }


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    work_dir = "../"
    os.chdir(work_dir)
    print("Current working dir: ", os.getcwd())

    params_path = "params/PSO_reconstruct.yml"
    print_system_info()

    base_params = load_params(params_path, validate=True)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Options
    solver_options = ["born", "gmres1", "multislice"]
    slice_thickness_options = [10.0]

    tasks = []
    for solver in solver_options:
        for thickness in slice_thickness_options:
            tasks.append((solver, thickness))

    # Dynamically detect available GPUs, fallback to 1 if none found
    num_gpus = max(1, torch.cuda.device_count())
    print(f"\nDetected {num_gpus} GPUs. Initializing Process Pool...")

    overall_start_time = time.time()
    benchmark_results = []

    # max_workers = num_gpus ensures optimal load balancing without thrashing VRAM
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
        future_to_task = {
            executor.submit(
                run_reconstruction, solver, thickness, base_params, start_time, i % num_gpus
            ): (solver, thickness)
            for i, (solver, thickness) in enumerate(tasks)
        }

        # Wrap as_completed in tqdm for a gorgeous terminal progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_task),
            total=len(tasks),
            desc="Benchmarking Solvers",
        ):
            solver, thickness = future_to_task[future]
            try:
                result_data = future.result()
                benchmark_results.append(result_data)
                # We optionally suppress individual print statements here because tqdm provides
                # the status, but if it failed, we definitely want to print it.
                if result_data["status"] != "Success":
                    tqdm.write(
                        f"GPU {result_data['gpu_id']}] FAILED: {solver} | {thickness}A -> {result_data['error']}"
                    )
            except Exception as e:
                tqdm.write(f"CRITICAL PROCESS ERROR for {solver} at {thickness}A: {e}")
                benchmark_results.append(
                    {
                        "solver": solver,
                        "slice_thickness_A": thickness,
                        "n_layers": "N/A",
                        "gpu_id": "Unknown",
                        "elapsed_time_sec": 0,
                        "status": "Process_Crash",
                        "error": str(e),
                    }
                )

    overall_end_time = time.time()
    print("\n--- ALL RUNS COMPLETE ---")
    print(
        f"Total benchmark wall-clock time: {round(overall_end_time - overall_start_time, 2)} seconds"
    )

    # --- SAVE RESULTS TO CSV ---
    csv_filename = f"output/{start_time}/benchmark_summary_{start_time}.csv"

    # Sort results nicely by solver, then by thickness
    benchmark_results.sort(key=lambda x: (x["solver"], x["slice_thickness_A"]))

    with open(csv_filename, mode="w", newline="") as file:
        fieldnames = [
            "solver",
            "slice_thickness_A",
            "n_layers",
            "gpu_id",
            "elapsed_time_sec",
            "status",
            "error",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in benchmark_results:
            writer.writerow(row)

    print(f"Benchmark data successfully saved to: {csv_filename}")
