# Multiple GPUs

Hypertune mode distributes many individual reconstruction tasks on multiple GPUs, where each task utilizes exactly 1 GPU. *PtyRAD* also supports running the **reconstuction mode with multiple GPUs**, which split the workload of a single large reconstuction task on multiple GPUs. *PtyRAD* uses the convenient API wrapper from *HuggingFace Accelerate* that wraps around PyTorch's Distributed Data Parallel (DDP) framework.

## Reconstruciton mode on multiple GPUs

To reconstruct with multiple GPUs, you'll need the following:

- Linux system
- more than 1 NVIDIA GPU on the same machine (multi-instance GPU, or MIGs are NOT considered as multi GPU and would not work, see [here](https://discuss.pytorch.org/t/parallel-training-with-invidia-migs/159445).)
- `accelerate` package from HuggingFace (it's currently listed as a core dependency of PtyRAD so if you follow the README instruction while creating environment, it should already be included)

This is an example Slurm script.

```bash
#!/bin/bash
#SBATCH --job-name=ptyrad
#SBATCH --nodes=1                            # number of nodes requested
#SBATCH --ntasks=1                           # number of tasks to run in parallel
#SBATCH --cpus-per-task=32                    # number of CPUs required for each task. 4 for 10GB, 8 for 20GB, 32 for 80GB of A100.
#SBATCH --gres=gpu:a100:2                 # request a GPU #gpu:a100:1, gpu:2g.20gb:1
#SBATCH --time=168:00:00                     # Time limit hrs:min:sec
#SBATCH --output=log_job_%j_ptyrad_PSO_reconstruct.txt  # Standard output and error log

pwd; hostname; date

module load cuda/11.8

source activate ptyrad

## Set the params_path variable
PARAMS_PATH="params/PSO_reconstruct.yml"
echo params_path = ${PARAMS_PATH}

## Assuming you are under `ptyrad/` root and calling `sbatch scripts/slurm_run_ptyrad.sub`
## Change directory to ptyrad/demo/ so the relative path of data location specified in params.yml is correct
cd demo/; pwd;

## For multi-GPU and mixed-precision, explicitly pass `--gpuid acc` so that we can defer the device assignment to the "HuggingFace accelerate" package
## These capabilities are only available while launched via `accelerate``, and are only supported on non-MIG nodes so we can only do c0001 (full A100s)
accelerate launch --multi_gpu --num_processes=2 --mixed_precision='no' -m ptyrad run --params_path "${PARAMS_PATH}" --gpuid acc 2>&1 # This runs DDP on 2 GPUs without mixed precision

```

> 💡 Note that 2 GPUs are requested `--gres=gpu:a100:2`, and the `ptyrad run` command is launched via `accelerate launch --multi_gpu`.
