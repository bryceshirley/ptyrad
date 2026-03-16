# Slurm script on HPC

If you have access to some HPC (High Performance Computing) systems (a.k.a clusters, supercomputers), very likely you will need to submit a job script via some scheduler software like [Slurm](https://slurm.schedmd.com/documentation.html)

*PtyRAD* can be easily launched on those machines using the previously mentioned CLI commands.

## Reconstruction mode on 1 GPU

Example Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=ptyrad
#SBATCH --nodes=1                            # number of nodes requested
#SBATCH --ntasks=1                           # number of tasks to run in parallel
#SBATCH --cpus-per-task=4                    # number of CPUs required for each task. 4 for 10GB, 8 for 20GB, 32 for 80GB of A100.
#SBATCH --gres=gpu:2g.20gb:1                 # request a GPU #gpu:a100:1, gpu:2g.20gb:1
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

## Execute the ptyrad CLI command `ptyrad run`
ptyrad run --params_path "${PARAMS_PATH}" --jobid "${JOBID:-0}" 2>&1

```
(The *reconstuction mode* is solely configured by the params file by setting `if_hypertune: false`.)

> 💡 This is the same example as `ptyrad/scripts/slurm_run_ptyrad.sub`.
