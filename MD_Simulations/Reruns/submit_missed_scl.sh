#!/bin/bash
#SBATCH --job-name=Calc_SCL
#SBATCH --output=slurm_scl_%j.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=h200:1  # Request 1 GPU
#SBATCH --mem=8G

module load miniforge3
mamba activate super_salt_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_scl_pipeline.py --comp 0.21NaCl-0.41KCl-0.38MgCl2 --temp 660 --density 1.7918
