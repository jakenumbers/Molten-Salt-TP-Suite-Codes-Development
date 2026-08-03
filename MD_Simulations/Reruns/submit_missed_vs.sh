#!/bin/bash
#SBATCH --job-name=Vs_Array
#SBATCH --output=slurm_vs_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=h200:1  
#SBATCH --mem=4G
#SBATCH --array=41-43        

module load miniforge3
mamba activate super_salt_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_vs_pipeline.py --comp 1.0MgCl2 --temp 987 --density 1.6934 --seed $SLURM_ARRAY_TASK_ID
