#!/bin/bash
#SBATCH --job-name=Calc_Kt
#SBATCH --account=acf-utk0011          # <--- CRITICAL: Slurm needs to know who to bill!
#SBATCH --partition=campus-gpu                # <--- CRITICAL: Tells Slurm to send this to ISAAC GPU nodes
#SBATCH --qos=campus-gpu               # <--- ADD THIS
#SBATCH --gpus=1                       # <--- FIX THIS
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_kt_%j.out
#SBATCH --time=04:00:00                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # <--- INCREASED from 1 to speed up neighbor lists
#SBATCH --mem=24G                  # <--- INCREASED from 6G to survive the memory spike              

module load anaconda3                  # <--- FIX THIS
module load cuda
source activate super_salt_env         # <--- FIX THIS
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # <--- (Fixes the OpenMP error)

COMP=$1
TEMP=$2

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DENSITY=$(cat "$SCRIPT_DIR/density_${COMP}_${TEMP}K.txt")

python "$PROJECT_ROOT/Pipeline/run_kt_pipeline.py" --comp $COMP --temp $TEMP --density $DENSITY --model "$PROJECT_ROOT/SuperSalt-swa.model"
