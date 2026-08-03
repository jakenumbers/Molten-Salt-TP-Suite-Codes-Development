#!/bin/bash
#SBATCH --job-name=Find_Density
#SBATCH --account=acf-utk0011          
#SBATCH --partition=campus-gpu         
#SBATCH --qos=campus-gpu               # <--- ADD THIS
#SBATCH --gres=gpu:1
#SBATCH --gpus=1                       # <--- FIX THIS
#SBATCH --output=slurm_md_density_%j.out
#SBATCH --time=03:00:00                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            
#SBATCH --mem=6G                

module load anaconda3                  # <--- FIX THIS
module load cuda
source activate super_salt_env         # <--- FIX THIS
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   # <--- (Fixes the OpenMP error)

# Define your massive 11-cation mixture and temperature
COMP=$1
TEMP=$2

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Execute the Density Finder
python "$PROJECT_ROOT/Pipeline/MD_density.py" --comp $COMP --temp $TEMP --out density_${COMP}_${TEMP}K.txt