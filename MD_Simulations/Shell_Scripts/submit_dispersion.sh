#!/bin/bash
#SBATCH --job-name=Calc_Disp
#SBATCH --account=acf-utk0011          
#SBATCH --partition=campus             # CPU partition is fine, numpy handles this well
#SBATCH --qos=campus                   
#SBATCH --output=slurm_disp_%j.out
#SBATCH --time=04:00:00                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # High CPU count for numpy vectorization
#SBATCH --mem=32G                      # Buffer to load the entire trajectory into RAM

module load anaconda3                  
source activate super_salt_env         
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# CRITICAL: Pass the CPU allocation to OpenMP to fix your previous warnings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK   

COMP=$1
TEMP=$2

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Calculating Dispersion Relation (Current Correlation) for $COMP at $TEMP K..."

# dt=5.0 assumes you dumped NVE frames every 5 fs. Adjust if different!
python "$PROJECT_ROOT/Pipeline/run_dispersion_pipeline.py" --comp "$COMP" --temp "$TEMP" --dt 5.0