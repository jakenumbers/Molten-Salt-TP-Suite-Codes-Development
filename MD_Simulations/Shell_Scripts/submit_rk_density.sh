#!/bin/bash
#SBATCH --job-name=Calc_Density
#SBATCH --account=acf-utk0011          
#SBATCH --partition=campus         
#SBATCH --qos=campus               
#SBATCH --output=slurm_rk_density_%j.out  
#SBATCH --time=00:10:00                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            
#SBATCH --mem=4G                

module load anaconda3                  
source activate super_salt_env         
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

COMP=$1
TEMP=$2

# Find the exact folder where this shell script lives, no matter where you launched it from
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Explicitly point to the Pipeline folder
PYTHON_SCRIPT="$PROJECT_ROOT/Pipeline/RK_density.py"

# Call the Python script (assuming FLiNaK fractions for example, adjust logic as needed)
# In production, you might want to pass x1 and x2 from the master script too!
python "$PYTHON_SCRIPT" --comp $COMP --temp $TEMP --out density_${COMP}_${TEMP}K.txt
