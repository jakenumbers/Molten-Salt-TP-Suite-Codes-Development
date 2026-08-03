#!/bin/bash
#SBATCH --job-name=Aggregate
#SBATCH --account=acf-utk0011          
#SBATCH --partition=campus         
#SBATCH --qos=campus               
#SBATCH --output=slurm_aggregate_%j.out  
#SBATCH --time=00:30:00                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            
#SBATCH --mem=4G                

module load anaconda3                  
source activate super_salt_env         
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Capture the variables passed from the master pipeline
COMP=$1
TEMP=$2

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

DENSITY=$(cat "$SCRIPT_DIR/density_${COMP}_${TEMP}K.txt")

echo "Starting aggregation for $COMP at $TEMP K..."

python "$PROJECT_ROOT/Pipeline/run_aggregate_pipeline.py" --comp $COMP --temp $TEMP

echo "Aggregation complete!"