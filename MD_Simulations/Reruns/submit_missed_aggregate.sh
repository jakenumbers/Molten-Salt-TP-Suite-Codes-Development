#!/bin/bash
#SBATCH --job-name=Aggregate_k
#SBATCH --output=slurm_aggregate_%j.out
#SBATCH --time=00:10:00        # 10 minutes is more than enough time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1      # CPU only, no GPUs required
#SBATCH --mem=4G

# Load your environment
module load miniforge3
mamba activate super_salt_env

echo "Starting aggregation..."

# Edit the composition and temperature on this line directly!
python run_aggregate_pipeline.py --comp 2LiCl-4NaCl-5KCl-3RbCl-2CsCl-3MgCl2-9CaCl2-2SrCl2-2BaCl2-1ZnCl2-6ZrCl4 --temp 1400

echo "Aggregation complete!"
