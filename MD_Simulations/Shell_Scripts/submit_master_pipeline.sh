#!/bin/bash
# submit_master_pipeline.sh
# Usage: ./submit_master_pipeline.sh 0.5NaCl-0.5KCl 1100

COMP=$1
TEMP=$2

echo "Submitting full thermal conductivity pipeline for $COMP at $TEMP K"

# Load your environment so the Python parser runs correctly on the submit node
module load anaconda3
source activate super_salt_env

# --- PATH FIX: Dynamically track directories regardless of where script is launched ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 1. Parse the composition and check the CSV
# The python script outputs exactly "RK" or "MD"
METHOD=$(python "$PROJECT_ROOT/Pipeline/check_density_method.py" "$COMP")

# 2. Queue the appropriate Density Calculation based on the parser output
if [ "$METHOD" == "RK" ]; then
    echo "Using RK Empirical method for density (CPU only)..."
    DENS_ID=$(sbatch --parsable submit_rk_density.sh $COMP $TEMP)
else
    echo "Using MD method for density (GPU required)..."
    DENS_ID=$(sbatch --parsable submit_md_density.sh $COMP $TEMP)
fi

echo "Density Job ($METHOD): $DENS_ID"

# 3. Parallel Properties (1 GPU each, dependent on Density)
SCL_ID=$(sbatch --parsable --dependency=afterok:$DENS_ID submit_scl.sh $COMP $TEMP)
CP_ID=$(sbatch --parsable --dependency=afterok:$DENS_ID submit_cp.sh $COMP $TEMP)
KT_ID=$(sbatch --parsable --dependency=afterok:$DENS_ID submit_kt.sh $COMP $TEMP)

echo "SCL Job: $SCL_ID | Cp Job: $CP_ID | Kt Job: $KT_ID"

# 4. Vs Array (3 GPUs handled dynamically, dependent on Density)
VS_ID=$(sbatch --parsable --dependency=afterok:$DENS_ID submit_vs.sh $COMP $TEMP)
echo "Vs Array Job: $VS_ID"

# # 5. Participation Ratio
# PART_ID=$(sbatch --parsable --dependency=afterok:$DENS_ID submit_pr.sh $COMP $TEMP)
# if [ -z "$PART_ID" ]; then
#     echo "ERROR: Failed to submit Participation Ratio job. Check your QOS limits."
#     exit 1
# fi
# echo "Participation Ratio Job: $PART_ID"

# 6. Final Aggregation
# AGG_ID=$(sbatch --parsable --dependency=afterok:$SCL_ID:$CP_ID:$KT_ID:$VS_ID:$PART_ID submit_aggregate.sh $COMP $TEMP)
AGG_ID=$(sbatch --parsable --dependency=afterok:$SCL_ID:$CP_ID:$KT_ID:$VS_ID submit_aggregate.sh $COMP $TEMP)
if [ -z "$AGG_ID" ]; then
    echo "ERROR: Failed to submit Aggregate job."
    exit 1
fi
echo "All jobs queued successfully! Final Aggregate Job: $AGG_ID"

echo "All jobs queued successfully!"
