# Integrated Structural Coherence Model (SCM) and Machine-Learning Interatomic Potential (MLIP)

This repository contains the Python scripts and SLURM batch files required to run the Integrated Structural Coherence Model (ISM) pipeline through an HPC Cluster. 

This pipeline is a highly automated computational framework designed to extract macroscopic thermodynamics, structural geometry, and microscale vibrational/spectral dynamics of molten salts directly from a universal Machine Learning Interatomic Potential (MLIP).

This pipeline originates from:
> **Integration of a Structural Coherence Model with a Semi-Universal Machine Learning Interatomic Potential to Predict Molten Salt Thermal Conductivity** (Walker et al., 2026). DOI: 10.21203/rs.3.rs-10223172/v1

## 🚀 Pipeline Capabilities
The automated pipeline executes parallel Molecular Dynamics (MD) simulations to extract:
- **Thermodynamics:** Heat Capacity ($C_p$), Isothermal Bulk Modulus ($K_T$), and Volumetric Thermal Expansion ($\alpha$).
- **Liquid Structure:** Structural Coherence Length (SCL), Radial Distribution Functions (RDFs), Coordination Number Distributions (CND), and Bond Angle Distributions (BAD).
- **Phonon/Spectral Dynamics:** Speed of Sound ($V_s$), Vibrational Density of States (VDOS), Phonon Transfer/Overlap Factors ($b_{PH}$), and Longitudinal Dispersion Relations.
- **Normal Modes:** Instantaneous Normal Modes (INM) and Participation Ratios (PR).

## 📁 Repository Structure
```text
2026-ISM-Integrated-SCM-MLIP-phonon_study/
│
├── SuperSalt-swa.model          # MACE MLIP Model (Requires Download)
├── super_salt_env.yml           # Conda/Mamba environment configuration
│
├── Pipeline/                    # Core Python MD scripts
│   ├── run_scl_pipeline.py      # Calculates SCL, CND, and BAD
│   ├── run_cp_pipeline.py       # Temperature sweep for Enthalpy/Cp
│   ├── run_kt_pipeline.py       # Pressure sweep for Bulk Modulus
│   ├── run_vs_pipeline.py       # NVE production for Vs and VDOS overlap
│   ├── run_dispersion_pipeline.py # J_L(q,w) spectral dispersion
│   ├── run_pr_pipeline.py       # Hessian/Participation Ratio calculation
│   ├── run_aggregate_pipeline.py# Final Thermal Conductivity calculation
│   └── (Density calculation scripts)
│
├── Shell_Scripts/               # SLURM submission scripts for HPC scheduling
│   ├── submit_master_pipeline.sh
│   └── (Individual submit_*.sh files)
│
└── Reruns/                      # Helper scripts for failed or single-job resubmissions
```

## ⚙️ Set-up Instructions
### 1. Environment and MLIP
1. Create your Python virtual environment using the provided configuration:
``` Bash
conda env create -f super_salt_env.yml
```
2. Download the `SuperSalt-swa.model` from the data availability section of the original SuperSalt paper: https://doi.org/10.5281/zenodo.15734798. Place this file in the root of the repository (`$PROJECT_ROOT`).
### 3. Density Databases (Optional)
If you wish to use the highly accurate Redlich-Kister (RK) empirical density models to bypass the MD density burn-in:
1. Request access to the MSD-TP through: https://msd.ornl.gov/access-instructions/
2. Download Molten_Salt_Thermophysical_Properties_rho_RK.csv and Molten_Salt_Thermophysical_Properties.csv.
3. Place them in the Shell_Scripts directory.

*Note: If these files are missing or the target composition is highly complex, the pipeline automatically falls back to calculating the density dynamically via NPT MD simulation.*

## 🧪 Running the Pipeline
The pipeline is completely automated. It features generalized regex parsing, meaning it can accept chlorides, fluorides, bromides, or mixed halide salts seamlessly.

To launch the full pipeline (Density → Structure/Thermo/Dynamics → Aggregation), execute the master shell script from the `Shell_Scripts` directory with your target composition and temperature (in Kelvin):

``` Bash
cd Shell_Scripts
./submit_master_pipeline.sh 0.32MgCl2-0.68KCl 800
```
#### Example for Fluorides:

``` Bash
./submit_master_pipeline.sh 0.67LiF-0.33BeF2 900
```
### Statistical Averaging (Array Jobs)
Because dynamic spectral properties (like VDOS, $V_s$, and acoustic dispersion) are highly sensitive to thermal noise in liquids, the pipeline automatically runs vs and dispersion as SLURM array jobs. Multiple independent seeds are generated, and the final aggregation scripts automatically parse, average, and smooth the power spectra and velocities to provide highly converged, noise-free outputs.

## 🛠️ Debugging & Reruns
If an individual job fails (e.g., due to a node crashing or running out of wall time), the dependent `aggregate` script will cancel itself. You do not need to rerun the entire master pipeline.

1. Navigate to the `Reruns/` directory.
2. Manually submit the missed property using its specific script (e.g., `sbatch submit_missed_vs.sh`).
3. Once all required `.txt` property files (and `.csv` VDOS/Dispersion files) are present in the output folder, calculate the final thermal conductivity manually:

``` Bash
./submit_missed_aggregate.sh COMPOSITION TEMPERATURE
```
*Note: Current shell scripts are configured for NVIDIA H200 GPUs. Modify the* `#SBATCH --partition` *and* `#SBATCH --time` *flags inside the* `.sh` *files to match your institution's specific cluster hardware and queue limits.*
