import argparse
import numpy as np
import pandas as pd
import glob
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Aggregate MD properties and calculate k(T)")
    parser.add_argument('--comp', required=True, type=str, help="Salt composition (e.g., 0.5NaCl-0.5KCl)")
    parser.add_argument('--temp', required=True, type=float, help="MD Simulation Temp / Melting Temp (K)")
    args = parser.parse_args()

    print(f"--- AGGREGATING THERMAL CONDUCTIVITY FOR {args.comp} ---")

    # 1. Dynamically construct filenames using the EXACT composition string (periods included)
    temp_int = int(args.temp)
    
    scl_file = f"scl_{args.comp}_{temp_int}K.txt"
    cp_file = f"cp_{args.comp}_{temp_int}K.txt"
    kt_file = f"Kt_{args.comp}_{temp_int}K.txt" 
    pr_file = f"PR_{args.comp}_{temp_int}K.csv"
    
    # Use args.comp directly for the glob pattern
    vs_pattern = f"Vs_{args.comp}_seed_*_{temp_int}K.txt"
    vs_files = glob.glob(vs_pattern)

    # 2. Check if files exist before trying to read them
    missing_files = []
    if not os.path.exists(scl_file): missing_files.append(scl_file)
    if not os.path.exists(cp_file): missing_files.append(cp_file)
    if not os.path.exists(kt_file): missing_files.append(kt_file)
    # if not os.path.exists(pr_file): missing_files.append(pr_file)
    if not vs_files: missing_files.append(vs_pattern)

    if missing_files:
        print("ERROR: The following required files are missing:")
        for mf in missing_files:
            print(f"  - {mf}")
        sys.exit(1)

    # 3. Read SCL (Angstroms -> meters)
    with open(scl_file, 'r') as f:
        l_scl_A = float(f.read().strip())
    l_scl_m = l_scl_A * 1e-10

    print(f"scl: {l_scl_A} Angstroms")

    # 4. Read Cp and Alpha 
    with open(cp_file, 'r') as f:
        cp_str, alpha_str = f.read().strip().split(',')
        cp_v = float(cp_str)      # J / (m^3 * K)
        alpha = float(alpha_str)  # 1 / K

    print(f"Cp: {cp_v}")
    print(f"alpha: {alpha}")

    # 5. Read Kt (GPa -> Pascals)
    with open(kt_file, 'r') as f:
        kt_gpa = float(f.read().strip())
    kt_pa = kt_gpa * 1e9

    print(f"Kt: {kt_gpa} GPa")

    # 6 Read Participation Ratio (PR) and calculate average PR for real modes
    pr_df = pd.read_csv(pr_file)
    real_modes_df = pr_df[pr_df['Type'] == 'Real']
    avg_real_pr = real_modes_df['Participation_Ratio'].mean() if not real_modes_df.empty else 0.0
    
    print(f"Participation Ratio (Avg Real Modes PR): {avg_real_pr:.4f}")

    # ==========================================
    # 7. Read and Average Vs (Safely handling NaN)
    # ==========================================
    vs_vals = []
    for v_file in vs_files:
        with open(v_file, 'r') as f:
            val = float(f.read().strip())
            vs_vals.append(val)
            if np.isnan(val):
                print(f"  [Warning] NaN value detected in {v_file}. Skipping this trial.")

    # Convert to a numpy array to easily filter and count
    vs_array = np.array(vs_vals)
    valid_trials = np.count_nonzero(~np.isnan(vs_array))
    
    if valid_trials == 0:
        print("\nERROR: All Vs trials returned NaN. Cannot calculate thermal conductivity.")
        sys.exit(1)

    # np.nanmean safely ignores the NaN values and averages the rest
    vs_avg = np.nanmean(vs_array)

    print(f"Averaged v_s from {valid_trials} valid trials (out of {len(vs_files)} total): {vs_avg:.2f} m/s")

    # ==========================================
    # PHYSICS CALCULATIONS
    # ==========================================
    
    # Base Thermal Conductivity at T_m
    k_tm = (1.0 / 3.0) * cp_v * vs_avg * l_scl_m

    # Thermodynamic Conversions
    cv_v = cp_v - (args.temp * (alpha**2) * kt_pa)
    if cv_v <= 0:
        raise ValueError(
            f"Unphysical Cv={cv_v:.6e} J/(m^3 K). "
            "Check Cp, alpha, Kt, unit conversions, and fitting quality."
        )
    
    # Gruneisen Parameter (gamma = alpha * K_T / C_v)
    gamma = (alpha * kt_pa) / cv_v

    print("\n--- FINAL RESULTS ---")
    print(f"k at T_m ({args.temp}K): {k_tm:.4f} W/(m*K)")
    print(f"Gruneisen Param (gamma): {gamma:.4f}")
    print(f"C_v (volumetric): {cv_v:.2f} J/(m^3*K)")
    
    # Gheribi Equation Output
    bracket_term = alpha * (gamma + (1.0/3.0))
    print(f"\n--- k(T) FUNCTION ---")
    print(f"k(T) = {k_tm:.4f} * [ 1 - {bracket_term:.6e} * (T - {args.temp}) ]")
    
    output_filename = f"k(T)_{args.comp}_{int(args.temp)}K.txt"
    with open(output_filename, "w") as f:
        f.write(f"k(T) = {k_tm:.4f} * [ 1 - {bracket_term:.6e} * (T - {args.temp}) ]")
    print(f"Pipeline data successfully saved to {output_filename}", flush=True)

if __name__ == "__main__":
    main()