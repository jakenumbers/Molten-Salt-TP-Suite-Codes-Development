import pandas as pd
import argparse
import sys
import re
import itertools
import os

def calculate_dynamic_density(comp_str, T, base_db_path, rk_db_path):
    # ---------------------------------------------------------
    # 1. Load the Databases
    # ---------------------------------------------------------
    base_df = pd.read_csv(base_db_path)
    rk_df = pd.read_csv(rk_db_path)
    pure_df = base_df[base_df['comp'] == 'Pure Salt']
    
    # ---------------------------------------------------------
    # 2. Extract Pure Salt Properties (Baseline)
    # ---------------------------------------------------------
    def get_pure_density(salt_name):
        salt_data = pure_df[pure_df['formula'] == salt_name].iloc[0]
        rho_a = float(salt_data['rho_a'])
        rho_b = float(salt_data['rho_b'])
        # Subtract the temperature coefficient
        density = rho_a - (rho_b * T)
        return density

    # ---------------------------------------------------------
    # 3. Parse Composition using Regex
    # ---------------------------------------------------------
    components = comp_str.split('-')
    salts = []
    fractions = []
    
    for comp in components:
        # Regex captures the decimal/fraction/integer, then the alphanumeric salt formula
        match = re.match(r"([0-9.]+)([A-Za-z0-9]+)", comp)
        if not match:
            return f"Error: Invalid format {comp}. Must be like '0.5NaCl-0.5KCl' or '2LiCl-4NaCl'."
        
        fractions.append(float(match.group(1)))
        salts.append(match.group(2))

    # Normalize fractions to guarantee they sum to 1.0 (for the RK pathway)
    tot_frac = sum(fractions)
    normalized_fractions = [f / tot_frac for f in fractions]

    # ---------------------------------------------------------
    # 3b. High-Order Override (Pre-Simulated Data)
    # ---------------------------------------------------------
    # Construct the expected filename based on the inputs
    override_filename = f"density_{comp_str}_{int(T)}K.txt"
    
    # 1. Highest Priority: If the file exists, always use it.
    if os.path.exists(override_filename):
        print(f"  [Info] Override file found: {override_filename}. Bypassing empirical RK model.")
        try:
            with open(override_filename, 'r') as f:
                content = f.read().strip()
                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    return float(match.group())
                else:
                    return f"Error: Could not parse a numerical density value from {override_filename}"
        except Exception as e:
            return f"Error reading override file {override_filename}: {e}"
            
    # 2. Safety Net: If the file doesn't exist, but the mixture is massive, abort.
    elif len(salts) > 10:
        return f"Error: >10 components detected but override file '{override_filename}' was not found. RK evaluation is not supported for this size."
        
    # 3. If file doesn't exist and components <= 10, proceed to empirical calculation...

    # ---------------------------------------------------------
    # 4. Calculate Ideal Density
    # ---------------------------------------------------------
    rho_ideal = 0.0
    try:
        for i in range(len(salts)):
            rho_ideal += normalized_fractions[i] * get_pure_density(salts[i])
    except IndexError:
        return f"Error: Could not find pure salt properties for one of your endmembers."

    # ---------------------------------------------------------
    # 5. Binary Excess Helper Function (Muggianu Model)
    # ---------------------------------------------------------
    def get_binary_excess(sA, sB, fracA, fracB):
        if fracA == 0 or fracB == 0:
            return 0.0
            
        rk_row = rk_df[(rk_df['C1'] == sA) & (rk_df['C2'] == sB)]
        reverse_order = False
        
        if rk_row.empty:
            rk_row = rk_df[(rk_df['C1'] == sB) & (rk_df['C2'] == sA)]
            reverse_order = True
            
        if rk_row.empty:
            print(f"  [Warning] No RK parameters for {sA}-{sB}. Assuming ideal mixing (0 excess) for this pair.")
            return 0.0
        
        rk_row = rk_row.iloc[0]
        
        A1, B1 = float(rk_row['A1']), float(rk_row['B1'])
        A2, B2 = float(rk_row['A2']), float(rk_row['B2'])
        A3, B3 = float(rk_row['A3']), float(rk_row['B3'])
        
        L0 = A1 + (B1 * T)
        L1 = A2 + (B2 * T)
        L2 = A3 + (B3 * T)
        
        if reverse_order:
            x_c1, x_c2 = fracB, fracA
        else:
            x_c1, x_c2 = fracA, fracB
            
        excess = x_c1 * x_c2 * (L0 + L1*(x_c1 - x_c2) + L2*(x_c1 - x_c2)**2)
        return excess

    # ---------------------------------------------------------
    # 6. Calculate Excess Density Based on Components
    # ---------------------------------------------------------
    rho_excess_total = 0.0
    
    # Dynamically generate all unique pairs of indices for N-components
    for i, j in itertools.combinations(range(len(salts)), 2):
        rho_excess_total += get_binary_excess(salts[i], salts[j], normalized_fractions[i], normalized_fractions[j])

    # ---------------------------------------------------------
    # 7. Final Density Calculation
    # ---------------------------------------------------------
    mixture_density = rho_ideal + rho_excess_total
    
    return mixture_density

# ==========================================
# Command-Line Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dynamic molten salt density for HPC pipeline.")
    parser.add_argument("--comp", required=True, type=str, help="Composition (e.g., 1.0NaCl or 0.417NaCl-0.525CaCl2-0.058KCl)")
    parser.add_argument("--temp", required=True, type=float, help="Temperature in Kelvin")
    parser.add_argument("--out", default="density_output.txt", type=str, help="Output file for the density value")
    
    args = parser.parse_args()
    
    BASE_CSV = "Molten_Salt_Thermophysical_Properties.csv"
    RK_CSV = "Molten_Salt_Thermophysical_Properties_rho_RK.csv"
    
    result = calculate_dynamic_density(args.comp, args.temp, BASE_CSV, RK_CSV)
    
    if isinstance(result, str):
        # Print the error message to the standard Slurm error log
        print(result, file=sys.stderr)
        # Exit with code 1 to prevent dependent jobs from starting with missing data
        sys.exit(1) 
    else:
        # Print human-readable output to the standard Slurm output log
        print(f"Calculated Density for {args.comp} at {args.temp}K: {result:.4f} g/cm^3")
        
        # Write ONLY the numerical density value to a text file for the next jobs
        with open(args.out, "w") as f:
            f.write(str(result))
