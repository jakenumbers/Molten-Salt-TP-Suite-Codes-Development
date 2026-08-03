import sys
import csv
import os
import re
from itertools import combinations

def check_density_method(comp_string, csv_filename):
    # 1. Parse user input properly
    # First, split the mixture into its individual fraction-salt blocks
    parts = comp_string.split('-')
    user_salts = set()
    
    for part in parts:
        # Strip ONLY the leading digits and decimals (e.g., "0.3MgCl2" -> "MgCl2")
        salt = re.sub(r'^[\d\.]+', '', part)
        if salt:
            user_salts.add(salt)
            
    user_salts_list = list(user_salts)

    # 2. Check if file exists
    if not os.path.exists(csv_filename):
        sys.stderr.write(f"INFO: CSV file '{csv_filename}' not found. Defaulting to MD.\n")
        return "MD"

    # 3. Generate all required binary pairs
    # If the user input 3 salts, this generates the 3 unique binary pairs.
    if len(user_salts_list) > 1:
        required_pairs = [set(pair) for pair in combinations(user_salts_list, 2)]
    else:
        # Fallback in case they input a pure salt (unary)
        required_pairs = [{user_salts_list[0]}]

    # 4. Read the CSV and store all available pairs
    available_csv_pairs = []
    try:
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            
            for row in reader:
                if len(row) >= 2:
                    # Create a set from the first two columns.
                    # This ensures order doesn't matter: {'NaCl', 'KCl'} == {'KCl', 'NaCl'}
                    pair = {col.strip() for col in row[:2] if col.strip()}
                    if pair:
                        available_csv_pairs.append(pair)
                        
    except Exception as e:
        sys.stderr.write(f"ERROR: Could not process CSV: {e}. Defaulting to MD.\n")
        return "MD"

    # 5. Verify EVERY required pair exists in the CSV
    missing_pairs = [pair for pair in required_pairs if pair not in available_csv_pairs]
    
    if not missing_pairs:
        sys.stderr.write(f"INFO: All required interaction pairs found in {csv_filename}. Using RK.\n")
        return "RK"
    else:
        # Format the missing pairs nicely for the terminal output so the user knows exactly why MD was chosen
        missing_str = ", ".join(["-".join(p) for p in missing_pairs])
        sys.stderr.write(f"INFO: Missing RK parameters in CSV for pair(s): {missing_str}. Defaulting to MD.\n")
        return "MD"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("ERROR: No composition string provided.\n")
        print("MD")
        sys.exit(1)
        
    COMP = sys.argv[1]
    CSV_NAME = "../Molten_Salt_Thermophysical_Properties_rho_RK.csv"
    
    # Print ONLY the result to stdout for the bash script
    print(check_density_method(COMP, CSV_NAME))