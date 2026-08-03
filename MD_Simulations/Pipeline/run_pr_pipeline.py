import os
import argparse
import numpy as np
import pandas as pd
from ase.io import read
from ase.vibrations import Vibrations
from mace.calculators import MACECalculator

def main():
    parser = argparse.ArgumentParser(description="Instantaneous Normal Modes & Participation Ratio Calculator")
    parser.add_argument('--comp', type=str, required=True, help='Composition e.g., "0.417NaCl-0.058KCl-0.525CaCl2"')
    parser.add_argument('--temp', type=float, required=True, help='Target production temperature in K')
    parser.add_argument('--model', type=str, default='SuperSalt-swa.model', help='Path to MACE model')
    parser.add_argument('--frame', type=int, default=-1, help='Trajectory frame to analyze (default: last frame)')
    args = parser.parse_args()

    TEMP_K = args.temp
    SYSTEM_NAME = args.comp
    TRAJ_FILE = f"{SYSTEM_NAME}_{TEMP_K}K_NVE_seed_42.extxyz"
    
    print("# =========================================================")
    print("# INSTANTANEOUS NORMAL MODES (INM) & PARTICIPATION RATIO")
    print("# =========================================================")
    print(f"Reading topology and positions from: {TRAJ_FILE} (Frame: {args.frame})")

    # 1. Load the specific snapshot from the MD trajectory
    try:
        atoms = read(TRAJ_FILE, index=args.frame)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {TRAJ_FILE}. Ensure MD has completed first.")

    N = len(atoms)
    print(f"System loaded: {N} atoms.")

    # 2. Attach the MACE Calculator
    print(f"Loading MACE MLIP from {args.model}...")
    calc = MACECalculator(model_paths=args.model, device='cuda')
    atoms.calc = calc

    # 3. Calculate the Hessian Matrix (Finite Differences)
    # ASE Vibrations displaces each atom by ~0.01 A in x, y, z and measures forces
    vib_dir = f"inm_data_{SYSTEM_NAME}_{TEMP_K}K"
    os.makedirs(vib_dir, exist_ok=True)
    
    print("\nCalculating Hessian matrix via finite displacements...")
    print(f"(This requires 6*N = {6 * N} force evaluations. Please wait...)")
    
    vib = Vibrations(atoms, name=f'{vib_dir}/vib')
    # vib.run() only computes if the files don't already exist in the folder
    vib.run() 
    print("Hessian matrix computed and cached.")

    # 4. Diagonalize to get Frequencies and Eigenvectors
    print("\nDiagonalizing Dynamical Matrix...")
    # ASE returns negative values for imaginary/unstable frequencies
    frequencies = vib.get_frequencies() 
    
    print("Calculating Participation Ratio (PR) for all 3N modes...")
    data = []

    # Loop through all 3N degrees of freedom
    for i in range(3 * N):
        freq = frequencies[i]
        
        # get_mode(i) returns an (N, 3) array of displacement vectors
        mode = vib.get_mode(i) 
        
        # Calculate u_i^2 (squared displacement magnitude for each atom)
        u2 = np.sum(mode**2, axis=1)
        
        # Participation Ratio Formula
        num = np.sum(u2)**2
        den = N * np.sum(u2**2)
        pr = num / den if den > 0 else 0.0
        
        # Classify as Real (stable) or Imaginary (unstable diffusive)
        mode_type = "Real" if freq >= 0 else "Imaginary"
        
        data.append({
            'Mode_Index': i,
            'Frequency_cm-1': freq,
            'Type': mode_type,
            'Participation_Ratio': pr
        })

    # 5. Export to CSV
    df = pd.DataFrame(data)
    csv_filename = f"PR_{SYSTEM_NAME}_{int(TEMP_K)}K.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\nAnalysis complete!")
    print(f"Participation Ratio data saved to: {csv_filename}")
    
    # Optional: Print a quick summary of the real modes
    real_df = df[df['Type'] == 'Real']
    if not real_df.empty:
        print(f"\n--- Real Modes Summary ---")
        print(f"Total Real Modes: {len(real_df)}")
        print(f"Max Frequency: {real_df['Frequency_cm-1'].max():.2f} cm^-1")
        print(f"Avg PR: {real_df['Participation_Ratio'].mean():.4f}")

if __name__ == "__main__":
    main()