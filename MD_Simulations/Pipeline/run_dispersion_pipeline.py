import argparse
import numpy as np
import pandas as pd
from ase.io import read
import os
import glob

def autocorrelate(x):
    n = len(x)
    f = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= np.arange(n, 0, -1) 
    return acf / acf[0]        

def main():
    parser = argparse.ArgumentParser(description="Calculate Longitudinal Current Correlation Function for Dispersion")
    parser.add_argument('--comp', type=str, required=True, help='Composition (e.g., 0.5NaCl-0.5KCl)')
    parser.add_argument('--temp', type=float, required=True, help='Temperature in K')
    parser.add_argument('--dt', type=float, default=5.0, help='Time step between frames in fs (default: 5.0)')
    parser.add_argument('--qmax', type=int, default=5, help='Number of q-vector harmonics to calculate')
    args = parser.parse_args()

    # 1. USE GLOB TO FIND ALL SEEDED TRAJECTORIES
    # (Adjust this path if your output directory structure differs slightly)
    traj_pattern = f"{args.comp}_{args.temp}K_NVE_seed_*.extxyz"
    traj_files = glob.glob(traj_pattern)

    if not traj_files:
        print(f"ERROR: Could not find any trajectories matching {traj_pattern}")
        return

    print(f"--- CALCULATING DISPERSION RELATION FOR {args.comp} ---")
    print(f"Found {len(traj_files)} independent trajectories. Averaging results to reduce noise.")

    # We will accumulate the spectral densities across all seeds here
    # Dictionary to hold the sum of J_L_omega for each q
    accumulated_spectra = {n: None for n in range(1, args.qmax + 1)}
    frequencies_cm = None 

    for traj_file in traj_files:
        print(f"\nProcessing: {os.path.basename(traj_file)}...")
        traj = read(traj_file, index=':')
        
        n_frames = len(traj)
        
        # Define q-vectors based on this specific trajectory's average box size
        L_avg = np.mean([frame.cell.lengths()[0] for frame in traj])
        q_min = 2.0 * np.pi / L_avg
        
        # J_L_data[n_q][frame][axis]
        j_L_data = np.zeros((args.qmax, n_frames, 3), dtype=complex)

        # Calculate spatial FFT of currents
        for f_idx, frame in enumerate(traj):
            pos = frame.get_positions()
            vel = frame.get_velocities() 
            
            for n in range(1, args.qmax + 1):
                q_mag = n * q_min
                j_L_data[n-1, f_idx, 0] = np.sum(vel[:, 0] * np.exp(-1j * q_mag * pos[:, 0]))
                j_L_data[n-1, f_idx, 1] = np.sum(vel[:, 1] * np.exp(-1j * q_mag * pos[:, 1]))
                j_L_data[n-1, f_idx, 2] = np.sum(vel[:, 2] * np.exp(-1j * q_mag * pos[:, 2]))

        # Calculate frequencies (only need to do this once)
        if frequencies_cm is None:
            frequencies = np.fft.rfftfreq(n_frames, d=args.dt * 1e-15) 
            frequencies_cm = frequencies / (3e10) 

        # Calculate Autocorrelation and FFT for this seed
        for n in range(1, args.qmax + 1):
            q_mag = n * q_min
            
            acf_x = autocorrelate(j_L_data[n-1, :, 0])
            acf_y = autocorrelate(j_L_data[n-1, :, 1])
            acf_z = autocorrelate(j_L_data[n-1, :, 2])
            acf_avg = (acf_x + acf_y + acf_z) / 3.0
            
            window = np.hanning(n_frames)
            acf_windowed = acf_avg * window
            
            J_L_omega = np.abs(np.fft.rfft(acf_windowed))
            
            # Add to our accumulator
            if accumulated_spectra[n] is None:
                accumulated_spectra[n] = J_L_omega
            else:
                accumulated_spectra[n] += J_L_omega

    # 4. Average the spectra and Export Results
    print("\nAveraging spectra across all seeds...")
    num_seeds = len(traj_files)
    
    df_results = pd.DataFrame({'Frequency_cm-1': frequencies_cm})
    
    for n in range(1, args.qmax + 1):
        # We estimate an average q_mag just for column naming
        avg_q_mag = n * (2.0 * np.pi / L_avg) 
        
        # Divide the accumulated sum by the number of seeds
        averaged_spectrum = accumulated_spectra[n] / num_seeds
        
        col_name = f"J_L_q{n}_mag{avg_q_mag:.3f}"
        df_results[col_name] = averaged_spectrum

    out_file = f"../Output/{args.comp}_{int(args.temp)}K/Dispersion_{args.comp}_{int(args.temp)}K.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df_results.to_csv(out_file, index=False)
    
    print(f"\nSuccess! Averaged dispersion data from {num_seeds} runs saved to {out_file}")

if __name__ == "__main__":
    main()