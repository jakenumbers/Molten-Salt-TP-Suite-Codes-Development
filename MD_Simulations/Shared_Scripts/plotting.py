import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- NEW IMPORTS FOR PDF & OMEGA_0 EXTRACTION ---
from ase.io import read
from ase.data import atomic_masses, atomic_numbers
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ====================================================================
# GLOBAL PLOT STYLING
# ====================================================================
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 12,
    'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.linewidth': 1.5,
    'xtick.labelsize': 11, 'ytick.labelsize': 12,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 1.75, 'ytick.major.width': 1.75,
    'legend.frameon': False, 'legend.fontsize': 11,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
})

# ====================================================================
# SCM MATH: DUAL-GAUSSIAN PMF TO FUNDAMENTAL FREQUENCY
# ====================================================================
def dual_gaussian(x, a1, m1, s1, a2, m2, s2, y0):
    return (a1 * np.exp(-0.5 * ((x - m1) / s1)**2) +
            a2 * np.exp(-0.5 * ((x - m2) / s2)**2) + y0)

def extract_pdf_and_omega0(traj_file, temp_k, cations, anion='Cl'):
    print(f"\n--- Extracting unweighted PDF from {traj_file} for omega_0 ---")
    
    # 1. Read last 200 frames via ASE for optimization
    traj = read(traj_file, index='-200:')
    
    # 2. Bridge ASE to MDAnalysis (Prevents topology errors with extxyz)
    n_atoms = len(traj[0])
    elements = traj[0].get_chemical_symbols()
    
    u = mda.Universe.empty(n_atoms, trajectory=True)
    u.add_TopologyAttr('name', elements)
    u.add_TopologyAttr('type', elements)
    
    coord_array = np.array([frame.get_positions() for frame in traj])
    u.load_new(coord_array, format=mda.coordinates.memory.MemoryReader)
    
    for i, frame in enumerate(traj):
        u.trajectory[i].dimensions = np.array([
            frame.cell.lengths()[0], frame.cell.lengths()[1], frame.cell.lengths()[2],
            frame.cell.angles()[0], frame.cell.angles()[1], frame.cell.angles()[2]
        ])

    omega0_dict = {}
    m_anion = atomic_masses[atomic_numbers[anion]]

    # 3. Calculate RDF, Fit, and Extract Math
    for cat in cations:
        ag_cat = u.select_atoms(f'name {cat}')
        ag_an = u.select_atoms(f'name {anion}')
        
        if len(ag_cat) == 0: continue
            
        # Execute MDAnalysis InterRDF
        irdf = rdf.InterRDF(ag_cat, ag_an, nbins=200, range=(1.5, 8.0))
        irdf.run()
        
        r, g = irdf.results.bins, irdf.results.rdf
        
        # Isolate First Peak
        peaks, _ = find_peaks(g, height=1.0, distance=10)
        if len(peaks) == 0: continue
        r_peak = r[peaks[0]]
        
        mask = (r > r_peak - 1.0) & (r < r_peak + 1.2)
        r_fit, g_fit = r[mask], g[mask]
        
        # Fit Dual Gaussian
        p0 = [g[peaks[0]], r_peak, 0.2, g[peaks[0]]*0.3, r_peak+0.5, 0.4, 0]
        try:
            popt, _ = curve_fit(dual_gaussian, r_fit, g_fit, p0=p0, bounds=(0, np.inf), maxfev=5000)
        except:
            print(f"Dual Gaussian fit failed for {cat}-{anion}. Using primary peak fallback.")
            popt = p0
            
        # Numerically locate the exact peak of the fitted Gaussian
        r_dense = np.linspace(r_peak - 0.5, r_peak + 0.5, 1000)
        g_dense = dual_gaussian(r_dense, *popt)
        r0_fit = r_dense[np.argmax(g_dense)]
        g0_fit = np.max(g_dense)
        
        # Calculate the second derivative (g'') via central difference
        dr = 1e-4
        g_plus = dual_gaussian(r0_fit + dr, *popt)
        g_minus = dual_gaussian(r0_fit - dr, *popt)
        d2g = (g_plus - 2*g0_fit + g_minus) / (dr**2)
        
        # PMF Stiffness: W''(r) = -k_B * T * (g'' / g)
        k_B = 1.380649e-23
        # Convert d2g from A^-2 to m^-2 (* 1e20)
        k_eff = -k_B * temp_k * (d2g / g0_fit) * 1e20  # N/m
        
        # Compute Fundamental Frequency (w_0)
        m_cat = atomic_masses[atomic_numbers[cat]]
        mu_kg = ((m_cat * m_anion) / (m_cat + m_anion)) * 1.660539e-27
        
        if k_eff > 0:
            omega_0_rad_s = np.sqrt(k_eff / mu_kg)
            omega_0_cm = omega_0_rad_s / (2 * np.pi * 29979245800.0)
            
            # Save BOTH the frequency and the fitted peak distance!
            omega0_dict[cat] = {
                'omega_0': omega_0_cm,
                'r_peak': r0_fit
            }
            print(f"[{cat}-{anion}] r_peak: {r0_fit:.2f} A | k_eff: {k_eff:.2f} N/m | omega_0: {omega_0_cm:.1f} cm-1")
        else:
            omega0_dict[cat] = {'omega_0': np.nan, 'r_peak': np.nan}

    return omega0_dict

# ====================================================================
# 1. COORDINATION NUMBER DISTRIBUTION (CND)
# ====================================================================
def plot_cnd(file_path, out_dir):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    cations = df['Cation'].unique()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8 / len(cations)
    
    # Group bars side-by-side if multiple cations exist
    for i, cat in enumerate(cations):
        sub = df[df['Cation'] == cat]['CN']
        counts = sub.value_counts(normalize=True).sort_index()
        
        # Offset x for grouped bars
        x = np.array(counts.index) + (i - len(cations)/2 + 0.5) * width
        ax.bar(x, counts.values, width=width, label=f'{cat}-Cl', alpha=0.7)
        ax.plot(counts.index, counts.values, marker='o', linestyle='-', linewidth=1.5, alpha=0.9)
        
    ax.set_xlabel('Coordination Number')
    ax.set_ylabel('Probability')
    
    min_c, max_c = int(df['CN'].min()), int(df['CN'].max())
    ax.set_xticks(range(min_c, max_c + 1))
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Plot_1_CND_Distribution.png'), dpi=300)
    plt.close()
    print("Saved CND Plot.")

# ====================================================================
# 2. BOND ANGLE DISTRIBUTION (BAD)
# ====================================================================
def plot_bad(file_path, out_dir):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    cations = df['Cation'].unique()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    for cat in cations:
        sub = df[df['Cation'] == cat]['Angle_deg']
        # KDE smoothly visualizes the probability density of angles
        sns.kdeplot(sub, ax=ax, label=f'Cl-{cat}-Cl', fill=True, bw_adjust=1.2, alpha=0.4)
        
    ax.set_xlabel('Bond Angle (Degrees)')
    ax.set_ylabel('Probability Density')
    ax.set_xlim(0, 180)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Plot_2_BAD_Distribution.png'), dpi=300)
    plt.close()
    print("Saved BAD Plot.")

# ====================================================================
# 3. DISPERSION RELATION (J_L HEATMAP) - Angular Freq (rad/ps) vs nm^-1
# ====================================================================
def plot_dispersion(file_path, out_dir, v_s_value=None, omega0_dict=None):
    if not os.path.exists(file_path): 
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    
    # 1. Convert Frequency from cm^-1 to angular frequency (rad/ps)
    # 1 cm^-1 = 0.029979 THz.  omega = 2 * pi * v
    freq_cm = df['Frequency_cm-1'].values
    freq_rad_ps = freq_cm * 0.0299792458 * 2 * np.pi
    
    q_cols = [c for c in df.columns if c.startswith('J_L_q')]
    
    # 2. Convert Wavenumber from A^-1 to nm^-1 (multiply by 10)
    q_vals_A = [float(col.split('mag')[-1]) for col in q_cols]
    q_vals_nm = [q * 10.0 for q in q_vals_A]
    
    Z = df[q_cols].values
    X, Y = np.meshgrid(q_vals_nm, freq_rad_ps)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot intensity heatmap
    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='Reds')
    
    # Extract and overlay actual dispersion peak curve
    peaks_rad_ps = freq_rad_ps[np.argmax(Z, axis=0)]
    ax.plot(q_vals_nm, peaks_rad_ps, color='black', marker='o', linestyle='--', 
            linewidth=2, label='MD Dispersion Peak')

    # ----------------------------------------------------------------
    # STRUCTURAL BOUNDARIES & ACOUSTIC LIMIT
    # ----------------------------------------------------------------
    max_q_line = 0
    colors = ['gray', 'dimgray', 'darkgray']
    
    if omega0_dict is not None:
        for idx, (cat, data) in enumerate(omega0_dict.items()):
            r_peak_A = data['r_peak']
            if not np.isnan(r_peak_A):
                # Convert Å to nm, then calculate Q = pi / r
                r_peak_nm = r_peak_A / 10.0
                q_peak = np.pi / r_peak_nm
                
                # Keep track of the highest Q to stop the Vs line
                if q_peak > max_q_line:
                    max_q_line = q_peak
                
                col_match = colors[idx % len(colors)]
                ax.axvline(q_peak, color=col_match, linestyle=':', linewidth=2, 
                           label=f'{cat}-Cl Peak ($Q={q_peak:.1f}$ nm$^{{-1}}$)')

    # 4. Overlay the theoretical Speed of Sound slope
    if v_s_value is not None and not np.isnan(v_s_value):
        # If we didn't find any peaks, default the line to stretch to Q=20
        domain_end = max_q_line if max_q_line > 0 else 20.0
        
        # Create a clean domain from 0 to the structural boundary
        v_s_q_domain = np.linspace(0, domain_end, 100)
        
        # Calculate expected angular frequencies
        # omega (rad/ps) = v_s (nm/ps) * Q (nm^-1)
        v_s_line = (v_s_value / 1000.0) * v_s_q_domain
        
        ax.plot(v_s_q_domain, v_s_line, color='#1f77b4', linestyle='-', 
                linewidth=2.5, label=f'Acoustic Limit ($V_s$ = {v_s_value:.0f} m/s)')
        
        ax.set_ylim(0, max(freq_rad_ps))

    # Update axis labels
    ax.set_xlabel(r'Wavenumber $Q$ (nm$^{-1}$)')
    ax.set_ylabel(r'Frequency $\omega$ (THz)') # Using THz label to match literature convention

    ax.set_ylim(0, 50)
    ax.set_xlim(0, 18)
    
    cbar = fig.colorbar(c, ax=ax, pad=0.02)
    cbar.set_label(r'Longitudinal Current Spectrum $J_L(Q, \omega)$', weight='bold')
    
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Plot_3_Dispersion_Map.png'), dpi=300)
    plt.close()
    print("Saved Dispersion Plot (Angular Freq vs nm^-1) with Structural Boundaries.")

# ====================================================================
# 4 & 5. PARTICIPATION RATIO AND VDOS (COMBINED PLOT)
# ====================================================================
def plot_pr_vdos(pr_file, vdos_file, cnd_file, out_dir, omega0_dict=None):
    if not os.path.exists(pr_file) or not os.path.exists(vdos_file): return
    
    # ---------------------------------------------------------
    # NEW: Calculate SPECIES-SPECIFIC PR Thresholds from CND
    # ---------------------------------------------------------
    species_thresholds = {}
    if os.path.exists(cnd_file):
        df_cnd = pd.read_csv(cnd_file)
        cations = df_cnd['Cation'].unique()
        
        for cat in cations:
            sub = df_cnd[df_cnd['Cation'] == cat]['CN']
            probs = sub.value_counts(normalize=True)
            cat_z_avg = np.sum(probs.index * probs.values)
            
            # Species-specific Ioffe-Regel Localization Threshold
            pr_thresh = 1.0 / (cat_z_avg + 1.0)
            species_thresholds[cat] = pr_thresh
            print(f"[{cat}] Z_avg = {cat_z_avg:.2f} -> PR Limit = {pr_thresh:.3f}")
    else:
        # Fallback if no CND file is found
        species_thresholds['Global Default'] = 0.20
    # ---------------------------------------------------------

    # Load PR Data
    df_pr = pd.read_csv(pr_file)
    
    def parse_freq(f):
        if isinstance(f, str):
            f = f.replace('i', 'j').replace(' ', '')
            try:
                c = complex(f)
                return -abs(c.imag) if c.imag != 0 else c.real
            except:
                return float(f)
        return float(f)
        
    df_pr['Freq_parsed'] = df_pr['Frequency_cm-1'].apply(parse_freq)
    
    # Load VDOS Data
    df_vdos = pd.read_csv(vdos_file)
    if 'Freq_rad_ps' in df_vdos.columns:
        df_vdos['Freq_cm-1'] = df_vdos['Freq_rad_ps'] * 5.3088
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    # --- Plot VDOS on Left Axis ---
    vdos_cols = [c for c in df_vdos.columns if c.startswith('VDOS_')]
    species_colors = {} 
    
    for col in vdos_cols:
        cat = col.split('_')[-1]
        p = ax1.plot(df_vdos['Freq_cm-1'], df_vdos[col], label=f'{cat} VDOS')
        c = p[0].get_color() 
        species_colors[cat] = c 
        ax1.fill_between(df_vdos['Freq_cm-1'], df_vdos[col], color=c, alpha=0.15)
        
    ax1.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
    ax1.set_ylabel(r'Density of States $D(\omega)$', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    # ---------------------------------------------------------
    # NEW: OVERLAY FUNDAMENTAL FREQUENCIES ON VDOS AXIS
    # ---------------------------------------------------------
    if omega0_dict is not None:
        for cat, data in omega0_dict.items():
            omega0 = data['omega_0']
            if not np.isnan(omega0):
                matched_color = species_colors.get(cat, 'black')
                ax1.axvline(omega0, color=matched_color, linestyle='-.', linewidth=2.0, alpha=0.8,
                            label=f'{cat}-Cl $\\omega_0$ ({omega0:.0f} cm$^{{-1}}$)')
    
    # --- Plot PR on Right Axis ---
    # --- Plot PR on Right Axis ---
    ax2 = ax1.twinx()
    mask_real = df_pr['Freq_parsed'] >= 0
    mask_imag = df_pr['Freq_parsed'] < 0

    # ---------------------------------------------------------
    # PLOT THE DYNAMIC THRESHOLD LINES FOR EACH SPECIES
    # ---------------------------------------------------------
    linestyles = ['--', '-.', ':']  # Different styles for variety
    for cat, thresh in species_thresholds.items():
        # Retrieve the matched color (fallback to 'gray' if not found)
        matched_color = species_colors.get(cat, 'gray') 
        
        ax2.axhline(thresh, color=matched_color, linestyle='--', linewidth=2.0, alpha=0.9, 
                    label=f'{cat} Localization ($PR={thresh:.2f}$)')
    
    ax2.scatter(df_pr.loc[mask_real, 'Freq_parsed'], df_pr.loc[mask_real, 'Participation_Ratio'], 
                color="#02080b", s=12, alpha=0.4, label='Real Modes', edgecolors='none')
    if mask_imag.any():
        ax2.scatter(df_pr.loc[mask_imag, 'Freq_parsed'], df_pr.loc[mask_imag, 'Participation_Ratio'], 
                    color='#d62728', marker='o', s=12, alpha=0.5, label='Imaginary Modes', edgecolors='none')
                    
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.5)
    
    ax2.set_ylabel('Participation Ratio $PR$')
    ax1.set_xlim(-100, 1000)
    ax1.set_ylim(0, 0.06)
    ax2.set_ylim(0, 1.0)
    
    # Consolidate Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Plot_4_5_PR_VDOS_Combined.png'), dpi=300)
    plt.close()
    print("Saved PR + VDOS Plot with omega_0 markers.")
 
# ====================================================================
# 6. TRANSPORT METRICS (MULTIPANEL)
# ====================================================================
def plot_transport(file_path, out_dir):
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    
    j_cols = [c for c in df.columns if c.startswith('J_')]
    has_j = len(j_cols) > 0
    
    # Creates 3 stacked plots if currents exist, else 2
    n_panels = 3 if has_j else 2
    fig, axs = plt.subplots(n_panels, 1, figsize=(8, 3*n_panels), sharex=True)
    
    # 1. Volume Fluctuation
    axs[0].plot(df['Step'], df['Volume_A3'], color='#2ca02c', linewidth=1.2)
    axs[0].set_ylabel(r'Volume ($\AA^3$)')
    axs[0].set_title('NVE Production Metrics')
    
    # 2. Stress Components (Checking for convergence/fluctuations)
    axs[1].plot(df['Step'], df['P_yz_eV_A3'], label=r'$P_{yz}$', alpha=0.8, linewidth=1)
    axs[1].plot(df['Step'], df['P_xz_eV_A3'], label=r'$P_{xz}$', alpha=0.8, linewidth=1)
    axs[1].plot(df['Step'], df['P_xy_eV_A3'], label=r'$P_{xy}$', alpha=0.8, linewidth=1)
    axs[1].set_ylabel(r'Off-Diag Stress (eV/$\AA^3$)')
    axs[1].legend(loc='upper right')
    
    # 3. Particle Current Magnitudes
    if has_j:
        species = list(set([c.split('_')[1] for c in j_cols]))
        for sp in species:
            jx, jy, jz = f'J_{sp}_x', f'J_{sp}_y', f'J_{sp}_z'
            if jx in df.columns:
                j_mag = np.sqrt(df[jx]**2 + df[jy]**2 + df[jz]**2)
                axs[2].plot(df['Step'], j_mag, label=f'{sp} Mass Current', alpha=0.7, linewidth=1)
        axs[2].set_ylabel(r'Current Magnitude')
        axs[2].set_xlabel('MD Step')
        axs[2].legend(loc='upper right')
    else:
        axs[1].set_xlabel('MD Step')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'Plot_6_Transport_Metrics.png'), dpi=300)
    plt.close()
    print("Saved Transport Plot.")


if __name__ == "__main__":
    import re
    
    # -------------------------------------------------------------
    # Shared Plotting Script
    # Usage: python plotting.py --system Output/0.5NaCl-0.5KCl_930K
    # Reads data from the system folder, and dumps plots into
    # Output/Plots/<system_folder>/
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Shared Plotting Script (runs on a system folder)")
    parser.add_argument('--system', type=str, required=True,
                        help='Path to the system folder, e.g. Output/0.5NaCl-0.5KCl_930K')
    args = parser.parse_args()

    SYSTEM_DIR = os.path.abspath(args.system)
    if not os.path.isdir(SYSTEM_DIR):
        raise ValueError(f"System folder '{SYSTEM_DIR}' does not exist.")

    # Run inside the system folder so all data I/O is local
    os.chdir(SYSTEM_DIR)

    # --- Auto-detect the prefix from the folder name ---
    PREFIX = os.path.basename(SYSTEM_DIR)  # e.g. "0.68KCl-0.32MgCl2_703.0K"
    print(f"Auto-detected prefix '{PREFIX}' from folder '{SYSTEM_DIR}'.")

    # -------------------------------------------------------------
    # RESOLVE OUTPUT PLOTS FOLDER (Output/Plots/<folder_name>/)
    # -------------------------------------------------------------
    plots_dir = os.path.join(os.path.dirname(SYSTEM_DIR), 'Plots', PREFIX)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. AUTO-DETECT CATIONS AND TEMPERATURE
    comp_str, temp_str = PREFIX.split('_')
    TEMP_K = float(temp_str.replace('K', ''))

    # Find all element symbols in the composition string, ignoring numbers and 'Cl'
    all_elements = re.findall(r'[A-Z][a-z]?', comp_str)
    CATIONS = list(set([el for el in all_elements if el != 'Cl']))
    print(f"\nAuto-detected Cations: {CATIONS} | Temperature: {TEMP_K} K")

    # -------------------------------------------------------------
    # HELPER: Build a glob pattern that matches both '930K' and '930.0K'
    # -------------------------------------------------------------
    def temp_glob(prefix, suffix=''):
        """Return a glob pattern matching both int and float temperature formats.
        e.g. temp_glob('Vs_0.5NaCl-0.5KCl_seed_*_', '.txt') matches
        'Vs_0.5NaCl-0.5KCl_seed_*_930K.txt' and '..._930.0K.txt'
        """
        temp_int = int(TEMP_K)
        # Match both '930K' and '930.0K' (and any other decimal variant like 930.00K)
        return f"{prefix}{temp_int}*K{suffix}"

    # 2. LOCATE Vs SEED FILES (handles both 930K and 930.0K)
    vs_pattern = temp_glob(f"Vs_{comp_str}_seed_*_", '.txt')
    vs_files = glob.glob(vs_pattern)

    avg_vs = None
    if vs_files:
        vs_values = []
        for vf in vs_files:
            with open(vf, 'r') as f:
                try:
                    val = float(f.read().strip())
                    if not np.isnan(val):
                        vs_values.append(val)
                except ValueError:
                    pass
        if vs_values:
            avg_vs = np.mean(vs_values)
            print(f"Averaged Speed of Sound ({len(vs_values)} seeds): {avg_vs:.2f} m/s")

    # 3. LOCATE DISPERSION FILE (Handles '703K' vs '703.0K' mismatch)
    disp_files = glob.glob(f"Dispersion_{comp_str}_*.csv")
    disp_file = disp_files[0] if disp_files else f"Dispersion_{PREFIX}.csv"

    # 4. EXTRACT PDF MATH & PEAKS (Do this BEFORE plotting!)
    # Use glob to find any matching NVE seed_42 file regardless of
    # whether the temperature is written as 738K or 738.0K
    extxyz_candidates = sorted(glob.glob(f"{comp_str}_*_NVE_seed_42.extxyz"))
    extxyz_file = extxyz_candidates[0] if extxyz_candidates else f"{PREFIX}_NVE_seed_42.extxyz"
    
    if os.path.exists(extxyz_file):
        omega0_results = extract_pdf_and_omega0(extxyz_file, temp_k=TEMP_K, cations=CATIONS)
    else:
        print(f"Trajectory {extxyz_file} not found. Skipping omega_0 calculation.")
        omega0_results = None

    # 5. EXECUTE PLOTS
    # CND file: use glob to handle both 930K and 930.0K formats
    cnd_files = glob.glob(f"CND_{comp_str}_*.csv")
    if cnd_files:
        plot_cnd(cnd_files[0], plots_dir)
    
    bad_files = glob.glob(f"BAD_{comp_str}_*.csv")
    if bad_files: plot_bad(bad_files[0], plots_dir)
    
    # PASS THE EXTRACTED RESULTS HERE!
    plot_dispersion(disp_file, plots_dir, v_s_value=avg_vs, omega0_dict=omega0_results)
    
    transport_files = glob.glob(f"Transport_{comp_str}_*.csv")
    if transport_files: plot_transport(transport_files[0], plots_dir)
        
    pr_files = glob.glob(f"PR_{comp_str}_*.csv")
    vdos_files = glob.glob(f"VDOS_{comp_str}_*.csv")
    
    if pr_files and vdos_files:
        plot_pr_vdos(pr_file=pr_files[0], 
                     vdos_file=vdos_files[0], 
                     cnd_file=cnd_files[0] if cnd_files else "", 
                     out_dir=plots_dir,
                     omega0_dict=omega0_results)
    
    print("\nAll plots generated successfully!")
