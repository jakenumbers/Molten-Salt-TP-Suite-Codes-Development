import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
# 1. COORDINATION NUMBER DISTRIBUTION (CND)
# ====================================================================
def plot_cnd(file_path):
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
    plt.savefig('Plot_1_CND_Distribution.png', dpi=300)
    plt.close()
    print("Saved CND Plot.")

# ====================================================================
# 2. BOND ANGLE DISTRIBUTION (BAD)
# ====================================================================
def plot_bad(file_path):
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
    plt.savefig('Plot_2_BAD_Distribution.png', dpi=300)
    plt.close()
    print("Saved BAD Plot.")

# ====================================================================
# 3. DISPERSION RELATION (J_L HEATMAP) - Angular Freq (rad/ps) vs nm^-1
# ====================================================================
def plot_dispersion(file_path, v_s_value=None):
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
    
    # 3. Define the real-space RDF peak distances in Angstroms
    # !! UPDATE THESE with the exact first-peak r-values from your RDF !!
    r_nacl_A = 2.70 
    r_kcl_A = 3.07  

    # Convert to nm
    r_nacl_nm = r_nacl_A / 10.0
    r_kcl_nm = r_kcl_A / 10.0

    # Calculate Wavenumber Q = pi / r (Derived from lambda_min = 2r)
    q_nacl = np.pi / r_nacl_nm
    q_kcl = np.pi / r_kcl_nm
    
    # Plot vertical dotted lines for the structural peaks
    ax.axvline(q_kcl, color='green', linestyle=':', linewidth=2, 
               label=f'K-Cl Peak ($Q={q_kcl:.1f}$ nm$^{{-1}}$)')
    ax.axvline(q_nacl, color='red', linestyle=':', linewidth=2, 
               label=f'Na-Cl Peak ($Q={q_nacl:.1f}$ nm$^{{-1}}$)')

    # 4. Overlay the theoretical Speed of Sound slope
    if v_s_value is not None and not np.isnan(v_s_value):
        # Limit the acoustic line to stop at the second (higher) dotted line
        max_q_line = max(q_nacl, q_kcl)
        
        # Create a clean domain from 0 to the structural boundary
        v_s_q_domain = np.linspace(0, max_q_line, 100)
        
        # Calculate expected angular frequencies
        # v_s in m/s divided by 1000 -> nm/ps
        # omega (rad/ps) = v_s (nm/ps) * Q (nm^-1)
        v_s_line = (v_s_value / 1000.0) * v_s_q_domain
        
        ax.plot(v_s_q_domain, v_s_line, color='#1f77b4', linestyle='-', 
                linewidth=2.5, label=f'Acoustic Limit ($V_s$ = {v_s_value:.0f} m/s)')
        
        # Keep the plot focused on the actual MD data bounds
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
    plt.savefig('Plot_3_Dispersion_Map.png', dpi=300)
    plt.close()
    print("Saved Dispersion Plot (Angular Freq vs nm^-1) with Structural Boundaries.")

# ====================================================================
# 4 & 5. PARTICIPATION RATIO AND VDOS (COMBINED PLOT)
# ====================================================================
def plot_pr_vdos(pr_file, vdos_file, cnd_file):
    if not os.path.exists(pr_file) or not os.path.exists(vdos_file): 
        print("Missing PR or VDOS file.")
        return
    
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
        
        # Plot the line and capture the object to extract its color
        p = ax1.plot(df_vdos['Freq_cm-1'], df_vdos[col], label=f'{cat} VDOS')
        c = p[0].get_color() 
        species_colors[cat] = c # Save it for later!
        
        # Fill under the curve using that exact color
        ax1.fill_between(df_vdos['Freq_cm-1'], df_vdos[col], color=c, alpha=0.15)
        
    ax1.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)')
    ax1.set_ylabel(r'Density of States $D(\omega)$', color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')
    
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
        
        ax2.axhline(thresh, color=matched_color, linestyle=linestyles[list(species_thresholds.keys()).index(cat) % len(linestyles)], linewidth=2.0, alpha=0.9, 
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
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('Plot_4_5_PR_VDOS_Combined.png', dpi=300)
    plt.close()
    print("Saved PR + VDOS Plot with Species-Specific Limits.")
 
# ====================================================================
# 6. TRANSPORT METRICS (MULTIPANEL)
# ====================================================================
def plot_transport(file_path):
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
    plt.savefig('Plot_6_Transport_Metrics.png', dpi=300)
    plt.close()
    print("Saved Transport Plot.")


if __name__ == "__main__":
    import glob
    
    # --- Edit these prefixes to match your files ---
    PREFIX = "0.5NaCl-0.5KCl_1130K"
    
    # 1. Parse the prefix to locate the matching Vs seed files
    try:
        comp_str, temp_str = PREFIX.split('_')
        vs_pattern = f"Vs_{comp_str}_seed_*_{temp_str}.txt"
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
                print(f"Averaged Speed of Sound from {len(vs_values)} seed files: {avg_vs:.2f} m/s")
        else:
            print(f"No Vs files found matching pattern: {vs_pattern}. Slope will not be plotted.")
    except Exception as e:
        print(f"Could not parse Vs files: {e}")
        avg_vs = None
        
    # 2. Execute all plotters
    plot_cnd(f"CND_{PREFIX}.csv")
    plot_bad(f"BAD_{PREFIX}.csv")
    
    # Pass the calculated avg_vs to the dispersion plotter!
    plot_dispersion(f"Dispersion_{PREFIX}.csv", v_s_value=avg_vs)
    
    plot_pr_vdos(pr_file=f"PR_{PREFIX}.csv", vdos_file=f"VDOS_{PREFIX}.csv", cnd_file=f"CND_{PREFIX}.csv")
    plot_transport(f"Transport_{PREFIX}.csv")
    
    print("\nAll plots generated successfully!")