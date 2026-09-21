import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Element properties for sorting: (Group, Period)
ELEMENT_PROPS = {
    'Li': (1, 2), 'Na': (1, 3), 'K': (1, 4), 'Rb': (1, 5), 'Cs': (1, 6),
    'Be': (2, 2), 'Mg': (2, 3), 'Ca': (2, 4), 'Sr': (2, 5), 'Ba': (2, 6),
    'Al': (13, 3), 'Zr': (4, 5), 
    'U': (99, 7), 'Th': (99, 7),
    'F': (17, 2), 'Cl': (17, 3)
}

def get_sort_key(comp_str):
    """
    Creates a sort key based on:
    1. Has Actinides (forces U and Th to the end)
    2. Complexity (number of distinct cations)
    3. Group of the cations
    4. Period of the cations
    """
    has_actinide = 1 if ('U' in comp_str or 'Th' in comp_str) else 0
    
    cations = re.findall(r'([A-Z][a-z]?)(?=F|Cl)', comp_str)
    if not cations:
        cations = [e for e in ELEMENT_PROPS.keys() if e in comp_str]
        
    cations = list(dict.fromkeys(cations))
    complexity = len(cations)
    
    props = sorted([ELEMENT_PROPS.get(c, (100, 100)) for c in cations])
    
    # First sort factor is actinide presence, so they always go last
    flat_props = [has_actinide, complexity]
    for p in props:
        flat_props.extend(p)
        
    return tuple(flat_props)

def format_label(label):
    """
    Formats labels to use LaTeX subscripts for stoichiometry.
    Example: 0.5MgCl2-0.5KCl -> 0.5MgCl$_{2}$-0.5KCl
    """
    return re.sub(r'([a-zA-Z])(\d+)', r'\1$_{\2}$', label)

def plot_all_bph_variations(csv_file="SCL_results.csv"):
    """
    Reads the combined SCL results and plots an ablation study for each b_PH variation.
    """
    # ---------------------------------------------------------
    # Set up plot style to match existing plots
    # ---------------------------------------------------------
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

    if not os.path.exists(csv_file):
        print(f"Could not find the CSV file: {csv_file}")
        return

    # Load dataset
    df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Drop rows with missing or 0 experimental boundaries
    exp_col = 'Experimental SCL (A)'
    df = df.dropna(subset=[exp_col])
    df = df[df[exp_col] != 0.0].copy()
    
    if df.empty:
        print("No valid experimental data found in CSV to plot against.")
        return

    # Calculate baseline (bPH = 0) deviation
    df['err_0'] = (df['Avg SCL_0 (A)'] - df[exp_col]) / df[exp_col] * 100

    # Define the 4 variations we want to plot
    variations = {
        '1': 'Nominal ($x_j$)',
        'Lc': 'Classical VDOS ($L_c$)',
        'Lq': 'Quantum VDOS ($L_q$)',
        'Oa': 'Anion VDOS ($O_a$)'
    }

    for var_key, var_title in variations.items():
        col_name = f'Avg SCL_{var_key} (A)'
        if col_name not in df.columns:
            print(f"Column {col_name} not found, skipping {var_key} plot.")
            continue
            
        # Calculate variation deviation
        df[f'err_{var_key}'] = (df[col_name] - df[exp_col]) / df[exp_col] * 100
        
        # Apply custom periodic table & actinide sorting
        df_plot = df.copy()
        df_plot['Sort_Key'] = df_plot['Composition'].apply(get_sort_key)
        df_plot = df_plot.sort_values(by=['Sort_Key', f'err_{var_key}'])
        
        raw_labels = df_plot['Composition'].values
        labels = [format_label(l) for l in raw_labels]
        
        bph_errors = df_plot[f'err_{var_key}'].values
        no_bph_errors = df_plot['err_0'].values
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # --- ADDED: 15% Error Band and Bold Zero Line ---
        ax.axhspan(-15, 15, color='#D0F0C0', zorder=1, label=r'$\pm 15\%$ Error Band')
        ax.axhline(0, color='black', linewidth=1.2, zorder=1)
        # ------------------------------------------------

        for i, raw_label in enumerate(raw_labels):
            family = 'Fluoride' if 'F' in raw_label and 'Cl' not in raw_label else 'Chloride'
            complexity = get_sort_key(raw_label)[1] # Complexity is the 2nd item in the sort key tuple
            
            # UNARY SALTS (Only plot the "With bPH" variant, centered)
            if complexity == 1:
                if family == 'Fluoride':
                    ax.bar(x[i], bph_errors[i], width, color='tab:blue', edgecolor='white', hatch='///', zorder=3)
                    ax.bar(x[i], bph_errors[i], width, color='none', edgecolor='black', linewidth=1.1, zorder=4)
                else:
                    ax.bar(x[i], bph_errors[i], width, color='tab:blue', edgecolor='black', linewidth=1.1, zorder=3)
            
            # MIXTURES (Plot Grouped Bars)
            else:
                pos_bph = x[i] - width/2
                pos_nobph = x[i] + width/2
                
                if family == 'Fluoride':
                    # With bPH (Blue base, White Hatch, Black Edge)
                    ax.bar(pos_bph, bph_errors[i], width, color='tab:blue', edgecolor='white', hatch='///', zorder=3)
                    ax.bar(pos_bph, bph_errors[i], width, color='none', edgecolor='black', linewidth=1.1, zorder=4)
                    
                    # No bPH (Red base, White Hatch, Black Edge)
                    ax.bar(pos_nobph, no_bph_errors[i], width, color='tab:red', edgecolor='white', hatch='///', zorder=3)
                    ax.bar(pos_nobph, no_bph_errors[i], width, color='none', edgecolor='black', linewidth=1.1, zorder=4)
                else:
                    # Solid Blue
                    ax.bar(pos_bph, bph_errors[i], width, color='tab:blue', edgecolor='black', linewidth=1.1, zorder=3)
                    # Solid Red
                    ax.bar(pos_nobph, no_bph_errors[i], width, color='tab:red', edgecolor='black', linewidth=1.1, zorder=3)
        
        # Formatting Y-Axis
        max_err = max(np.max(np.abs(bph_errors)), np.max(np.abs(no_bph_errors)))
        y_lim = max(max_err * 1.15, 20)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_ylabel(r'$\ell_{{\mathrm{{sc}}}}$ Deviation (%)')
        ax.set_title(f'SCM Accuracy: Variation [{var_title}] vs [$b_{{PH}}=0$]\n(Actinides Grouped at Right)', pad=15)
        
        # Formatting X-Axis
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # ---------------------------------------------------------
        # Group Dividers and Average Bias
        # ---------------------------------------------------------
        def assign_group(comp):
            has_act, comp_len = get_sort_key(comp)[:2]
            if has_act == 1: return "Actinide"
            is_f = 'F' in comp and 'Cl' not in comp
            if comp_len == 1 and is_f: return "Fluoride\nUnaries"
            if comp_len == 1 and not is_f: return "Chloride\nUnaries"
            if comp_len > 1 and is_f: return "Fluoride\nMixtures"
            return "Chloride\nMixtures"

        df_plot['Group'] = df_plot['Composition'].apply(assign_group)
        groups = df_plot['Group'].unique()
        
        for idx, g in enumerate(groups):
            group_mask = df_plot['Group'] == g
            group_indices = np.where(group_mask)[0]
            if len(group_indices) == 0: continue
            
            x_min = group_indices[0] - 0.5
            x_max = group_indices[-1] + 0.5
            x_mid = (x_min + x_max) / 2
            
            group_df = df_plot[group_mask]
            
            # Calculate and draw Average Bias for w/ bPH
            mean_bph = group_df[f'err_{var_key}'].mean()
            ax.hlines(mean_bph, xmin=x_min, xmax=x_max, color='tab:blue', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
            
            # Calculate and draw Average Bias for pen bPH (Only for mixtures)
            mixtures_mask = group_df['Composition'].apply(lambda c: get_sort_key(c)[1] > 1)
            if mixtures_mask.any():
                mean_nobph = group_df.loc[mixtures_mask, 'err_0'].mean()
                ax.hlines(mean_nobph, xmin=x_min, xmax=x_max, color='tab:red', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
                
                # Plot Text (offsetting to prevent overlap)
                y_offset = -y_lim * 0.025 if (mean_bph - mean_nobph) > (-y_lim * 0.05) else y_lim * 0.025
                va_align = 'top' if y_offset < 0 else 'bottom'
                ax.text(x_max - 0.1, mean_nobph + y_offset, f'{mean_nobph:.1f}%', color='tab:red', fontsize=10, ha='right', va=va_align, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
            
            # Draw bPH mean text after to ensure it overlays cleanly
            ax.text(x_max - 0.1, mean_bph + (y_lim * 0.025), f'{mean_bph:.1f}%', color='tab:blue', fontsize=10, ha='right', va='bottom', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))

            # Add cleanly formatted Group labels at the BOTTOM
            ax.text(x_mid, -y_lim + (y_lim * 0.05), g, ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=5)
            
            # Draw vertical dotted line separators
            if idx < len(groups) - 1:
                ax.axvline(x_max, ymin=0, ymax=1, color='gray', linestyle=':', linewidth=1.5, zorder=0, clip_on=False)
                ax.plot([x_max, x_max], [-y_lim, -y_lim + (y_lim*0.2)], color='gray', linestyle=':', linewidth=1.5, clip_on=False)
                
        # ---------------------------------------------------------
        
        ax.grid(axis='y', linestyle='--', color='#E0E0E0', zorder=0)

        # Create Custom Legend
        custom_legend = [
            Patch(facecolor='tab:blue', edgecolor='black', label=r'With $b_{PH}$'),
            Patch(facecolor='tab:red', edgecolor='black', label=r'No $b_{PH}$'),
            Patch(facecolor='gray', edgecolor='black', label='Chloride'),
            Patch(facecolor='gray', edgecolor='white', hatch='///', label='Fluoride'),
            Line2D([0], [0], color='tab:blue', lw=2, linestyle='-.', label=r'Avg Bias (With $b_{PH}$)'),
            Line2D([0], [0], color='tab:red', lw=2, linestyle='-.', label=r'Avg Bias (No $b_{PH}$)')
        ]
        
        ax.legend(handles=custom_legend, loc='upper left', ncol=2, frameon=True, facecolor='white', edgecolor='none')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20) 
        
        output_filename = f'Structural_Coherence_Length/SCL_Summary_Deviation_bPH_{var_key}.png'
        os.makedirs('Structural_Coherence_Length', exist_ok=True)
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Saved comparison plot to {output_filename}")
        plt.close(fig)

if __name__ == "__main__":
    plot_all_bph_variations()