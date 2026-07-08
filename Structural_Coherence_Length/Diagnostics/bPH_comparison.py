import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib.lines import Line2D

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
    Analyzes the composition string to extract properties like complexity
    and actinide presence.
    """
    has_actinide = 1 if ('U' in comp_str or 'Th' in comp_str) else 0
    
    cations = re.findall(r'([A-Z][a-z]?)(?=F|Cl)', comp_str)
    if not cations:
        cations = [e for e in ELEMENT_PROPS.keys() if e in comp_str]

    anions = re.findall(r'(F|Cl)', comp_str)
    if not anions:
        anions = ['F'] if 'F' in comp_str else ['Cl'] if 'Cl' in comp_str else []
        
    cations = list(dict.fromkeys(cations))
    anions = list(dict.fromkeys(anions))
    complexity = len(cations)
    
    props_c = sorted([ELEMENT_PROPS.get(c, (100, 100)) for c in cations])
    props_a = sorted([ELEMENT_PROPS.get(a, (100, 100)) for a in anions])
    
    return (has_actinide, complexity) + tuple(
        item for gp in props_a + props_c for item in gp
    )

def format_label(label):
    """
    Formats labels to use LaTeX subscripts for stoichiometry.
    Example: 0.5MgCl2-0.5KCl -> 0.5MgCl$_{2}$-0.5KCl
    """
    return re.sub(r'([a-zA-Z])(\d+)', r'\1$_{\2}$', label)

def plot_bph_ablation_study(csv_with_bph="Structural_Coherence_Length/Diagnostics/SCL_results_with_bPH.csv", csv_no_bph="Structural_Coherence_Length/Diagnostics/SCL_results_pen_bPH.csv"):
    """
    Plots a grouped bar chart comparing the model deviation with and without b_PH.
    """
    # ---------------------------------------------------------
    # Set up plot style to match existing plots
    # ---------------------------------------------------------
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'], 'font.size': 12,
        'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.linewidth': 1.5,
        'xtick.labelsize': 11, 'ytick.labelsize': 12,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': 1.75, 'ytick.major.width': 1.75,
        'legend.frameon': False, 'legend.fontsize': 11,
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Helvetica',
        'mathtext.it': 'Helvetica:italic', 'mathtext.bf': 'Helvetica:bold',
    })

    if not os.path.exists(csv_with_bph) or not os.path.exists(csv_no_bph):
        print(f"Could not find the CSV files: {csv_with_bph} and/or {csv_no_bph}")
        return

    # Load datasets[cite: 4]
    df_bph = pd.read_csv(csv_with_bph, encoding='cp1252')
    
    # Store the exact order of the provided CSV[cite: 4]
    df_bph['original_order'] = range(len(df_bph))
    
    df_no_bph = pd.read_csv(csv_no_bph, encoding='cp1252')
    
    comp_col = 'Composition'
    err_col = 'Deviation'
    exp_col = 'Experimental SCL (A)_with'
    
    # Merge datasets
    df_merged = pd.merge(df_bph, df_no_bph, on=comp_col, suffixes=('_with', '_none'))
    
    # Re-apply the original CSV order immediately after merge
    df_merged = df_merged.sort_values('original_order')
    
    # Drop rows with missing deviations or 0 experimental boundaries[cite: 4]
    df_merged = df_merged.dropna(subset=[f'{err_col}_with', f'{err_col}_none'])
    if exp_col in df_merged.columns:
        df_merged = df_merged[df_merged[exp_col] != 0.0]
        
    raw_labels = df_merged[comp_col].values
    labels = [format_label(l) for l in raw_labels]
    
    # Preserve negative deviations
    bph_errors = df_merged[f'{err_col}_with'].values
    no_bph_errors = df_merged[f'{err_col}_none'].values
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # --- ADDED: 15% Error Band and Bold Zero Line ---
    ax.axhspan(-15, 15, color='#D0F0C0', zorder=1, label=r'$\pm 15\%$ Error Band')
    ax.axhline(0, color='black', linewidth=1.2, zorder=1)
    # ------------------------------------------------
    
    # Track standard bars to avoid legend duplication
    plotted_unary = False
    plotted_mixture_bph = False
    plotted_mixture_nobph = False

    for i, raw_label in enumerate(raw_labels):
        complexity = get_sort_key(raw_label)[1] # Complexity is the 2nd item in the sort key tuple
        
        lbl_bph = r"w/ $b_{PH}$"
        lbl_nobph = r"pen $b_{PH}$"
        
        if complexity == 1:
            # UNARY SALTS: Plot a single bar centered on the tick mark
            _lbl = lbl_bph if not plotted_unary else ""
            ax.bar(x[i], bph_errors[i], width, 
                   color='tab:blue', edgecolor='black', linewidth=1.1, zorder=3,
                   label=_lbl)
            plotted_unary = True
        else:
            # MIXTURES: Plot the grouped/side-by-side bars
            # With bPH (left bar) - blue[cite: 4]
            _lbl1 = lbl_bph if not plotted_mixture_bph else ""
            ax.bar(x[i] - width/2, bph_errors[i], width, 
                   color='tab:blue', edgecolor='black', linewidth=1.1, zorder=3,
                   label=_lbl1)
            plotted_mixture_bph = True
            
            # No bPH (right bar) - red[cite: 4]
            _lbl2 = lbl_nobph if not plotted_mixture_nobph else ""
            ax.bar(x[i] + width/2, no_bph_errors[i], width, 
                   color='tab:red', edgecolor='black', linewidth=1.1, zorder=3,
                   label=_lbl2)
            plotted_mixture_nobph = True
    
    # Formatting Y-Axis
    ax.set_ylim(-65, 110)
    ax.set_ylabel(r'$\ell_{{\mathrm{{sc}}}}$ Deviation (%)')
    
    # Formatting X-Axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # ---------------------------------------------------------
    # Group Dividers and Average Bias
    # ---------------------------------------------------------
    def assign_group(comp):
        """Maps the composition to the exact 5 groups naturally[cite: 4]"""
        has_act, comp_len = get_sort_key(comp)[:2]
        if has_act == 1: return "Actinide"
        is_f = 'F' in comp and 'Cl' not in comp
        if comp_len == 1 and is_f: return "Fluoride\nUnaries"
        if comp_len == 1 and not is_f: return "Chloride\nUnaries"
        if comp_len > 1 and is_f: return "Fluoride\nMixtures"
        return "Chloride\nMixtures"

    df_merged['Group'] = df_merged[comp_col].apply(assign_group)
    groups = df_merged['Group'].unique()
    
    for idx, g in enumerate(groups):
        group_mask = df_merged['Group'] == g
        group_indices = np.where(group_mask)[0]
        if len(group_indices) == 0: continue
        
        x_min = group_indices[0] - 0.5
        x_max = group_indices[-1] + 0.5
        x_mid = (x_min + x_max) / 2
        
        group_df = df_merged[group_mask]
        
        # Calculate and draw Average Bias for w/ bPH
        mean_bph = group_df[f'{err_col}_with'].mean()
        if g == "Actinide":
            ax.hlines(mean_bph, xmin=x_min, xmax=x_max+0.2, color='tab:blue', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
            ax.text(x_max + 0.9, mean_bph + 1.5, f'{mean_bph:.1f}%', color='tab:blue', fontsize=10, ha='right', va='bottom', fontweight='bold')
        elif g == "Fluoride\nUnaries":
            ax.hlines(mean_bph, xmin=x_min, xmax=x_max, color='tab:blue', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
            ax.text(x_max - 0.1, mean_bph - 8, f'{mean_bph:.1f}%', color='tab:blue', fontsize=10, ha='right', va='bottom', fontweight='bold')
        elif g == "Fluoride\nMixtures":
            ax.hlines(mean_bph, xmin=x_min, xmax=x_max, color='tab:blue', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
            ax.text(x_max - 1.1, mean_bph - 8, f'{mean_bph:.1f}%', color='tab:blue', fontsize=10, ha='right', va='bottom', fontweight='bold')
        elif g == "Chloride\nMixtures":
            ax.hlines(mean_bph, xmin=x_min, xmax=x_max, color='tab:blue', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
            ax.text(x_max - 0.1, mean_bph - 10, f'{mean_bph:.1f}%', color='tab:blue', fontsize=10, ha='right', va='bottom', fontweight='bold')        
        else:
            ax.hlines(mean_bph, xmin=x_min, xmax=x_max, color='tab:blue', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
            ax.text(x_max - 0.1, mean_bph + 1.5, f'{mean_bph:.1f}%', color='tab:blue', fontsize=10, ha='right', va='bottom', fontweight='bold')
        
        # Calculate and draw Average Bias for pen bPH (Only for mixtures)
        mixtures_mask = group_df[comp_col].apply(lambda c: get_sort_key(c)[1] > 1)
        if mixtures_mask.any():
            if g == "Actinide":
                mean_nobph = group_df.loc[mixtures_mask, f'{err_col}_none'].mean()
                ax.hlines(mean_nobph, xmin=x_min, xmax=x_max+0.2, color='tab:red', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
                ax.text(x_max - 0.1, mean_nobph + 1.5, f'{mean_nobph:.1f}%', color='tab:red', fontsize=10, ha='right', va='bottom', fontweight='bold')
            else:
                mean_nobph = group_df.loc[mixtures_mask, f'{err_col}_none'].mean()
                ax.hlines(mean_nobph, xmin=x_min, xmax=x_max, color='tab:red', linestyle='-.', linewidth=2, alpha=0.9, zorder=4)
            
                # Offset text slightly to prevent overlap if means are close
                y_offset = -1.5 if (mean_bph - mean_nobph) > -4 else 1.5
                va_align = 'top' if y_offset < 0 else 'bottom'
                ax.text(x_max - 0.1, mean_nobph + y_offset, f'{mean_nobph:.1f}%', color='tab:red', fontsize=10, ha='right', va=va_align, fontweight='bold')
        
        # Add cleanly formatted Group labels at the BOTTOM
        if g == "Actinide":
            ax.text(x_mid + 0.55, -48, g, ha='center', va='top', fontsize=11, fontweight='bold', zorder=5)
        else:
            ax.text(x_mid, -48, g, ha='center', va='top', fontsize=11, fontweight='bold', zorder=5)
        
        # Draw vertical dotted line separators that extend past the x-axis down to the labels
        if idx < len(groups) - 1:
            ax.axvline(x_max, ymin=0, ymax=1, color='gray', linestyle=':', linewidth=1.5, zorder=0, clip_on=False)
            ax.plot([x_max, x_max], [-65, -40], color='gray', linestyle=':', linewidth=1.5, clip_on=False) # Extends divider down
            
    # ---------------------------------------------------------
    
    ax.grid(axis='y', linestyle='--', color='#E0E0E0', zorder=0)

    # Clean up the legend and add custom bias lines
    handles, legend_labels = ax.get_legend_handles_labels()
    
    custom_lines = [
        Line2D([0], [0], color='tab:blue', lw=2, linestyle='-.', label=r'Avg Bias (w/ $b_{PH}$)'),
        Line2D([0], [0], color='tab:red', lw=2, linestyle='-.', label=r'Avg Bias (pen $b_{PH}$)')
    ]
    
    # Filter out empty string labels
    valid_handles_labels = [(h, l) for h, l in zip(handles, legend_labels) if l != ""]
    
    if valid_handles_labels:
        handles, legend_labels = zip(*valid_handles_labels)
        handles = list(handles) + custom_lines
        legend_labels = list(legend_labels) + [h.get_label() for h in custom_lines]
        
        # Deduplicate final legend list
        final_handles, final_labels = [], []
        for h, l in zip(handles, legend_labels):
            if l not in final_labels:
                final_labels.append(l)
                final_handles.append(h)
                
        # Positioned cleanly out of the way in the upper left, 2 columns wide
        ax.legend(final_handles, final_labels, loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    # Pushes the bottom margin up slightly so the group labels aren't cut off
    plt.subplots_adjust(bottom=0.25) 
    
    output_filename = 'SCL_Summary_Deviation_bPH_Comparison_Sorted.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Saved comparison plot to {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_bph_ablation_study()