"""
PDF Comparison Plot
===================
Plots the partial pair distribution functions g(r) for selected
cation-anion pairs across multiple molten salt compositions.

Shows how the same ion pair (e.g., Li-F) has different peak positions,
heights, and widths depending on the mixture environment.

Requires the PDF CSV files in ../PDF_Analysis/PDF_CSV/ or ./PDF_CSV/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# =============================================================================
# 0. PLOT FORMATTING
# =============================================================================
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman"],
    "mathtext.fontset":     "cm",
    "axes.labelsize":       12,
    "xtick.labelsize":      11,
    "ytick.labelsize":      11,
    "legend.fontsize":       10,
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
})


# =============================================================================
# 1. FILE LOADING
# =============================================================================

def find_pdf_file(filename: str) -> str:
    """Locate a PDF CSV file by searching common relative paths."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    search_paths = [
        os.path.join(script_dir, '..', 'PDF_Analysis', 'PDF_CSV', filename),
        os.path.join(script_dir, 'PDF_CSV', filename),
        os.path.join(script_dir, filename),
        filename,
    ]
    for path in search_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Could not find PDF file '{filename}'.")


def load_pdf_pair(filename: str, ion_pair_str: str):
    """
    Load r and g(r) for a specific ion pair from a PDF CSV.

    Returns (r_angstrom, g_r) arrays, or (None, None) if not found.
    """
    full_path = find_pdf_file(filename)

    with open(full_path, 'r') as f:
        first_line = f.readline().strip()
    pair_names = [name.strip() for name in first_line.split(',')
                  if name.strip() and "Unnamed" not in name]

    df = pd.read_csv(full_path, header=1)

    # Build column names
    new_cols = []
    pair_idx = 0
    for i in range(0, len(df.columns), 2):
        if pair_idx < len(pair_names):
            pair = pair_names[pair_idx]
            new_cols.extend([f"r_{pair}", f"rdf_{pair}"])
            pair_idx += 1
        else:
            new_cols.extend([f"col_{i}", f"col_{i+1}"])
    df.columns = new_cols[:len(df.columns)]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    r_col = f"r_{ion_pair_str}"
    g_col = f"rdf_{ion_pair_str}"

    if r_col not in df.columns or g_col not in df.columns:
        print(f"  Warning: '{ion_pair_str}' not found in {filename}")
        print(f"  Available pairs: {pair_names}")
        return None, None

    sub = df[[r_col, g_col]].dropna()
    return sub[r_col].values, sub[g_col].values


# =============================================================================
# 2. PLOT CONFIGURATION
# =============================================================================

# Each entry: (pdf_filename, ion_pair_str, label, color, linestyle, linewidth)
PLOT_CURVES = [
    # Li-F family (blue)
    {
        'file': '0.465LiF-0.115NaF-0.42KF_Frandsen_2020_873.0_AP.csv',
        'pair': 'Li-F',
        'label': 'Li-F (FLiNaK)',
        'color': '#1f77b4',     # blue
        'linestyle': '-',
        'linewidth': 1.8,
    },
    {
        'file': '0.66LiF-0.34BeF2_Fayfar_2024_973.0_AP.csv',
        'pair': 'Li-F',
        'label': 'Li-F (FLiBe)',
        'color': '#1f77b4',
        'linestyle': '--',
        'linewidth': 1.8,
    },
    {
        'file': 'LiF_Walz_2019_1121.0_PIM.csv',
        'pair': 'Li-F',
        'label': 'Li-F (LiF)',
        'color': '#1f77b4',
        'linestyle': '-.',
        'linewidth': 1.8,
    },
    # Na-F family (orange)
    {
        'file': '0.465LiF-0.115NaF-0.42KF_Frandsen_2020_873.0_AP.csv',
        'pair': 'Na-F',
        'label': 'Na-F (FLiNaK)',
        'color': '#ff7f0e',     # orange
        'linestyle': '-',
        'linewidth': 1.8,
    },
    {
        'file': 'NaF_Walz_2019_1266.0_PIM.csv',
        'pair': 'Na-F',
        'label': 'Na-F (NaF)',
        'color': '#ff7f0e',
        'linestyle': ':',
        'linewidth': 2.0,
    },
]


# =============================================================================
# 3. MAIN PLOT
# =============================================================================

def make_pdf_comparison(r_max=6.0):
    """Generate the PDF comparison figure."""

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    for curve in PLOT_CURVES:
        try:
            r, g = load_pdf_pair(curve['file'], curve['pair'])
        except FileNotFoundError as e:
            print(f"  Skipping {curve['label']}: {e}")
            continue

        if r is None:
            continue

        # Trim to r_max
        mask = r <= r_max
        ax.plot(r[mask], g[mask],
                color=curve['color'],
                linestyle=curve['linestyle'],
                linewidth=curve['linewidth'],
                label=curve['label'])

    ax.set_xlabel('Distance (\\AA)')
    ax.set_ylabel('$g(r)$')
    ax.set_xlim(1.0, r_max)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='lightgray')

    # Save
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    out_dir = os.path.join(script_dir, 'Figures')
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, 'PDF_comparison_LiF_NaF.pdf')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")

    plt.show()
    return fig


if __name__ == '__main__':
    make_pdf_comparison()
