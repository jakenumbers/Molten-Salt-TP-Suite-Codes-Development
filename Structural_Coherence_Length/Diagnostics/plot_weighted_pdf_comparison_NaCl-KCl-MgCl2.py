"""
plot_weighted_pdf_comparison.py — Weighted partial PDF + SCL comparison plot
for a single dataset analyzed with and without b_PH (concentration-only vs b_PH=0).

Solid lines  → with b_PH
Dotted lines → b_PH = 0

This is a self-contained script. Place it in the same folder as a prepared PDF CSV
and run.

Required files in the same folder:
  - This script
  - A prepared PDF CSV file
  - Python packages: numpy, pandas, matplotlib, scipy, mendeleev

Usage:
    python plot_weighted_pdf_comparison.py
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from mendeleev import element


# ==========================================
# Inlined utility functions (from scl_utils)
# ==========================================

def standardize_ion_pair(ion_pair):
    """Standardize ion pair strings to cation-anion order (e.g., 'K-Cl' not 'Cl-K')."""
    if not isinstance(ion_pair, str):
        ion_pair = str(ion_pair)
    parts = ion_pair.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid ion pair format: {ion_pair}")

    def extract_elements(compound):
        elements = re.findall(r'([A-Z][a-z]?\d*)', compound)
        return [re.sub(r'\d', '', el) for el in elements]

    elems1 = extract_elements(parts[0])
    elems2 = extract_elements(parts[1])

    if not elems1 or not elems2:
        raise ValueError(f"Could not extract elements from: {ion_pair}")

    el1, el2 = element(elems1[0]), element(elems2[0])
    state1 = el1.oxistates[0] if el1.oxistates else 0
    state2 = el2.oxistates[0] if el2.oxistates else 0

    if state1 > 0 and state2 < 0:
        return f"{elems1[0]}-{elems2[0]}"
    elif state1 < 0 and state2 > 0:
        return f"{elems2[0]}-{elems1[0]}"
    else:
        return '-'.join(sorted([elems1[0], elems2[0]]))


def parse_composition(comp_str):
    """Parse composition string into (fractions, ion_counts, sorted_comp_str)."""
    fractions = {}
    components = comp_str.split('-')
    for comp in components:
        match = re.match(r"([0-9.]+)?([A-Za-z0-9]+)", comp)
        if match:
            frac_str, salt = match.groups()
            frac = float(frac_str) if frac_str else 1.0
            fractions[salt] = frac

    ion_counts = {}
    all_salts_matches = re.findall(r'([0-9.]*)([A-Z][a-z]?\d*[A-Z]?[a-z]?\d*)', comp_str)
    for _, salt in all_salts_matches:
        if not salt:
            continue
        elements = re.findall(r'([A-Z][a-z]?)([0-9]*)', salt)
        i_counts = {}
        for el, count in elements:
            cnt = int(count) if count else 1
            i_counts[el] = i_counts.get(el, 0) + cnt
        ion_counts[salt] = i_counts

    def get_cation_atomic_number(s):
        try:
            elems = re.findall(r'([A-Z][a-z]?)', s)
            if elems:
                return element(elems[0]).atomic_number
        except Exception:
            pass
        return 999

    sorted_salts = sorted(fractions.keys(), key=get_cation_atomic_number)
    sorted_comp_str = '-'.join([f"{fractions[s]}{s}" for s in sorted_salts])

    return fractions, ion_counts, sorted_comp_str


# ==========================================
# IonPairData (from SCL_calc.py)
# ==========================================

class IonPairData:
    """Stores PDF data for a single ion pair on the common grid."""

    def __init__(self, name, x_grid, y_grid, weight):
        self.name = name
        self.x = x_grid
        self.y = y_grid
        self.weight = weight
        self.spline = interp1d(x_grid, y_grid, kind='linear', bounds_error=False, fill_value=0)
        self.type = self._determine_type()
        self.peak = (None, None)
        self.minima = (None, None)

    def _determine_type(self):
        try:
            parts = self.name.split('-')
            el1, el2 = element(parts[0]), element(parts[1])
            s1 = el1.oxistates[0] if el1.oxistates else 0
            s2 = el2.oxistates[0] if el2.oxistates else 0
            if s1 > 0 and s2 < 0:
                return "ca"
            if s1 < 0 and s2 > 0:
                return "ca"
            if s1 > 0 and s2 > 0:
                return "cc_sim" if parts[0] == parts[1] else "cc_diff"
            if s1 < 0 and s2 < 0:
                return "aa"
        except Exception:
            pass
        return "other"

    def find_features_on_grid(self, x_grid):
        """Find peak and first minimum after peak on the weighted g(r)."""
        y_grid = self.spline(x_grid) * self.weight

        x_spacing = x_grid[1] - x_grid[0]
        min_dist = int(0.5 / x_spacing)

        peaks, _ = find_peaks(
            y_grid,
            prominence=0.1 * np.max(y_grid) if np.max(y_grid) > 0 else 0,
            distance=min_dist,
            width=2,
        )

        peak_pt = (None, None)
        min_pt = (None, None)

        if len(peaks) > 0:
            p_idx = peaks[0]
            peak_pt = (x_grid[p_idx], y_grid[p_idx])

            y_after = y_grid[p_idx:]
            mins, _ = find_peaks(-y_after, prominence=0.01 * np.max(y_grid), distance=min_dist)
            if len(mins) > 0:
                m_idx = mins[0] + p_idx
                min_pt = (x_grid[m_idx], y_grid[m_idx])
            else:
                search_end = min(len(x_grid) - 1, int(p_idx + (2.0 / x_spacing)))
                if search_end > p_idx:
                    m_idx = p_idx + np.argmin(y_grid[p_idx:search_end])
                    min_pt = (x_grid[m_idx], y_grid[m_idx])

        self.peak = peak_pt
        self.minima = min_pt
        return peak_pt, min_pt


# ==========================================
# Dataset analysis
# ==========================================

def analyze_dataset(filepath, comp_str, use_bph=True):
    """Load a prepared CSV and run the SCL analysis.

    Parameters
    ----------
    use_bph : bool
        If True, use concentration-only b_PH.
        If False, force b_PH = 0.

    Returns:
        dict with keys:
            x_grid, weighted_data (dict of pair_name -> weighted g(r)),
            ion_pairs (dict of IonPairData), avg_SCL, pair_SCLs (dict name->SCL),
            S_curves (dict name->S_y_grid)
    """
    fractions, ion_counts, comp = parse_composition(comp_str)

    # Calculate weights
    el_conc = {}
    for salt, frac in fractions.items():
        for el, count in ion_counts[salt].items():
            el_conc[el] = el_conc.get(el, 0) + frac * count
    total_conc = sum(el_conc.values())
    rel_conc = {k: v / total_conc for k, v in el_conc.items()}

    # Accumulate weights correctly:
    # For like pairs (el-el): weight = c_el * c_el
    # For unlike pairs (el1-el2): weight = c_el1 * c_el2 + c_el2 * c_el1 = 2 * c_el1 * c_el2
    #   (because both orderings contribute physically)
    weights = {}
    for el1, c1 in rel_conc.items():
        for el2, c2 in rel_conc.items():
            pair = standardize_ion_pair(f"{el1}-{el2}")
            weights[pair] = weights.get(pair, 0) + c1 * c2
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    # Load CSV
    df = pd.read_csv(filepath, comment='#')
    x_grid = df['r (A)'].values
    ion_pairs = {}
    for col in df.columns:
        if col == 'r (A)':
            continue
        name = col
        y = df[col].values
        weight = weights.get(name, 0)
        ion_pairs[name] = IonPairData(name, x_grid, y, weight)

    # Find features
    for pair in ion_pairs.values():
        pair.find_features_on_grid(x_grid)

    # Weighted splines
    weighted_splines = {}
    for name, p in ion_pairs.items():
        weighted_splines[name] = lambda x, s=p.spline, w=p.weight: s(x) * w

    # Weighted data for plotting
    weighted_data = {}
    for name, p in ion_pairs.items():
        weighted_data[name] = weighted_splines[name](x_grid)

    # --- SCL calculation ---
    ca_pairs = [p for _, p in ion_pairs.items() if p.type == 'ca']
    sum_ca_weights = sum(p.weight for p in ca_pairs)

    total_weighted_scl = 0
    total_weight_norm = 0
    pair_sc_ls = {}
    s_curves = {}

    for pair in ca_pairs:
        name = pair.name
        r_peak = pair.peak[0]
        if r_peak is None or r_peak <= 0:
            continue

        delta_r = r_peak
        transfer_points = np.arange(delta_r, x_grid[-1], delta_r)
        if len(transfer_points) == 0:
            continue

        # --- Disruption factors ---
        g_peak = pair.peak[1]
        g_min = pair.minima[1] if pair.minima[0] else 0
        kf_val = 1.0
        if g_peak > 1e-6:
            kf_val = 1 - (g_peak - g_min) / g_peak
        kf_val = np.clip(kf_val, 0, 1)

        # b_PH: concentration-only or forced to 0
        if use_bph:
            if sum_ca_weights > 0:
                ph_val = np.clip(1.0 - pair.weight / sum_ca_weights, 0, 1)
            else:
                ph_val = 1.0
        else:
            ph_val = 0.0  # b_PH = 0

        cation = name.split('-')[0]
        cc_name = standardize_ion_pair(f"{cation}-{cation}")
        has_cc = cc_name in ion_pairs

        beta_vals = []

        for m, r_m in enumerate(transfer_points, 1):
            g_tot_val = 0
            for pn, pair_obj in ion_pairs.items():
                if cation in pn.split('-'):
                    g_tot_val += weighted_splines[pn](r_m)

            if m % 2 == 1:
                g_ideal_val = weighted_splines[name](r_m)
            else:
                if has_cc:
                    g_ideal_val = weighted_splines[cc_name](r_m)
                else:
                    g_ideal_val = weights.get(cc_name, 0)

            ni_val = 1.0
            if g_tot_val > 1e-6:
                ni_val = 1 - (g_ideal_val / g_tot_val)
            ni_val = np.clip(ni_val, 0, 1)

            if kf_val == 1 or ph_val == 1 or ni_val == 1:
                beta = float('inf')
            else:
                beta = (kf_val / (1 - kf_val)) + (ph_val / (1 - ph_val)) + (ni_val / (1 - ni_val))
            beta_vals.append(beta)

        # Survival function
        S_discrete = [1.0]
        int_beta = 0
        for beta in beta_vals:
            if beta == float('inf'):
                int_beta = -float('inf')
            else:
                int_beta -= beta * delta_r
            S_discrete.append(np.exp(int_beta))

        scl_pair = delta_r * sum(S_discrete[:-1])
        RTE = pair.weight / sum_ca_weights if sum_ca_weights > 0 else 0
        total_weighted_scl += scl_pair * RTE
        total_weight_norm += RTE

        # Map S(r) to fine grid
        S_y_grid = np.zeros_like(x_grid)
        curr_s_idx = 0
        for i, x_val in enumerate(x_grid):
            if curr_s_idx < len(transfer_points):
                if x_val >= transfer_points[curr_s_idx]:
                    curr_s_idx += 1
            if curr_s_idx < len(S_discrete):
                S_y_grid[i] = S_discrete[curr_s_idx]
            else:
                S_y_grid[i] = S_discrete[-1]

        pair_sc_ls[name] = scl_pair
        s_curves[name] = S_y_grid

    avg_scl = total_weighted_scl / total_weight_norm if total_weight_norm > 0 else 0

    return {
        'x_grid': x_grid,
        'weighted_data': weighted_data,
        'ion_pairs': ion_pairs,
        'avg_SCL': avg_scl,
        'pair_SCLs': pair_sc_ls,
        'S_curves': s_curves,
    }


# ==========================================
# Helper: create a lighter version of a color
# ==========================================
def lighten_color(color, factor=0.5):
    """
    Lighten a color by blending with white.
    factor=0 gives the original color, factor=1 gives white.
    """
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return (r, g, b)


# ==========================================
# Main
# ==========================================

def main():
    # ══════════════════════════════════════════
    # CONFIGURATION — edit these for your data
    # ══════════════════════════════════════════

    csv_file = '0.21NaCl_0.38MgCl2_0.41KCl_Walker_660K.csv'

    label_bph_on = 'w/ b$_{\\mathrm{PH}}$'
    label_bph_off = 'w/o b$_{\\mathrm{PH}}$'

    style_bph_on = '-'      # solid
    style_bph_off = ':'     # dotted

    comp_str = '0.21NaCl-0.38MgCl2-0.41KCl'

    output_base = 'PDF_SCL_bPH_Comparison_0.21NaCl_0.38MgCl2_0.41KCl_Walker'

    # ══════════════════════════════════════════
    # Load data & run SCL analysis
    # ══════════════════════════════════════════

    script_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(script_dir, csv_file)

    if not os.path.exists(csv_path):
        print(f"Error: {csv_file} not found at {csv_path}")
        print(f"Please copy the CSV file to: {script_dir}")
        return

    print("=" * 60)
    print("With b_PH (concentration-only):")
    result_bph = analyze_dataset(csv_path, comp_str, use_bph=True)
    print(f"  Average SCL: {result_bph['avg_SCL']:.4f} Å")
    for name, scl in result_bph['pair_SCLs'].items():
        print(f"    {name}: SCL = {scl:.3f} Å")

    print(f"\nWithout b_PH (b_PH = 0):")
    result_nobph = analyze_dataset(csv_path, comp_str, use_bph=False)
    print(f"  Average SCL: {result_nobph['avg_SCL']:.4f} Å")
    for name, scl in result_nobph['pair_SCLs'].items():
        print(f"    {name}: SCL = {scl:.3f} Å")
    print("=" * 60)

    # Verify pairs
    pairs = sorted(result_bph['weighted_data'].keys())
    print(f"Ion pairs: {pairs}")

    # ══════════════════════════════════════════
    # Plot — matching SCL_calc.py style
    # ══════════════════════════════════════════

    # plt.rcParams.update({
    #     'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica'], 'font.size': 12,
    #     'axes.labelsize': 12, 'axes.labelweight': 'bold', 'axes.linewidth': 1.5,
    #     'xtick.labelsize': 11, 'ytick.labelsize': 11,
    #     'xtick.direction': 'out', 'ytick.direction': 'out',
    #     'xtick.major.width': 1.75, 'ytick.major.width': 1.75,
    #     'legend.frameon': False, 'legend.fontsize': 10,
    #     'mathtext.fontset': 'custom', 'mathtext.rm': 'Helvetica',
    #     'mathtext.it': 'Helvetica:italic', 'mathtext.bf': 'Helvetica:bold',
    # })

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 15,
        "axes.labelsize": 15, 'axes.labelweight': 'bold', 
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10
    })

    fig, ax = plt.subplots(figsize=(5.5, 5.2)) #plt.subplots(figsize=(4.75, 4.25))

    x_grid = result_bph['x_grid']

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    pair_colors = {}

    # Determine which pairs to show: only cation-anion ('ca') type
    # (cc and aa pairs are still used in the SCL calculation, just not plotted)
    ion_pairs_data = result_bph['ion_pairs']
    show_pairs = [n for n in sorted(result_bph['weighted_data'].keys())
                  if ion_pairs_data[n].type == 'ca']

    print(f"Showing pairs (only ca): {show_pairs}")

    # The g(r) curves are the same for both analyses — plot once
    for i, pair in enumerate(show_pairs):
        if pair in result_bph['weighted_data']:
            color = color_cycle[i % len(color_cycle)]
            pair_colors[pair] = color
            ax.plot(x_grid, result_bph['weighted_data'][pair],
                    linestyle='-', color=color, linewidth=1.5)

    # S(r) step curves: with b_PH (solid) and without b_PH (dotted)
    # Use lighter versions of the respective cation-anion g(r) colors
    for pair in show_pairs:
        if ion_pairs_data[pair].type != 'ca':
            continue

        gr_color = pair_colors.get(pair, 'black')
        # Create a lighter, muted version for the S(r) curves
        sr_color = lighten_color(gr_color, factor=0.6)

        # With b_PH — solid
        if pair in result_bph['S_curves']:
            ax.plot(x_grid, result_bph['S_curves'][pair],
                    linestyle=style_bph_on, color=sr_color, linewidth=1.3, alpha=0.8)

        # Without b_PH — dotted
        if pair in result_nobph['S_curves']:
            ax.plot(x_grid, result_nobph['S_curves'][pair],
                    linestyle=style_bph_off, color=sr_color, linewidth=1.3, alpha=0.8)

    # Average SCL vertical lines
    ax.axvline(x=result_bph['avg_SCL'], color='g', linestyle=style_bph_on,
               label=f"$\\ell_{{\\mathrm{{sc}}}}$ (w/ b$_{{\\mathrm{{PH}}}}$) = {result_bph['avg_SCL']:.2f} Å",
               linewidth=1.5)
    ax.axvline(x=result_nobph['avg_SCL'], color='r', linestyle=style_bph_off,
               label=f"$\\ell_{{\\mathrm{{sc}}}}$ (w/o b$_{{\\mathrm{{PH}}}}$) = {result_nobph['avg_SCL']:.2f} Å",
               linewidth=1.5)

    # Axis labels
    ax.set_xlabel('r [Å]')
    ax.set_ylabel('g(r)')

    # X-limits: auto-detect
    x_max = x_grid[-1]
    first_nz = x_max
    for arr in result_bph['weighted_data'].values():
        nz = np.where(arr > 0.01)[0]
        if len(nz):
            first_nz = min(first_nz, x_grid[nz[0]])
    x_start = max(0, first_nz - 0.75)
    x_pad = (x_max - 0) * 0.05
    ax.set_xlim(x_start, x_max + x_pad)

    # Y-axis: round up
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, np.ceil(ymax * 10) / 10)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    # ══════════════════════════════════════════
    # Legend
    # ══════════════════════════════════════════

    legend_elements = [
        # b_PH mode line-style guides
        Line2D([0], [0], linestyle=style_bph_on, color='black', linewidth=1.5, label=label_bph_on),
        Line2D([0], [0], linestyle=style_bph_off, color='black', linewidth=1.5, label=label_bph_off),
        Line2D([0], [0], linestyle='', color='none', label=''),
        # Ion pairs (g(r) curves)
        *[Line2D([0], [0], linestyle='-', color=pair_colors[p], linewidth=2.5, label=p)
          for p in show_pairs],
        # S(r) step curves — each with its own lighter colour
        *[Line2D([0], [0], linestyle='-', color=lighten_color(pair_colors.get(p, 'gray'), 0.6),
                 linewidth=1.3, alpha=0.8, label=f'S(r): {p}')
          for p in show_pairs if ion_pairs_data[p].type == 'ca'],
        Line2D([0], [0], linestyle='', color='none', label=''),
        # SCL vertical lines
        Line2D([0], [0], linestyle=style_bph_on, color='g', linewidth=1.5,
               label=f"$\\ell_{{\\mathrm{{sc}}}}$ (w/ b$_{{\\mathrm{{PH}}}}$) = {result_bph['avg_SCL']:.2f} Å"),
        Line2D([0], [0], linestyle=style_bph_off, color='r', linewidth=1.7,
               label=f"$\\ell_{{\\mathrm{{sc}}}}$ (w/o b$_{{\\mathrm{{PH}}}}$) = {result_nobph['avg_SCL']:.2f} Å"),
    ]

    ax.legend(handles=legend_elements, ncol=1, loc='upper right',
              bbox_to_anchor=(1, 1),
              facecolor='white', frameon=True, framealpha=0.75, edgecolor='none',
              borderpad=0.5, handletextpad=0.5, columnspacing=0.6,
              handlelength=1.5, labelspacing=0.15)

    plt.tight_layout()

    # ══════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════

    out_path = os.path.join(script_dir, output_base)
    plt.savefig(f'{out_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_path}.eps', bbox_inches='tight')
    print(f"\nSaved: {out_path}.*")
    plt.close()


if __name__ == "__main__":
    main()