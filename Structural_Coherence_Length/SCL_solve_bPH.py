"""
bph_discover.py — Discover interpretable equations for the phonon transfer
disruption factor b_PH from PDF features and experimental SCL data.

Strategy:
  1. For each multi-component salt with experimental SCL, re-run a simplified
     SCL calculation sweeping S_ij (pairwise transfer efficiency) from 0 to 1.
  2. Find S_ij that makes predicted SCL match experimental SCL.
  3. Extract pairwise PDF features (masses, peak positions, heights, widths, etc.)
  4. Fit interpretable models (polynomial LASSO) to discover S_ij = f(features).

Usage:
    python bph_discover.py

Requires:
    - Prepared PDF CSVs in Prepared_PDF_CSV/
    - SCL_results.csv from scl_analysis.py (for b_KF, b_NI values)
    - scl_utils.py in the same directory
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from itertools import combinations
from datetime import datetime

from mendeleev import element

from scl_utils import (
    get_scl_dir,
    standardize_ion_pair,
    parse_composition,
    format_composition_with_subscripts,
)

# Attempt sklearn imports
try:
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.model_selection import LeaveOneOut
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. ML fitting will be skipped.")
    print("  Install with: pip install scikit-learn")


# ============================================================
# Constants
# ============================================================

ANION_POLAR = {'F': 1.040, 'Cl': 3.660, 'Br': 4.770, 'I': 7.100}
ANION_RADII = {'F': 1.33, 'Cl': 1.81, 'Br': 1.96, 'I': 2.20}


# ============================================================
# 1. Simplified SCL calculator (standalone, no class needed)
# ============================================================

def compute_scl_for_pair(delta_r, b_KF, b_NI_vals, b_PH, x_grid_max):
    """Compute SCL for a single ca pair given disruption factors.

    Args:
        delta_r: peak position (transfer distance)
        b_KF: bond strength factor (scalar, constant for all transfer points)
        b_NI_vals: array of b_NI at each transfer point
        b_PH: phonon transfer factor (scalar, constant for all transfer points)
        x_grid_max: maximum r on the grid

    Returns:
        scl: structural coherence length for this pair
    """
    transfer_points = np.arange(delta_r, x_grid_max, delta_r)
    if len(transfer_points) == 0:
        return delta_r

    S_discrete = [1.0]
    int_beta = 0.0

    for m, r_m in enumerate(transfer_points):
        kf = b_KF
        ni = b_NI_vals[m] if m < len(b_NI_vals) else b_NI_vals[-1]
        ph = b_PH

        if kf >= 1.0 or ph >= 1.0 or ni >= 1.0:
            beta = float('inf')
        else:
            beta = (kf / (1 - kf)) + (ph / (1 - ph)) + (ni / (1 - ni))

        if beta == float('inf'):
            int_beta = -float('inf')
        else:
            int_beta -= beta * delta_r
        S_discrete.append(np.exp(int_beta))

    scl = delta_r * sum(S_discrete[:-1])
    return scl


def compute_mixture_scl(pair_data_list, s_ij_matrix):
    """Compute mixture SCL given pair data and pairwise S_ij values.

    Args:
        pair_data_list: list of dicts, each with:
            'name', 'delta_r', 'b_KF', 'b_NI_vals', 'weight', 'x_grid_max'
        s_ij_matrix: dict of (name_i, name_j) -> S_ij

    Returns:
        avg_scl: weighted average SCL
    """
    ca_weights = [p['weight'] for p in pair_data_list]
    sum_ca_w = sum(ca_weights)
    if sum_ca_w <= 0:
        return 0.0

    total_weighted_scl = 0.0
    total_weight_norm = 0.0

    for pair in pair_data_list:
        # Compute b_PH from S_ij matrix
        numerator = 0.0
        denominator = 0.0
        for other in pair_data_list:
            x_j = other['weight']
            denominator += x_j
            if pair['name'] == other['name']:
                S_ij = 1.0
            else:
                S_ij = s_ij_matrix.get((pair['name'], other['name']), 0.0)
            numerator += x_j * S_ij

        if denominator > 0:
            b_PH = np.clip(1.0 - numerator / denominator, 0.0, 0.999)
        else:
            b_PH = 0.0

        scl_pair = compute_scl_for_pair(
            pair['delta_r'], pair['b_KF'], pair['b_NI_vals'],
            b_PH, pair['x_grid_max'],
        )
        RTE = pair['weight'] / sum_ca_w
        total_weighted_scl += scl_pair * RTE
        total_weight_norm += RTE

    return total_weighted_scl / total_weight_norm if total_weight_norm > 0 else 0.0


# ============================================================
# 2. Load salt data from prepared PDFs and rebuild disruption factors
# ============================================================

def load_pair_data_from_pdf(prepared_csv, comp_str, source, temp):
    """Load a prepared PDF and compute b_KF, b_NI for each ca pair.

    Returns list of pair data dicts, or None if loading fails.
    """
    if not os.path.exists(prepared_csv):
        return None

    fractions, ion_counts, comp = parse_composition(comp_str)

    # Calculate weights
    el_conc = {}
    for salt, frac in fractions.items():
        for el, count in ion_counts[salt].items():
            el_conc[el] = el_conc.get(el, 0) + frac * count
    total_conc = sum(el_conc.values())
    rel_conc = {k: v / total_conc for k, v in el_conc.items()}

    weights = {}
    for el1, c1 in rel_conc.items():
        for el2, c2 in rel_conc.items():
            pair = standardize_ion_pair(f"{el1}-{el2}")
            weights[pair] = weights.get(pair, 0) + c1 * c2
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    # Load PDF
    df = pd.read_csv(prepared_csv, comment='#')
    x_grid = df['r (A)'].values
    dx = x_grid[1] - x_grid[0]

    # Build ion pair data
    ion_pairs = {}
    for col in df.columns:
        if col == 'r (A)':
            continue
        name = col
        y = df[col].values
        weight = weights.get(name, 0)

        # Determine type
        try:
            parts = name.split('-')
            el1, el2 = element(parts[0]), element(parts[1])
            s1 = el1.oxistates[0] if el1.oxistates else 0
            s2 = el2.oxistates[0] if el2.oxistates else 0
            if (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0):
                pair_type = 'ca'
            else:
                pair_type = 'other'
        except Exception:
            pair_type = 'other'

        ion_pairs[name] = {
            'y': y, 'weight': weight, 'type': pair_type,
            'spline': interp1d(x_grid, y, kind='linear', bounds_error=False, fill_value=0),
        }

    # Find peaks and compute disruption factors for ca pairs
    ca_pairs = {n: p for n, p in ion_pairs.items() if p['type'] == 'ca'}
    sum_ca_w = sum(p['weight'] for p in ca_pairs.values())

    min_dist = int(0.5 / dx)
    pair_data_list = []

    for name, pair in ca_pairs.items():
        y_weighted = pair['spline'](x_grid) * pair['weight']

        # Find peak
        peaks, _ = find_peaks(
            y_weighted,
            prominence=0.1 * np.max(y_weighted) if np.max(y_weighted) > 0 else 0,
            distance=min_dist, width=2,
        )
        if len(peaks) == 0:
            continue

        p_idx = peaks[0]
        peak_x = x_grid[p_idx]
        peak_y = y_weighted[p_idx]

        # Find minimum
        y_after = y_weighted[p_idx:]
        mins, _ = find_peaks(-y_after, prominence=0.01 * np.max(y_weighted), distance=min_dist)
        if len(mins) > 0:
            min_y = y_weighted[mins[0] + p_idx]
        else:
            search_end = min(len(x_grid) - 1, int(p_idx + (2.0 / dx)))
            if search_end > p_idx:
                min_y = y_weighted[p_idx + np.argmin(y_weighted[p_idx:search_end])]
            else:
                min_y = 0.0

        # b_KF
        b_KF = min_y / peak_y if peak_y > 1e-6 else 1.0
        b_KF = np.clip(b_KF, 0, 1)

        # Cation info
        cation = name.split('-')[0]
        cc_name = standardize_ion_pair(f"{cation}-{cation}")
        has_cc = cc_name in ion_pairs

        # Weighted spline functions
        w_splines = {}
        for n2, p2 in ion_pairs.items():
            w_splines[n2] = lambda x, s=p2['spline'], w=p2['weight']: s(x) * w

        # b_NI at each transfer point
        delta_r = peak_x
        transfer_points = np.arange(delta_r, x_grid[-1], delta_r)
        b_NI_vals = []

        for m, r_m in enumerate(transfer_points, 1):
            g_tot = sum(w_splines[pn](r_m) for pn in ion_pairs if cation in pn.split('-'))

            if m % 2 == 1:
                g_ideal = w_splines[name](r_m)
            else:
                if has_cc:
                    g_ideal = w_splines[cc_name](r_m)
                else:
                    g_ideal = weights.get(cc_name, 0)

            ni = 1.0 - g_ideal / g_tot if g_tot > 1e-6 else 1.0
            b_NI_vals.append(np.clip(ni, 0, 1))

        # Unweighted g(r) features for this pair
        g_unw = pair['spline'](x_grid)
        peaks_unw, _ = find_peaks(
            g_unw,
            prominence=0.1 * np.max(g_unw) if np.max(g_unw) > 0 else 0,
            distance=min_dist, width=2,
        )
        if len(peaks_unw) == 0:
            continue
        p_idx_unw = peaks_unw[0]
        g_peak_unw = g_unw[p_idx_unw]
        r_peak_unw = x_grid[p_idx_unw]

        # Find unweighted minimum
        g_after_unw = g_unw[p_idx_unw:]
        mins_unw, _ = find_peaks(-g_after_unw, prominence=0.01 * np.max(g_unw), distance=min_dist)
        if len(mins_unw) > 0:
            g_base_unw = g_unw[mins_unw[0] + p_idx_unw]
        else:
            search_end = min(len(x_grid) - 1, int(p_idx_unw + (2.0 / dx)))
            g_base_unw = g_unw[p_idx_unw + np.argmin(g_unw[p_idx_unw:search_end])] if search_end > p_idx_unw else 0
        g_base_unw = max(g_base_unw, 0)

        # FWHM (split: left baseline=0, right baseline=g_base)
        half_left = g_peak_unw / 2.0
        left_crosses = np.where(g_unw[:p_idx_unw] <= half_left)[0]
        if len(left_crosses) > 0:
            li = left_crosses[-1]
            if li + 1 < p_idx_unw and g_unw[li + 1] != g_unw[li]:
                frac = (half_left - g_unw[li]) / (g_unw[li + 1] - g_unw[li])
                x_left = x_grid[li] + frac * dx
            else:
                x_left = x_grid[li]
        else:
            x_left = x_grid[0]
        hwhm_left = r_peak_unw - x_left

        half_right = (g_peak_unw + g_base_unw) / 2.0
        right_crosses = np.where(g_unw[p_idx_unw:] <= half_right)[0]
        if len(right_crosses) > 0:
            ri = right_crosses[0] + p_idx_unw
            if ri > p_idx_unw and g_unw[ri - 1] != g_unw[ri]:
                frac = (half_right - g_unw[ri - 1]) / (g_unw[ri] - g_unw[ri - 1])
                x_right = x_grid[ri - 1] + frac * dx
            else:
                x_right = x_grid[ri]
        else:
            x_right = x_grid[-1]
        hwhm_right = x_right - r_peak_unw

        hwhm_left = max(hwhm_left, dx)
        hwhm_right = max(hwhm_right, dx)
        fwhm = hwhm_left + hwhm_right

        # Anion info
        anion = name.split('-')[1]
        alpha_a = ANION_POLAR.get(anion, 2.0)
        r_a = ANION_RADII.get(anion, 1.5)
        chi = alpha_a / (r_a ** 3) if r_a > 0 else 0

        # Cation mass
        try:
            m_cat = element(cation).mass
        except Exception:
            m_cat = 20.0

        # Reduced mass
        try:
            m_an = element(anion).mass
            mu = (m_cat * m_an) / (m_cat + m_an)
        except Exception:
            mu = 15.0

        pair_data_list.append({
            'name': name,
            'delta_r': peak_x,
            'b_KF': b_KF,
            'b_NI_vals': b_NI_vals,
            'weight': pair['weight'],
            'x_grid_max': x_grid[-1],
            # PDF features
            'r_peak': r_peak_unw,
            'g_peak': g_peak_unw,
            'g_base': g_base_unw,
            'b_kf_unw': g_base_unw / g_peak_unw if g_peak_unw > 0 else 1.0,
            'fwhm': fwhm,
            'hwhm_left': hwhm_left,
            'hwhm_right': hwhm_right,
            'cation': cation,
            'anion': anion,
            'm_cat': m_cat,
            'mu': mu,
            'alpha_a': alpha_a,
            'chi': chi,
        })

    return pair_data_list if pair_data_list else None


# ============================================================
# 3. Back-calculate optimal S_ij
# ============================================================

def find_optimal_S(pair_data_list, scl_exp, tol=1e-5):
    """Find the uniform S_ij that makes computed SCL match experimental.

    Uses scalar optimization since Brent's method requires sign change.

    Args:
        pair_data_list: from load_pair_data_from_pdf
        scl_exp: experimental SCL

    Returns:
        (S_opt, scl_pred, status)
        status: 'converged', 'boundary_low', 'boundary_high'
    """
    ca_names = [p['name'] for p in pair_data_list]

    def scl_at_S(S_val):
        s_matrix = {}
        for a in ca_names:
            for b in ca_names:
                s_matrix[(a, b)] = 1.0 if a == b else S_val
        return compute_mixture_scl(pair_data_list, s_matrix)

    # Check boundaries
    scl_at_0 = scl_at_S(0.0)
    scl_at_1 = scl_at_S(1.0)

    if scl_at_0 >= scl_exp:
        return 0.0, scl_at_0, 'boundary_low'
    if scl_at_1 <= scl_exp:
        return 1.0, scl_at_1, 'boundary_high'

    # Brent's method
    try:
        S_opt = brentq(lambda S: scl_at_S(S) - scl_exp, 0.0, 1.0, xtol=tol)
        return S_opt, scl_at_S(S_opt), 'converged'
    except Exception:
        # Fallback to minimize
        result = minimize_scalar(
            lambda S: (scl_at_S(S) - scl_exp) ** 2,
            bounds=(0, 1), method='bounded',
        )
        return result.x, scl_at_S(result.x), 'minimized'


# ============================================================
# 4. Feature extraction for pairwise S_ij
# ============================================================

FEATURE_NAMES = [
    'delta_m_rel',    # |m1-m2| / mean(m1,m2)
    'mass_ratio',     # min/max of cation masses
    'mu_ratio',       # min/max of reduced masses
    'delta_r_rel',    # |r1-r2| / mean(r1,r2)
    'r_ratio',        # min/max of peak positions
    'delta_g_rel',    # |g1-g2| / max(g1,g2)
    'g_ratio',        # min/max of peak heights
    'delta_w_rel',    # |w1-w2| / mean(w1,w2)
    'w_ratio',        # min/max of FWHM
    'delta_bkf',      # |bkf1-bkf2|
    'bkf_mean',       # mean(bkf1, bkf2)
    'alpha_a',        # anion polarizability
    'chi',            # alpha_a / r_a^3
    'delta_r_over_w', # |r1-r2| / mean(w1,w2) — gap in FWHM units
    'stiffness_ratio',# ratio of (1-bkf) values
]


def extract_pairwise_features(pair_A, pair_B):
    """Extract features for the pair (A, B). Returns dict."""
    m1, m2 = pair_A['m_cat'], pair_B['m_cat']
    r1, r2 = pair_A['r_peak'], pair_B['r_peak']
    g1, g2 = pair_A['g_peak'], pair_B['g_peak']
    w1, w2 = pair_A['fwhm'], pair_B['fwhm']
    bkf1, bkf2 = pair_A['b_kf_unw'], pair_B['b_kf_unw']
    mu1, mu2 = pair_A['mu'], pair_B['mu']

    m_mean = (m1 + m2) / 2
    r_mean = (r1 + r2) / 2
    w_mean = (w1 + w2) / 2
    g_max = max(g1, g2)

    k1 = max(1 - bkf1, 1e-6)
    k2 = max(1 - bkf2, 1e-6)

    # Use shared anion properties (should be same for single-anion systems)
    alpha = pair_A['alpha_a']  # same anion
    chi = pair_A['chi']

    return {
        'delta_m_rel': abs(m1 - m2) / m_mean if m_mean > 0 else 0,
        'mass_ratio': min(m1, m2) / max(m1, m2) if max(m1, m2) > 0 else 1,
        'mu_ratio': min(mu1, mu2) / max(mu1, mu2) if max(mu1, mu2) > 0 else 1,
        'delta_r_rel': abs(r1 - r2) / r_mean if r_mean > 0 else 0,
        'r_ratio': min(r1, r2) / max(r1, r2) if max(r1, r2) > 0 else 1,
        'delta_g_rel': abs(g1 - g2) / g_max if g_max > 0 else 0,
        'g_ratio': min(g1, g2) / max(g1, g2) if max(g1, g2) > 0 else 1,
        'delta_w_rel': abs(w1 - w2) / w_mean if w_mean > 0 else 0,
        'w_ratio': min(w1, w2) / max(w1, w2) if max(w1, w2) > 0 else 1,
        'delta_bkf': abs(bkf1 - bkf2),
        'bkf_mean': (bkf1 + bkf2) / 2,
        'alpha_a': alpha,
        'chi': chi,
        'delta_r_over_w': abs(r1 - r2) / w_mean if w_mean > 0 else 0,
        'stiffness_ratio': min(k1, k2) / max(k1, k2) if max(k1, k2) > 0 else 1,
    }


# ============================================================
# 5. Confidence weighting
# ============================================================

# Higher weight = more trusted experimental data
CONFIDENCE_WEIGHTS = {
    # Well-measured salts (multiple reliable datasets, well-studied)
    '0.465LiF-0.115NaF-0.42KF':            1.0,   # FLiNaK — many measurements
    '0.66LiF-0.34BeF2':                     1.0,   # FLiBe — well characterized
    '0.64NaCl-0.36UCl3':                    0.9,   # NaCl-UCl3 — recent measurements
    '0.5NaCl-0.5KCl':                       1.0,   # NaCl-KCl — simple system
    '0.6LiF-0.4NaF':                        0.8,   # LiF-NaF — one dataset
    '0.4903NaCl-0.5097CaCl2':               0.7,   # NaCl-CaCl2
    '0.535NaCl-0.315MgCl2-0.15CaCl2':       0.5,   # Ternary — uncertain k_exp
    '0.345NaF-0.59KF-0.065MgF2':            0.4,   # FMgNaK — questionable k_exp
    '0.32MgCl2-0.68KCl':                    0.7,   # MgCl2-KCl
    '0.38MgCl2-0.21NaCl-0.41KCl':           0.6,   # Ternary MgCl2
}

def get_confidence(comp_str):
    """Get confidence weight for a composition. Default = 0.5."""
    # Try exact match first
    if comp_str in CONFIDENCE_WEIGHTS:
        return CONFIDENCE_WEIGHTS[comp_str]
    # Try parsing and matching sorted form
    _, _, sorted_comp = parse_composition(comp_str)
    for key, val in CONFIDENCE_WEIGHTS.items():
        _, _, sorted_key = parse_composition(key)
        if sorted_comp == sorted_key:
            return val
    return 0.5


# ============================================================
# 6. Manual functional form search
# ============================================================

def evaluate_manual_forms(features_df, S_targets, weights):
    """Test a battery of physically motivated functional forms."""
    results = []
    X = features_df.values
    y = np.array(S_targets)
    w = np.array(weights)
    n = len(y)

    def weighted_r2(y_true, y_pred, w):
        ss_res = np.sum(w * (y_true - y_pred) ** 2)
        ss_tot = np.sum(w * (y_true - np.average(y_true, weights=w)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf

    def weighted_mae(y_true, y_pred, w):
        return np.average(np.abs(y_true - y_pred), weights=w)

    f = {name: features_df[name].values for name in FEATURE_NAMES}

    # --- Form 1: Klemens defect scattering ---
    Gamma_m = f['delta_m_rel'] ** 2
    Gamma_k = (1 - f['stiffness_ratio']) ** 2
    Gamma_e = f['delta_r_rel'] ** 2
    chi_arr = f['chi']
    Gamma_tot = Gamma_m + (Gamma_k + Gamma_e) * np.maximum(1 - chi_arr, 0.01)
    y_pred = 1.0 / (1.0 + Gamma_tot)
    results.append(('Klemens defect scattering',
                     weighted_r2(y, y_pred, w), weighted_mae(y, y_pred, w)))

    # --- Form 2: Franck-Condon (polarizability-screened) ---
    sqrt2ln2 = np.sqrt(2 * np.log(2))
    sig1 = features_df.apply(lambda r: 0, axis=1).values  # placeholder
    # Use FWHM/2 as sigma proxy, screened by chi
    sig_eff = (f['delta_r_rel'] + 0.01) * (1 + f['chi'])
    FC = f['w_ratio'] * np.exp(-0.5 * (f['delta_r_rel'] / np.maximum(sig_eff, 0.01)) ** 2)
    # Frequency match
    nu_ratio = f['r_ratio'] * np.sqrt(f['mu_ratio'])
    f_omega = 2 * np.sqrt(nu_ratio) / (1 + nu_ratio)
    y_pred = np.clip(FC * f_omega, 0, 1)
    results.append(('Franck-Condon (polarizability-screened)',
                     weighted_r2(y, y_pred, w), weighted_mae(y, y_pred, w)))

    # --- Form 3: Peak similarity (f_nu × f_w × f_k) ---
    nu_i = 1 / (features_df.apply(lambda r: 1, axis=1).values)  # placeholder
    f_nu = 2 * np.sqrt(f['mu_ratio'] * f['r_ratio']) / (1 + f['mu_ratio'] * f['r_ratio'])
    f_w = f['w_ratio']
    f_k = 1.0 - f['delta_bkf']
    y_pred = np.clip(f_nu * f_w * f_k, 0, 1)
    results.append(('f_ν × f_w × f_k (peak similarity)',
                     weighted_r2(y, y_pred, w), weighted_mae(y, y_pred, w)))

    # --- Form 4: Impedance mismatch + polarizability ---
    k1 = 1 - f['bkf_mean'] - f['delta_bkf'] / 2
    k2 = 1 - f['bkf_mean'] + f['delta_bkf'] / 2
    k1 = np.maximum(k1, 0.01)
    k2 = np.maximum(k2, 0.01)
    # Approximate impedance ratio using mass_ratio and stiffness_ratio
    Z_ratio = np.sqrt(f['stiffness_ratio'] * f['mu_ratio'])
    R_Z = ((1 - Z_ratio) / (1 + Z_ratio)) ** 2
    R_e = (2 * f['delta_r_rel']) ** 2
    R_0 = 1 - (1 - R_Z) * (1 - np.minimum(R_e, 1))
    R_eff = np.power(np.maximum(R_0, 1e-10), 1 + f['chi'])
    y_pred = np.clip(1 - R_eff, 0, 1)
    results.append(('Impedance mismatch + polarizability',
                     weighted_r2(y, y_pred, w), weighted_mae(y, y_pred, w)))

    # --- Form 5: Penalty-based exp(-(m_p² + K_p²) / P) ---
    m_p = f['delta_m_rel']
    bkf_safe = np.clip(f['bkf_mean'], 0, 0.9999)
    k_prime = -np.log(1.0 - bkf_safe + 1e-10)
    delta_k = f['delta_bkf']
    K_p = delta_k / np.maximum(k_prime, 0.01)
    P = f['alpha_a']
    y_pred = np.exp(-(m_p ** 2 + K_p ** 2) / np.maximum(P, 0.01))
    results.append(('exp(-(m_p² + K_p²) / P)',
                     weighted_r2(y, y_pred, w), weighted_mae(y, y_pred, w)))

    # --- Form 6: Simple product of ratios ---
    y_pred = np.clip(f['mass_ratio'] * f['r_ratio'] * f['w_ratio'] * (1 - f['delta_bkf']), 0, 1)
    results.append(('mass_ratio × r_ratio × w_ratio × (1-Δb_KF)',
                     weighted_r2(y, y_pred, w), weighted_mae(y, y_pred, w)))

    # --- Form 7: Klemens with Grüneisen amplification ---
    gamma = 2.0  # typical for molten salts
    Gamma_e_amp = (6.4 * gamma * f['delta_r_rel']) ** 2
    Gamma_tot_amp = Gamma_m + (Gamma_k + Gamma_e_amp) * np.maximum(1 - chi_arr, 0.01)
    y_pred = 1.0 / (1.0 + Gamma_tot_amp)
    results.append(('Klemens + Grüneisen (γ=2)',
                     weighted_r2(y, y_pred, w), weighted_mae(y, y_pred, w)))

    # --- Form 8: Exponential decay with delta_r/FWHM ---
    y_pred = np.exp(-f['delta_r_over_w'] ** 2) * f['mass_ratio']
    y_pred = np.clip(y_pred, 0, 1)
    results.append(('exp(-Δr/w²) × mass_ratio',
                     weighted_r2(y, y_pred, w), weighted_mae(y, y_pred, w)))

    # Sort by R²
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ============================================================
# 7. ML fitting: exhaustive feature combination search
# ============================================================

def exhaustive_combination_search(features_df, S_targets, weights,
                                  min_features=2, max_features=4,
                                  max_degree=2, top_n=20):
    """Try all combinations of 2-N features at polynomial degrees 1-max_degree.

    Uses weighted Ridge regression with LOO cross-validation.
    Reports the top_n models by weighted R²_LOO.

    Returns list of (r2_loo, mae_loo, degree, feature_names, model_info) tuples.
    """
    if not HAS_SKLEARN:
        print("  sklearn not available — skipping combination search.")
        return []

    all_feature_names = list(features_df.columns)
    n_features = len(all_feature_names)
    y = np.array(S_targets)
    w = np.array(weights)
    n_samples = len(y)

    results = []
    total_combos = 0

    for n_feat in range(min_features, min(max_features + 1, n_features + 1)):
        combos = list(combinations(range(n_features), n_feat))
        total_combos += len(combos) * max_degree

    print(f"\n  Searching {total_combos} combinations "
          f"({min_features}-{max_features} features, degree 1-{max_degree})...")

    combo_count = 0
    for n_feat in range(min_features, min(max_features + 1, n_features + 1)):
        for feat_indices in combinations(range(n_features), n_feat):
            feat_names = [all_feature_names[i] for i in feat_indices]
            X_raw = features_df.iloc[:, list(feat_indices)].values

            for degree in range(1, max_degree + 1):
                combo_count += 1
                if combo_count % 500 == 0:
                    print(f"    ... {combo_count}/{total_combos}")

                # Generate polynomial features
                if degree > 1:
                    poly = PolynomialFeatures(degree=degree, include_bias=False,
                                              interaction_only=False)
                    X_poly = poly.fit_transform(X_raw)
                    poly_names = poly.get_feature_names_out(feat_names)
                else:
                    X_poly = X_raw
                    poly_names = feat_names

                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_poly)

                # Skip if too many features relative to samples
                if X_scaled.shape[1] >= n_samples - 1:
                    continue

                # LOO cross-validation with Ridge
                loo = LeaveOneOut()
                y_pred_loo = np.zeros(n_samples)

                try:
                    for train_idx, test_idx in loo.split(X_scaled):
                        X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
                        y_tr = y[train_idx]
                        w_tr = w[train_idx]

                        # Weighted Ridge
                        ridge = RidgeCV(
                            alphas=np.logspace(-4, 4, 30),
                            fit_intercept=True,
                        )
                        ridge.fit(X_tr, y_tr, sample_weight=w_tr)
                        y_pred_loo[test_idx] = ridge.predict(X_te)
                except Exception:
                    continue

                # Weighted LOO metrics
                ss_res = np.sum(w * (y - y_pred_loo) ** 2)
                ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
                r2_loo = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
                mae_loo = np.average(np.abs(y - y_pred_loo), weights=w)

                if r2_loo > -1:  # filter out complete failures
                    results.append((r2_loo, mae_loo, degree, feat_names, poly_names))

    # Sort by R²_LOO descending
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_n]


def fit_and_report_best(features_df, S_targets, weights, best_result):
    """Refit the best model on all data and print the equation."""
    if not HAS_SKLEARN:
        return None

    r2_loo, mae_loo, degree, feat_names, poly_names = best_result
    feat_indices = [list(features_df.columns).index(f) for f in feat_names]
    X_raw = features_df.iloc[:, feat_indices].values
    y = np.array(S_targets)
    w = np.array(weights)

    if degree > 1:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_raw)
        poly_names = poly.get_feature_names_out(feat_names)
    else:
        X_poly = X_raw
        poly_names = feat_names

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Fit Ridge on all data
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 30))
    ridge.fit(X_scaled, y, sample_weight=w)

    # Also try LASSO for sparsity
    try:
        lasso = LassoCV(cv=min(5, len(y)), max_iter=200000, tol=1e-3, random_state=42)
        lasso.fit(X_scaled, y, sample_weight=w)
    except Exception:
        lasso = None

    # Report Ridge
    print(f"\n  RIDGE (degree {degree}, features: {feat_names})")
    print(f"  R²_LOO = {r2_loo:.4f}, MAE_LOO = {mae_loo:.4f}")
    print(f"  S_ij = {ridge.intercept_:.6f}")
    for i, (name, coef) in enumerate(zip(poly_names, ridge.coef_)):
        if abs(coef) > 1e-6:
            print(f"         {'+' if coef >= 0 else '-'} {abs(coef):.6f} * {name}")

    # Report LASSO if sparser
    if lasso is not None:
        nonzero = np.sum(np.abs(lasso.coef_) > 1e-6)
        if nonzero < len(ridge.coef_):
            print(f"\n  LASSO ({nonzero} nonzero terms, α={lasso.alpha_:.2e})")
            print(f"  S_ij = {lasso.intercept_:.6f}")
            for name, coef in zip(poly_names, lasso.coef_):
                if abs(coef) > 1e-6:
                    print(f"         {'+' if coef >= 0 else '-'} {abs(coef):.6f} * {name}")

    return {
        'ridge': ridge,
        'lasso': lasso,
        'scaler': scaler,
        'poly_names': poly_names,
        'feat_names': feat_names,
        'degree': degree,
        'scaler_means': dict(zip(poly_names, scaler.mean_)),
        'scaler_stds': dict(zip(poly_names, scaler.scale_)),
    }


# ============================================================
# 8. Diagnostic plots
# ============================================================

def plot_diagnostics(features_df, S_targets, weights, compositions,
                     manual_results, combo_results, output_dir):
    """Generate all diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    y = np.array(S_targets)
    w = np.array(weights)

    # --- 1. Feature correlations with S_target ---
    n_feat = len(FEATURE_NAMES)
    ncols = 4
    nrows = int(np.ceil(n_feat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten()

    for i, fname in enumerate(FEATURE_NAMES):
        ax = axes[i]
        x_vals = features_df[fname].values
        sizes = w * 80 + 20  # scale marker size by weight

        ax.scatter(x_vals, y, s=sizes, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax.set_xlabel(fname, fontsize=9)
        ax.set_ylabel('S_target', fontsize=9)
        ax.set_ylim(-0.1, 1.1)

        # Linear correlation
        if len(x_vals) > 2 and np.std(x_vals) > 1e-10:
            r = np.corrcoef(x_vals, y)[0, 1]
            ax.set_title(f'r = {r:.3f}', fontsize=9)
        ax.grid(True, alpha=0.3)

    for i in range(n_feat, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Feature Correlations with S_target\n(marker size = confidence weight)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_correlations.png'), dpi=200)
    plt.close(fig)

    # --- 2. Manual model comparison ---
    if manual_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r[0] for r in manual_results]
        r2s = [r[1] for r in manual_results]
        maes = [r[2] for r in manual_results]

        x_pos = np.arange(len(names))
        colors = ['green' if r > 0 else 'red' for r in r2s]
        ax.barh(x_pos, r2s, color=colors, alpha=0.7, edgecolor='k')
        ax.set_yticks(x_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('R² (weighted)')
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title('Manual Functional Form Comparison')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=200)
        plt.close(fig)

    # --- 3. Training data table ---
    out_df = features_df.copy()
    out_df['S_target'] = y
    out_df['confidence'] = w
    out_df['composition'] = compositions
    out_df.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False, float_format='%.6f')

    print(f"\n  Diagnostics saved to {output_dir}/")


# ============================================================
# 9. Salt definitions
# ============================================================

def _prep_path(comp_str, source, temp):
    _, _, sorted_comp = parse_composition(comp_str)
    safe_source = ''.join(c if c.isalnum() else '_' for c in source.split(',')[0].strip())
    fname = f"{sorted_comp.replace('-', '_')}_{safe_source}_{int(temp)}K.csv"
    return os.path.join(get_scl_dir(), 'Prepared_PDF_CSV', fname)


# Define mixtures with experimental SCL
MIXTURE_DEFS = [
    # (comp_str, source, temp, scl_exp, confidence_override)
    ("0.6LiF-0.4NaF",                     'Grizzi, 2024',    1473, 2.63857, None),
    ("0.66LiF-0.34BeF2",                  'Fayfar, 2024',     973, 1.90187, None),
    ("0.66LiF-0.34BeF2",                  'Yin, 2025',        973, 1.90187, None),
    ("0.5NaCl-0.5KCl",                    'Manga, 2014',     1100, 4.32778, None),
    ("0.5NaCl-0.5KCl",                    'Walker, 2026',    1100, 4.32778, None),
    # ("0.6NaCl-0.4KCl",                    'Walker, 2026',    1100, 4.32778, None),
    # ("0.3NaCl-0.7KCl",                    'Walker, 2026',    1100, 4.32778, None),
    ("0.4903NaCl-0.5097CaCl2",            'Wei, 2022',       1023, 3.76913, None),
    ("0.465LiF-0.115NaF-0.42KF",          'Frandsen, 2020',   873, 2.26059, None),
    ("0.345NaF-0.59KF-0.065MgF2",         'Solano, 2021',    1073, 3.92263, None),
    ("0.535NaCl-0.315MgCl2-0.15CaCl2",    'Wei, 2022',       1023, 3.52027, None),
    ("0.64NaCl-0.36UCl3",                 'Andersson, 2022', 1250, 2.53930, None),
    ("0.32MgCl2-0.68KCl",                 'Walker, 2026',     723, 3.91988, None),
    ("0.38MgCl2-0.21NaCl-0.41KCl",        'Jiang, 2024',      750, 3.65358, None),
    ("0.38MgCl2-0.21NaCl-0.41KCl",        'Walker, 2026',     660, 3.65358, None),
]


# ============================================================
# 10. Main
# ============================================================

def main():
    print("=" * 60)
    print("  b_PH FUNCTIONAL FORM DISCOVERY")
    print("=" * 60)

    # ----------------------------------------------------------
    # Stage 1: Back-calculate optimal S_ij for each mixture
    # ----------------------------------------------------------
    print("\nBack-calculating optimal S_ij from experimental SCL...")

    training_records = []  # (features_dict, S_target, weight, comp_label)

    for comp_str, source, temp, scl_exp, conf_override in MIXTURE_DEFS:
        csv_path = _prep_path(comp_str, source, temp)

        pair_data = load_pair_data_from_pdf(csv_path, comp_str, source, temp)
        if pair_data is None:
            print(f"  {comp_str} ({source}): SKIPPED — PDF not found")
            continue

        ca_pairs = [p for p in pair_data]
        if len(ca_pairs) < 2:
            continue  # unary salt, no S_ij to optimize

        S_opt, scl_pred, status = find_optimal_S(pair_data, scl_exp)

        conf = conf_override if conf_override is not None else get_confidence(comp_str)

        tag = ""
        if status == 'boundary_low':
            tag = " [S=0 boundary — SCL still overestimates]"
            # Keep in dataset with reduced weight
            conf *= 0.3
        elif status == 'boundary_high':
            tag = " [S=1 boundary]"
            conf *= 0.3

        print(f"  {comp_str} ({source}): S_opt={S_opt:.4f}, "
              f"SCL_pred={scl_pred:.3f} vs SCL_exp={scl_exp:.3f} "
              f"(conf={conf:.2f}){tag}")

        # Extract pairwise features
        for i, pA in enumerate(ca_pairs):
            for pB in ca_pairs[i + 1:]:
                feats = extract_pairwise_features(pA, pB)
                comp_label = f"{comp_str}|{source}|{pA['name']}-{pB['name']}"
                training_records.append((feats, S_opt, conf, comp_label))

    if not training_records:
        print("\nNo training data collected! Check PDF paths.")
        return

    # Build dataframe
    features_list = [r[0] for r in training_records]
    S_targets = [r[1] for r in training_records]
    weights = [r[2] for r in training_records]
    comp_labels = [r[3] for r in training_records]

    features_df = pd.DataFrame(features_list)[FEATURE_NAMES]

    n_mixtures = len(set(c.rsplit('|', 1)[0] for c in comp_labels))
    print(f"\nCollected {len(training_records)} pairwise training samples "
          f"from {n_mixtures} mixtures.")

    # ----------------------------------------------------------
    # Stage 2: Manual functional form search
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  MANUAL FUNCTIONAL FORM SEARCH ({len(S_targets)} data points)")
    print(f"{'=' * 60}")

    manual_results = evaluate_manual_forms(features_df, S_targets, weights)

    print(f"\n  {'Functional Form':<50s}  {'R²':>8s}  {'MAE':>8s}")
    print(f"  {'─' * 50}  {'─' * 8}  {'─' * 8}")
    for name, r2, mae in manual_results:
        marker = " ◀ best" if r2 == manual_results[0][1] else ""
        print(f"  {name:<50s}  {r2:8.4f}  {mae:8.4f}{marker}")

    # ----------------------------------------------------------
    # Stage 3: Exhaustive feature combination search
    # ----------------------------------------------------------
    if HAS_SKLEARN:
        print(f"\n{'=' * 60}")
        print(f"  EXHAUSTIVE COMBINATION SEARCH")
        print(f"  {len(S_targets)} samples, {len(FEATURE_NAMES)} features")
        print(f"{'=' * 60}")

        combo_results = exhaustive_combination_search(
            features_df, S_targets, weights,
            min_features=2, max_features=4,
            max_degree=2, top_n=30,
        )

        if combo_results:
            print(f"\n  TOP 20 MODELS (by weighted R²_LOO):")
            print(f"  {'Rank':>4s}  {'R²_LOO':>8s}  {'MAE_LOO':>8s}  {'Deg':>3s}  Features")
            print(f"  {'─' * 4}  {'─' * 8}  {'─' * 8}  {'─' * 3}  {'─' * 40}")
            for rank, (r2, mae, deg, fnames, _) in enumerate(combo_results[:20], 1):
                print(f"  {rank:4d}  {r2:8.4f}  {mae:8.4f}  {deg:3d}  {', '.join(fnames)}")

            # Fit and report the best
            print(f"\n{'=' * 60}")
            print(f"  BEST MODEL DETAILS")
            print(f"{'=' * 60}")

            best_model = fit_and_report_best(
                features_df, S_targets, weights, combo_results[0],
            )
        else:
            combo_results = []
            print("\n  No valid models found.")
    else:
        combo_results = []

    # ----------------------------------------------------------
    # Stage 4: Diagnostics
    # ----------------------------------------------------------
    output_dir = os.path.join(get_scl_dir(), 'bPH_discovery')
    plot_diagnostics(
        features_df, S_targets, weights, comp_labels,
        manual_results, combo_results, output_dir,
    )


if __name__ == "__main__":
    main()
