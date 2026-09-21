"""
scl_analysis.py — Structural Coherence Length analysis from prepared PDF data.

Reads prepared CSV files produced by pdf_prepare.py (from Prepared_PDF_CSV/),
computes the structural coherence length, VDOS overlap, ω₀, and generates
publication-quality plots.

Usage:
    python scl_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import csv

from mendeleev import element

from scl_utils import (
    get_scl_dir,
    standardize_ion_pair,
    parse_composition,
    format_composition_with_subscripts,
)


# ==========================================
# Anion polarizability database (Å³)
# ==========================================
ANION_POLAR = {
    'F':  1.040,
    'Cl': 3.660,
    'Br': 4.770,
    'I':  7.100,
}
POLAR_REF =2# 2.0  # α₀ reference scale (Å³), geometric mean of F⁻ and Cl⁻


# ==========================================
# 1. Core Classes
# ==========================================

class IonPairData:
    """Stores PDF data and properties for a single ion pair on the common grid."""

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


class MoltenSaltPDF:
    """Analyse a single molten salt composition from a prepared PDF CSV."""

    def __init__(self, prepared_csv, comp_str, source, temp, scl_bc,
                 vdos_method='gaussian', b_ph_factors=None):
        self.original_comp_str = comp_str
        self.source = source
        self.temp = temp
        self.scl_bc = scl_bc
        self.vdos_method = vdos_method  # 'gaussian' or 'pmf'

        # b_PH factor selection
        # Valid factors: 'vdos', 'coordination', 'polarizability'
        # Default: {'vdos'} — VDOS overlap only (matches previous behavior)
        # Empty set / None: original concentration-only b_PH
        if b_ph_factors is None:
            self.b_ph_factors = {'vdos'}
        else:
            self.b_ph_factors = set(b_ph_factors)

        # Parse composition
        self.fractions, self.ion_counts, self.comp = parse_composition(comp_str)
        self.weights = self._calculate_weights()

        # Load prepared PDF
        self.ion_pairs, self.x_grid = self._load_prepared_pdf(prepared_csv)

        # Find features on grid
        for pair in self.ion_pairs.values():
            pair.find_features_on_grid(self.x_grid)

        self.weighted_splines = self._create_weighted_splines()

        # Results storage
        self.plot_data = []
        self.ion_pair_results = {}

        # VDOS (populated during analyze_pdf)
        self.vdos_data = {}       # pair_name -> (omega_grid, D_omega)
        self.vdos_overlaps = {}   # (pairA, pairB) -> S_AB

        # Coordination numbers (populated during analyze_pdf)
        self.coordination_numbers = {}  # pair_name -> CN_hat
        self.coord_mismatches = {}      # (pairA, pairB) -> G_AB

        # Anion polarizability factor
        self.polarizability_factor = self._calculate_polarizability_factor()

    # ------------------------------------------------------------------
    # Weight calculation
    # ------------------------------------------------------------------
    def _calculate_weights(self):
        el_conc = {}
        for salt, frac in self.fractions.items():
            for el, count in self.ion_counts[salt].items():
                el_conc[el] = el_conc.get(el, 0) + frac * count
        total_conc = sum(el_conc.values())
        rel_conc = {k: v / total_conc for k, v in el_conc.items()}

        weights = {}
        for el1, c1 in rel_conc.items():
            for el2, c2 in rel_conc.items():
                pair = standardize_ion_pair(f"{el1}-{el2}")
                weights[pair] = c1 * c2
        total_w = sum(weights.values())
        return {k: v / total_w for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Load prepared CSV
    # ------------------------------------------------------------------
    def _load_prepared_pdf(self, filepath):
        """Read a prepared CSV (with # comment header) into IonPairData objects."""
        if not os.path.exists(filepath):
            print(f"Error: Prepared file not found: {filepath}")
            return {}, np.linspace(0, 10, 2000)

        # Read metadata lines (skip # lines), then read the data
        df = pd.read_csv(filepath, comment='#')

        x_grid = df['r (A)'].values
        pairs = {}
        for col in df.columns:
            if col == 'r (A)':
                continue
            name = col  # already standardized by pdf_prepare
            y = df[col].values
            weight = self.weights.get(name, 0)
            pairs[name] = IonPairData(name, x_grid, y, weight)

        return pairs, x_grid

    # ------------------------------------------------------------------
    # Weighted splines
    # ------------------------------------------------------------------
    def _create_weighted_splines(self):
        splines = {}
        for name, p in self.ion_pairs.items():
            splines[name] = lambda x, s=p.spline, w=p.weight: s(x) * w
        return splines

    # ------------------------------------------------------------------
    # Gaussian fit to first peak of unweighted g(r)
    # ------------------------------------------------------------------
    def _fit_gaussian_to_peak(self, ion_pair_name):
        """Fit a split Gaussian to the first peak of unweighted g(r).

        Measures left and right half-widths at half-maximum separately to
        capture peak asymmetry (steep repulsive wall vs broad attractive tail).

        The split Gaussian is:
            Left  (r < r0): g_peak * exp(-(r-r0)²/(2σ_L²))           [baseline = 0]
            Right (r ≥ r0): (g_peak - g_base) * exp(-(r-r0)²/(2σ_R²)) + g_base

        Returns:
            dict with keys: r_peak, g_peak, sigma_left, sigma_right, g_base,
                            fwhm, hwhm_left, hwhm_right
            or None if no peak found.
        """
        if ion_pair_name not in self.ion_pairs:
            return None

        pair = self.ion_pairs[ion_pair_name]
        g = pair.spline(self.x_grid)  # unweighted g(r)

        dx = self.x_grid[1] - self.x_grid[0]
        min_dist = int(0.5 / dx)

        peaks, _ = find_peaks(
            g,
            prominence=0.1 * np.max(g) if np.max(g) > 0 else 0,
            distance=min_dist,
            width=2,
        )
        if len(peaks) == 0:
            return None

        p_idx = peaks[0]
        r_peak = self.x_grid[p_idx]
        g_peak = g[p_idx]
        if g_peak <= 0:
            return None

        # Find minimum after peak (g_base)
        g_after = g[p_idx:]
        mins, _ = find_peaks(-g_after, prominence=0.01 * np.max(g), distance=min_dist)
        if len(mins) > 0:
            m_idx = mins[0] + p_idx
            g_base = g[m_idx]
        else:
            search_end = min(len(self.x_grid) - 1, int(p_idx + (2.0 / dx)))
            if search_end > p_idx:
                m_idx = p_idx + np.argmin(g[p_idx:search_end])
                g_base = g[m_idx]
            else:
                g_base = 0.0
        g_base = max(g_base, 0.0)

        # --- Left side: baseline = 0 ---
        half_max_left = g_peak / 2.0
        left_region = g[:p_idx]
        left_crossings = np.where(left_region <= half_max_left)[0]
        if len(left_crossings) > 0:
            li = left_crossings[-1]
            if li + 1 < p_idx and g[li + 1] != g[li]:
                frac = (half_max_left - g[li]) / (g[li + 1] - g[li])
                x_left = self.x_grid[li] + frac * dx
            else:
                x_left = self.x_grid[li]
        else:
            x_left = self.x_grid[0]
        hwhm_left = r_peak - x_left

        # --- Right side: baseline = g_base ---
        half_max_right = (g_peak + g_base) / 2.0
        right_region = g[p_idx:]
        right_crossings = np.where(right_region <= half_max_right)[0]
        if len(right_crossings) > 0:
            ri = right_crossings[0] + p_idx
            if ri > p_idx and g[ri - 1] != g[ri]:
                frac = (half_max_right - g[ri - 1]) / (g[ri] - g[ri - 1])
                x_right = self.x_grid[ri - 1] + frac * dx
            else:
                x_right = self.x_grid[ri]
        else:
            x_right = self.x_grid[-1]
        hwhm_right = x_right - r_peak

        hwhm_left = max(hwhm_left, dx)
        hwhm_right = max(hwhm_right, dx)

        sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
        sigma_left = hwhm_left / sqrt_2ln2
        sigma_right = hwhm_right / sqrt_2ln2
        fwhm = hwhm_left + hwhm_right

        # --- Analytical PMF slopes at idealized Gaussian inflection points ---
        k_B = 8.314462618e-3
        exp_half = np.exp(-0.5)

        # Left side (pure Gaussian)
        center_left = r_peak - sigma_left
        g_val_left = g_peak * exp_half
        pmf_val_left = -k_B * self.temp * np.log(g_val_left)
        slope_left = -k_B * self.temp / sigma_left

        # Right side (Gaussian + baseline)
        center_right = r_peak + sigma_right
        g_val_right = (g_peak - g_base) * exp_half + g_base
        pmf_val_right = -k_B * self.temp * np.log(g_val_right)
        slope_right = (k_B * self.temp / sigma_right) * ((g_peak - g_base) * exp_half) / g_val_right

        return {
            'r_peak': r_peak,
            'g_peak': g_peak,
            'g_base': g_base,
            'sigma_left': sigma_left,
            'sigma_right': sigma_right,
            'hwhm_left': hwhm_left,
            'hwhm_right': hwhm_right,
            'fwhm': fwhm,
            'pmf_slope_left': slope_left,
            'pmf_slope_right': slope_right,
            'pmf_center_left': center_left,
            'pmf_center_right': center_right,
            'pmf_val_left': pmf_val_left,
            'pmf_val_right': pmf_val_right,
            'g_val_left': g_val_left,
            'g_val_right': g_val_right,
        }

    # ------------------------------------------------------------------
    # Anion polarizability factor
    # ------------------------------------------------------------------
    def _calculate_polarizability_factor(self):
        """p_a = α_a / (α_a + α₀) for the dominant anion in the mixture."""
        anion_conc = {}
        for salt, frac in self.fractions.items():
            for el, count in self.ion_counts[salt].items():
                if el in ANION_POLAR:
                    anion_conc[el] = anion_conc.get(el, 0) + frac * count
        if not anion_conc:
            return 0.5  # fallback
        dominant_anion = max(anion_conc, key=anion_conc.get)
        alpha_a = ANION_POLAR[dominant_anion]
        return alpha_a / (alpha_a + POLAR_REF)

    # ------------------------------------------------------------------
    # PMF & bond strength (from split Gaussian)
    # ------------------------------------------------------------------
    def calculate_pmf_and_bond_strength(self, ion_pair_name):
        """Bond strength from split-Gaussian fit: k = k_B T / σ_L²."""
        if ion_pair_name not in self.ion_pairs:
            return np.zeros_like(self.x_grid), None, None

        pair = self.ion_pairs[ion_pair_name]
        g_values = pair.spline(self.x_grid)
        g_safe = np.maximum(g_values, 1e-10)
        k_B = 8.314462618e-3  # kJ/(mol·K)
        pmf_values = -k_B * self.temp * np.log(g_safe)

        fit = self._fit_gaussian_to_peak(ion_pair_name)
        if fit is None:
            return pmf_values, None, None

        slope_L = fit.get('pmf_slope_left')
        slope_R = fit.get('pmf_slope_right')
        center_L = fit.get('pmf_center_left')
        center_R = fit.get('pmf_center_right')

        if slope_L is not None and slope_R is not None and center_R > center_L:
            # k = change in PMF slope / change in position = d²W/dr²
            bond_strength = (slope_R - slope_L) / (center_R - center_L)
        else:
            # Fallback to Gaussian σ_L method
            bond_strength = k_B * self.temp / (fit['sigma_left'] ** 2)
        return pmf_values, bond_strength, fit['r_peak']

    # ------------------------------------------------------------------
    # Reduced mass
    # ------------------------------------------------------------------
    def calculate_reduced_mass(self, ion_pair_name):
        elements_list = ion_pair_name.split('-')
        if len(elements_list) != 2:
            return None
        try:
            el1 = element(elements_list[0])
            el2 = element(elements_list[1])
            m1 = el1.mass * 1e-3
            m2 = el2.mass * 1e-3
            return (m1 * m2) / (m1 + m2)
        except Exception as e:
            print(f"Error calculating reduced mass for {ion_pair_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Fundamental frequency ω₀
    # ------------------------------------------------------------------
    def calculate_fundamental_frequency(self, ion_pair_name, bond_strength):
        """ω₀ = √(k/μ) in rad/ps."""
        reduced_mass = self.calculate_reduced_mass(ion_pair_name)
        if reduced_mass is None or bond_strength is None or bond_strength <= 0:
            return None
        N_A = 6.02214076e23
        k_si = bond_strength * 1000 / 1e-20
        k_mol = k_si / N_A
        mu_mol = reduced_mass / N_A
        omega = np.sqrt(k_mol / mu_mol) * 1e-12
        return omega

    # ------------------------------------------------------------------
    # VDOS (Gaussian approximation — for visualization)
    # ------------------------------------------------------------------
    def calculate_vdos_gaussian(self, ion_pair_name):
        """Gaussian VDOS centered at ω₀ with anharmonicity-informed width."""
        fit = self._fit_gaussian_to_peak(ion_pair_name)
        if fit is None:
            return None, None
        _, bond_strength, _ = self.calculate_pmf_and_bond_strength(ion_pair_name)
        omega_0 = self.calculate_fundamental_frequency(ion_pair_name, bond_strength)
        if omega_0 is None or omega_0 <= 0:
            return None, None

        b_kf = fit['g_base'] / fit['g_peak'] if fit['g_peak'] > 0 else 0
        sigma_avg = (fit['sigma_left'] + fit['sigma_right']) / 2.0
        sigma_omega = omega_0 * (sigma_avg / fit['r_peak']) * (1.0 + b_kf)
        sigma_omega = max(sigma_omega, omega_0 * 0.01)

        omega_grid = np.linspace(0, omega_0 + 6 * sigma_omega, 500)
        D_omega = np.exp(-0.5 * ((omega_grid - omega_0) / sigma_omega) ** 2)
        D_omega /= (sigma_omega * np.sqrt(2 * np.pi))
        area = np.trapezoid(D_omega, omega_grid)
        if area > 0:
            D_omega /= area
        return omega_grid, D_omega

    # ------------------------------------------------------------------
    # VDOS (PMF-based)
    # ------------------------------------------------------------------
    def calculate_vdos_pmf(self, ion_pair_name):
        """PMF-based VDOS using Savgol second derivative + Jacobian mapping."""
        if ion_pair_name not in self.ion_pairs:
            return None, None
        pair = self.ion_pairs[ion_pair_name]
        g = pair.spline(self.x_grid)
        g_safe = np.maximum(g, 1e-10)
        k_B = 8.314462618e-3
        pmf = -k_B * self.temp * np.log(g_safe)

        dx = self.x_grid[1] - self.x_grid[0]
        wl = max(int(0.5 / dx) | 1, 5)
        if wl % 2 == 0:
            wl += 1
        po = min(3, wl - 1)
        if len(pmf) <= wl:
            return None, None

        d2pmf = savgol_filter(pmf, window_length=wl, polyorder=po, deriv=2, delta=dx)
        reduced_mass = self.calculate_reduced_mass(ion_pair_name)
        if reduced_mass is None:
            return None, None

        N_A = 6.02214076e23
        mu_kg = reduced_mass / N_A

        fit = self._fit_gaussian_to_peak(ion_pair_name)
        if fit is None:
            return None, None

        peak_idx = np.argmin(np.abs(self.x_grid - fit['r_peak']))
        left = peak_idx
        while left > 0 and d2pmf[left] > 0:
            left -= 1
        right = peak_idx
        while right < len(d2pmf) - 1 and d2pmf[right] > 0:
            right += 1
        if right - left < 3:
            return None, None

        r_region = self.x_grid[left:right]
        g_region = g[left:right]
        k_region = d2pmf[left:right]
        valid = k_region > 0
        if valid.sum() < 3:
            return None, None

        r_valid = r_region[valid]
        g_valid = g_region[valid]
        k_valid = k_region[valid]

        k_si = k_valid * 1000 / 1e-20 / N_A
        omega_r = np.sqrt(k_si / mu_kg) * 1e-12
        weights_r = r_valid ** 2 * g_valid

        n_bins = min(100, max(20, len(omega_r) // 2))
        omega_min_v, omega_max_v = omega_r.min(), omega_r.max()
        if omega_max_v <= omega_min_v:
            return None, None

        bin_edges = np.linspace(omega_min_v * 0.9, omega_max_v * 1.1, n_bins + 1)
        D_omega = np.zeros(n_bins)
        omega_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for w, om in zip(weights_r, omega_r):
            bi = np.searchsorted(bin_edges, om) - 1
            bi = np.clip(bi, 0, n_bins - 1)
            D_omega[bi] += w

        area = np.trapezoid(D_omega, omega_centers)
        if area > 0:
            D_omega /= area

        sm_wl = min(max(n_bins // 5, 5), n_bins)
        if sm_wl % 2 == 0:
            sm_wl += 1
        if sm_wl >= 5 and len(D_omega) > sm_wl:
            D_omega = savgol_filter(D_omega, sm_wl, min(3, sm_wl - 1))
            D_omega = np.maximum(D_omega, 0)
            area = np.trapezoid(D_omega, omega_centers)
            if area > 0:
                D_omega /= area

        return omega_centers, D_omega

    def calculate_vdos(self, ion_pair_name):
        if self.vdos_method == 'pmf':
            return self.calculate_vdos_pmf(ion_pair_name)
        return self.calculate_vdos_gaussian(ion_pair_name)

    # ------------------------------------------------------------------
    # VDOS overlap (for visualization + 'vdos' b_PH factor)
    # ------------------------------------------------------------------
    def _compute_vdos_overlaps(self):
        """Compute VDOS for all ca pairs and build the overlap matrix."""
        ca_pairs = [n for n, p in self.ion_pairs.items() if p.type == 'ca']
        self.vdos_data = {}
        for name in ca_pairs:
            omega, D = self.calculate_vdos(name)
            if omega is not None and D is not None:
                self.vdos_data[name] = (omega, D)

        self.vdos_overlaps = {}
        computed = [n for n in ca_pairs if n in self.vdos_data]
        for i, a in enumerate(computed):
            self.vdos_overlaps[(a, a)] = 1.0
            for b in computed[i + 1:]:
                om_a, D_a = self.vdos_data[a]
                om_b, D_b = self.vdos_data[b]
                lo = min(om_a[0], om_b[0])
                hi = max(om_a[-1], om_b[-1])
                om_common = np.linspace(lo, hi, 1000)
                Da_i = np.interp(om_common, om_a, D_a, left=0, right=0)
                Db_i = np.interp(om_common, om_b, D_b, left=0, right=0)
                overlap = np.trapezoid(np.minimum(Da_i, Db_i), om_common)
                overlap = np.clip(overlap, 0, 1)
                self.vdos_overlaps[(a, b)] = overlap
                self.vdos_overlaps[(b, a)] = overlap

        if len(computed) > 1:
            print(f"  VDOS overlap matrix ({self.vdos_method}):")
            header = "          " + "  ".join(f"{n:>8s}" for n in computed)
            print(header)
            for a in computed:
                row = f"  {a:>8s}"
                for b in computed:
                    s = self.vdos_overlaps.get((a, b), 0)
                    row += f"  {s:8.3f}"
                print(row)

    # ------------------------------------------------------------------
    # Penalty-based similarity
    # ------------------------------------------------------------------
    def _compute_penalty_similarity(self):
        """Compute pairwise S_ij using mass penalty, bond penalty, and polarizability.

        m_p = |m_c1 - m_c2| / mean(m_c1, m_c2)    (relative cation mass mismatch)
        K_p = |k'_1 - k'_2| / mean(k'_1, k'_2)    (relative bond stiffness mismatch)
            where k' = -ln(1 - b_KF),  b_KF = 1 - (g_peak - g_min) / g_peak
        P   = anion polarizability (Å³)

        S_ij = exp(-(m_p² + K_p²) / P)
        """
        ca_pairs = [p for p in self.ion_pairs.values() if p.type == 'ca']
        ca_names = [p.name for p in ca_pairs]

        # --- Gather per-pair properties ---
        pair_props = {}
        for p in ca_pairs:
            cation = p.name.split('-')[0]
            anion = p.name.split('-')[1]
            m_cat = element(cation).mass  # amu
            m_an = element(anion).mass  # amu

            # b_KF from unweighted g(r)
            fit = self._fit_gaussian_to_peak(p.name)
            if fit is not None and fit['g_peak'] > 1e-6:
                b_kf = fit['g_base'] / fit['g_peak']
            else:
                b_kf = 0.0
            b_kf = np.clip(b_kf, 0, 0.9999)  # avoid ln(0)

            k_prime = -np.log(b_kf)

            fwhm = fit['fwhm'] if fit is not None else None

            r_peak = fit['r_peak'] if fit is not None else None

            # Anion polarizability
            polar = ANION_POLAR.get(anion, 2.0)

            pair_props[p.name] = {
                'm_cat': m_cat,
                'm_an': m_an,
                'k_prime': k_prime,
                'b_kf': b_kf,
                'polar': polar,
                'anion': anion,
                'r_peak': r_peak,
                'fwhm': fwhm
            }

        # --- Compute pairwise S_ij ---
        self.penalty_overlaps = {}
        for i, a in enumerate(ca_names):
            self.penalty_overlaps[(a, a)] = 1.0
            for b in ca_names[i + 1:]:
                pa, pb = pair_props[a], pair_props[b]

                # Reduced mass of each pair
                m_A, m_B, m_an = pa['m_cat'], pb['m_cat'], pa['m_an']
                mu_A = (m_A * m_an) / (m_A + m_an)
                mu_B = (m_B * m_an) / (m_B + m_an)
                m_mean = (mu_A + mu_B) / 2.0
                delta_m_rel = abs(mu_A - mu_B) / m_mean if m_mean > 0 else 0.0
                mu_ratio = min(mu_A, mu_B) / max(mu_A, mu_B)

                # Relative peak radius
                r_A, r_B = pa.get('r_peak', 0), pb.get('r_peak', 0)
                r_ratio = min(r_A, r_B) / max(r_A, r_B) if max(r_A, r_B) > 0 else 1.0
                r_mean = (r_A + r_B) / 2.0
                delta_r_rel = abs(r_A - r_B) / r_mean

                # Mass penalty
                m1, m2 = pa['m_cat'], pb['m_cat']
                m_mean = (m1 + m2) / 2.0
                m_p = abs(m1 - m2) / m_mean if m_mean > 0 else 0.0
                m_tri_p = (27 * m1 * m2 * m_an) / ((m1 + m2 + m_an)**3)  ########################################################

                # b_KF
                bkf1, bkf2 = pa['b_kf'], pb['b_kf']
                bkf_mean = (bkf1 + bkf2) / 2.0
                bkf1_log = -np.log(bkf1) if bkf1 > 0 else 0.0
                bkf2_log = -np.log(bkf2) if bkf2 > 0 else 0.0
                bkf_mean_log = 1 - np.sqrt((bkf1_log - bkf2_log)**2) / np.maximum(np.maximum(bkf1_log, bkf2_log), 1e-10) #

                # fwhm
                w1, w2 = pa.get('fwhm', 0), pb.get('fwhm', 0)
                w_mean = (w1 + w2) / 2.0
                delta_w_rel = abs(w1 - w2) / w_mean if w_mean > 0 else 0.0

                # Bond penalty
                k1, k2 = pa['k_prime'], pb['k_prime']
                k_mean = (k1 + k2) / 2.0
                k1_log = -np.log(1 - k1) if k1 > 0 else 0.0
                k2_log = -np.log(1 - k2) if k2 > 0 else 0.0
                K_p = abs(k1 - k2) / k_mean if k_mean > 1e-10 else 0.0
                # k_mean_log = (k1_log - k2_log) / np.maximum(np.maximum(k1_log, k2_log), 1e-10) ########################################################

                # Anion polarizability (use the shared anion, or average)
                if pa['anion'] == pb['anion']:
                    P = pa['polar']
                else:
                    P = (pa['polar'] + pb['polar']) / 2.0
                P = max(P, 0.01)  # prevent division by zero

                # Average polarizability over radial separation
                r_mean = (pa.get('r_peak', 0) + pb.get('r_peak', 0)) / 2.0
                if r_mean > 0:
                    P_factor = (P / r_mean)**3   

                S_ij = np.minimum(bkf_mean_log,m_tri_p)     #np.exp(-(m_p**3 * K_p**3) / P_factor) #P)
                self.penalty_overlaps[(a, b)] = S_ij
                self.penalty_overlaps[(b, a)] = S_ij

        # Print matrix
        if len(ca_names) > 1:
            print(f"  Penalty similarity matrix:")
            header = "          " + "  ".join(f"{n:>8s}" for n in ca_names)
            print(header)
            for a in ca_names:
                row = f"  {a:>8s}"
                for b in ca_names:
                    s = self.penalty_overlaps.get((a, b), 0)
                    row += f"  {s:8.3f}"
                print(row)
                # Print components for off-diagonal
                if a != ca_names[-1]:
                    for b in ca_names:
                        if b != a and (a, b) in self.penalty_overlaps:
                            pa_p, pb_p = pair_props[a], pair_props[b]
                            m1, m2 = pa_p['m_cat'], pb_p['m_cat']
                            mp = abs(m1 - m2) / ((m1 + m2) / 2)
                            k1, k2 = pa_p['k_prime'], pb_p['k_prime']
                            km = (k1 + k2) / 2
                            Kp = abs(k1 - k2) / km if km > 1e-10 else 0
                            P = pa_p['polar'] if pa_p['anion'] == pb_p['anion'] else (pa_p['polar'] + pb_p['polar']) / 2
                            print(f"    {a}-{b}: m_p={mp:.3f}, K_p={Kp:.3f}, P={P:.2f}")

    # ==================================================================
    # DISPERSION-BASED PHONON TRANSFER
    # ==================================================================
    #
    # Physical model:
    #   Each cation-anion pair is modelled as a 1D diatomic chain with:
    #     - masses m_c, m_a (amu) from element data
    #     - lattice constant a = r_peak (nearest-neighbour distance from PDF)
    #     - relative force constant k_rel = 1 - b_KF (from peak/min ratio)
    #
    #   The BZ boundary is at q_ZB = π/(2a).
    #
    #   The FWHM of g(r) represents structural disorder — a spread of
    #   effective lattice constants.  This maps to a q-window around q_ZB.
    #
    #   Transfer efficiency S_ij is the fraction of chain i's q-window
    #   frequencies that chain j also supports in its q-window.
    #   If frequency doesn't match: total reflection.
    #   Wavevector tolerance comes from the FWHM q-window.
    #
    # ==================================================================

    def _get_chain_params(self, pair_name):
        """Extract 1D diatomic chain parameters from PDF peak data."""
        fit = self._fit_gaussian_to_peak(pair_name)
        if fit is None:
            return None
        parts = pair_name.split('-')
        if len(parts) != 2:
            return None
        try:
            el_c = element(parts[0])
            el_a = element(parts[1])
        except Exception:
            return None

        b_kf = fit['g_base'] / fit['g_peak'] if fit['g_peak'] > 0 else 1.0
        return {
            'name':  pair_name,
            'm_c':   el_c.mass,          # amu
            'm_a':   el_a.mass,          # amu
            'a':     fit['r_peak'],      # Å  (nearest-neighbour distance)
            'k_rel': max(1.0 - b_kf, 1e-6),
            'fwhm':  fit['fwhm'],        # Å
            'b_kf':  b_kf,
        }

    @staticmethod
    def _diatomic_dispersion(k_rel, m_c, m_a, a, n_q=500):
        """1D diatomic chain dispersion relation.

        BZ boundary at q_ZB = π/(2a)  where a is nearest-neighbour distance.
        Returns q, ω_acoustic, ω_optic in consistent relative units
        (sqrt(k_rel / amu)).
        """
        q_max = np.pi / (2.0 * a)
        q = np.linspace(0, q_max, n_q)

        inv_mc = 1.0 / m_c
        inv_ma = 1.0 / m_a
        mass_sum = inv_mc + inv_ma

        sin2 = np.sin(q * a) ** 2
        discriminant = mass_sum ** 2 - 4.0 * sin2 / (m_c * m_a)
        discriminant = np.maximum(discriminant, 0)

        omega_ac = np.sqrt(np.maximum(k_rel * (mass_sum - np.sqrt(discriminant)), 0))
        omega_op = np.sqrt(np.maximum(k_rel * (mass_sum + np.sqrt(discriminant)), 0))

        return q, omega_ac, omega_op

    @staticmethod
    def _q_window_from_fwhm(a, fwhm):
        """Compute q-range around BZ boundary from FWHM of g(r) peak.

        Bond lengths vary from  a - FWHM/2  to  a + FWHM/2.
        Since q = π/(2a), larger a → smaller q and vice-versa.

        Returns (q_lo, q_hi, q_center).
        """
        q_center = np.pi / (2.0 * a)
        a_min = max(a - fwhm / 2.0, 0.1)   # avoid div-by-zero
        a_max = a + fwhm / 2.0
        q_hi = np.pi / (2.0 * a_min)        # smaller a → larger q
        q_lo = np.pi / (2.0 * a_max)        # larger a → smaller q
        return q_lo, q_hi, q_center

    def _compute_dispersion_overlaps(self):
        """Compute S_ij from q-windowed frequency matching.

        For each pair i we:
          1. Compute the dispersion (acoustic + optic branches)
          2. Define the q-window from the FWHM of the peak
          3. Extract the frequency ranges accessible in that window

        Transfer fraction T_{i→j} = fraction of chain i's q-window
        frequencies that fall within chain j's allowed frequency bands
        (in j's q-window).

        S_ij = sqrt(T_{i→j} · T_{j→i}).
        """
        ca_pairs = [n for n, p in self.ion_pairs.items() if p.type == 'ca']

        # Collect chain parameters
        chain_params = {}
        for name in ca_pairs:
            cp = self._get_chain_params(name)
            if cp is not None:
                chain_params[name] = cp

        # Initialise self-overlap
        self.dispersion_overlaps = {}
        self.dispersion_chain_data = {}   # stored for plotting

        for name in chain_params:
            self.dispersion_overlaps[(name, name)] = 1.0

        if len(chain_params) < 2:
            return

        # Compute dispersion + q-window data for each chain
        chain_data = {}
        for name, cp in chain_params.items():
            q, omega_ac, omega_op = self._diatomic_dispersion(
                cp['k_rel'], cp['m_c'], cp['m_a'], cp['a'],
            )
            q_lo, q_hi, q_center = self._q_window_from_fwhm(cp['a'], cp['fwhm'])

            # Clip q-window to BZ
            q_lo = max(q_lo, q[0])
            q_hi = min(q_hi, q[-1])

            # Frequencies in q-window
            in_window = (q >= q_lo) & (q <= q_hi)
            if in_window.sum() < 2:
                # Window too narrow — widen slightly
                margin = (q_hi - q_lo) * 0.2
                in_window = (q >= q_lo - margin) & (q <= q_hi + margin)

            ac_win = omega_ac[in_window]
            op_win = omega_op[in_window]

            # Band ranges in q-window
            ac_band = (float(ac_win.min()), float(ac_win.max())) if len(ac_win) > 0 else (0, 0)
            op_band = (float(op_win.min()), float(op_win.max())) if len(op_win) > 0 else (0, 0)

            chain_data[name] = {
                'q': q,
                'omega_ac': omega_ac,
                'omega_op': omega_op,
                'q_lo': q_lo,
                'q_hi': q_hi,
                'q_center': q_center,
                'ac_band': ac_band,
                'op_band': op_band,
                'ac_freqs': ac_win,   # frequency samples in q-window
                'op_freqs': op_win,
                'params': cp,
            }

        self.dispersion_chain_data = chain_data

        # Pairwise transfer efficiency
        names = list(chain_data.keys())
        for i, a_name in enumerate(names):
            for b_name in names[i + 1:]:
                T_ab = self._transfer_fraction(chain_data[a_name], chain_data[b_name])
                T_ba = self._transfer_fraction(chain_data[b_name], chain_data[a_name])
                S = np.sqrt(max(T_ab * T_ba, 0))
                self.dispersion_overlaps[(a_name, b_name)] = S
                self.dispersion_overlaps[(b_name, a_name)] = S

        # Print matrix
        print(f"  Dispersion transfer matrix (q-windowed):")
        header = "          " + "  ".join(f"{n:>8s}" for n in names)
        print(header)
        for a in names:
            row = f"  {a:>8s}"
            for b in names:
                s = self.dispersion_overlaps.get((a, b), 0)
                row += f"  {s:8.3f}"
            print(row)

    def _calculate_penalty_b_ph(self, pair_name, ca_pairs, sum_ca_weights):
        """b_PH using penalty-based similarity."""
        pair = self.ion_pairs[pair_name]

        if not hasattr(self, 'penalty_overlaps') or pair_name not in [p.name for p in ca_pairs]:
            if sum_ca_weights > 0:
                return np.clip(1 - (pair.weight / sum_ca_weights), 0, 1)
            return 1.0

        numerator = 0.0
        denominator = 0.0
        for other in ca_pairs:
            x_j = other.weight
            S_ij = self.penalty_overlaps.get((pair_name, other.name), 0)
            if other.name == pair_name:
                S_ij = 1.0
            numerator += x_j * S_ij
            denominator += x_j

        if denominator <= 0:
            return 1.0
        return np.clip(1.0 - numerator / denominator, 0, 1)

    @staticmethod
    def _transfer_fraction(source, target):
        """Fraction of source's q-window frequencies that fall in target's allowed bands.

        A frequency ω from source chain can transmit if target chain has an
        allowed mode at that frequency (in target's q-window).  The allowed
        bands are the acoustic and optic frequency intervals of the target
        restricted to its q-window.
        """
        source_freqs = np.concatenate([source['ac_freqs']])#, source['op_freqs']])
        if len(source_freqs) == 0:
            return 0.0

        ac_lo, ac_hi = target['ac_band']
        op_lo, op_hi = target['op_band']

        in_ac = (source_freqs >= ac_lo) & (source_freqs <= ac_hi)
        in_op = (source_freqs >= op_lo) & (source_freqs <= op_hi)
        transmitted = (in_ac).sum() # (in_ac | in_op).sum()
        return float(transmitted) / len(source_freqs)

    # ------------------------------------------------------------------
    # Coordination number from PDF
    # ------------------------------------------------------------------
    def _compute_coordination_numbers(self):
        """Compute relative CN by integrating r²g(r) under the first peak."""
        ca_pairs = [n for n, p in self.ion_pairs.items() if p.type == 'ca']
        self.coordination_numbers = {}

        for name in ca_pairs:
            pair = self.ion_pairs[name]
            g = pair.spline(self.x_grid)
            fit = self._fit_gaussian_to_peak(name)
            if fit is None:
                continue

            r_peak = fit['r_peak']
            r_min = pair.minima[0] if pair.minima[0] else r_peak * 1.5

            # Find onset (where g first exceeds 0.05)
            onset_idx = np.where(g > 0.05)[0]
            if len(onset_idx) == 0:
                continue
            r_onset = self.x_grid[onset_idx[0]]

            mask = (self.x_grid >= r_onset) & (self.x_grid <= r_min)
            if mask.sum() < 3:
                continue

            r_sel = self.x_grid[mask]
            g_sel = g[mask]
            cn_hat = np.trapezoid(r_sel ** 2 * g_sel, r_sel)
            self.coordination_numbers[name] = cn_hat

        # Build pairwise mismatch G_AB = min/max
        self.coord_mismatches = {}
        computed = [n for n in ca_pairs if n in self.coordination_numbers]
        for i, a in enumerate(computed):
            self.coord_mismatches[(a, a)] = 1.0
            for b in computed[i + 1:]:
                cn_a = self.coordination_numbers[a]
                cn_b = self.coordination_numbers[b]
                mx = max(cn_a, cn_b)
                G = min(cn_a, cn_b) / mx if mx > 0 else 1.0
                self.coord_mismatches[(a, b)] = G
                self.coord_mismatches[(b, a)] = G

        if len(computed) > 1 and 'coordination' in self.b_ph_factors:
            print(f"  Coordination numbers (relative ∫r²g dr):")
            for n in computed:
                print(f"    {n}: CN_hat = {self.coordination_numbers[n]:.3f}")

    # ------------------------------------------------------------------
    # Combined b_PH
    # ------------------------------------------------------------------
    def _calculate_combined_b_ph(self, pair_name, ca_pairs, sum_ca_weights):
        """Compute b_PH using the user-selected combination of factors.

        Factors (selected via self.b_ph_factors):
            'vdos'           — VDOS overlap S_ij (Gaussian or PMF)
            'dispersion'     — q-windowed dispersion transfer S_ij
            'coordination'   — coordination geometry match G_ij
            'polarizability' — anion polarizability accommodation p_a
            'no_bPH'           — forces b_PH = 0 (no phase difference disruption)
            'penalty'        — mass + bond penalty + polarizability S_ij 

        If no factors selected (empty set), falls back to original
        concentration-only formula.
        """
        pair = self.ion_pairs[pair_name]
        factors = self.b_ph_factors

        # --- Force b_PH = 0 ---
        if 'no_bPH' in factors:
            return 0.0

        # --- Fallback: original concentration-only ---
        if not factors:
            if sum_ca_weights > 0:
                return np.clip(1.0 - pair.weight / sum_ca_weights, 0, 1)
            return 1.0

        # --- Determine S_ij source ---
        use_dispersion = 'dispersion' in factors
        use_vdos = 'vdos' in factors and not use_dispersion
        use_coord = 'coordination' in factors
        use_polar = 'polarizability' in factors
        use_penalty = 'penalty' in factors
        numerator = 0.0
        denominator = 0.0

        for other in ca_pairs:
            x_j = other.weight
            denominator += x_j

            # S_ij: frequency / dispersion overlap
            if pair_name == other.name:
                S_ij = 1.0
            elif use_dispersion:
                S_ij = self.dispersion_overlaps.get((pair_name, other.name), 0)
            elif use_vdos:
                S_ij = self.vdos_overlaps.get((pair_name, other.name), 0)
                if pair_name == other.name:
                    S_ij = 1.0
            elif use_penalty:
                S_ij = self.penalty_overlaps.get((pair_name, other.name), 0)
                if pair_name == other.name:
                    S_ij = 1.0
            else:
                S_ij = 1.0   # no frequency factor

            # G_ij: coordination mismatch
            if use_coord:
                G_ij = self.coord_mismatches.get((pair_name, other.name), 1.0)
            else:
                G_ij = 1.0

            # p_a: polarizability accommodation of the mismatch
            if use_polar and G_ij < 1.0:
                p_a = self.polarizability_factor
                C_eff = G_ij + (1.0 - G_ij) * p_a
            else:
                C_eff = G_ij

            numerator += x_j * S_ij * C_eff

        if denominator <= 0:
            return 1.0
        b_ph = 1.0 - numerator / denominator
        return np.clip(b_ph, 0, 1)

    def _load_vdos_metrics(self):
        """Loads VDOS interaction metrics M from the CSV for the current salt system."""
        csv_path = os.path.join(get_scl_dir(), 'vdos_metrics_summary.csv')
        self.vdos_metrics = {'L_classical': {}, 'L_quantum': {}, 'O_anionv': {}}
        if not os.path.exists(csv_path):
            return

        try:
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                # Parse composition safely to catch arbitrary salt ordering
                _, _, row_comp = parse_composition(str(row['composition']))
                if row_comp == self.comp and abs(float(row['temperature_K']) - self.temp) < 1.0:
                    for i in range(1, 10):  # Check up to 9 possible cation-cation pairs
                        label_col = f'ca_{i}_label'
                        if label_col in row and pd.notna(row[label_col]):
                            cc_label = str(row[label_col]).strip()
                            if '-' not in cc_label:
                                continue
                            c1, c2 = cc_label.split('-')
                            pair_key = tuple(sorted([c1, c2]))
                            
                            lc_col = f'{i}__L_classical__mean'
                            lq_col = f'{i}__L_quantum__mean'
                            oa_col = f'{i}__O_anionv__mean'
                            
                            if lc_col in row and pd.notna(row[lc_col]):
                                self.vdos_metrics['L_classical'][pair_key] = float(row[lc_col])
                            if lq_col in row and pd.notna(row[lq_col]):
                                self.vdos_metrics['L_quantum'][pair_key] = float(row[lq_col])
                            if oa_col in row and pd.notna(row[oa_col]):
                                self.vdos_metrics['O_anionv'][pair_key] = float(row[oa_col])
                    break # Break early once the matching composition and temp are found
        except Exception as e:
            print(f"Warning: Could not load VDOS metrics correctly: {e}")

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze_pdf(self):
        print(f"\n### Analysis for {self.comp} ###")
        self._load_vdos_metrics() # Fetch the metrics from the CSV

        # Pre-compute overlaps / coordination as needed
        self._compute_vdos_overlaps()
        if 'penalty' in self.b_ph_factors:
            self._compute_penalty_similarity()
        if 'dispersion' in self.b_ph_factors:
            self._compute_dispersion_overlaps()
        if 'coordination' in self.b_ph_factors:
            self._compute_coordination_numbers()

        var_keys = ['1', '0', 'Lc', 'Lq', 'Oa']
        total_weighted_scl = {k: 0.0 for k in var_keys}
        total_weight_norm = 0

        ca_pairs = [p for _, p in self.ion_pairs.items() if p.type == 'ca']
        sum_ca_weights = sum(p.weight for p in ca_pairs)

        for pair in ca_pairs:
            name = pair.name
            print(f"Analyzing Pair: {name}")

            r_peak = pair.peak[0]
            if r_peak is None or r_peak <= 0:
                print(f"  Skipping {name}: No valid peak found.")
                continue

            delta_r = r_peak
            transfer_points = np.arange(delta_r, self.x_grid[-1], delta_r)
            if len(transfer_points) == 0:
                continue

            # --- Disruption factors ---
            g_peak = pair.peak[1]
            g_min = pair.minima[1] if pair.minima[0] else 0
            kf_val = 1.0
            if g_peak > 1e-6:
                kf_val = 1 - (g_peak - g_min) / g_peak
            kf_val = np.clip(kf_val, 0, 1)

            cation = name.split('-')[0]
            cc_name = standardize_ion_pair(f"{cation}-{cation}")
            has_cc = cc_name in self.ion_pairs

            # Calculate b_PH for the 5 variations concurrently
            ph_vals = {k: 0.0 for k in var_keys}
            for other in ca_pairs:
                if other.name == name:
                    continue
                x_j = other.weight / sum_ca_weights if sum_ca_weights > 0 else 0
                c_j = other.name.split('-')[0]
                pair_key = tuple(sorted([cation, c_j]))
                
                ph_vals['1'] += x_j * 1.0
                ph_vals['0'] += x_j * 0.0
                ph_vals['Lc'] += x_j * (1 - self.vdos_metrics['L_classical'].get(pair_key, 1.0))
                ph_vals['Lq'] += x_j * (1 - self.vdos_metrics['L_quantum'].get(pair_key, 1.0))
                ph_vals['Oa'] += x_j * (1 - self.vdos_metrics['O_anionv'].get(pair_key, 1.0))

            for k in var_keys:
                ph_vals[k] = np.clip(ph_vals[k], 0, 1)

            print(f"  b_PH_1: {ph_vals['1']:.4f} | b_PH_0: {ph_vals['0']:.4f} | b_PH_Lc: {ph_vals['Lc']:.4f} | b_PH_Lq: {ph_vals['Lq']:.4f} | b_PH_Oa: {ph_vals['Oa']:.4f}")

            b_NI_vals = []
            beta_vals = {k: [] for k in var_keys}

            for m, r_m in enumerate(transfer_points, 1):
                g_tot_val = 0
                for pn, pd in self.ion_pairs.items():
                    if cation in pn.split('-'):
                        g_tot_val += self.weighted_splines[pn](r_m)

                if m % 2 == 1:
                    g_ideal_val = self.weighted_splines[name](r_m)
                else:
                    if has_cc:
                        g_ideal_val = self.weighted_splines[cc_name](r_m)
                    else:
                        g_ideal_val = self.weights.get(cc_name, 0)

                ni_val = 1.0
                if g_tot_val > 1e-6:
                    ni_val = 1 - (g_ideal_val / g_tot_val)
                ni_val = np.clip(ni_val, 0, 1)

                b_NI_vals.append(ni_val)

                for k in var_keys:
                    ph = ph_vals[k]
                    if kf_val == 1 or ph == 1 or ni_val == 1:
                        beta = float('inf')
                    else:
                        beta = (kf_val / (1 - kf_val)) + (ph / (1 - ph)) + (ni_val / (1 - ni_val))
                    beta_vals[k].append(beta)

            # --- Survival function ---
            scl_pairs = {k: 0.0 for k in var_keys}
            S_y_grids = {k: np.zeros_like(self.x_grid) for k in var_keys}

            for k in var_keys:
                S_discrete = [1.0]
                int_beta = 0
                for beta in beta_vals[k]:
                    if beta == float('inf'):
                        int_beta = -float('inf')
                    else:
                        int_beta -= beta * delta_r
                    S_discrete.append(np.exp(int_beta))

                scl_pairs[k] = delta_r * sum(S_discrete[:-1])
                
                # Keep S(r) for Variation 1 active for the SCL plotting tools
                if k == '1':
                    curr_s_idx = 0
                    for i, x in enumerate(self.x_grid):
                        if curr_s_idx < len(transfer_points):
                            if x >= transfer_points[curr_s_idx]:
                                curr_s_idx += 1
                        if curr_s_idx < len(S_discrete):
                            S_y_grids[k][i] = S_discrete[curr_s_idx]
                        else:
                            S_y_grids[k][i] = S_discrete[-1]

            RTE = pair.weight / sum_ca_weights if sum_ca_weights > 0 else 0
            for k in var_keys:
                total_weighted_scl[k] += scl_pairs[k] * RTE
            total_weight_norm += RTE

            print(f"  SCL_1: {scl_pairs['1']:.3f} A (Weight: {RTE:.3f})")

            # PMF and ω₀
            pmf_values, bond_strength, peak_x = self.calculate_pmf_and_bond_strength(name)
            reduced_mass = self.calculate_reduced_mass(name)
            fundamental_frequency = self.calculate_fundamental_frequency(name, bond_strength)
            fit = self._fit_gaussian_to_peak(name)

            self.ion_pair_results[name] = {
                'scl_1': scl_pairs['1'],
                'scl_0': scl_pairs['0'],
                'scl_Lc': scl_pairs['Lc'],
                'scl_Lq': scl_pairs['Lq'],
                'scl_Oa': scl_pairs['Oa'],
                'peak_x': pair.peak[0],
                'peak_y': pair.peak[1],
                'minima_x': pair.minima[0],
                'minima_y': pair.minima[1],
                'cc_peak_x': self.ion_pairs[cc_name].peak[0] if has_cc else None,
                'cc_peak_y': self.ion_pairs[cc_name].peak[1] if has_cc else None,
                'pmf_at_peak': pmf_values[np.argmin(np.abs(self.x_grid - pair.peak[0]))] if bond_strength is not None else None,
                'bond_strength': bond_strength,
                'reduced_mass': reduced_mass * 1000 if reduced_mass else None,
                'fundamental_frequency': fundamental_frequency,
                'b_ph_1': ph_vals['1'],
                'b_ph_0': ph_vals['0'],
                'b_ph_Lc': ph_vals['Lc'],
                'b_ph_Lq': ph_vals['Lq'],
                'b_ph_Oa': ph_vals['Oa'],
                'gaussian_fit': fit,
            }

            self.plot_data.append({
                'ion_pair_ca_i': name,
                'x_range': self.x_grid,
                'S_i': S_y_grids['1'],
                'x_SCL_pair': scl_pairs['1'],
            })

        self.avg_SCL_1 = total_weighted_scl['1'] / total_weight_norm if total_weight_norm > 0 else 0
        self.avg_SCL_0 = total_weighted_scl['0'] / total_weight_norm if total_weight_norm > 0 else 0
        self.avg_SCL_Lc = total_weighted_scl['Lc'] / total_weight_norm if total_weight_norm > 0 else 0
        self.avg_SCL_Lq = total_weighted_scl['Lq'] / total_weight_norm if total_weight_norm > 0 else 0
        self.avg_SCL_Oa = total_weighted_scl['Oa'] / total_weight_norm if total_weight_norm > 0 else 0
        
        # Link avg_SCL so plotting and output scripts remain completely unbroken
        self.avg_SCL = self.avg_SCL_1
        
        self.plot_data.append({'avg_SCL': self.avg_SCL})
        print(f"Average SCL_1: {self.avg_SCL_1:.4f} A")
        self._save_csv_results()

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    def _save_csv_results(self):
        filename = os.path.join(get_scl_dir(), 'SCL_results.csv')
        file_exists = os.path.isfile(filename)

        deviation = ""
        if self.scl_bc > 0:
            deviation = round(((self.avg_SCL_1 - self.scl_bc) / self.scl_bc) * 100, 2)

        base_headers = [
            'Composition', 'Source', 'Temperature (K)', 
            'Avg SCL_1 (A)', 'Avg SCL_0 (A)', 'Avg SCL_Lc (A)', 'Avg SCL_Lq (A)', 'Avg SCL_Oa (A)',
            'Experimental SCL (A)', 'Deviation'
        ]
        
        row = [
            self.comp, self.source, self.temp, 
            round(self.avg_SCL_1, 5), round(self.avg_SCL_0, 5), round(self.avg_SCL_Lc, 5),
            round(self.avg_SCL_Lq, 5), round(self.avg_SCL_Oa, 5),
            self.scl_bc, deviation
        ]

        sorted_pairs = sorted(self.ion_pair_results.items())
        headers = base_headers.copy()
        data = row.copy()

        for i in range(1, 7):
            headers.extend([
                f'Pair {i} Label', 
                f'Pair {i} SCL_1 (A)', f'Pair {i} SCL_0 (A)', f'Pair {i} SCL_Lc (A)', 
                f'Pair {i} SCL_Lq (A)', f'Pair {i} SCL_Oa (A)',
                f'Pair {i} Peak X (A)', f'Pair {i} Peak Y',
                f'Pair {i} Min X (A)', f'Pair {i} Min Y',
                f'Pair {i} CC Peak X (A)', f'Pair {i} CC Peak Y',
                f'Pair {i} PMF at Peak (kJ/mol)', f'Pair {i} Bond Strength (kJ/mol/A^2)',
                f'Pair {i} Reduced Mass (g/mol)', f'Pair {i} Fundamental Frequency (rad/ps)',
                f'Pair {i} b_PH_1', f'Pair {i} b_PH_0', f'Pair {i} b_PH_Lc', 
                f'Pair {i} b_PH_Lq', f'Pair {i} b_PH_Oa',
            ])
            if i <= len(sorted_pairs):
                nm, res = sorted_pairs[i - 1]
                data.extend([
                    nm, 
                    round(res['scl_1'], 5), round(res['scl_0'], 5), round(res['scl_Lc'], 5), 
                    round(res['scl_Lq'], 5), round(res['scl_Oa'], 5),
                    round(res['peak_x'] or 0, 5), round(res['peak_y'] or 0, 5),
                    round(res['minima_x'] or 0, 5), round(res['minima_y'] or 0, 5),
                    round(res['cc_peak_x'] or 0, 5), round(res['cc_peak_y'] or 0, 5),
                    round(res['pmf_at_peak'] or 0, 5), round(res['bond_strength'] or 0, 5),
                    round(res['reduced_mass'] or 0, 5), round(res['fundamental_frequency'] or 0, 5),
                    round(res.get('b_ph_1', 0), 5), round(res.get('b_ph_0', 0), 5), 
                    round(res.get('b_ph_Lc', 0), 5), round(res.get('b_ph_Lq', 0), 5), 
                    round(res.get('b_ph_Oa', 0), 5),
                ])
            else:
                data.extend([''] * 21)

        if file_exists:
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                existing = list(reader)
            filtered = [r for r in existing if len(r) > 1 and not (r[0] == str(self.comp) and r[1] == str(self.source))]
            if len(filtered) != len(existing):
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(filtered)
        else:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    # ------------------------------------------------------------------
    # Save plot data
    # ------------------------------------------------------------------
    def save_plot_data(self, folder='SCL_plot_data'):
        folder = os.path.join(get_scl_dir(), folder)
        os.makedirs(folder, exist_ok=True)
        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        filename = os.path.join(folder, f'{self.comp.replace("-", "_")}_{safe_source}_plot_data.csv')
        df = pd.DataFrame({'r (A)': self.x_grid})
        for name, func in self.weighted_splines.items():
            df[f'g(r)_weighted_{name}'] = func(self.x_grid)
        for data in self.plot_data:
            if 'ion_pair_ca_i' in data:
                df[f"S_i_{data['ion_pair_ca_i']}"] = data['S_i']
        df['Average_SCL'] = self.avg_SCL
        df['SCL_BC'] = self.scl_bc
        df.to_csv(filename, index=False)
        print(f"Plot data saved to {filename}")

    # ------------------------------------------------------------------
    # PDF plot
    # ------------------------------------------------------------------
    def plot_pdf(self, show_plot=True, save_plot=True, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(get_scl_dir(), 'SCL_plots')
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)

        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        base_filename = f'PDF_{self.comp}_{safe_source}'

        plt.rcParams.update({
            'font.family': 'Times New Roman', 'font.size': 14,
            'axes.labelsize': 14, 'axes.labelweight': 'bold', 'axes.linewidth': 1.5,
            'xtick.labelsize': 14, 'ytick.labelsize': 14,
            'xtick.direction': 'out', 'ytick.direction': 'out',
            'xtick.major.width': 1.75, 'ytick.major.width': 1.75,
            'legend.frameon': False, 'legend.fontsize': int(round(0.95 * 12)),
            'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
            'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        })

        plt.figure(figsize=(4.75, 4.25))
        ion_pair_colors = {}
        for ion_pair in self.ion_pairs:
            if ion_pair in self.weighted_splines:
                wy = self.weighted_splines[ion_pair](self.x_grid)
                line, = plt.plot(self.x_grid, wy, label=f"{ion_pair}")
                ion_pair_colors[ion_pair] = line.get_color()

        # S(r) bars + dotted lines
        for data in self.plot_data:
            if 'ion_pair_ca_i' not in data:
                continue
            ip = data['ion_pair_ca_i']
            color = ion_pair_colors.get(ip, 'black')
            S_i = data.get('S_i')
            if S_i is None or len(S_i) != len(self.x_grid):
                continue
            step_idx = np.where(np.diff(S_i) != 0)[0] + 1
            step_idx = np.concatenate(([0], step_idx, [len(S_i) - 1]))
            for si in range(len(step_idx) - 1):
                s0 = step_idx[si]
                s1 = step_idx[si + 1] if si < len(step_idx) - 1 else len(self.x_grid) - 1
                xs = self.x_grid[s0]
                xe = self.x_grid[s1] if s1 < len(self.x_grid) else self.x_grid[-1]
                bw = self.x_grid[step_idx[si + 1]] - xs if si < len(step_idx) - 2 else xe - xs
                sv = S_i[s0]
                alpha = min(1.0, 0.1 + 0.45 * sv)
                plt.bar((xs + xe) / 2, sv, width=bw, color=color, alpha=alpha * 0.6,
                        edgecolor='none', align='center', zorder=0)
            plt.plot(self.x_grid, S_i, label=f"S(r): {ip}", color=color, linestyle='dotted')

        for data in self.plot_data:
            if 'avg_SCL' in data:
                plt.axvline(x=data['avg_SCL'], color='g', linestyle='-.',
                            label=f"$\\ell_{{\\mathrm{{sc}}}}$ = {round(data['avg_SCL'], 2)}")
        if self.scl_bc > 0:
            plt.axvline(x=self.scl_bc, color='k', linestyle='--',
                        label=f"$\\ell_{{\\mathrm{{exp}}}}$ = {round(self.scl_bc, 2)}")

        plt.xlabel('r [Å]')
        plt.ylabel('g(r)')
        x_min, x_max = self.x_grid[0], self.x_grid[-1]
        first_nz = x_max
        for ip in self.ion_pairs:
            if ip in self.weighted_splines:
                nz = np.where(self.weighted_splines[ip](self.x_grid) > 0.01)[0]
                if len(nz):
                    first_nz = min(first_nz, self.x_grid[nz[0]])
        x_start = max(x_min, first_nz - 0.75)
        x_pad = (x_max - x_min) * 0.05
        plt.xlim(x_start, x_max + x_pad)
        ax = plt.gca()

        if len(self.fractions) == 1:
            ca_plot = [(ip, r) for ip, r in self.ion_pair_results.items()
                       if self.ion_pairs[ip].type == 'ca' and r.get('peak_x')]
            for idx, (ip, res) in enumerate(ca_plot):
                rp = res.get('peak_x')
                if not rp or rp <= 0:
                    continue
                pts = np.arange(0, (x_max + x_pad) + 0.5 * rp, rp)
                lbls = ["" if i == 0 else rf"$r_{{{i}}}$" for i in range(len(pts))]
                v = (pts >= x_start) & (pts <= (x_max + x_pad))
                pts, lbls = pts[v], [l for l, vv in zip(lbls, v) if vv]
                if len(pts) > 8:
                    s = int(np.ceil(len(pts) / 8))
                    pts, lbls = pts[::s], lbls[::s]
                sec = ax.twiny()
                sec.set_xlim(ax.get_xlim())
                sec.set_xticks(pts)
                sec.set_xticklabels(lbls)
                c = ion_pair_colors.get(ip, 'k')
                for l in sec.get_xticklabels():
                    l.set_color(c)
                sec.set_xlabel("")
                if 'top' in sec.spines:
                    sec.spines['top'].set_visible(False)
                sec.tick_params(axis='x', which='major', length=5, width=1.25, colors=c, pad=6 + 12 * idx)

        ax.legend(ncol=2, loc='upper right', bbox_to_anchor=(1, 1),
                  facecolor='white', frameon=True, framealpha=0.75, edgecolor='none',
                  borderpad=0.5, handletextpad=0.5, columnspacing=0.6, handlelength=1.5, labelspacing=0.3)
        ymin, ymax = plt.ylim()
        plt.ylim(ymin, np.ceil(ymax * 10) / 10)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(output_dir, f'{base_filename}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, f'{base_filename}.pdf'), bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------
    # VDOS plot
    # ------------------------------------------------------------------
    def plot_vdos(self, show_plot=False, save_plot=True, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(get_scl_dir(), 'SCL_plots')
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
        ca_names = list(self.vdos_data.keys())
        if not ca_names:
            return
        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        base = f'VDOS_{self.comp}_{safe_source}'
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        pc = {n: cc[i % len(cc)] for i, n in enumerate(ca_names)}
        for n in ca_names:
            om, D = self.vdos_data[n]
            ax.plot(om, D, color=pc[n], lw=1.5, label=f"D(ω): {n}")
        for i, a in enumerate(ca_names):
            for b in ca_names[i + 1:]:
                S = self.vdos_overlaps.get((a, b), 0)
                if S < 0.001:
                    continue
                oa, Da = self.vdos_data[a]
                ob, Db = self.vdos_data[b]
                lo, hi = min(oa[0], ob[0]), max(oa[-1], ob[-1])
                oc = np.linspace(lo, hi, 500)
                da = np.interp(oc, oa, Da, left=0, right=0)
                db = np.interp(oc, ob, Db, left=0, right=0)
                ov = np.minimum(da, db)
                blend = tuple((np.array(matplotlib.colors.to_rgb(pc[a])) +
                               np.array(matplotlib.colors.to_rgb(pc[b]))) / 2)
                ax.fill_between(oc, 0, ov, color=blend, alpha=0.3)
                pi = np.argmax(ov)
                if ov[pi] > 0:
                    ax.annotate(f"S = {S:.2f}", xy=(oc[pi], ov[pi]),
                                fontsize=9, ha='center', va='bottom', color='dimgray')
        ax.set_xlabel('ω [rad/ps]')
        ax.set_ylabel('D(ω) [ps/rad]')
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.7, edgecolor='none')
        fig.tight_layout()
        if save_plot:
            fig.savefig(os.path.join(output_dir, f'{base}.png'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(output_dir, f'{base}.pdf'), bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Gaussian fit overlay plot
    # ------------------------------------------------------------------
    def plot_gaussian_fits(self, show_plot=False, save_plot=True, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(get_scl_dir(), 'SCL_plots')
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
        ca = [(n, r) for n, r in self.ion_pair_results.items()
              if self.ion_pairs[n].type == 'ca' and r.get('gaussian_fit')]
        if not ca:
            return
        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        base = f'GaussianFit_{self.comp}_{safe_source}'
        nc = len(ca)
        fig, axes = plt.subplots(1, nc, figsize=(6.0 * nc, 5.0), squeeze=False)
        axes = axes[0]
        cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        k_B = 8.314462618e-3 # kJ/(mol*K) for PMF calculations
        
        for idx, (name, res) in enumerate(ca):
            ax = axes[idx]
            fit = res['gaussian_fit']
            g = self.ion_pairs[name].spline(self.x_grid)
            r0, gp, gb = fit['r_peak'], fit['g_peak'], fit['g_base']
            sL, sR = fit['sigma_left'], fit['sigma_right']
            color = cc[idx % len(cc)]
            
            # --- Primary Axis: g(r) ---
            ax.plot(self.x_grid, g, color=color, lw=1.5, label=f"g(r) Raw: {name}")
            
            # Generate the idealized Split Gaussian
            rf = np.linspace(r0 - 4 * sL, r0 + 4 * sR, 500)
            gf = np.where(rf < r0, gp * np.exp(-0.5 * ((rf - r0) / sL) ** 2),
                          (gp - gb) * np.exp(-0.5 * ((rf - r0) / sR) ** 2) + gb)
            ax.plot(rf, gf, 'k--', lw=1.5, label="Split Gaussian")
            
            hml_y = gp / 2; hml_x = r0 - fit['hwhm_left']
            hmr_y = (gp + gb) / 2; hmr_x = r0 + fit['hwhm_right']
            ax.plot([hml_x, r0], [hml_y, hml_y], 'r-', lw=1.2)
            ax.plot([r0, hmr_x], [hmr_y, hmr_y], 'b-', lw=1.2)
            ax.annotate(f"HWHM_L={fit['hwhm_left']:.3f}", xy=((hml_x + r0) / 2, hml_y),
                        fontsize=7, ha='center', va='top', color='red')
            ax.annotate(f"HWHM_R={fit['hwhm_right']:.3f}", xy=((r0 + hmr_x) / 2, hmr_y),
                        fontsize=7, ha='center', va='top', color='blue')
            ax.plot(r0, gp, 'o', color=color, ms=6, zorder=5)
            ax.axhline(gb, color='gray', ls=':', lw=0.8, alpha=0.6)
            
            # --- Secondary Axis: PMF & Tangent Lines ---
            ax_pmf = ax.twinx()
            
            # 1. Raw PMF (Faint Background)
            g_safe = np.maximum(g, 1e-10)
            pmf_raw = -k_B * self.temp * np.log(g_safe)
            ax_pmf.plot(self.x_grid, pmf_raw, color='gray', linestyle=':', lw=1.0, alpha=0.3, label="Raw PMF")
            
            # 2. Idealized PMF from Split Gaussian (Bold)
            gf_safe = np.maximum(gf, 1e-10)
            pmf_ideal = -k_B * self.temp * np.log(gf_safe)
            ax_pmf.plot(rf, pmf_ideal, color='gray', linestyle='-', lw=1.5, alpha=0.6, label="Ideal PMF")
            
            # 3. Left side analytical PMF tangent at inflection (Magenta)
            slope_L = fit.get('pmf_slope_left')
            center_L = fit.get('pmf_center_left')
            pmf_L = fit.get('pmf_val_left')
            g_L = fit.get('g_val_left')
            
            if slope_L is not None and center_L is not None:
                x_ext_L = np.linspace(center_L - 0.25, center_L + 0.25, 50)
                pmf_ext_L = slope_L * (x_ext_L - center_L) + pmf_L
                ax_pmf.plot(x_ext_L, pmf_ext_L, color='magenta', lw=2.5, label=f"Tangent L (slope={slope_L:.0f})")
                
                # Highlight the analytical inflection point
                ax_pmf.plot(center_L, pmf_L, 'o', color='magenta', ms=6, markeredgecolor='black', zorder=10)
                ax.plot(center_L, g_L, 'o', color='magenta', ms=6, markeredgecolor='black', zorder=10)

            # 4. Right side analytical PMF tangent at inflection (Cyan)
            slope_R = fit.get('pmf_slope_right')
            center_R = fit.get('pmf_center_right')
            pmf_R = fit.get('pmf_val_right')
            g_R = fit.get('g_val_right')
            
            if slope_R is not None and center_R is not None:
                x_ext_R = np.linspace(center_R - 0.25, center_R + 0.25, 50)
                pmf_ext_R = slope_R * (x_ext_R - center_R) + pmf_R
                ax_pmf.plot(x_ext_R, pmf_ext_R, color='cyan', lw=2.5, label=f"Tangent R (slope={slope_R:.0f})")
                
                # Highlight the analytical inflection point
                ax_pmf.plot(center_R, pmf_R, 'o', color='cyan', ms=6, markeredgecolor='black', zorder=10)
                ax.plot(center_R, g_R, 'o', color='cyan', ms=6, markeredgecolor='black', zorder=10)

            # Format PMF Axis Limits
            pmf_at_peak = -k_B * self.temp * np.log(gp) if gp > 0 else 0
            ax_pmf.set_ylim(pmf_at_peak - 5, pmf_at_peak + 50)
            ax_pmf.set_ylabel('PMF [kJ/mol]', color='gray', fontsize=11, fontweight='bold')
            ax_pmf.tick_params(axis='y', labelcolor='gray')
            
            # Combine Legends from both axes
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax_pmf.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=7, loc='upper left', framealpha=0.85)

            # --- Info Box ---
            b_kf = gb / gp if gp > 0 else 0
            info = (f"$r_{{peak}}$={r0:.3f} Å\n$g_{{peak}}$={gp:.2f}\n"
                    f"$g_{{base}}$={gb:.2f}\n$\\sigma_L$={sL:.4f}\n"
                    f"$\\sigma_R$={sR:.4f}\n$b_{{KF}}$={b_kf:.3f}")
            k_val = res.get('bond_strength')
            w0 = res.get('fundamental_frequency')
            if k_val: info += f"\n$k$={k_val:.2f}"
            if w0: info += f"\n$\\omega_0$={w0:.2f}"
            
            ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))
            
            ax.set_xlabel('r [Å]')
            ax.set_ylabel('g(r)')
            ax.set_title(name, fontsize=11)
            ax.set_xlim(r0 - 5 * sL, r0 + 6 * sR)
            ax.set_ylim(bottom=0)
            
        fig.suptitle(f'{format_composition_with_subscripts(self.comp)} ({self.temp}K) — Analytical Inflection & PMF', fontsize=12)
        fig.tight_layout()
        if save_plot:
            fig.savefig(os.path.join(output_dir, f'{base}.png'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(output_dir, f'{base}.pdf'), bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Dispersion overlap plot
    # ------------------------------------------------------------------
    def plot_dispersion(self, show_plot=False, save_plot=True, output_dir=None):
        """Plot dispersion curves for each ca-pair combination with q-windows.

        Shows acoustic (solid) and optic (dashed) branches for each chain,
        with shaded q-windows and annotated S_ij values.
        """
        if output_dir is None:
            output_dir = os.path.join(get_scl_dir(), 'SCL_plots')
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)

        cd = getattr(self, 'dispersion_chain_data', {})
        if len(cd) < 2:
            return

        names = list(cd.keys())
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        if not pairs:
            return

        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        base = f'Dispersion_{self.comp}_{safe_source}'

        n_plots = len(pairs)
        ncols = min(n_plots, 3)
        nrows = int(np.ceil(n_plots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for pi, (a_name, b_name) in enumerate(pairs):
            ax = axes_flat[pi]
            a_d = cd[a_name]
            b_d = cd[b_name]
            S_val = self.dispersion_overlaps.get((a_name, b_name), 0)

            # Chain A — blue
            ax.plot(a_d['q'], a_d['omega_ac'], '-', color='#1f77b4', lw=2, label=f'{a_name} acoustic')
            ax.plot(a_d['q'], a_d['omega_op'], '--', color='#1f77b4', lw=2, label=f'{a_name} optic')

            # Chain B — red
            ax.plot(b_d['q'], b_d['omega_ac'], '-', color='#d62728', lw=2, label=f'{b_name} acoustic')
            ax.plot(b_d['q'], b_d['omega_op'], '--', color='#d62728', lw=2, label=f'{b_name} optic')

            # Q-windows
            ax.axvspan(a_d['q_lo'], a_d['q_hi'], alpha=0.12, color='#1f77b4', label=f'{a_name} q-window')
            ax.axvspan(b_d['q_lo'], b_d['q_hi'], alpha=0.12, color='#d62728', label=f'{b_name} q-window')

            # Allowed frequency bands in q-windows (horizontal spans)
            a_ac_lo, a_ac_hi = a_d['ac_band']
            a_op_lo, a_op_hi = a_d['op_band']
            b_ac_lo, b_ac_hi = b_d['ac_band']
            b_op_lo, b_op_hi = b_d['op_band']

            # Light frequency-band shading on y-axis
            ymax = max(a_d['omega_op'].max(), b_d['omega_op'].max()) * 1.05
            ax.axhspan(a_ac_lo, a_ac_hi, alpha=0.06, color='#1f77b4')
            ax.axhspan(a_op_lo, a_op_hi, alpha=0.06, color='#1f77b4')
            ax.axhspan(b_ac_lo, b_ac_hi, alpha=0.06, color='#d62728')
            ax.axhspan(b_op_lo, b_op_hi, alpha=0.06, color='#d62728')

            # BZ boundary markers
            ax.axvline(a_d['q_center'], color='#1f77b4', ls=':', lw=0.8, alpha=0.5)
            ax.axvline(b_d['q_center'], color='#d62728', ls=':', lw=0.8, alpha=0.5)

            q_max = max(a_d['q'][-1], b_d['q'][-1])
            ax.set_xlim(0, q_max * 1.05)
            ax.set_ylim(0, ymax)
            ax.set_xlabel('q [rad/Å]')
            ax.set_ylabel('ω [relative]')
            ax.set_title(f'{a_name} vs {b_name}  —  $S$ = {S_val:.3f}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=7, loc='upper left', framealpha=0.7, ncol=2)
            ax.grid(True, ls=':', alpha=0.4)

        # Hide unused subplots
        for pi in range(len(pairs), len(axes_flat)):
            axes_flat[pi].set_visible(False)

        fig.suptitle(f'{format_composition_with_subscripts(self.comp)} ({self.temp}K) — Dispersion Overlap',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        if save_plot:
            fig.savefig(os.path.join(output_dir, f'{base}.png'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(output_dir, f'{base}.pdf'), bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close(fig)


# ==========================================
# 2. Analyzer (batch runner)
# ==========================================

class PDFAnalyzer:
    def __init__(self, save_plot_data=False, show_plot=False,
                 plot_vdos=False, plot_gaussian=False, plot_dispersion=False,
                 plot_summary=True, vdos_method='gaussian', b_ph_factors=None):
        self.salts = []
        self.save_plot_data = save_plot_data
        self.show_plot = show_plot
        self.plot_vdos = plot_vdos
        self.plot_gaussian = plot_gaussian
        self.plot_dispersion = plot_dispersion
        self.plot_summary = plot_summary
        self.vdos_method = vdos_method
        self.b_ph_factors = b_ph_factors

    def add_molten_salt(self, prepared_csv, comp_str, source, temp, scl_bc):
        self.salts.append(MoltenSaltPDF(
            prepared_csv, comp_str, source, temp, scl_bc,
            vdos_method=self.vdos_method,
            b_ph_factors=self.b_ph_factors,
        ))

    def analyze_all(self):
        for salt in self.salts:
            salt.analyze_pdf()
            if self.save_plot_data:
                salt.save_plot_data()

    def plot_all(self):
        for salt in self.salts:
            salt.plot_pdf(show_plot=self.show_plot)
            if self.plot_vdos:
                salt.plot_vdos(show_plot=self.show_plot)
            if self.plot_gaussian:
                salt.plot_gaussian_fits(show_plot=self.show_plot)
            if self.plot_dispersion:
                salt.plot_dispersion(show_plot=self.show_plot)
        
        if self.plot_summary:
            self.plot_summary_deviation(show_plot=self.show_plot)

    def plot_summary_deviation(self, show_plot=False, output_dir=None):
        """Plot deviation of avg_SCL from scl_bc for all salts.
        
        Shows each salt's deviation (%) from experimental value with a shaded ±15% region.
        Bars are color-coded: blue for fluoride salts, green for chloride salts.
        """
        if output_dir is None:
            output_dir = os.path.join(get_scl_dir(), 'SCL_plots')
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect data from all salts
        salt_labels = []
        deviations = []
        colors_list = []
        
        for idx, salt in enumerate(self.salts):
            if salt.scl_bc <= 0:  # Skip salts without experimental data
                continue
            
            # Calculate percentage deviation: (avg_SCL - scl_bc) / scl_bc * 100
            dev = ((salt.avg_SCL - salt.scl_bc) / salt.scl_bc) * 100
            deviations.append(dev)
            salt_labels.append(format_composition_with_subscripts(salt.comp))
            
            # Determine anion type and assign color
            if 'F' in salt.comp:
                color = '#1f77b4'  # Blue for fluoride
            elif 'Cl' in salt.comp:
                color = '#2ca02c'  # Green for chloride
            else:
                color = '#808080'  # Gray for other
            colors_list.append(color)
        
        if not deviations:
            print("No salts with experimental SCL values to plot.")
            return
        
        # Set up plot style to match existing plots
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
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(deviations))
        
        # Plot bars
        bars = ax.bar(x_pos, deviations, color=colors_list, alpha=0.75, edgecolor='black', linewidth=1.0)
        
        # Add centerline at 0%
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=0)
        
        # Add shaded ±15% region
        ax.axhspan(-15, 15, alpha=0.15, color='green', zorder=0, label='±15% region')
        ax.axhline(y=15, color='green', linestyle='--', linewidth=1.0, alpha=0.6, zorder=0)
        ax.axhline(y=-15, color='green', linestyle='--', linewidth=1.0, alpha=0.6, zorder=0)
        
        # Labels and formatting
        ax.set_xlabel('Salt Composition', fontweight='bold', fontsize=13)
        ax.set_ylabel('Deviation from Experiment (%)', fontweight='bold', fontsize=13)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(salt_labels, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, (bar, dev) in enumerate(zip(bars, deviations)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dev:.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=9, fontweight='bold')
        
        # Grid
        # ax.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        # Set y-axis limits with some padding
        y_max = max(abs(min(deviations)), abs(max(deviations))) * 1.2
        y_max = max(y_max, 20)  # Ensure at least ±20 on y-axis
        ax.set_ylim(-100,100)#(-y_max, y_max)
        
        fig.tight_layout()
        
        # Save plot
        base_filename = 'SCL_Summary_Deviation'
        fig.savefig(os.path.join(output_dir, f'{base_filename}.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, f'{base_filename}.pdf'), bbox_inches='tight')
        print(f"Summary deviation plot saved to {output_dir}/{base_filename}.*")
        
        if show_plot:
            plt.show()
        plt.close(fig)


# ==========================================
# 3. Helper: build prepared CSV path
# ==========================================

def _prep_path(comp_str, source, temp):
    _, _, sorted_comp = parse_composition(comp_str)
    safe_source = ''.join(c if c.isalnum() else '_' for c in source.split(',')[0].strip())
    fname = f"{sorted_comp.replace('-', '_')}_{safe_source}_{int(temp)}K.csv"
    return os.path.join(get_scl_dir(), 'Prepared_PDF_CSV', fname)


# ==========================================
# 4. Main
# ==========================================

def main():
    analyzer = PDFAnalyzer(
        save_plot_data=True,
        show_plot=False,
        plot_vdos=True,
        plot_gaussian=True,
        plot_dispersion=True,
        vdos_method='gaussian',
        # Select b_PH factors:
        #   'vdos'           — VDOS overlap (Gaussian approx)
        #   'dispersion'     — q-windowed dispersion transfer (recommended)
        #   'coordination'   — coordination number mismatch
        #   'polarizability' — anion polarizability accommodation
        #   'no_bPH'           — forces b_PH = 0 (no phase difference disruption)
        #   'penalty'        — mass, bond, and polarizability penalty (default)
        #   None / empty     — original concentration-only
        b_ph_factors={},
    )

    # --- Structural Studies (18) ---
    analyzer.add_molten_salt(_prep_path("1.0LiCl", 'Li7Numbers, 2026', 900), "1.0LiCl", 'Li7Numbers, 2026', 900, 4.10511)
    analyzer.add_molten_salt(_prep_path("1.0LiCl", 'Li6Numbers, 2026', 900), "1.0LiCl", 'Li6Numbers, 2026', 900, 4.10511)
    # analyzer.add_molten_salt(_prep_path("1.0RbCl", 'Numbers, 2026', 1015), "1.0RbCl", 'Numbers, 2026', 1015, 0)
    analyzer.add_molten_salt(_prep_path("0.68KCl-0.32MgCl2", 'Numbers, 2026', 903), "0.68KCl-0.32MgCl2", 'Numbers, 2026', 903, 3.91988)
    # analyzer.add_molten_salt(_prep_path("0.60LiCl-0.40MgCl2", 'Numbers, 2026', 870), "0.60LiCl-0.40MgCl2", 'Numbers, 2026', 870, 0)
    analyzer.add_molten_salt(_prep_path("0.68KCl-0.32MgCl2", 'Numbers, 2026', 703), "0.68KCl-0.32MgCl2", 'Numbers, 2026', 703, 3.91988)
    analyzer.add_molten_salt(_prep_path("0.5NaCl-0.5KCl", 'Numbers, 2026', 950), "0.5NaCl-0.5KCl", 'Numbers, 2026', 950, 4.32778)
    # analyzer.add_molten_salt(_prep_path("0.5LiCl-0.5KCl", 'Numbers, 2026', 760), "0.5LiCl-0.5KCl", 'Numbers, 2026', 760, 0)
    # analyzer.add_molten_salt(_prep_path("0.56NaCl-0.44MgCl2", 'Numbers, 2026', 760), "0.56NaCl-0.44MgCl2", 'Numbers, 2026', 760, 0)
    analyzer.add_molten_salt(_prep_path("0.535NaCl-0.315MgCl2-0.15CaCl2", 'Numbers, 2026', 855), "0.535NaCl-0.315MgCl2-0.15CaCl2", 'Numbers, 2026', 855, 3.52027)
    # analyzer.add_molten_salt(_prep_path("0.535LiCl-0.086NaCl-0.379KCl", 'Numbers, 2026', 650), "0.535LiCl-0.086NaCl-0.379KCl", 'Numbers, 2026', 650, 0)
    analyzer.add_molten_salt(_prep_path("0.4903NaCl-0.5097CaCl2", 'Numbers, 2026', 800), "0.4903NaCl-0.5097CaCl2", 'Numbers, 2026', 800, 3.76913)
    # analyzer.add_molten_salt(_prep_path("0.43KCl-0.57MgCl2", 'Numbers, 2026', 750), "0.43KCl-0.57MgCl2", 'Numbers, 2026', 750, 0)
    # analyzer.add_molten_salt(_prep_path("0.33NaCl-0.22KCl-0.45MgCl2", 'Numbers, 2026', 750), "0.33NaCl-0.22KCl-0.45MgCl2", 'Numbers, 2026', 750, 0)
    # analyzer.add_molten_salt(_prep_path("0.2839NaCl-0.2725KCl-0.4436MgCl2", 'Numbers, 2026', 680), "0.2839NaCl-0.2725KCl-0.4436MgCl2", 'Numbers, 2026', 680, 0)
    # analyzer.add_molten_salt(_prep_path("0.275NaCl-0.325KCl-0.40MgCl2", 'Numbers, 2026', 700), "0.275NaCl-0.325KCl-0.40MgCl2", 'Numbers, 2026', 700, 0)
    # analyzer.add_molten_salt(_prep_path("0.25LiCl-0.75KCl", 'Numbers, 2026', 934), "0.25LiCl-0.75KCl", 'Numbers, 2026', 934, 0)

    analyzer.add_molten_salt(_prep_path("0.637LiCl-0.363KCl", 'Numbers, 2026', 690), "0.637LiCl-0.363KCl", 'Numbers, 2026', 690, 4.05630)
    analyzer.add_molten_salt(_prep_path("1.0CaCl2", 'Numbers, 2026', 1060), "1.0CaCl2", 'Numbers, 2026', 1060, 7.72598)
    # analyzer.add_molten_salt(_prep_path("1.00CsCl", 'Numbers, 2026', 935), "1.00CsCl", 'Numbers, 2026', 935, 0)
    analyzer.add_molten_salt(_prep_path("1.00LiCl", 'Numbers, 2026', 900), "1.00LiCl", 'Numbers, 2026', 900, 4.10511)
    analyzer.add_molten_salt(_prep_path("1.00NaCl", 'Numbers, 2026', 1090), "1.00NaCl", 'Numbers, 2026', 1090, 4.48028)
    analyzer.add_molten_salt(_prep_path("1.00MgCl2", 'Numbers, 2026', 1000), "1.00MgCl2", 'Numbers, 2026', 1000, 4.76796)
    analyzer.add_molten_salt(_prep_path("1.00KCl", 'Numbers, 2026', 1060), "1.00KCl", 'Numbers, 2026', 1060, 4.47675)
    

    # # --- Unary Salts (11) ---
    # analyzer.add_molten_salt(_prep_path("1.0LiF", 'Walz, 2019', 1121), "1.0LiF", 'Walz, 2019', 1121, 3.28553)
    # analyzer.add_molten_salt(_prep_path("1.0NaF", 'Walz, 2019', 1266), "1.0NaF", 'Walz, 2019', 1266, 5.22361)
    # analyzer.add_molten_salt(_prep_path("1.0KF", 'Walz, 2019', 1131), "1.0KF", 'Walz, 2019', 1131, 4.63533)
    # analyzer.add_molten_salt(_prep_path("1.0LiCl", 'Walz, 2019', 878), "1.0LiCl", 'Walz, 2019', 878, 4.10511)
    # analyzer.add_molten_salt(_prep_path("1.0NaCl", 'Lu, 2021', 1200), "1.0NaCl", 'Lu, 2021', 1200, 4.48028)
    # analyzer.add_molten_salt(_prep_path("1.0KCl", 'Walz, 2019', 1043), "1.0KCl", 'Walz, 2019', 1043, 4.47675)
    # analyzer.add_molten_salt(_prep_path("1.0KCl", 'Walker, 2026', 1043), "1.0KCl", 'Walker, 2026', 1043, 4.47675)
    # analyzer.add_molten_salt(_prep_path("1.0MgCl2", 'Roy, 2021', 1073), "1.0MgCl2", 'Roy, 2021', 1073, 4.76796)
    # analyzer.add_molten_salt(_prep_path("1.0MgCl2", 'Walker, 2026', 987), "1.0MgCl2", 'Walker, 2026', 987, 4.76796)
    # analyzer.add_molten_salt(_prep_path("1.0CaCl2", 'Bu, 2021', 1100), "1.0CaCl2", 'Bu, 2021', 1100, 7.72598)
    # analyzer.add_molten_salt(_prep_path("1.0SrCl2", 'McGreevy, 1987', 1198), "1.0SrCl2", 'McGreevy, 1987', 1198, 6.408743)

    # # --- Mixtures (16) ---
    # analyzer.add_molten_salt(_prep_path("0.6LiF-0.4NaF", 'Grizzi, 2024', 1473), "0.6LiF-0.4NaF", 'Grizzi, 2024', 1473, 2.63857)
    # analyzer.add_molten_salt(_prep_path("0.5LiF-0.5BeF2", 'Sun, 2024', 900), "0.5LiF-0.5BeF2", 'Sun, 2024', 900, 0)
    # analyzer.add_molten_salt(_prep_path("0.66LiF-0.34BeF2", 'Fayfar, 2024', 973), "0.66LiF-0.34BeF2", 'Fayfar, 2024', 973, 1.90187)
    # analyzer.add_molten_salt(_prep_path("0.66LiF-0.34BeF2", 'Yin, 2025', 973), "0.66LiF-0.34BeF2", 'Yin, 2025', 973, 1.90187)
    # analyzer.add_molten_salt(_prep_path("0.32MgCl2-0.68KCl", 'Walker, 2026', 723), "0.32MgCl2-0.68KCl", 'Walker, 2026', 723, 3.91988)
    # analyzer.add_molten_salt(_prep_path("0.5LiCl-0.5KCl", 'Jiang, 2016', 727), "0.5LiCl-0.5KCl", 'Jiang, 2016', 727, 0)
    # analyzer.add_molten_salt(_prep_path("0.637LiCl-0.363KCl", 'Jiang, 2016', 750), "0.637LiCl-0.363KCl", 'Jiang, 2016', 750,  4.05630)
    # analyzer.add_molten_salt(_prep_path("0.5NaCl-0.5KCl", 'Manga, 2014', 1100), "0.5NaCl-0.5KCl", 'Manga, 2014', 1100, 4.32778)
    # analyzer.add_molten_salt(_prep_path("0.5NaCl-0.5KCl", 'Walker, 2026', 1100), "0.5NaCl-0.5KCl", 'Walker, 2026', 1100, 4.32778)
    # # analyzer.add_molten_salt(_prep_path("0.5LiCl-0.5KCl", 'NumbersNVE, 2026', 738), "0.5LiCl-0.5KCl", 'NumbersNVE, 2026', 738, 0)
    # # analyzer.add_molten_salt(_prep_path("0.5LiCl-0.5KCl", 'NumbersNVT, 2026', 738), "0.5LiCl-0.5KCl", 'NumbersNVT, 2026', 738, 0)
    # analyzer.add_molten_salt(_prep_path("0.6NaCl-0.4KCl", 'Walker, 2026', 1100), "0.6NaCl-0.4KCl", 'Walker, 2026', 1100, 4.32778)
    # analyzer.add_molten_salt(_prep_path("0.3NaCl-0.7KCl", 'Walker, 2026', 1100), "0.3NaCl-0.7KCl", 'Walker, 2026', 1100, 4.32778)
    # analyzer.add_molten_salt(_prep_path("0.7LiCl-0.3CaCl2", 'Liang, 2024', 1073), "0.7LiCl-0.3CaCl2", 'Liang, 2024', 1073, 0)
    # analyzer.add_molten_salt(_prep_path("0.4903NaCl-0.5097CaCl2", 'Wei, 2022', 1023), "0.4903NaCl-0.5097CaCl2", 'Wei, 2022', 1023, 3.76913)
    # analyzer.add_molten_salt(_prep_path("0.718KCl-0.282CaCl2", 'Wei, 2022', 1300), "0.718KCl-0.282CaCl2", 'Wei, 2022', 1300, 0)
    # analyzer.add_molten_salt(_prep_path("0.465LiF-0.115NaF-0.42KF", 'Frandsen, 2020', 873), "0.465LiF-0.115NaF-0.42KF", 'Frandsen, 2020', 873, 2.26059)
    # analyzer.add_molten_salt(_prep_path("0.345NaF-0.59KF-0.065MgF2", 'Solano, 2021', 1073), "0.345NaF-0.59KF-0.065MgF2", 'Solano, 2021', 1073, 3.92263)
    # analyzer.add_molten_salt(_prep_path("0.45MgCl2-0.33NaCl-0.22KCl", 'Jiang, 2024', 750), "0.45MgCl2-0.33NaCl-0.22KCl", 'Jiang, 2024', 750, 0)
    # analyzer.add_molten_salt(_prep_path("0.38MgCl2-0.21NaCl-0.41KCl", 'Jiang, 2024', 750), "0.38MgCl2-0.21NaCl-0.41KCl", 'Jiang, 2024', 750, 3.65358)
    # analyzer.add_molten_salt(_prep_path("0.38MgCl2-0.21NaCl-0.41KCl", 'Walker, 2026', 660), "0.38MgCl2-0.21NaCl-0.41KCl", 'Walker, 2026', 660, 3.65358)    
    # analyzer.add_molten_salt(_prep_path("0.417NaCl-0.525CaCl2-0.058KCl", 'Wei, 2022', 1023), "0.417NaCl-0.525CaCl2-0.058KCl", 'Wei, 2022', 1023, 0)
    # analyzer.add_molten_salt(_prep_path("0.535NaCl-0.315MgCl2-0.15CaCl2", 'Wei, 2022', 1023), "0.535NaCl-0.315MgCl2-0.15CaCl2", 'Wei, 2022', 1023, 3.52027)

    # # --- Actinides (15) ---
    # analyzer.add_molten_salt(_prep_path("1.0ThF4", 'Dai, 2015', 1633), "1.0ThF4", 'Dai, 2015', 1633, 0)
    # analyzer.add_molten_salt(_prep_path("1.0UF4", 'OcadizFlores, 2021', 1357), "1.0UF4", 'OcadizFlores, 2021', 1357, 0)
    # analyzer.add_molten_salt(_prep_path("0.64NaCl-0.36UCl3", 'Andersson, 2022', 1250), "0.64NaCl-0.36UCl3", 'Andersson, 2022', 1250, 2.5393)
    # analyzer.add_molten_salt(_prep_path("0.85KCl-0.15UCl3", 'Andersson, 2024', 1250), "0.85KCl-0.15UCl3", 'Andersson, 2024', 1250, 0)
    # analyzer.add_molten_salt(_prep_path("0.75KCl-0.25UCl3", 'Andersson, 2024', 1250), "0.75KCl-0.25UCl3", 'Andersson, 2024', 1250, 0)
    # analyzer.add_molten_salt(_prep_path("0.65KCl-0.35UCl3", 'Andersson, 2024', 1250), "0.65KCl-0.35UCl3", 'Andersson, 2024', 1250, 0)
    # analyzer.add_molten_salt(_prep_path("0.5KCl-0.5UCl3", 'Andersson, 2024', 1250), "0.5KCl-0.5UCl3", 'Andersson, 2024', 1250, 0)
    # analyzer.add_molten_salt(_prep_path("0.625LiF-0.3125BeF2-0.0625ThF4", 'Yin, 2025', 973), "0.625LiF-0.3125BeF2-0.0625ThF4", 'Yin, 2025', 973, 0)
    # analyzer.add_molten_salt(_prep_path("0.60LiF-0.30BeF2-0.10ThF4", 'Yin, 2025', 973), "0.60LiF-0.30BeF2-0.10ThF4", 'Yin, 2025', 973, 0)
    # analyzer.add_molten_salt(_prep_path("0.5455LiF-0.2727BeF2-0.1818ThF4", 'Yin, 2025', 973), "0.5455LiF-0.2727BeF2-0.1818ThF4", 'Yin, 2025', 973, 0)
    # analyzer.add_molten_salt(_prep_path("0.5454LiF-0.3636NaF-0.091UF4", 'Grizzi, 2024', 1473), "0.5454LiF-0.3636NaF-0.091UF4", 'Grizzi, 2024', 1473, 0)
    # analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '900K-AIMD-Zhang, 2026', 900), "0.78NaF-0.22UF4", '900K-AIMD-Zhang, 2026', 900, 0)
    # analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '900K-CMD-Zhang, 2026', 900), "0.78NaF-0.22UF4", '900K-CMD-Zhang, 2026', 900, 0)
    # analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '1000K-CMD-Zhang, 2026', 1000), "0.78NaF-0.22UF4", '1000K-CMD-Zhang, 2026', 1000, 0)
    # analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '1100K-CMD-Zhang, 2026', 1100), "0.78NaF-0.22UF4", '1100K-CMD-Zhang, 2026', 1100, 0)
    # analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '1200K-CMD-Zhang, 2026', 1200), "0.78NaF-0.22UF4", '1200K-CMD-Zhang, 2026', 1200, 0)
    # analyzer.add_molten_salt(_prep_path("0.57NaF-0.16KF-0.27UF4", '900K-AIMD-Zhang, 2026', 900), "0.57NaF-0.16KF-0.27UF4", '900K-AIMD-Zhang, 2026', 900, 0)
    # analyzer.add_molten_salt(_prep_path("0.57NaF-0.16KF-0.27UF4", '1000K-AIMD-Zhang, 2026', 1000), "0.57NaF-0.16KF-0.27UF4", '1000K-AIMD-Zhang, 2026', 1000, 0)
    # analyzer.add_molten_salt(_prep_path("0.57NaF-0.16KF-0.27UF4", '1100K-AIMD-Zhang, 2026', 1100), "0.57NaF-0.16KF-0.27UF4", '1100K-AIMD-Zhang, 2026', 1100, 0)
    # analyzer.add_molten_salt(_prep_path("0.57NaF-0.16KF-0.27UF4", '1200K-AIMD-Zhang, 2026', 1200), "0.57NaF-0.16KF-0.27UF4", '1200K-AIMD-Zhang, 2026', 1200, 0)
    # analyzer.add_molten_salt(_prep_path("0.63NaCl-0.37UCl3", 'AIMD-Zhang, 2026', 1100), "0.63NaCl-0.37UCl3", 'AIMD-Zhang, 2026', 1100, 0)

    # Run
    analyzer.analyze_all()
    analyzer.plot_all()


if __name__ == "__main__":
    main()
