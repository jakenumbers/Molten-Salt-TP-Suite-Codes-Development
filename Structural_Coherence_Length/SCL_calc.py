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

    def __init__(self, prepared_csv, comp_str, source, temp, gamma_bc,
                 vdos_method='gaussian'):
        self.original_comp_str = comp_str
        self.source = source
        self.temp = temp
        self.gamma_bc = gamma_bc
        self.vdos_method = vdos_method  # 'gaussian' or 'pmf'

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
            g_fit(r) = (g_peak - g_base) * exp(-(r-r0)²/(2σ²)) + g_base
        where σ = σ_L for r < r0 and σ = σ_R for r >= r0.

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

        # Find first peak
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
        # Half-max on left = g_peak / 2
        half_max_left = g_peak / 2.0
        left_region = g[:p_idx]
        left_crossings = np.where(left_region <= half_max_left)[0]
        if len(left_crossings) > 0:
            li = left_crossings[-1]
            # Linear interpolation for sub-grid accuracy
            if li + 1 < p_idx and g[li + 1] != g[li]:
                frac = (half_max_left - g[li]) / (g[li + 1] - g[li])
                x_left = self.x_grid[li] + frac * dx
            else:
                x_left = self.x_grid[li]
        else:
            x_left = self.x_grid[0]

        hwhm_left = r_peak - x_left

        # --- Right side: baseline = g_base ---
        # Half-max on right = (g_peak - g_base) / 2 + g_base = (g_peak + g_base) / 2
        half_max_right = (g_peak + g_base) / 2.0
        right_region = g[p_idx:]
        right_crossings = np.where(right_region <= half_max_right)[0]
        if len(right_crossings) > 0:
            ri = right_crossings[0] + p_idx
            # Linear interpolation
            if ri > p_idx and g[ri - 1] != g[ri]:
                frac = (half_max_right - g[ri - 1]) / (g[ri] - g[ri - 1])
                x_right = self.x_grid[ri - 1] + frac * dx
            else:
                x_right = self.x_grid[ri]
        else:
            x_right = self.x_grid[-1]

        hwhm_right = x_right - r_peak

        # Ensure positive widths
        hwhm_left = max(hwhm_left, dx)
        hwhm_right = max(hwhm_right, dx)

        # Convert HWHM to σ: HWHM = σ√(2 ln 2)
        sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
        sigma_left = hwhm_left / sqrt_2ln2
        sigma_right = hwhm_right / sqrt_2ln2

        fwhm = hwhm_left + hwhm_right

        return {
            'r_peak': r_peak,
            'g_peak': g_peak,
            'g_base': g_base,
            'sigma_left': sigma_left,
            'sigma_right': sigma_right,
            'hwhm_left': hwhm_left,
            'hwhm_right': hwhm_right,
            'fwhm': fwhm,
        }

    # ------------------------------------------------------------------
    # PMF & bond strength (from split Gaussian)
    # ------------------------------------------------------------------
    def calculate_pmf_and_bond_strength(self, ion_pair_name):
        """Calculate bond strength from split-Gaussian fit to unweighted g(r).

        Left side (repulsive wall) determines bond stiffness:
            k = k_B T / σ_L²

        Since the left-side baseline is 0, the amplitude factor cancels.

        Returns:
            (pmf_values, bond_strength, peak_position)
            pmf_values: array of PMF on the grid (from raw g(r))
            bond_strength: k in kJ/(mol·Å²) or None
            peak_position: r_peak or None
        """
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

        sigma_L = fit['sigma_left']

        # Left side baseline = 0, so k = k_B T / σ_L²
        bond_strength = k_B * self.temp / (sigma_L ** 2)

        return pmf_values, bond_strength, fit['r_peak']

    # ------------------------------------------------------------------
    # Reduced mass
    # ------------------------------------------------------------------
    def calculate_reduced_mass(self, ion_pair_name):
        """Reduced mass μ in kg/mol."""
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
        k_si = bond_strength * 1000 / 1e-20       # kJ/(mol·Å²) → J/(mol·m²)
        k_mol = k_si / N_A                         # J/m² per molecule
        mu_mol = reduced_mass / N_A                 # kg per molecule

        omega = np.sqrt(k_mol / mu_mol)             # rad/s
        omega *= 1e-12                               # → rad/ps
        return omega

    # ------------------------------------------------------------------
    # VDOS (Gaussian approximation)
    # ------------------------------------------------------------------
    def calculate_vdos_gaussian(self, ion_pair_name):
        """Gaussian VDOS centered at ω₀ with anharmonicity-informed width.

        Width: σ_ω = ω₀ · (σ_L + σ_R) / (2 · r_peak) · (1 + b_KF)
        """
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
    # VDOS (PMF-based, Jacobian mapping)
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
        k_r = d2pmf

        reduced_mass = self.calculate_reduced_mass(ion_pair_name)
        if reduced_mass is None:
            return None, None

        N_A = 6.02214076e23
        mu_kg = reduced_mass / N_A

        # Find contiguous concave region around peak
        fit = self._fit_gaussian_to_peak(ion_pair_name)
        if fit is None:
            return None, None

        peak_idx = np.argmin(np.abs(self.x_grid - fit['r_peak']))
        left = peak_idx
        while left > 0 and k_r[left] > 0:
            left -= 1
        right = peak_idx
        while right < len(k_r) - 1 and k_r[right] > 0:
            right += 1

        if right - left < 3:
            return None, None

        r_region = self.x_grid[left:right]
        g_region = g[left:right]
        k_region = k_r[left:right]
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
        omega_min, omega_max = omega_r.min(), omega_r.max()
        if omega_max <= omega_min:
            return None, None

        bin_edges = np.linspace(omega_min * 0.9, omega_max * 1.1, n_bins + 1)
        D_omega = np.zeros(n_bins)
        omega_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        for w, om in zip(weights_r, omega_r):
            bi = np.searchsorted(bin_edges, om) - 1
            bi = np.clip(bi, 0, n_bins - 1)
            D_omega[bi] += w

        d_omega = bin_edges[1] - bin_edges[0]
        area = np.trapezoid(D_omega, omega_centers)
        if area > 0:
            D_omega /= area

        # Light smoothing
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

    # ------------------------------------------------------------------
    # VDOS dispatcher
    # ------------------------------------------------------------------
    def calculate_vdos(self, ion_pair_name):
        if self.vdos_method == 'pmf':
            return self.calculate_vdos_pmf(ion_pair_name)
        return self.calculate_vdos_gaussian(ion_pair_name)

    # ------------------------------------------------------------------
    # VDOS overlap computation
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
    # b_PH with VDOS overlap
    # ------------------------------------------------------------------
    def _calculate_vdos_b_ph(self, pair_name, ca_pairs, sum_ca_weights):
        """b_PH using VDOS overlap weighting."""
        pair = self.ion_pairs[pair_name]

        if pair_name not in self.vdos_data:
            if sum_ca_weights > 0:
                return np.clip(1 - (pair.weight / sum_ca_weights), 0, 1)
            return 1.0

        numerator = 0.0
        denominator = 0.0
        for other in ca_pairs:
            x_j = other.weight
            S_ij = self.vdos_overlaps.get((pair_name, other.name), 0)
            if other.name == pair_name:
                S_ij = 1.0
            numerator += x_j * S_ij
            denominator += x_j

        if denominator <= 0:
            return 1.0
        b_ph = 1.0 - numerator / denominator
        return np.clip(b_ph, 0, 1)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze_pdf(self):
        print(f"\n### Analysis for {self.comp} ###")

        # Compute VDOS overlaps before the pair loop
        self._compute_vdos_overlaps()

        total_weighted_scl = 0
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

            # b_PH with VDOS overlap
            ph_val = self._calculate_vdos_b_ph(name, ca_pairs, sum_ca_weights)
            print(f"  b_PH (VDOS): {ph_val:.4f}")

            cation = name.split('-')[0]
            cc_name = standardize_ion_pair(f"{cation}-{cation}")
            has_cc = cc_name in self.ion_pairs

            b_KF_vals, b_NI_vals, b_PH_vals, beta_vals = [], [], [], []

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

                b_KF_vals.append(kf_val)
                b_PH_vals.append(ph_val)
                b_NI_vals.append(ni_val)

                if kf_val == 1 or ph_val == 1 or ni_val == 1:
                    beta = float('inf')
                else:
                    beta = (kf_val / (1 - kf_val)) + (ph_val / (1 - ph_val)) + (ni_val / (1 - ni_val))
                beta_vals.append(beta)

            # --- Survival function ---
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
            print(f"  SCL: {scl_pair:.3f} A (Weight: {RTE:.3f})")

            # Map S(r) to fine grid
            S_y_grid = np.zeros_like(self.x_grid)
            curr_s_idx = 0
            for i, x in enumerate(self.x_grid):
                if curr_s_idx < len(transfer_points):
                    if x >= transfer_points[curr_s_idx]:
                        curr_s_idx += 1
                if curr_s_idx < len(S_discrete):
                    S_y_grid[i] = S_discrete[curr_s_idx]
                else:
                    S_y_grid[i] = S_discrete[-1]

            # PMF and ω₀
            pmf_values, bond_strength, peak_x = self.calculate_pmf_and_bond_strength(name)
            reduced_mass = self.calculate_reduced_mass(name)
            fundamental_frequency = self.calculate_fundamental_frequency(name, bond_strength)

            # Gaussian fit info
            fit = self._fit_gaussian_to_peak(name)

            if fundamental_frequency is not None and fit is not None:
                print(f"  ω₀: {fundamental_frequency:.4f} rad/ps "
                      f"(σ_L={fit['sigma_left']:.4f} Å, σ_R={fit['sigma_right']:.4f} Å, "
                      f"k={bond_strength:.4f} kJ/mol/Å²)")

            self.ion_pair_results[name] = {
                'scl': scl_pair,
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
                'b_ph': ph_val,
                'gaussian_fit': fit,
            }

            self.plot_data.append({
                'ion_pair_ca_i': name,
                'x_range': self.x_grid,
                'S_i': S_y_grid,
                'x_SCL_pair': scl_pair,
            })

        self.avg_SCL = total_weighted_scl / total_weight_norm if total_weight_norm > 0 else 0
        self.plot_data.append({'avg_SCL': self.avg_SCL})
        print(f"Average SCL: {self.avg_SCL:.4f} A")

        self._save_csv_results()

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    def _save_csv_results(self):
        filename = os.path.join(get_scl_dir(), 'SCL_results.csv')
        file_exists = os.path.isfile(filename)

        base_headers = ['Composition', 'Source', 'Temperature (K)', 'Average SCL (A)']
        row = [self.comp, self.source, self.temp, round(self.avg_SCL, 5)]

        sorted_pairs = sorted(self.ion_pair_results.items())
        headers = base_headers.copy()
        data = row.copy()

        for i in range(1, 7):
            headers.extend([
                f'Pair {i} Label', f'Pair {i} SCL_i (A)',
                f'Pair {i} Peak X (A)', f'Pair {i} Peak Y',
                f'Pair {i} Min X (A)', f'Pair {i} Min Y',
                f'Pair {i} CC Peak X (A)', f'Pair {i} CC Peak Y',
                f'Pair {i} PMF at Peak (kJ/mol)', f'Pair {i} Bond Strength (kJ/mol/A²)',
                f'Pair {i} Reduced Mass (g/mol)', f'Pair {i} Fundamental Frequency (rad/ps)',
                f'Pair {i} b_PH (VDOS)',
            ])
            if i <= len(sorted_pairs):
                name, res = sorted_pairs[i - 1]
                data.extend([
                    name, round(res['scl'], 5),
                    round(res['peak_x'] or 0, 5), round(res['peak_y'] or 0, 5),
                    round(res['minima_x'] or 0, 5), round(res['minima_y'] or 0, 5),
                    round(res['cc_peak_x'] or 0, 5), round(res['cc_peak_y'] or 0, 5),
                    round(res['pmf_at_peak'] or 0, 5), round(res['bond_strength'] or 0, 5),
                    round(res['reduced_mass'] or 0, 5), round(res['fundamental_frequency'] or 0, 5),
                    round(res.get('b_ph', 0), 5),
                ])
            else:
                data.extend([''] * 13)

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
        df['lambda_BC'] = self.gamma_bc
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
            'font.family': 'Times New Roman',
            'font.size': 14,
            'axes.labelsize': 14,
            'axes.labelweight': 'bold',
            'axes.linewidth': 1.5,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.major.width': 1.75,
            'ytick.major.width': 1.75,
            'legend.frameon': False,
            'legend.fontsize': int(round(0.95 * 12)),
            'mathtext.fontset': 'custom',
            'mathtext.rm': 'Times New Roman',
            'mathtext.it': 'Times New Roman:italic',
            'mathtext.bf': 'Times New Roman:bold',
        })

        plt.figure(figsize=(4.75, 4.25))
        ion_pair_colors = {}

        for ion_pair, pdf_data in self.ion_pairs.items():
            if ion_pair in self.weighted_splines:
                weighted_y = self.weighted_splines[ion_pair](self.x_grid)
                spline, = plt.plot(self.x_grid, weighted_y, label=f"{ion_pair}")
                ion_pair_colors[ion_pair] = spline.get_color()

        # S(r) bars
        for data in self.plot_data:
            if 'ion_pair_ca_i' in data:
                ion_pair = data['ion_pair_ca_i']
                color = ion_pair_colors.get(ion_pair, 'black')
                if 'S_i' in data and len(data['S_i']) == len(self.x_grid):
                    S_i = data['S_i']
                elif 'S_i' in data and 'x_range' in data:
                    S_i = np.interp(self.x_grid, data['x_range'], data['S_i'])
                else:
                    continue
                step_indices = np.where(np.diff(S_i) != 0)[0] + 1
                step_indices = np.concatenate(([0], step_indices, [len(S_i) - 1]))
                for i in range(len(step_indices) - 1):
                    start_idx = step_indices[i]
                    end_idx = step_indices[i + 1] if i < len(step_indices) - 1 else len(self.x_grid) - 1
                    x_start = self.x_grid[start_idx]
                    x_end = self.x_grid[end_idx] if end_idx < len(self.x_grid) else self.x_grid[-1]
                    if i < len(step_indices) - 2:
                        bar_width = self.x_grid[step_indices[i + 1]] - x_start
                    else:
                        bar_width = x_end - x_start
                    s_value = S_i[start_idx]
                    alpha = 0.1 + 0.45 * s_value
                    bar_x = (x_start + x_end) / 2
                    plt.bar(bar_x, s_value, width=bar_width,
                            color=color, alpha=alpha * 0.6, edgecolor='none',
                            align='center', zorder=0)

        # S(r) dotted lines
        for data in self.plot_data:
            if 'ion_pair_ca_i' in data:
                ion_pair = data['ion_pair_ca_i']
                color = ion_pair_colors.get(ion_pair, 'black')
                if 'S_i' in data and len(data['S_i']) == len(self.x_grid):
                    plt.plot(self.x_grid, data['S_i'], label=f"S(r): {ion_pair}",
                             color=color, linestyle='dotted')

        # SCL and experimental lines
        for data in self.plot_data:
            if 'avg_SCL' in data:
                plt.axvline(x=data['avg_SCL'], color='g', linestyle='-.',
                            label=f"$\\ell_{{\\mathrm{{sc}}}}$ = {round(data['avg_SCL'], 2)}")
        if self.gamma_bc > 0:
            plt.axvline(x=self.gamma_bc, color='k', linestyle='--',
                        label=f"$\\ell_{{\\mathrm{{exp}}}}$ = {round(self.gamma_bc, 2)}")

        plt.xlabel('r [Å]')
        plt.ylabel('g(r)')

        x_min, x_max = self.x_grid[0], self.x_grid[-1]
        first_nonzero_x = x_max
        for ion_pair in self.ion_pairs:
            if ion_pair in self.weighted_splines:
                wy = self.weighted_splines[ion_pair](self.x_grid)
                nz = np.where(wy > 0.01)[0]
                if len(nz) > 0:
                    first_nonzero_x = min(first_nonzero_x, self.x_grid[nz[0]])
        x_start = max(x_min, first_nonzero_x - 0.75)
        x_pad = (x_max - x_min) * 0.05
        plt.xlim(x_start, x_max + x_pad)

        ax = plt.gca()

        if len(self.fractions) == 1:
            ca_pairs_plot = [(ip, res) for ip, res in self.ion_pair_results.items()
                             if self.ion_pairs[ip].type == 'ca' and res.get('peak_x')]
            for idx, (ion_pair, result) in enumerate(ca_pairs_plot):
                rep_peak = result.get('peak_x')
                if not rep_peak or rep_peak <= 0:
                    continue
                r_points_full = np.arange(0.0, (x_max + x_pad) + 0.5 * rep_peak, rep_peak)
                r_labels_full = ["" if i == 0 else rf"$r_{{{i}}}$" for i in range(len(r_points_full))]
                valid = (r_points_full >= x_start) & (r_points_full <= (x_max + x_pad))
                r_points = r_points_full[valid]
                r_labels = [lbl for lbl, v in zip(r_labels_full, valid) if v]
                max_labels = 8
                if len(r_points) > max_labels:
                    step = int(np.ceil(len(r_points) / max_labels))
                    r_points = r_points[::step]
                    r_labels = r_labels[::step]
                secax = ax.twiny()
                secax.set_xlim(ax.get_xlim())
                secax.set_xticks(r_points)
                secax.set_xticklabels(r_labels)
                color = ion_pair_colors.get(ion_pair, 'black')
                for lbl in secax.get_xticklabels():
                    lbl.set_color(color)
                secax.set_xlabel("")
                if 'top' in secax.spines:
                    secax.spines['top'].set_visible(False)
                secax.tick_params(axis='x', which='major', length=5, width=1.25,
                                 colors=color, pad=6 + 12 * idx)

        legend = ax.legend(
            ncol=2, loc='upper right', bbox_to_anchor=(1.0, 1.0),
            facecolor='white', frameon=True, framealpha=0.75, edgecolor='none',
            borderpad=0.5, borderaxespad=0.5, handletextpad=0.5,
            columnspacing=0.6, handlelength=1.5, labelspacing=0.3,
        )

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

        ca_names = [n for n in self.vdos_data]
        if len(ca_names) == 0:
            print(f"  No VDOS data for {self.comp}")
            return

        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        base_filename = f'VDOS_{self.comp}_{safe_source}'

        plt.rcParams.update({
            'font.family': 'Times New Roman', 'font.size': 12,
            'axes.labelsize': 13, 'axes.labelweight': 'bold',
            'axes.linewidth': 1.3,
        })

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        pair_colors = {}
        for i, name in enumerate(ca_names):
            pair_colors[name] = color_cycle[i % len(color_cycle)]

        for name in ca_names:
            om, D = self.vdos_data[name]
            ax.plot(om, D, color=pair_colors[name], lw=1.5, label=f"D(ω): {name}")

        # Overlap shading
        for i, a in enumerate(ca_names):
            for b in ca_names[i + 1:]:
                S = self.vdos_overlaps.get((a, b), 0)
                if S < 0.001:
                    continue
                om_a, D_a = self.vdos_data[a]
                om_b, D_b = self.vdos_data[b]
                lo = min(om_a[0], om_b[0])
                hi = max(om_a[-1], om_b[-1])
                om_c = np.linspace(lo, hi, 500)
                Da = np.interp(om_c, om_a, D_a, left=0, right=0)
                Db = np.interp(om_c, om_b, D_b, left=0, right=0)
                overlap_curve = np.minimum(Da, Db)
                c_a = np.array(matplotlib.colors.to_rgb(pair_colors[a]))
                c_b = np.array(matplotlib.colors.to_rgb(pair_colors[b]))
                blend = tuple((c_a + c_b) / 2)
                ax.fill_between(om_c, 0, overlap_curve, color=blend, alpha=0.3)
                peak_idx = np.argmax(overlap_curve)
                if overlap_curve[peak_idx] > 0:
                    ax.annotate(f"S = {S:.2f}",
                                xy=(om_c[peak_idx], overlap_curve[peak_idx]),
                                fontsize=9, ha='center', va='bottom',
                                color='dimgray')

        ax.set_xlabel('ω [rad/ps]')
        ax.set_ylabel('D(ω) [ps/rad]')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.7, edgecolor='none')
        fig.tight_layout()

        if save_plot:
            fig.savefig(os.path.join(output_dir, f'{base_filename}.png'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(output_dir, f'{base_filename}.pdf'), bbox_inches='tight')
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

        ca_pairs = [(n, r) for n, r in self.ion_pair_results.items()
                     if self.ion_pairs[n].type == 'ca' and r.get('gaussian_fit')]
        if not ca_pairs:
            return

        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        base_filename = f'GaussianFit_{self.comp}_{safe_source}'

        n_pairs = len(ca_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(5.0 * n_pairs, 4.5), squeeze=False)
        axes = axes[0]

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx, (name, result) in enumerate(ca_pairs):
            ax = axes[idx]
            fit = result['gaussian_fit']
            pair = self.ion_pairs[name]
            g = pair.spline(self.x_grid)

            r0 = fit['r_peak']
            gp = fit['g_peak']
            gb = fit['g_base']
            sL = fit['sigma_left']
            sR = fit['sigma_right']

            color = color_cycle[idx % len(color_cycle)]
            ax.plot(self.x_grid, g, color=color, lw=1.5, label=f"g(r): {name}")

            # Build split Gaussian with asymmetric baselines
            r_fit = np.linspace(r0 - 4 * sL, r0 + 4 * sR, 500)
            g_fit = np.zeros_like(r_fit)
            left_mask = r_fit < r0
            right_mask = ~left_mask
            # Left side: baseline = 0, amplitude = g_peak
            g_fit[left_mask] = gp * np.exp(-0.5 * ((r_fit[left_mask] - r0) / sL) ** 2)
            # Right side: baseline = g_base, amplitude = g_peak - g_base
            g_fit[right_mask] = (gp - gb) * np.exp(-0.5 * ((r_fit[right_mask] - r0) / sR) ** 2) + gb

            ax.plot(r_fit, g_fit, 'k--', lw=1.5, label="Split Gaussian fit")

            # FWHM markers
            # Left half-max = g_peak / 2 (baseline 0)
            hm_left_y = gp / 2.0
            hm_left_x = r0 - fit['hwhm_left']
            # Right half-max = (g_peak + g_base) / 2
            hm_right_y = (gp + gb) / 2.0
            hm_right_x = r0 + fit['hwhm_right']

            # Draw FWHM lines for left and right
            ax.plot([hm_left_x, r0], [hm_left_y, hm_left_y], 'r-', lw=1.2)
            ax.plot([r0, hm_right_x], [hm_right_y, hm_right_y], 'b-', lw=1.2)
            ax.annotate(f"HWHM_L = {fit['hwhm_left']:.3f} Å",
                        xy=((hm_left_x + r0) / 2, hm_left_y),
                        fontsize=7, ha='center', va='top', color='red')
            ax.annotate(f"HWHM_R = {fit['hwhm_right']:.3f} Å",
                        xy=((r0 + hm_right_x) / 2, hm_right_y),
                        fontsize=7, ha='center', va='top', color='blue')

            # Peak and baseline markers
            ax.plot(r0, gp, 'o', color=color, markersize=6, zorder=5)
            ax.axhline(gb, color='gray', ls=':', lw=0.8, alpha=0.6)
            ax.axvline(r0, color=color, ls=':', lw=0.8, alpha=0.4)

            # Annotation
            omega_0 = result.get('fundamental_frequency')
            k_val = result.get('bond_strength')
            b_kf = gb / gp if gp > 0 else 0
            info = (
                f"$r_{{peak}}$ = {r0:.3f} Å\n"
                f"$g_{{peak}}$ = {gp:.2f}\n"
                f"$g_{{base}}$ = {gb:.2f}\n"
                f"$\\sigma_L$ = {sL:.4f} Å\n"
                f"$\\sigma_R$ = {sR:.4f} Å\n"
                f"$b_{{KF}}$ = {b_kf:.3f}"
            )
            if k_val is not None:
                info += f"\n$k$ = {k_val:.2f} kJ/mol/Å²"
            if omega_0 is not None:
                info += f"\n$\\omega_0$ = {omega_0:.2f} rad/ps"

            ax.text(0.97, 0.97, info, transform=ax.transAxes,
                    fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.85))

            ax.set_xlabel('r [Å]')
            ax.set_ylabel('g(r)')
            ax.set_title(name, fontsize=11)
            ax.legend(fontsize=8, loc='upper left')

            # Set x limits around the peak
            ax.set_xlim(0,6)
            ax.set_ylim(0,12)
            # ax.set_xlim(r0 - 5 * sL, r0 + 6 * sR)
            # ax.set_ylim(bottom=0)

        comp_label = format_composition_with_subscripts(self.comp)
        fig.suptitle(f'{comp_label} ({self.temp}K) — Split Gaussian Fits', fontsize=12)
        fig.tight_layout()

        if save_plot:
            fig.savefig(os.path.join(output_dir, f'{base_filename}.png'), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(output_dir, f'{base_filename}.pdf'), bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close(fig)


# ==========================================
# 2. Analyzer (batch runner)
# ==========================================

class PDFAnalyzer:
    def __init__(self, save_plot_data=False, show_plot=False,
                 plot_vdos=False, plot_gaussian=False,
                 vdos_method='gaussian'):
        self.salts = []
        self.save_plot_data = save_plot_data
        self.show_plot = show_plot
        self.plot_vdos = plot_vdos
        self.plot_gaussian = plot_gaussian
        self.vdos_method = vdos_method

    def add_molten_salt(self, prepared_csv, comp_str, source, temp, gamma_bc):
        self.salts.append(MoltenSaltPDF(
            prepared_csv, comp_str, source, temp, gamma_bc,
            vdos_method=self.vdos_method,
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


# ==========================================
# 3. Helper: build prepared CSV path
# ==========================================

def _prep_path(comp_str, source, temp):
    """Build path to prepared CSV from composition, source, temperature."""
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
        vdos_method='gaussian', # 'gaussian' or 'pmf'
    )

    # --- Unary Salts (9) ---
    analyzer.add_molten_salt(_prep_path("1.0LiF", 'Walz, 2019', 1121), "1.0LiF", 'Walz, 2019', 1121, 3.28553)
    analyzer.add_molten_salt(_prep_path("1.0NaF", 'Walz, 2019', 1266), "1.0NaF", 'Walz, 2019', 1266, 5.22361)
    analyzer.add_molten_salt(_prep_path("1.0KF", 'Walz, 2019', 1131), "1.0KF", 'Walz, 2019', 1131, 4.63533)
    analyzer.add_molten_salt(_prep_path("1.0LiCl", 'Walz, 2019', 878), "1.0LiCl", 'Walz, 2019', 878, 4.10511)
    analyzer.add_molten_salt(_prep_path("1.0NaCl", 'Lu, 2021', 1200), "1.0NaCl", 'Lu, 2021', 1200, 4.48028)
    analyzer.add_molten_salt(_prep_path("1.0KCl", 'Walz, 2019', 1043), "1.0KCl", 'Walz, 2019', 1043, 4.47675)
    analyzer.add_molten_salt(_prep_path("1.0MgCl2", 'Roy, 2021', 1073), "1.0MgCl2", 'Roy, 2021', 1073, 4.76796)
    analyzer.add_molten_salt(_prep_path("1.0CaCl2", 'Bu, 2021', 1100), "1.0CaCl2", 'Bu, 2021', 1100, 7.72598)
    analyzer.add_molten_salt(_prep_path("1.0SrCl2", 'McGreevy, 1987', 1198), "1.0SrCl2", 'McGreevy, 1987', 1198, 0)

    # --- Mixtures (16) ---
    analyzer.add_molten_salt(_prep_path("0.6LiF-0.4NaF", 'Grizzi, 2024', 1473), "0.6LiF-0.4NaF", 'Grizzi, 2024', 1473, 2.63857)
    analyzer.add_molten_salt(_prep_path("0.5LiF-0.5BeF2", 'Sun, 2024', 900), "0.5LiF-0.5BeF2", 'Sun, 2024', 900, 0)
    analyzer.add_molten_salt(_prep_path("0.66LiF-0.34BeF2", 'Fayfar, 2024', 973), "0.66LiF-0.34BeF2", 'Fayfar, 2024', 973, 1.90187)
    analyzer.add_molten_salt(_prep_path("0.5LiCl-0.5KCl", 'Jiang, 2016', 727), "0.5LiCl-0.5KCl", 'Jiang, 2016', 727, 0)
    analyzer.add_molten_salt(_prep_path("0.637LiCl-0.363KCl", 'Jiang, 2016', 750), "0.637LiCl-0.363KCl", 'Jiang, 2016', 750, 0)
    analyzer.add_molten_salt(_prep_path("0.5NaCl-0.5KCl", 'Manga, 2014', 1100), "0.5NaCl-0.5KCl", 'Manga, 2014', 1100, 4.32778)
    analyzer.add_molten_salt(_prep_path("0.7LiCl-0.3CaCl2", 'Liang, 2024', 1073), "0.7LiCl-0.3CaCl2", 'Liang, 2024', 1073, 0)
    analyzer.add_molten_salt(_prep_path("0.4903NaCl-0.5097CaCl2", 'Wei, 2022', 1023), "0.4903NaCl-0.5097CaCl2", 'Wei, 2022', 1023, 3.76913)
    analyzer.add_molten_salt(_prep_path("0.718KCl-0.282CaCl2", 'Wei, 2022', 1300), "0.718KCl-0.282CaCl2", 'Wei, 2022', 1300, 0)
    analyzer.add_molten_salt(_prep_path("0.465LiF-0.115NaF-0.42KF", 'Frandsen, 2020', 873), "0.465LiF-0.115NaF-0.42KF", 'Frandsen, 2020', 873, 2.26059)
    analyzer.add_molten_salt(_prep_path("0.345NaF-0.59KF-0.065MgF2", 'Solano, 2021', 1073), "0.345NaF-0.59KF-0.065MgF2", 'Solano, 2021', 1073, 3.92263)
    analyzer.add_molten_salt(_prep_path("0.45MgCl2-0.33NaCl-0.22KCl", 'Jiang, 2024', 750), "0.45MgCl2-0.33NaCl-0.22KCl", 'Jiang, 2024', 750, 0)
    analyzer.add_molten_salt(_prep_path("0.38MgCl2-0.21NaCl-0.41KCl", 'Jiang, 2024', 750), "0.38MgCl2-0.21NaCl-0.41KCl", 'Jiang, 2024', 750, 0)
    analyzer.add_molten_salt(_prep_path("0.417NaCl-0.525CaCl2-0.058KCl", 'Wei, 2022', 1023), "0.417NaCl-0.525CaCl2-0.058KCl", 'Wei, 2022', 1023, 0)
    analyzer.add_molten_salt(_prep_path("0.535NaCl-0.315MgCl2-0.15CaCl2", 'Wei, 2022', 1023), "0.535NaCl-0.315MgCl2-0.15CaCl2", 'Wei, 2022', 1023, 3.52027)

    # --- Actinides (15) ---
    analyzer.add_molten_salt(_prep_path("1.0ThF4", 'Dai, 2015', 1633), "1.0ThF4", 'Dai, 2015', 1633, 0)
    analyzer.add_molten_salt(_prep_path("1.0UF4", 'OcadizFlores, 2021', 1357), "1.0UF4", 'OcadizFlores, 2021', 1357, 0)
    analyzer.add_molten_salt(_prep_path("0.64NaCl-0.36UCl3", 'Andersson, 2022', 1250), "0.64NaCl-0.36UCl3", 'Andersson, 2022', 1250, 2.5393)
    analyzer.add_molten_salt(_prep_path("0.85KCl-0.15UCl3", 'Andersson, 2024', 1250), "0.85KCl-0.15UCl3", 'Andersson, 2024', 1250, 0)
    analyzer.add_molten_salt(_prep_path("0.75KCl-0.25UCl3", 'Andersson, 2024', 1250), "0.75KCl-0.25UCl3", 'Andersson, 2024', 1250, 0)
    analyzer.add_molten_salt(_prep_path("0.65KCl-0.35UCl3", 'Andersson, 2024', 1250), "0.65KCl-0.35UCl3", 'Andersson, 2024', 1250, 0)
    analyzer.add_molten_salt(_prep_path("0.5KCl-0.5UCl3", 'Andersson, 2024', 1250), "0.5KCl-0.5UCl3", 'Andersson, 2024', 1250, 0)
    analyzer.add_molten_salt(_prep_path("0.5454LiF-0.3636NaF-0.091UF4", 'Grizzi, 2024', 1473), "0.5454LiF-0.3636NaF-0.091UF4", 'Grizzi, 2024', 1473, 0)
    analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '900K-AIMD-Zhang, 2026', 900), "0.78NaF-0.22UF4", '900K-AIMD-Zhang, 2026', 900, 0)
    analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '900K-CMD-Zhang, 2026', 900), "0.78NaF-0.22UF4", '900K-CMD-Zhang, 2026', 900, 0)
    analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '1000K-CMD-Zhang, 2026', 1000), "0.78NaF-0.22UF4", '1000K-CMD-Zhang, 2026', 1000, 0)
    analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '1100K-CMD-Zhang, 2026', 1100), "0.78NaF-0.22UF4", '1100K-CMD-Zhang, 2026', 1100, 0)
    analyzer.add_molten_salt(_prep_path("0.78NaF-0.22UF4", '1200K-CMD-Zhang, 2026', 1200), "0.78NaF-0.22UF4", '1200K-CMD-Zhang, 2026', 1200, 0)
    analyzer.add_molten_salt(_prep_path("0.57NaF-0.16KF-0.27UF4", '900K-AIMD-Zhang, 2026', 900), "0.57NaF-0.16KF-0.27UF4", '900K-AIMD-Zhang, 2026', 900, 0)
    analyzer.add_molten_salt(_prep_path("0.57NaF-0.16KF-0.27UF4", '1000K-AIMD-Zhang, 2026', 1000), "0.57NaF-0.16KF-0.27UF4", '1000K-AIMD-Zhang, 2026', 1000, 0)
    analyzer.add_molten_salt(_prep_path("0.57NaF-0.16KF-0.27UF4", '1100K-AIMD-Zhang, 2026', 1100), "0.57NaF-0.16KF-0.27UF4", '1100K-AIMD-Zhang, 2026', 1100, 0)
    analyzer.add_molten_salt(_prep_path("0.57NaF-0.16KF-0.27UF4", '1200K-AIMD-Zhang, 2026', 1200), "0.57NaF-0.16KF-0.27UF4", '1200K-AIMD-Zhang, 2026', 1200, 0)
    analyzer.add_molten_salt(_prep_path("0.63NaCl-0.37UCl3", 'AIMD-Zhang, 2026', 1100), "0.63NaCl-0.37UCl3", 'AIMD-Zhang, 2026', 1100, 0)

    # Run
    analyzer.analyze_all()
    analyzer.plot_all()


if __name__ == "__main__":
    main()
