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

    def __init__(self, prepared_csv, comp_str, source, temp, gamma_bc):
        self.original_comp_str = comp_str
        self.source = source
        self.temp = temp
        self.gamma_bc = gamma_bc

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
    # PMF / bond strength / reduced mass / ω₀
    # ------------------------------------------------------------------
    def calculate_pmf_and_bond_strength(self, ion_pair_name):
        """PMF from unweighted g(r); bond strength = d²V/dr² at peak."""
        if ion_pair_name not in self.ion_pairs:
            raise ValueError(f"Ion pair {ion_pair_name} not found")
        pair = self.ion_pairs[ion_pair_name]

        g_values = pair.spline(self.x_grid)
        g_values_safe = np.maximum(g_values, 1e-10)

        k_B = 8.314462618e-3  # kJ/(mol·K)
        pmf_values = -k_B * self.temp * np.log(g_values_safe)

        peak_x = pair.peak[0]
        if peak_x is None or peak_x <= 0:
            return pmf_values, None, None

        dx = self.x_grid[1] - self.x_grid[0]
        d1 = np.gradient(pmf_values, dx)
        d2 = np.gradient(d1, dx)
        peak_idx = np.argmin(np.abs(self.x_grid - peak_x))
        bond_strength = d2[peak_idx]

        return pmf_values, bond_strength, peak_x

    def calculate_reduced_mass(self, ion_pair_name):
        """Reduced mass μ = m1·m2/(m1+m2) in kg/mol."""
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

    def calculate_fundamental_frequency(self, ion_pair_name, bond_strength):
        """ω₀ = √(k/μ) in rad/ps."""
        reduced_mass = self.calculate_reduced_mass(ion_pair_name)
        if reduced_mass is None or bond_strength is None:
            return None
        if bond_strength <= 0:
            print(f"  Warning: Non-positive bond strength ({bond_strength:.4f}) for {ion_pair_name}, ω₀ undefined")
            return None

        k_si = bond_strength * 1000 / 1e-20          # J/(mol·m²)
        N_A = 6.02214076e23
        k_molecule = k_si / N_A                       # N/m per molecule
        mu_molecule = reduced_mass / N_A               # kg per molecule
        omega = np.sqrt(k_molecule / mu_molecule)      # rad/s
        omega *= 1e-12                                 # rad/ps
        return omega

    # ------------------------------------------------------------------
    # VDOS calculation (corrected)
    # ------------------------------------------------------------------
    def calculate_vdos(self, ion_pair_name, n_bins=200):
        """Compute the normalized vibrational density of states D(ω) for a ca pair.

        Uses Savgol derivatives for a smooth d²W/dr², restricts to the
        contiguous concave region around the first peak, maps r²g(r) into
        ω-space via histogram binning, and smooths the result.

        Returns:
            (omega_centers, D_omega)  or  (None, None) on failure.
        """
        if ion_pair_name not in self.ion_pairs:
            return None, None
        pair = self.ion_pairs[ion_pair_name]

        # --- 1. Unweighted g(r) and PMF ---
        g_values = pair.spline(self.x_grid)
        g_safe = np.maximum(g_values, 1e-10)
        k_B = 8.314462618e-3  # kJ/(mol·K)
        pmf = -k_B * self.temp * np.log(g_safe)

        # --- 2. Smooth second derivative of PMF ---
        # Use Savgol filter to compute d²PMF/dr² analytically from a local
        # polynomial fit, avoiding the massive noise amplification of two
        # successive np.gradient calls.
        dx = self.x_grid[1] - self.x_grid[0]
        # Window ~ 5 % of grid points, but at least 7 and at most 51, must be odd
        smooth_wl = min(len(pmf) // 20, 51)
        if smooth_wl % 2 == 0:
            smooth_wl += 1
        smooth_wl = max(smooth_wl, 7)
        # Savgol deriv=2 gives the second derivative directly
        k_r = savgol_filter(pmf, window_length=smooth_wl, polyorder=4, deriv=2, delta=dx)

        # --- 3. Identify peak and contiguous concave region ---
        peak_x = pair.peak[0]
        if peak_x is None:
            return None, None
        peak_idx = np.argmin(np.abs(self.x_grid - peak_x))

        # Walk left from peak until k(r) <= 0
        left_idx = 0
        for i in range(peak_idx - 1, -1, -1):
            if k_r[i] <= 0:
                left_idx = i + 1
                break

        # Walk right from peak until k(r) <= 0
        right_idx = len(k_r) - 1
        for i in range(peak_idx + 1, len(k_r)):
            if k_r[i] <= 0:
                right_idx = i - 1
                break

        if right_idx <= left_idx + 2:
            print(f"  VDOS: concave region too narrow for {ion_pair_name}")
            return None, None

        # Extract the shell
        r_shell = self.x_grid[left_idx:right_idx + 1]
        g_shell = g_values[left_idx:right_idx + 1]
        k_shell = np.maximum(k_r[left_idx:right_idx + 1], 0)   # clip residual negatives

        # --- 4. Reduced mass ---
        mu = self.calculate_reduced_mass(ion_pair_name)
        if mu is None:
            return None, None
        N_A = 6.02214076e23
        mu_molecule = mu / N_A   # kg per molecule

        # --- 5. ω(r) = √(k(r)/μ) in rad/ps ---
        k_si = k_shell * 1000 / 1e-20 / N_A    # N/m per molecule
        omega_r = np.sqrt(k_si / mu_molecule) * 1e-12   # rad/ps

        # --- 6. Probability weight: r²g(r) ---
        prob = r_shell**2 * g_shell

        # --- 7. Build D(ω) via weighted histogram ---
        omega_max = np.max(omega_r) * 1.05 if np.max(omega_r) > 0 else 1.0
        bin_edges = np.linspace(0, omega_max, n_bins + 1)
        D_omega, _ = np.histogram(omega_r, bins=bin_edges, weights=prob)
        omega_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # --- 8. Normalize so ∫D(ω)dω = 1 ---
        total = np.trapezoid(D_omega, omega_centers)
        if total > 0:
            D_omega /= total

        # --- 9. Light smoothing of histogram artifacts ---
        sm_wl = min(n_bins // 10, 21)
        if sm_wl % 2 == 0:
            sm_wl += 1
        sm_wl = max(sm_wl, 5)
        if len(D_omega) > sm_wl:
            D_omega = savgol_filter(D_omega, window_length=sm_wl, polyorder=3, mode='nearest')
            D_omega = np.maximum(D_omega, 0)
            # Re-normalize after smoothing
            total = np.trapezoid(D_omega, omega_centers)
            if total > 0:
                D_omega /= total

        return omega_centers, D_omega

    # ------------------------------------------------------------------
    # VDOS overlap computation
    # ------------------------------------------------------------------
    def _compute_vdos_overlaps(self):
        """Compute VDOS for all ca pairs, then build the pairwise overlap matrix."""
        ca_pairs = [n for n, p in self.ion_pairs.items() if p.type == 'ca']
        self.vdos_data = {}
        self.vdos_overlaps = {}

        # Compute VDOS for each ca pair
        for name in ca_pairs:
            omega, D = self.calculate_vdos(name)
            if omega is not None and D is not None:
                self.vdos_data[name] = (omega, D)
            else:
                print(f"  VDOS computation failed for {name}")

        # Build overlap matrix S_AB = ∫ min(D_A, D_B) dω
        successful = list(self.vdos_data.keys())
        for i, pA in enumerate(successful):
            omA, dA = self.vdos_data[pA]
            for j, pB in enumerate(successful):
                if j < i:
                    # Symmetric
                    self.vdos_overlaps[(pA, pB)] = self.vdos_overlaps[(pB, pA)]
                    continue
                if pA == pB:
                    self.vdos_overlaps[(pA, pB)] = 1.0
                    continue
                omB, dB = self.vdos_data[pB]
                # Interpolate both onto common grid
                om_min = min(omA[0], omB[0])
                om_max = max(omA[-1], omB[-1])
                om_common = np.linspace(om_min, om_max, 500)
                fA = interp1d(omA, dA, bounds_error=False, fill_value=0)(om_common)
                fB = interp1d(omB, dB, bounds_error=False, fill_value=0)(om_common)
                S = np.trapezoid(np.minimum(fA, fB), om_common)
                S = np.clip(S, 0, 1)
                self.vdos_overlaps[(pA, pB)] = S

        # Print overlap matrix
        if len(successful) > 1:
            print("\n  VDOS Overlap Matrix (S_AB):")
            header = "          " + "  ".join(f"{n:>10s}" for n in successful)
            print(header)
            for pA in successful:
                row = f"  {pA:>8s}"
                for pB in successful:
                    S = self.vdos_overlaps.get((pA, pB), 0)
                    row += f"  {S:10.4f}"
                print(row)

    def _calculate_vdos_b_ph(self, pair_name, ca_pairs, sum_ca_weights):
        """Calculate b_PH using VDOS overlap weighting.

        b_PH,ca = 1 - Σ_j (x_j · S_{ca,j}) / Σ_j x_j

        where j runs over all ca pairs (including self, where S=1).
        Falls back to original concentration-only formula if VDOS failed.
        """
        pair = self.ion_pairs[pair_name]
        has_vdos = pair_name in self.vdos_data

        if not has_vdos or len(self.vdos_data) < 2:
            # Original formula (no VDOS data or unary salt)
            if sum_ca_weights > 0:
                return np.clip(1.0 - pair.weight / sum_ca_weights, 0, 1)
            return 1.0

        numerator = 0.0
        denominator = 0.0
        for other_name, other_pair in self.ion_pairs.items():
            if other_pair.type != 'ca':
                continue
            x_j = other_pair.weight
            S_ij = self.vdos_overlaps.get((pair_name, other_name),
                    self.vdos_overlaps.get((other_name, pair_name), 0.0))
            numerator += x_j * S_ij
            denominator += x_j

        if denominator > 0:
            b_ph = 1.0 - numerator / denominator
        else:
            b_ph = 1.0

        return np.clip(b_ph, 0, 1)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze_pdf(self):
        print(f"\n### Analysis for {self.comp} ###")

        total_weighted_scl = 0
        total_weight_norm = 0

        # Identify CA pairs
        ca_pairs = [p for name, p in self.ion_pairs.items() if p.type == 'ca']
        sum_ca_weights = sum(p.weight for p in ca_pairs)

        # Compute VDOS overlaps before the pair loop
        self._compute_vdos_overlaps()

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

            # b_KF (Bond Strength)
            g_peak = pair.peak[1]
            g_min = pair.minima[1] if pair.minima[0] else 0
            kf_val = 1.0
            if g_peak > 1e-6:
                kf_val = 1 - (g_peak - g_min) / g_peak
            kf_val = np.clip(kf_val, 0, 1)

            # b_PH (Phonon Transfer — VDOS-weighted)
            ph_val = self._calculate_vdos_b_ph(name, ca_pairs, sum_ca_weights)
            print(f"  b_PH (VDOS): {ph_val:.4f}")

            # Cation-cation pair lookup
            cation = name.split('-')[0]
            cc_name = standardize_ion_pair(f"{cation}-{cation}")
            has_cc = cc_name in self.ion_pairs

            b_KF_vals = []
            b_NI_vals = []
            b_PH_vals = []
            beta_vals = []

            for m, r_m in enumerate(transfer_points, 1):
                g_tot_val = 0
                for pair_name, pair_data in self.ion_pairs.items():
                    if cation in pair_name.split('-'):
                        g_tot_val += self.weighted_splines[pair_name](r_m)

                g_ideal_val = 0
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
                    beta = kf_val / (1 - kf_val) + ph_val / (1 - ph_val) + ni_val / (1 - ni_val)
                beta_vals.append(beta)

            # Cumulative survival S(r)
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
            for i, x_val in enumerate(self.x_grid):
                if curr_s_idx < len(transfer_points):
                    if x_val >= transfer_points[curr_s_idx]:
                        curr_s_idx += 1
                if curr_s_idx < len(S_discrete):
                    S_y_grid[i] = S_discrete[curr_s_idx]
                else:
                    S_y_grid[i] = S_discrete[-1]

            # PMF / bond strength / ω₀
            pmf_values, bond_strength, peak_x = self.calculate_pmf_and_bond_strength(name)
            reduced_mass = self.calculate_reduced_mass(name)
            fundamental_frequency = self.calculate_fundamental_frequency(name, bond_strength)

            if fundamental_frequency is not None:
                print(f"  ω₀: {fundamental_frequency:.4f} rad/ps "
                      f"(k={bond_strength:.4f} kJ/mol/Å², "
                      f"μ={reduced_mass * 1000:.4f} g/mol)")

            # Collect VDOS overlap values for this pair
            pair_overlaps = {}
            for other_name in self.vdos_data:
                if other_name == name:
                    continue
                S_val = self.vdos_overlaps.get((name, other_name),
                        self.vdos_overlaps.get((other_name, name), None))
                if S_val is not None:
                    pair_overlaps[other_name] = S_val

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
                'reduced_mass': reduced_mass * 1000 if reduced_mass is not None else None,
                'fundamental_frequency': fundamental_frequency,
                'b_ph': ph_val,
                'vdos_overlaps': pair_overlaps,
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

        mode = 'a'
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
    # Plot data export
    # ------------------------------------------------------------------
    def save_plot_data(self, folder='SCL_plot_data'):
        folder = os.path.join(get_scl_dir(), folder)
        os.makedirs(folder, exist_ok=True)
        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        filename = os.path.join(folder, f'{self.comp.replace("-", "_")}_{safe_source}_plot_data.csv')

        df = pd.DataFrame({'r (A)': self.x_grid})
        for name, func in self.weighted_splines.items():
            df[f'g(r)_weighted_{name}'] = func(self.x_grid)
        for d in self.plot_data:
            if 'ion_pair_ca_i' in d:
                df[f"S_i_{d['ion_pair_ca_i']}"] = d['S_i']
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

        # Publication style
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.major.width'] = 1.75
        plt.rcParams['ytick.major.width'] = 1.75
        _small_text = int(round(0.95 * 12))
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.fontsize'] = _small_text
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

        plt.figure(figsize=(4.75, 4.25))
        ion_pair_colors = {}
        ca_pair_count = sum(1 for p in self.ion_pairs.values() if p.type == 'ca')

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
                for si in range(len(step_indices) - 1):
                    start_idx = step_indices[si]
                    end_idx = step_indices[si + 1] if si < len(step_indices) - 1 else len(self.x_grid) - 1
                    x_start = self.x_grid[start_idx]
                    x_end = self.x_grid[end_idx] if end_idx < len(self.x_grid) else self.x_grid[-1]
                    if si < len(step_indices) - 2:
                        bar_width = self.x_grid[step_indices[si + 1]] - x_start
                    else:
                        bar_width = x_end - x_start
                    s_value = S_i[start_idx]
                    alpha = 0.1 + 0.45 * s_value
                    bar_x = (x_start + x_end) / 2
                    plt.bar(bar_x, s_value, width=bar_width, color=color,
                            alpha=alpha * 0.6, edgecolor='none', align='center', zorder=0)

        # S(r) lines
        for data in self.plot_data:
            if 'ion_pair_ca_i' in data:
                ion_pair = data['ion_pair_ca_i']
                color = ion_pair_colors.get(ion_pair, 'black')
                if 'S_i' in data and len(data['S_i']) == len(self.x_grid):
                    plt.plot(self.x_grid, data['S_i'], label=f"S(r): {ion_pair}",
                             color=color, linestyle='dotted')
                elif 'S_i' in data and 'x_range' in data:
                    interp_S_i = np.interp(self.x_grid, data['x_range'], data['S_i'])
                    plt.plot(self.x_grid, interp_S_i, label=f"S(r): {ion_pair}",
                             color=color, linestyle='dotted')

        # SCL and exp lines
        if self.plot_data:
            last = self.plot_data[-1]
            if 'avg_SCL' in last:
                plt.axvline(x=last['avg_SCL'], color='g', linestyle='-.',
                            label=f"$\\ell_{{\\mathrm{{sc}}}}$ = {round(last['avg_SCL'], 2)}")
        if self.gamma_bc > 0:
            plt.axvline(x=self.gamma_bc, color='k', linestyle='--',
                        label=f"$\\ell_{{\\mathrm{{exp}}}}$ = {round(self.gamma_bc, 2)}")

        plt.xlabel('r [Å]')
        plt.ylabel('g(r)')

        x_min, x_max = self.x_grid[0], self.x_grid[-1]
        first_nonzero_x = x_max
        for ion_pair in self.weighted_splines:
            wy = self.weighted_splines[ion_pair](self.x_grid)
            nz = np.where(wy > 0.01)[0]
            if len(nz) > 0:
                first_nonzero_x = min(first_nonzero_x, self.x_grid[nz[0]])
        x_start = max(x_min, first_nonzero_x - 0.75)
        x_pad = (x_max - x_min) * 0.05
        plt.xlim(x_start, x_max + x_pad)

        ax = plt.gca()
        if len(self.fractions) == 1:
            ca_res = [(ip, res) for ip, res in self.ion_pair_results.items()
                      if self.ion_pairs[ip].type == 'ca' and res.get('peak_x')]
            max_labels = 8
            for idx, (ion_pair, result) in enumerate(ca_res):
                rep_peak = result.get('peak_x')
                if not rep_peak or rep_peak <= 0:
                    continue
                r_points_full = np.arange(0.0, (x_max + x_pad) + 0.5 * rep_peak, rep_peak)
                r_labels_full = ["" if i == 0 else rf"$r_{{{i}}}$" for i in range(len(r_points_full))]
                valid = (r_points_full >= x_start) & (r_points_full <= (x_max + x_pad))
                r_points = r_points_full[valid]
                r_labels = [lbl for lbl, v in zip(r_labels_full, valid) if v]
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
            plot_filename = os.path.join(output_dir, f'{base_filename}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_filename}")
            pdf_filename = os.path.join(output_dir, f'{base_filename}.pdf')
            plt.savefig(pdf_filename, bbox_inches='tight')
            print(f"Plot saved to {pdf_filename}")
        if show_plot:
            plt.show()
        plt.close()

        # Store colors for VDOS plot to reuse
        self._ion_pair_colors = ion_pair_colors

    # ------------------------------------------------------------------
    # VDOS plot
    # ------------------------------------------------------------------
    def plot_vdos(self, show_plot=True, save_plot=True, output_dir=None):
        """Plot the VDOS D(ω) for each ca pair with overlap shading."""
        if not self.vdos_data:
            print(f"No VDOS data for {self.comp}")
            return
        if len(self.vdos_data) < 1:
            return

        if output_dir is None:
            output_dir = os.path.join(get_scl_dir(), 'SCL_plots')
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)

        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        base_filename = f'VDOS_{self.comp}_{safe_source}'

        # Publication style (inherit from plot_pdf if already set)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.major.width'] = 1.75
        plt.rcParams['ytick.major.width'] = 1.75
        _small_text = int(round(0.95 * 12))
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.fontsize'] = _small_text
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

        fig, ax = plt.subplots(figsize=(4.75, 4.25))

        # Reuse colors from PDF plot if available
        colors = getattr(self, '_ion_pair_colors', {})
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ci = 0

        pair_colors = {}
        ca_names = sorted(self.vdos_data.keys())

        # Plot each D(ω)
        for name in ca_names:
            omega, D = self.vdos_data[name]
            c = colors.get(name, color_cycle[ci % len(color_cycle)])
            ci += 1
            pair_colors[name] = c
            ax.plot(omega, D, label=f"D(ω): {name}", color=c, linewidth=1.5)

        # Shade overlaps between pairs
        for i, pA in enumerate(ca_names):
            omA, dA = self.vdos_data[pA]
            for j, pB in enumerate(ca_names):
                if j <= i:
                    continue
                omB, dB = self.vdos_data[pB]
                S_val = self.vdos_overlaps.get((pA, pB),
                        self.vdos_overlaps.get((pB, pA), 0))

                # Interpolate onto common grid
                om_min = min(omA[0], omB[0])
                om_max = max(omA[-1], omB[-1])
                om_common = np.linspace(om_min, om_max, 500)
                fA = interp1d(omA, dA, bounds_error=False, fill_value=0)(om_common)
                fB = interp1d(omB, dB, bounds_error=False, fill_value=0)(om_common)
                min_AB = np.minimum(fA, fB)

                # Blend colors
                cA = np.array(plt.matplotlib.colors.to_rgb(pair_colors[pA]))
                cB = np.array(plt.matplotlib.colors.to_rgb(pair_colors[pB]))
                blend = 0.5 * (cA + cB)

                ax.fill_between(om_common, 0, min_AB, color=blend, alpha=0.3)

                # Annotate S value near the overlap peak
                peak_idx = np.argmax(min_AB)
                if min_AB[peak_idx] > 0:
                    ax.annotate(f"S = {S_val:.2f}",
                                xy=(om_common[peak_idx], min_AB[peak_idx]),
                                xytext=(0, 8), textcoords='offset points',
                                fontsize=_small_text, ha='center', color=blend * 0.7)

        ax.set_xlabel('ω [rad/ps]')
        ax.set_ylabel('D(ω) [ps/rad]')
        ax.legend(loc='upper right', facecolor='white', frameon=True,
                  framealpha=0.75, edgecolor='none')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        plt.tight_layout()

        if save_plot:
            png_path = os.path.join(output_dir, f'{base_filename}.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            print(f"VDOS plot saved to {png_path}")
            pdf_path = os.path.join(output_dir, f'{base_filename}.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"VDOS plot saved to {pdf_path}")
        if show_plot:
            plt.show()
        plt.close()


# ==========================================
# 2. Batch Analyzer
# ==========================================

class SCLAnalyzer:
    """Batch analyse multiple molten salt compositions."""

    def __init__(self, save_plot_data=False, show_plot=False, plot_vdos=False):
        self.salts = []
        self.save_plot_data = save_plot_data
        self.show_plot = show_plot
        self.plot_vdos = plot_vdos

    def _prep_path(self, comp_str, source, temp):
        """Construct the expected Prepared_PDF_CSV filename."""
        _, _, comp = parse_composition(comp_str)
        safe_source = ''.join(c if c.isalnum() else '_' for c in source.split(',')[0].strip())
        fname = f"{comp.replace('-', '_')}_{safe_source}_{int(temp)}K.csv"
        return os.path.join(get_scl_dir(), 'Prepared_PDF_CSV', fname)

    def add(self, comp_str, source, temp, gamma_bc, prepared_csv=None):
        """Add a salt for analysis.

        Parameters
        ----------
        comp_str : str   e.g. '0.5NaCl-0.5KCl'
        source   : str   e.g. 'Manga, 2014'
        temp     : float Temperature in K
        gamma_bc : float Experimental SCL (0 if unknown)
        prepared_csv : str or None
            Explicit path to prepared CSV. If None, auto-constructed from the
            other arguments.
        """
        if prepared_csv is None:
            prepared_csv = self._prep_path(comp_str, source, temp)
        self.salts.append(MoltenSaltPDF(prepared_csv, comp_str, source, temp, gamma_bc))

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


# ==========================================
# 3. Main
# ==========================================

def main():
    analyzer = SCLAnalyzer(save_plot_data=True, show_plot=False, plot_vdos=True)

    # Unary Salts
    analyzer.add("1.0LiF", 'Walz, 2019', 1121, 3.28553)
    analyzer.add("1.0NaF", 'Walz, 2019', 1266, 5.22361)
    analyzer.add("1.0KF", 'Walz, 2019', 1131, 4.63533)
    analyzer.add("1.0LiCl", 'Walz, 2019', 878, 4.10511)
    analyzer.add("1.0NaCl", 'Lu, 2021', 1200, 4.48028)
    analyzer.add("1.0KCl", 'Walz, 2019', 1043, 4.47675)
    analyzer.add("1.0MgCl2", 'Roy, 2021', 1073, 4.76796)
    analyzer.add("1.0CaCl2", 'Bu, 2021', 1100, 7.72598)
    analyzer.add("1.0SrCl2", 'McGreevy, 1987', 1198, 0)

    # Mixtures
    analyzer.add("0.6LiF-0.4NaF", 'Grizzi, 2024', 1473, 2.63857)
    analyzer.add("0.5LiF-0.5BeF2", 'Sun, 2024', 900, 0)
    analyzer.add("0.66LiF-0.34BeF2", 'Fayfar, 2024', 973, 1.90187)
    analyzer.add("0.5LiCl-0.5KCl", 'Jiang, 2016', 727, 0)
    analyzer.add("0.637LiCl-0.363KCl", 'Jiang, 2016', 750, 0)
    analyzer.add("0.5NaCl-0.5KCl", 'Manga, 2014', 1100, 4.32778)
    analyzer.add("0.7LiCl-0.3CaCl2", 'Liang, 2024', 1073, 0)
    analyzer.add("0.4903NaCl-0.5097CaCl2", 'Wei, 2022', 1023, 3.76913)
    analyzer.add("0.718KCl-0.282CaCl2", 'Wei, 2022', 1300, 0)
    analyzer.add("0.465LiF-0.115NaF-0.42KF", 'Frandsen, 2020', 873, 2.26059)
    analyzer.add("0.345NaF-0.59KF-0.065MgF2", 'Solano, 2021', 1073, 3.92263)
    analyzer.add("0.45MgCl2-0.33NaCl-0.22KCl", 'Jiang, 2024', 750, 0)
    analyzer.add("0.38MgCl2-0.21NaCl-0.41KCl", 'Jiang, 2024', 750, 0)
    analyzer.add("0.417NaCl-0.525CaCl2-0.058KCl", 'Wei, 2022', 1023, 0)
    analyzer.add("0.535NaCl-0.315MgCl2-0.15CaCl2", 'Wei, 2022', 1023, 3.52027)

    # Actinides
    analyzer.add("1.0ThF4", 'Dai, 2015', 1633, 0)
    analyzer.add("1.0UF4", 'OcadizFlores, 2021', 1357, 0)
    analyzer.add("0.64NaCl-0.36UCl3", 'Andersson, 2022', 1250, 2.5393)
    analyzer.add("0.85KCl-0.15UCl3", 'Andersson, 2024', 1250, 0)
    analyzer.add("0.75KCl-0.25UCl3", 'Andersson, 2024', 1250, 0)
    analyzer.add("0.65KCl-0.35UCl3", 'Andersson, 2024', 1250, 0)
    analyzer.add("0.5KCl-0.5UCl3", 'Andersson, 2024', 1250, 0)
    analyzer.add("0.5454LiF-0.3636NaF-0.091UF4", 'Grizzi, 2024', 1473, 0)
    analyzer.add("0.78NaF-0.22UF4", '900K-AIMD-Zhang, 2026', 900, 0)
    analyzer.add("0.78NaF-0.22UF4", '900K-CMD-Zhang, 2026', 900, 0)
    analyzer.add("0.78NaF-0.22UF4", '1000K-CMD-Zhang, 2026', 1000, 0)
    analyzer.add("0.78NaF-0.22UF4", '1100K-CMD-Zhang, 2026', 1100, 0)
    analyzer.add("0.78NaF-0.22UF4", '1200K-CMD-Zhang, 2026', 1200, 0)
    analyzer.add("0.57NaF-0.16KF-0.27UF4", '900K-AIMD-Zhang, 2026', 900, 0)
    analyzer.add("0.57NaF-0.16KF-0.27UF4", '1000K-AIMD-Zhang, 2026', 1000, 0)
    analyzer.add("0.57NaF-0.16KF-0.27UF4", '1100K-AIMD-Zhang, 2026', 1100, 0)
    analyzer.add("0.57NaF-0.16KF-0.27UF4", '1200K-AIMD-Zhang, 2026', 1200, 0)
    analyzer.add("0.63NaCl-0.37UCl3", 'AIMD-Zhang, 2026', 1100, 0)

    # Run
    analyzer.analyze_all()
    analyzer.plot_all()


if __name__ == "__main__":
    main()
