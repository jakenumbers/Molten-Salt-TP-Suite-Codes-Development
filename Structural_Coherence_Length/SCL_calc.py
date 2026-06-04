import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from datetime import datetime
import re
from mendeleev import element
import csv

# ==========================================
# 1. Helper Functions
# ==========================================

def get_scl_dir():
    """Return the Structural_Coherence_Length directory, whether running from it or its parent."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == 'Structural_Coherence_Length':
        return script_dir
    candidate = os.path.join(script_dir, 'Structural_Coherence_Length')
    if os.path.isdir(candidate):
        return candidate
    return script_dir  # fallback

def standardize_ion_pair(ion_pair):
    """Standardizes ion pair strings (e.g., 'LiCl-KCl' -> 'Cl-Li')."""
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
    # Default to 0 if oxidation states are missing
    state1 = el1.oxistates[0] if el1.oxistates else 0
    state2 = el2.oxistates[0] if el2.oxistates else 0
    
    # Sort cation-anion
    if state1 > 0 and state2 < 0:
        return f"{elems1[0]}-{elems2[0]}"
    elif state1 < 0 and state2 > 0:
        return f"{elems2[0]}-{elems1[0]}"
    else:
        # Sort alphabetical for like-charged
        return '-'.join(sorted([elems1[0], elems2[0]]))

def format_composition_with_subscripts(composition_str):
    """Format composition string for plots (e.g., UCl3 -> UCl₃)."""
    def replace_with_subscript(match):
        el = match.group(1)
        num = match.group(2)
        if num:
            subs = ''.join([f'₀₁₂₃₄₅₆₇₈₉'[int(d)] for d in num])
            return f"{el}{subs}"
        return el
    
    parts = composition_str.split('-')
    formatted = [re.sub(r'([A-Z][a-z]?)(\d*)', replace_with_subscript, p) for p in parts]
    return '-'.join(formatted)

def estimate_fwhm(x, y):
    """Estimate FWHM of the first peak in g(r).
    
    Uses half-maximum interpolation on the raw data. Robust to moderate noise.
    
    Returns:
        fwhm: Full width at half maximum in same units as x, or None if no peak found.
    """
    peaks, _ = find_peaks(y, prominence=0.05 * np.max(y) if np.max(y) > 0 else 0, distance=10)
    if len(peaks) == 0:
        return None
    p_idx = peaks[0]
    half_max = y[p_idx] / 2.0

    # Left side: last point below half_max before peak
    left = np.where(y[:p_idx] <= half_max)[0]
    x_left = x[left[-1]] if len(left) > 0 else x[0]

    # Right side: first point below half_max after peak
    right = np.where(y[p_idx:] <= half_max)[0]
    x_right = x[p_idx + right[0]] if len(right) > 0 else x[-1]

    return x_right - x_left

def parse_composition(comp_str):
    """Parses composition string into fractions and ion counts."""
    # 1. Parse Molar Fractions
    fractions = {}
    components = comp_str.split('-')
    
    # Use regex to find number at start of string
    for comp in components:
        match = re.match(r"([0-9.]+)?([A-Za-z0-9]+)", comp)
        if match:
            frac_str, salt = match.groups()
            frac = float(frac_str) if frac_str else 1.0 # Default to 1.0 if no number
            fractions[salt] = frac

    # 2. Parse Ion Counts
    ion_counts = {}
    # Re-extract salts to handle string parsing purely
    all_salts_matches = re.findall(r'([0-9.]*)([A-Z][a-z]?\d*[A-Z]?[a-z]?\d*)', comp_str)
    
    for _, salt in all_salts_matches:
        if not salt: continue
        elements = re.findall(r'([A-Z][a-z]?)([0-9]*)', salt)
        i_counts = {}
        for el, count in elements:
            cnt = int(count) if count else 1
            i_counts[el] = i_counts.get(el, 0) + cnt
        ion_counts[salt] = i_counts

    # Sort string for consistency
    def get_cation_atomic_number(s):
        try:
            elems = re.findall(r'([A-Z][a-z]?)', s)
            if elems: return element(elems[0]).atomic_number
        except: pass
        return 999
        
    sorted_salts = sorted(fractions.keys(), key=get_cation_atomic_number)
    sorted_comp_str = '-'.join([f"{fractions[s]}{s}" for s in sorted_salts])
    
    return fractions, ion_counts, sorted_comp_str

# ==========================================
# 2. Core Classes
# ==========================================

class IonPairData:
    """Stores PDF data and properties for a single ion pair."""
    def __init__(self, name, x, y, weight):
        self.name = name
        self.x = x
        self.y = y
        self.weight = weight
        # Create spline for standard grid evaluation
        self.spline = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
        self.type = self._determine_type()
        # peak/minima will be set after common grid is created
        self.peak = (None, None)
        self.minima = (None, None)

    def _determine_type(self):
        try:
            parts = self.name.split('-')
            el1, el2 = element(parts[0]), element(parts[1])
            s1 = el1.oxistates[0] if el1.oxistates else 0
            s2 = el2.oxistates[0] if el2.oxistates else 0
            
            if s1 > 0 and s2 < 0: return "ca"
            if s1 < 0 and s2 > 0: return "ca" 
            if s1 > 0 and s2 > 0: return "cc_sim" if parts[0] == parts[1] else "cc_diff"
            if s1 < 0 and s2 < 0: return "aa"
        except:
            pass
        return "other"

    def find_features_on_grid(self, x_grid):
        """Find peak and minima after interpolating to common grid."""
        # Get weighted values on the common grid
        y_grid = self.spline(x_grid) * self.weight
        
        # Heuristics for peak finding
        x_spacing = x_grid[1] - x_grid[0]  # Uniform spacing on common grid
        min_dist = int(0.5 / x_spacing)
        
        peaks, _ = find_peaks(y_grid, prominence=0.1*np.max(y_grid) if np.max(y_grid) > 0 else 0, distance=min_dist, width=2)
        
        peak_pt = (None, None)
        min_pt = (None, None)
        
        if len(peaks) > 0:
            p_idx = peaks[0]
            peak_pt = (x_grid[p_idx], y_grid[p_idx])
            
            # Find minima after peak
            y_after = y_grid[p_idx:]
            mins, _ = find_peaks(-y_after, prominence=0.01*np.max(y_grid), distance=min_dist)
            if len(mins) > 0:
                m_idx = mins[0] + p_idx
                min_pt = (x_grid[m_idx], y_grid[m_idx])
            else:
                # If no clear minima, find global min in reasonable range (up to 2*peak)
                search_end = min(len(x_grid)-1, int(p_idx + (2.0/x_spacing)))
                if search_end > p_idx:
                    m_idx = p_idx + np.argmin(y_grid[p_idx:search_end])
                    min_pt = (x_grid[m_idx], y_grid[m_idx])
        
        self.peak = peak_pt
        self.minima = min_pt
        return peak_pt, min_pt

class MoltenSaltPDF:
    def __init__(self, pdf_file, comp_str, source, temp, gamma_bc, apply_savgol=False, savgol_window_length=None, savgol_polyorder=None):
        self.original_comp_str = comp_str
        self.source = source
        self.temp = temp
        self.gamma_bc = gamma_bc
        self.apply_savgol = apply_savgol

        # Store user overrides (None means auto-determine from FWHM)
        self.savgol_user_window = savgol_window_length
        self.savgol_user_polyorder = savgol_polyorder
        
        # Parse Composition
        self.fractions, self.ion_counts, self.comp = parse_composition(comp_str)
        self.weights = self._calculate_weights()
        
        # Load PDF Data
        self.ion_pairs = self._load_pdf(pdf_file)
        
        # Global Grid (Interpolate all to this grid for summation)
        self.x_grid = self._create_common_grid()
        
        # Find features on common grid AFTER grid is created and Savgol applied
        for pair in self.ion_pairs.values():
            pair.find_features_on_grid(self.x_grid)
        
        self.weighted_splines = self._create_weighted_splines()
        
        # Results Storage
        self.plot_data = []
        self.ion_pair_results = {}

    def _calculate_weights(self):
        el_conc = {}
        for salt, frac in self.fractions.items():
            for el, count in self.ion_counts[salt].items():
                el_conc[el] = el_conc.get(el, 0) + frac * count
                
        total_conc = sum(el_conc.values())
        rel_conc = {k: v/total_conc for k, v in el_conc.items()}
        
        weights = {}
        for el1, c1 in rel_conc.items():
            for el2, c2 in rel_conc.items():
                pair = standardize_ion_pair(f"{el1}-{el2}")
                weights[pair] = c1 * c2 # Initial weight
                
        # Normalize weights to sum to 1
        total_w = sum(weights.values())
        return {k: v/total_w for k, v in weights.items()}

    def _determine_savgol_params(self, x, y):
        """Determine Savgol filter parameters adaptively from FWHM, or use user overrides.
        
        Strategy:
            - If user provided both window_length and polyorder, use those.
            - Otherwise, estimate FWHM of first peak and set window = FWHM / 1.5 (in grid points).
            - User-provided values override individual auto-determined values.
        
        Returns:
            (window_length, polyorder) or None if smoothing should be skipped.
        """
        # Start with auto-determined values
        auto_wl = None
        auto_po = 3  # Default polyorder

        fwhm = estimate_fwhm(x, y)
        if fwhm is not None and len(x) > 1:
            dx = x[1] - x[0]
            if dx > 0:
                auto_wl = int(round(fwhm / (1.5 * dx)))
                auto_wl = max(auto_wl, 5)
                if auto_wl % 2 == 0:
                    auto_wl += 1
                print(f"    Auto Savgol: FWHM={fwhm:.4f} A, dx={dx:.5f} A -> window_length={auto_wl}")
        
        # Apply user overrides where provided
        wl = self.savgol_user_window if self.savgol_user_window is not None else auto_wl
        po = self.savgol_user_polyorder if self.savgol_user_polyorder is not None else auto_po

        # If we still have no window length, skip smoothing
        if wl is None:
            print(f"    Savgol skipped: could not determine window length (no peak found)")
            return None

        # Validate parameters
        if wl < 5:
            wl = 5
        if wl % 2 == 0:
            wl += 1
        if po >= wl:
            po = wl - 1
        po = max(po, 1)

        if self.savgol_user_window is not None or self.savgol_user_polyorder is not None:
            print(f"    Savgol params (user override): window_length={wl}, polyorder={po}")
        else:
            print(f"    Savgol params (auto): window_length={wl}, polyorder={po}")

        return (wl, po)

    def _load_pdf(self, filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
        path_candidates = [
            os.path.join(repo_root, "PDF_Analysis", "PDF_CSV", filename),
            os.path.join(script_dir, "PDF_CSV", filename)
        ]

        path = None
        for candidate in path_candidates:
            if os.path.exists(candidate):
                path = candidate
                break

        if path is None:
            print(f"Error: File {filename} not found in expected locations:")
            for candidate in path_candidates:
                print(f"  - {candidate}")
            return {}

        df = pd.read_csv(path, header=None)
        pairs = {}
        
        for i in range(0, df.shape[1], 2):
            raw_name = df.iloc[0, i]
            if pd.isna(raw_name): continue
            
            name = standardize_ion_pair(raw_name)
            x = df.iloc[2:, i].dropna().astype(float).values
            y = df.iloc[2:, i+1].dropna().astype(float).values
            
            # Sort and Clean
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
            x, u_idx = np.unique(x, return_index=True)
            y = y[u_idx]
            
            # Zero extension at start
            if x[0] > 0.1:
                x_ext = np.linspace(0, x[0], int(x[0]*20))
                x = np.concatenate([x_ext[:-1], x])
                y = np.concatenate([np.zeros(len(x_ext)-1), y])
            
            # Adaptive Savgol Smoothing (per ion pair)
            if self.apply_savgol and len(y) > 5:
                print(f"  Savgol for {name}:")
                params = self._determine_savgol_params(x, y)
                if params is not None:
                    wl, po = params
                    nonzero_idx = np.where(y > 0.001)[0]
                    if len(nonzero_idx) > wl:
                        start = nonzero_idx[0]
                        y[start:] = savgol_filter(y[start:], window_length=wl, polyorder=po, mode='nearest')

            # Ensure PDF doesn't go below zero by clipping negative values
            y = np.maximum(y, 0)

            weight = self.weights.get(name, 0)
            pairs[name] = IonPairData(name, x, y, weight)
            
        return pairs

    def _create_common_grid(self, points=2000):
        if not self.ion_pairs: return np.linspace(0, 10, points)
        max_x = min(max(p.x) for p in self.ion_pairs.values()) # Find the minimum max
        return np.linspace(0, max_x, points)

    def _create_weighted_splines(self):
        # Create lambda functions that return weighted values on the grid
        splines = {}
        for name, p in self.ion_pairs.items():
            splines[name] = lambda x, s=p.spline, w=p.weight: s(x) * w
        return splines

    def calculate_pmf_and_bond_strength(self, ion_pair_name):
        """
        Calculate Potential of Mean Force (PMF) and bond strength constant for a cation-anion pair.
        
        PMF: V(r) = -k_B*T*ln(g(r))  using the UNWEIGHTED g(r)
        Bond strength constant: k = d²V/dr² at first peak position
        
        Returns:
            pmf_values: array of PMF values on the grid
            bond_strength: second derivative of PMF at peak position (kJ/mol/Å²)
            peak_position: position of the first peak
        """
        if ion_pair_name not in self.ion_pairs:
            raise ValueError(f"Ion pair {ion_pair_name} not found")
        
        pair = self.ion_pairs[ion_pair_name]
        if pair.type != 'ca':
            print(f"Warning: {ion_pair_name} is not a cation-anion pair")
        
        # Use UNWEIGHTED g(r) for PMF — the potential of mean force is an
        # intrinsic pair property, not scaled by composition weights.
        g_values = pair.spline(self.x_grid)
        
        # Avoid log(0) by setting minimum value
        g_values_safe = np.maximum(g_values, 1e-10)
        
        # Boltzmann constant in kJ/(mol*K)
        k_B = 8.314462618e-3  # kJ/(mol*K)
        
        # Calculate PMF: V(r) = -k_B*T*ln(g(r))
        pmf_values = -k_B * self.temp * np.log(g_values_safe)
        
        # Get peak position
        peak_x = pair.peak[0]
        if peak_x is None or peak_x <= 0:
            return pmf_values, None, None
        
        # Calculate second derivative using finite differences
        dx = self.x_grid[1] - self.x_grid[0]
        first_derivative = np.gradient(pmf_values, dx)
        second_derivative = np.gradient(first_derivative, dx)
        
        # Find the index closest to the peak position
        peak_idx = np.argmin(np.abs(self.x_grid - peak_x))
        
        # Get bond strength (second derivative at peak)
        bond_strength = second_derivative[peak_idx]
        
        return pmf_values, bond_strength, peak_x

    def calculate_reduced_mass(self, ion_pair_name):
        """
        Calculate reduced mass of cation-anion pair in kg/mol.
        
        μ = (m1 * m2) / (m1 + m2)
        
        Returns:
            reduced_mass: reduced mass in kg/mol
        """
        if ion_pair_name not in self.ion_pairs:
            raise ValueError(f"Ion pair {ion_pair_name} not found")
        
        # Get elements from ion pair name
        elements = ion_pair_name.split('-')
        if len(elements) != 2:
            raise ValueError(f"Invalid ion pair format: {ion_pair_name}")
        
        try:
            el1 = element(elements[0])
            el2 = element(elements[1])
            
            # Get atomic masses in kg/mol (convert from g/mol)
            m1 = el1.mass * 1e-3  # g/mol to kg/mol
            m2 = el2.mass * 1e-3  # g/mol to kg/mol
            
            # Calculate reduced mass: μ = (m1 * m2) / (m1 + m2)
            reduced_mass = (m1 * m2) / (m1 + m2)
            
            return reduced_mass
            
        except Exception as e:
            print(f"Error calculating reduced mass for {ion_pair_name}: {e}")
            return None

    def calculate_fundamental_frequency(self, ion_pair_name, bond_strength):
        """
        Calculate fundamental frequency ω₀ = √(k/μ) where k is bond strength constant.
        
        Units:
            bond_strength: kJ/(mol·Å²)
            reduced_mass: kg/mol
            → ω₀ returned in rad/ps
        
        Returns:
            omega: fundamental frequency in rad/ps
        """
        reduced_mass = self.calculate_reduced_mass(ion_pair_name)
        
        if reduced_mass is None or bond_strength is None:
            return None
        
        if bond_strength <= 0:
            print(f"  Warning: Non-positive bond strength ({bond_strength:.4f}) for {ion_pair_name}, ω₀ undefined")
            return None
        
        # Convert bond strength from kJ/(mol·Å²) to J/(mol·m²)
        # 1 kJ = 1000 J, 1 Å = 1e-10 m → 1 Å² = 1e-20 m²
        k_si = bond_strength * 1000 / (1e-20)  # J/(mol·m²)
        
        # Convert to per-molecule quantities (divide by Avogadro's number)
        N_A = 6.02214076e23
        k_molecule = k_si / N_A       # J/m² per molecule (= N/m)
        mu_molecule = reduced_mass / N_A  # kg per molecule
        
        # ω₀ = √(k/μ) in rad/s
        omega = np.sqrt(k_molecule / mu_molecule)
        
        # Convert rad/s to rad/ps (1 ps = 1e-12 s)
        omega = omega * 1e-12  # rad/ps
        
        return omega

    def analyze_pdf(self):
        print(f"\n### Analysis for {self.comp} ###")
        
        total_weighted_scl = 0
        total_weight_norm = 0

        # Identify Cation-Anion Pairs
        ca_pairs = [p for name, p in self.ion_pairs.items() if p.type == 'ca']
        sum_ca_weights = sum(p.weight for p in ca_pairs)
        
        for pair in ca_pairs:
            name = pair.name
            print(f"Analyzing Pair: {name}")
            
            r_peak = pair.peak[0]
            if r_peak is None or r_peak <= 0:
                print(f"  Skipping {name}: No valid peak found.")
                continue

            # --- Formalism Step 1: Transfer Points ---
            # r_m = m * Delta_r (Eq. 18)
            # We calculate points until we run out of grid
            delta_r = r_peak
            transfer_points = np.arange(delta_r, self.x_grid[-1], delta_r)
            
            if len(transfer_points) == 0: continue
            
            # --- Formalism Step 2: Disruption Factors at Discrete Points ---
            b_KF_vals = []
            b_NI_vals = []
            b_PH_vals = []
            beta_vals = []
            
            # Constants for this pair
            # b_KF (Bond Strength) Eq. 22
            g_peak = pair.peak[1]
            g_min = pair.minima[1] if pair.minima[0] else 0
            # Note: Paper uses raw ratio. Since weight cancels out in ratio, we use weighted values is fine.
            kf_val = 1.0
            if g_peak > 1e-6:
                kf_val = 1 - (g_peak - g_min) / g_peak
            kf_val = np.clip(kf_val, 0, 1)

            # b_PH (Phonon Transfer) Eq. 23
            ph_val = 1.0
            if sum_ca_weights > 0:
                ph_val = 1 - (pair.weight / sum_ca_weights)
            ph_val = np.clip(ph_val, 0, 1)

            # Identify corresponding cation-cation pair
            cation = name.split('-')[0]
            cc_name = standardize_ion_pair(f"{cation}-{cation}")
            has_cc = cc_name in self.ion_pairs
            
            for m, r_m in enumerate(transfer_points, 1):
                g_tot_val = 0
                for pair_name, pair_data in self.ion_pairs.items():
                    if cation in pair_name.split('-'):
                        g_tot_val += self.weighted_splines[pair_name](r_m)
                
                g_ideal_val = 0
                if m % 2 == 1: # Odd: Cation-Anion (This pair)
                    g_ideal_val = self.weighted_splines[name](r_m)
                else:          # Even: Cation-Cation
                    if has_cc: # Check if cation-cation spline exists
                        g_ideal_val = self.weighted_splines[cc_name](r_m)
                    else: # Use the concentration of the cation-cation pair if not
                        g_ideal_val = self.weights[cc_name]
                        print(f"No cc data exists for {cc_name}, using concentration: {g_ideal_val}")
                
                # b_NI (Non-Ideal Recipient)
                ni_val = 1.0
                if g_tot_val > 1e-6:
                    ni_val = 1 - (g_ideal_val / g_tot_val)
                ni_val = np.clip(ni_val, 0, 1)
                
                # Store
                b_KF_vals.append(kf_val)
                b_PH_vals.append(ph_val)
                b_NI_vals.append(ni_val)
                
                # Beta
                if kf_val == 1 or ph_val == 1 or ni_val == 1:
                    beta = float('inf')
                else:
                    beta = (kf_val/(1-kf_val)) + (ph_val/(1-ph_val)) + (ni_val/(1-ni_val))
                beta_vals.append(beta)

            # --- Formalism Step 3: Cumulative Survival S(r) ---
            
            S_discrete = [1.0] # Value for first interval [0, r1]
            int_beta = 0
            
            for beta in beta_vals:
                if beta == float('inf'):
                    int_beta = -float('inf')
                else:
                    int_beta -= beta*delta_r
                S_discrete.append(np.exp(int_beta))
                
            # --- Formalism Step 4: SCL Integration ---
            # Eq. 16: Integral of S(r)dr
            # Sum of rectangles: Width * Height
            # Width is always delta_r
            # SCL = delta_r * (S[0] + S[1] + S[2] + ...)
            # We exclude the last tail if it goes to infinity, practically truncate when S is negligible
            
            # S_discrete has N+1 elements for N transfer points (intervals 0..N)
            # Sum S_discrete[:-1] because S_discrete[i] is the height of the i-th interval
            scl_pair = delta_r * sum(S_discrete[:-1])
            
            # Weighting for average
            RTE = pair.weight / sum_ca_weights if sum_ca_weights > 0 else 0
            total_weighted_scl += scl_pair * RTE
            total_weight_norm += RTE
            
            print(f"  SCL: {scl_pair:.3f} A (Weight: {RTE:.3f})")
            
            # --- Map S(r) to fine grid for Plotting ---
            S_y_grid = np.zeros_like(self.x_grid)
            curr_s_idx = 0
            for i, x in enumerate(self.x_grid):
                # Determine which interval we are in
                # interval 0: 0 <= x < r1
                # interval 1: r1 <= x < r2
                if curr_s_idx < len(transfer_points):
                    if x >= transfer_points[curr_s_idx]:
                        curr_s_idx += 1
                
                if curr_s_idx < len(S_discrete):
                    S_y_grid[i] = S_discrete[curr_s_idx]
                else:
                    S_y_grid[i] = S_discrete[-1]

            # Calculate PMF and bond strength for this pair
            pmf_values, bond_strength, peak_x = self.calculate_pmf_and_bond_strength(name)
            
            # Calculate reduced mass and fundamental frequency
            reduced_mass = self.calculate_reduced_mass(name)
            fundamental_frequency = self.calculate_fundamental_frequency(name, bond_strength)

            if fundamental_frequency is not None:
                print(f"  ω₀: {fundamental_frequency:.4f} rad/ps (k={bond_strength:.4f} kJ/mol/Å², μ={reduced_mass*1000:.4f} g/mol)")

            # Store Results
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
                'reduced_mass': reduced_mass * 1000 if reduced_mass is not None else None,  # Convert back to g/mol for display
                'fundamental_frequency': fundamental_frequency,
            }
            
            self.plot_data.append({
                'ion_pair_ca_i': name,
                'x_range': self.x_grid,
                'S_i': S_y_grid,
                'x_SCL_pair': scl_pair
            })

        self.avg_SCL = total_weighted_scl / total_weight_norm if total_weight_norm > 0 else 0
        self.plot_data.append({'avg_SCL': self.avg_SCL})
        
        print(f"Average SCL: {self.avg_SCL:.4f} A")
        
        # --- Generate Outputs ---
        self._save_csv_results()
        
    def _save_csv_results(self):
        filename = os.path.join(get_scl_dir(), 'SCL_results.csv')
        file_exists = os.path.isfile(filename)
        
        base_headers = ['Composition', 'Source', 'Temperature (K)', 'Average SCL (A)']
        row = [self.comp, self.source, self.temp, round(self.avg_SCL, 5)]
        
        # Prepare pair data (Limit to top 6 pairs to keep CSV consistent)
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
                f'Pair {i} Reduced Mass (g/mol)', f'Pair {i} Fundamental Frequency (rad/ps)'
            ])
            
            if i <= len(sorted_pairs):
                name, res = sorted_pairs[i-1]
                data.extend([
                    name, round(res['scl'], 5),
                    round(res['peak_x'] or 0, 5), round(res['peak_y'] or 0, 5),
                    round(res['minima_x'] or 0, 5), round(res['minima_y'] or 0, 5),
                    round(res['cc_peak_x'] or 0, 5), round(res['cc_peak_y'] or 0, 5),
                    round(res['pmf_at_peak'] or 0, 5), round(res['bond_strength'] or 0, 5),
                    round(res['reduced_mass'] or 0, 5), round(res['fundamental_frequency'] or 0, 5)
                ])
            else:
                data.extend([''] * 12)
                
        # Write mode handling (don't duplicate if same comp/source exists)
        mode = 'a'
        if file_exists:
            # Check duplicates
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                existing = list(reader)
            # Filter out current comp if exists to overwrite
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

    def save_plot_data(self, folder='SCL_plot_data'):
        folder = os.path.join(get_scl_dir(), folder)
        os.makedirs(folder, exist_ok=True)
        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        filename = os.path.join(folder, f'{self.comp.replace("-", "_")}_{safe_source}_plot_data.csv')
        
        df = pd.DataFrame({'r (A)': self.x_grid})
        
        # Add weighted PDFs
        for name, func in self.weighted_splines.items():
            df[f'g(r)_weighted_{name}'] = func(self.x_grid)
            
        # Add S(r) curves
        for data in self.plot_data:
            if 'ion_pair_ca_i' in data:
                df[f"S_i_{data['ion_pair_ca_i']}"] = data['S_i']
                
        df['Average_SCL'] = self.avg_SCL
        df['lambda_BC'] = self.gamma_bc
        
        df.to_csv(filename, index=False)
        print(f"Plot data saved to {filename}")


    def plot_pdf(self, show_plot=True, save_plot=True, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(get_scl_dir(), 'SCL_plots')
            
        # Create output directory if it doesn't exist
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            
        # Create a safe source string for filenames
        safe_source = ''.join(c if c.isalnum() else '_' for c in self.source.split(',')[0].strip())
        timestamp = datetime.now().strftime("%Y%m%d")
        base_filename = f'PDF_{self.comp}_{safe_source}' #_{timestamp}
        
        # Apply publication-style matplotlib settings to match tc_batch_cli.py
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
        # Legend and small text/annotation sizing (~0.85x of tick labels)
        _tick_size = 14
        _small_text = int(round(0.95 * 12))#_tick_size))
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.fontsize'] = _small_text
        # Ensure mathtext (e.g., $r_{i}$) uses Times New Roman
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

        plt.figure(figsize=(4.75, 4.25))
        ion_pair_colors = {}  # Dictionary to store colors for each ion pair

        # Count the number of cation-anion pairs
        ca_pair_count = sum(1 for pdf_data in self.ion_pairs.values() if pdf_data.type == "ca")

        for ion_pair, pdf_data in self.ion_pairs.items():
            # Always use the interpolated splines for plotting
            if ion_pair in self.weighted_splines:
                # Evaluate the spline at the interpolated x-range
                weighted_y = self.weighted_splines[ion_pair](self.x_grid)
                spline, = plt.plot(self.x_grid, weighted_y, label=f"{ion_pair}")
                spline_color = spline.get_color()
                ion_pair_colors[ion_pair] = spline_color  # Store the color for this ion pair

            # Access peak and minima
            peak = pdf_data.peak
            minima = pdf_data.minima

        # In the plot_pdf method, update the S_i plotting section to:
        for data in self.plot_data:
            if 'ion_pair_ca_i' in data:
                ion_pair = data['ion_pair_ca_i']
                color = ion_pair_colors.get(ion_pair, 'black')
                
                # Get S_i data
                if 'S_i' in data and len(data['S_i']) == len(self.x_grid):
                    S_i = data['S_i']
                elif 'S_i' in data and 'x_range' in data:
                    S_i = np.interp(self.x_grid, data['x_range'], data['S_i'])
                else:
                    continue
                    
                # Find the steps in S_i
                step_indices = np.where(np.diff(S_i) != 0)[0] + 1
                step_indices = np.concatenate(([0], step_indices, [len(S_i)-1]))
                
                # Plot bars for each step
                for i in range(len(step_indices)-1):
                    start_idx = step_indices[i]
                    end_idx = step_indices[i+1] if (i < len(step_indices)-1) else len(self.x_grid)-1
                    x_start = self.x_grid[start_idx]
                    x_end = self.x_grid[end_idx] if end_idx < len(self.x_grid) else self.x_grid[-1]
                    
                    # Calculate the midpoint between steps for bar width
                    if i < len(step_indices)-2:
                        next_x_start = self.x_grid[step_indices[i+1]]
                        bar_width = (next_x_start - x_start)# / 2
                    else:
                        bar_width = (x_end - x_start)# / 2
                        
                    s_value = S_i[start_idx]
                    # New approach - scale between min_alpha and max_alpha
                    min_alpha = 0.1
                    max_alpha = 0.55
                    alpha = min_alpha + (max_alpha - min_alpha) * s_value
                    
                    # Plot the bar (centered on the step)
                    bar_x = (x_start + x_end) / 2
                    plt.bar(bar_x, s_value, width=bar_width, 
                        color=color, alpha=alpha*0.6, edgecolor='none', 
                        align='center', zorder=0)

        for data in self.plot_data:
            # Check if 'ion_pair_ca_i' key exists
            if 'ion_pair_ca_i' in data:
                ion_pair = data['ion_pair_ca_i']
                color = ion_pair_colors.get(ion_pair, 'black')  # Default to black if not found
                
                # Ensure we're using the interpolated x-range for S_i
                if 'S_i' in data and len(data['S_i']) == len(self.x_grid):
                    plt.plot(self.x_grid, data['S_i'], label=f"S(r): {ion_pair}", color=color, linestyle='dotted')
                elif 'S_i' in data and 'x_range' in data:
                    # If S_i was calculated on a different x-range, interpolate it to self.x_grid
                    interp_S_i = np.interp(self.x_grid, data['x_range'], data['S_i'])
                    plt.plot(self.x_grid, interp_S_i, label=f"S(r): {ion_pair}", color=color, linestyle='dotted')
                
                # Only plot the line if there is more than one cation-anion pair
                # if ca_pair_count > 1 and 'x_SCL_pair' in data:
                #     plt.axvline(x=data['x_SCL_pair'], color='gray', linestyle='--', 
                #                label=f"$\\lambda_{{{ion_pair}}}$ = {round(data['x_SCL_pair'], 2)}")

        # Use dark green for the average SCL line
        plt.axvline(x=data['avg_SCL'], color='g', linestyle='-.', label=f"$\\ell_{{\\mathrm{{sc}}}}$ = {round(data['avg_SCL'], 2)}")
        if self.gamma_bc > 0:
            plt.axvline(x=self.gamma_bc, color='k', linestyle='--', label=f"$\\ell_{{\\mathrm{{exp}}}}$ = {round(self.gamma_bc,2)}")

        current_date = datetime.now().strftime("%Y%m%d")
        save_title = f'PDF_{self.comp}_{current_date}.png'

        plt.xlabel('r [Å]')
        plt.ylabel('g(r)')
        # plt.title(f'{format_composition_with_subscripts(self.comp)} ({self.temp}K)')
        # Calculate the x-range to start 0.5 Angstrom before the first nonzero data point
        x_min = min(self.x_grid)
        x_max = max(self.x_grid)

        # Find the first nonzero data point across all ion pairs
        first_nonzero_x = x_max  # Start with maximum as fallback
        for ion_pair, pdf_data in self.ion_pairs.items():
            if ion_pair in self.weighted_splines:
                weighted_y = self.weighted_splines[ion_pair](self.x_grid)
                # Find first index where y > 0.01 (small threshold to avoid numerical noise)
                nonzero_indices = np.where(weighted_y > 0.01)[0]
                if len(nonzero_indices) > 0:
                    first_nonzero_x = min(first_nonzero_x, self.x_grid[nonzero_indices[0]])

        # Set x-axis range to start 0.5 Angstrom before first nonzero data point
        x_start = max(x_min, first_nonzero_x - 0.75)
        x_pad = (x_max-x_min)*0.05
        plt.xlim(x_start, x_max+x_pad)

        ax = plt.gca()
        # Only create top axis for unary salts with a single endmember
        if len(self.fractions) == 1:  # Check if it's a unary salt
            # FIX: Check .type from self.ion_pairs, not from the results dict
            ca_pairs = [(ip, res) for ip, res in self.ion_pair_results.items() 
                        if self.ion_pairs[ip].type == 'ca' and res.get('peak_x')]
            
            max_labels = 8  # limit labels per pair to avoid horizontal collisions
            
            for idx, (ion_pair, result) in enumerate(ca_pairs):
                rep_peak = result.get('peak_x')
                if not rep_peak or rep_peak <= 0:
                    continue
                
                # Calculate points based on peak distance (ideal transfer steps)
                r_points_full = np.arange(0.0, (x_max + x_pad) + 0.5 * rep_peak, rep_peak)
                r_labels_full = ["" if i == 0 else rf"$r_{{{i}}}$" for i in range(len(r_points_full))]
                
                valid = (r_points_full >= x_start) & (r_points_full <= (x_max + x_pad))
                r_points = r_points_full[valid]
                r_labels = [lbl for lbl, v in zip(r_labels_full, valid) if v]
                
                # Thin labels to avoid horizontal overlap
                if len(r_points) > max_labels:
                    step = int(np.ceil(len(r_points) / max_labels))
                    r_points = r_points[::step]
                    r_labels = r_labels[::step]
                
                # Create a dedicated twin axis for this pair
                secax = ax.twiny()
                secax.set_xlim(ax.get_xlim())
                secax.set_xticks(r_points)
                secax.set_xticklabels(r_labels)
                
                # Color tick labels to match the ion pair curve
                color = ion_pair_colors.get(ion_pair, 'black')
                for lbl in secax.get_xticklabels():
                    lbl.set_color(color)
                
                # Styling: no axis label/spine, no tick lines, and offset pad to avoid overlap
                secax.set_xlabel("")
                if 'top' in secax.spines:
                    secax.spines['top'].set_visible(False)
                secax.tick_params(axis='x', which='major', length=5, width=1.25, colors=color, pad=6 + 12 * idx)

        # ax.legend(ncol=2, loc='upper right', bbox_to_anchor=(1.0, 1.0), facecolor='white', framealpha=1)
        legend = ax.legend(
            ncol=2,
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            facecolor='white',
            frameon=True,  # Remove the frame/border
            framealpha=0.75,
            edgecolor='none',  # Remove the border
            borderpad=0.5,  # Reduce padding around the legend
            borderaxespad=0.5,  # Reduce padding between border and axes
            handletextpad=0.5,  # Reduce space between legend line and text
            columnspacing=0.6,  # Reduce space between columns
            handlelength=1.5,   # Adjust the length of the legend lines
            labelspacing=0.3    # Reduce space between legend entries
        )
        
        # Get current y-axis limits
        ymin, ymax = plt.ylim()

        # Round up to nearest 0.1 for the maximum y-limit
        ymax_rounded = np.ceil(ymax * 10) / 10

        # Set the y-axis limits with the rounded max
        plt.ylim(ymin, ymax_rounded)

        # Format y-tick labels to show only one decimal place
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

        #plt.grid(True)
        plt.tight_layout()
        
        # Save the plot if requested
        if save_plot:
            # Save as PNG
            plot_filename = os.path.join(output_dir, f'{base_filename}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_filename}")
            
            # # Also save as PDF
            pdf_filename = os.path.join(output_dir, f'{base_filename}.pdf')
            plt.savefig(pdf_filename, bbox_inches='tight')
            print(f"Plot saved to {pdf_filename}")
        
        if show_plot:
            plt.show()
        plt.close()
   
class PDFAnalyzer:
    def __init__(self, save_plot_data=False, show_plot=False):
        self.salts = []
        self.save_plot_data = save_plot_data
        self.show_plot = show_plot

    def add_molten_salt(self, pdf_file, comp_str, source, temp, gamma_bc, apply_savgol=False, savgol_window_length=None, savgol_polyorder=None):
        self.salts.append(MoltenSaltPDF(
            pdf_file,
            comp_str,
            source,
            temp,
            gamma_bc,
            apply_savgol=apply_savgol,
            savgol_window_length=savgol_window_length,
            savgol_polyorder=savgol_polyorder
        ))

    def analyze_all(self):
        for salt in self.salts:
            salt.analyze_pdf()
            if self.save_plot_data:
                salt.save_plot_data()

    def plot_all(self):
        for salt in self.salts:
            salt.plot_pdf(show_plot=self.show_plot)

# ==========================================
# 3. Execution
# ==========================================

def main():
    analyzer = PDFAnalyzer(save_plot_data=True,show_plot=False)

    # --- Add your salts here (Copied from original script inputs) ---
    # Example:
    # analyzer.add_molten_salt('PDF_LiCl.csv', "1.0LiCl", 'Walz, 2019', 878, 4.10511)

    # The below entries will currently analyze all salts avaialable

    # Unary Salts
    analyzer.add_molten_salt('LiF_Walz_2019_1121.0_PIM.csv',"1.0LiF",'Walz, 2019', 1121, 3.28553, apply_savgol=True)
    analyzer.add_molten_salt('NaF_Walz_2019_1266.0_PIM.csv',"1.0NaF",'Walz, 2019', 1266, 5.22361, apply_savgol=True)
    analyzer.add_molten_salt('KF_Walz_2019_1131.0_PIM.csv',"1.0KF",'Walz, 2019', 1131, 4.63533, apply_savgol=True)
    analyzer.add_molten_salt('LiCl_Walz_2019_878.0_PIM.csv',"1.0LiCl",'Walz, 2019', 878, 4.10511, apply_savgol=True)
    analyzer.add_molten_salt('NaCl_Lu_2021_1200.0_PIM.csv', "1.0NaCl", 'Lu, 2021', 1200, 4.48028, apply_savgol=True)
    analyzer.add_molten_salt('KCl_Walz_2019_1043.0_PIM.csv',"1.0KCl",'Walz, 2019', 1043, 4.47675, apply_savgol=True)
    analyzer.add_molten_salt('MgCl2_Roy_2021_1073.0_AP.csv',"1.0MgCl2",'Roy, 2021', 1073, 4.76796, apply_savgol=True)
    analyzer.add_molten_salt('CaCl2_Bu_2021_1100.0_AP.csv',"1.0CaCl2",'Bu, 2021', 1100, 7.72598, apply_savgol=True)
    analyzer.add_molten_salt('SrCl2_McGreevy_1987_1198.0_Exp.csv',"1.0SrCl2",'McGreevy, 1987', 1198, 0, apply_savgol=True)
    
    # Mixtures
    analyzer.add_molten_salt('0.6LiF-0.4NaF_Grizzi_2024_1473.0_AP.csv',"0.6LiF-0.4NaF",'Grizzi, 2024', 1473, 2.63857, apply_savgol=True)
    analyzer.add_molten_salt('0.5LiF-0.5BeF2_Sun_2024_900.0_AP.csv',"0.5LiF-0.5BeF2",'Sun, 2024', 900, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.66LiF-0.34BeF2_Fayfar_2024_973.0_AP.csv',"0.66LiF-0.34BeF2",'Fayfar, 2024', 973, 1.90187, apply_savgol=True)
    analyzer.add_molten_salt('0.5LiCl-0.5KCl_Jiang_2016_727.0_RIM.csv', "0.5LiCl-0.5KCl", 'Jiang, 2016', 727, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.637LiCl-0.363KCl_Jiang_2016_750.0_RIM.csv',"0.637LiCl-0.363KCl",'Jiang, 2016', 750, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.5NaCl-0.5KCl_Manga_2014_1100.0_RIM.csv', "0.5NaCl-0.5KCl", 'Manga, 2014', 1100, 4.32778, apply_savgol=True)#, savgol_window_length=15, savgol_polyorder=3)
    analyzer.add_molten_salt('0.7LiCl-0.3CaCl2_Liang_2024_1073.0_RIM.csv', "0.7LiCl-0.3CaCl2", 'Liang, 2024', 1073, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.4903NaCl-0.5097CaCl2_Wei_2022_1023.0_RIM.csv', "0.4903NaCl-0.5097CaCl2", 'Wei, 2022', 1023, 3.76913, apply_savgol=True)
    analyzer.add_molten_salt('0.718KCl-0.282CaCl2_Wei_2022_1300.0_RIM.csv', "0.718KCl-0.282CaCl2", 'Wei, 2022', 1300, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.465LiF-0.115NaF-0.42KF_Frandsen_2020_873.0_AP.csv',"0.465LiF-0.115NaF-0.42KF",'Frandsen, 2020', 873, 2.26059, apply_savgol=True)
    analyzer.add_molten_salt('0.345NaF-0.065MgF2-0.59KF_Solano_2021_1073.0_AP.csv',"0.345NaF-0.59KF-0.065MgF2",'Solano, 2021', 1073, 3.92263, apply_savgol=True)
    analyzer.add_molten_salt('0.45MgCl2-0.33NaCl-0.22KCl_Jiang_2024_750.0_PIM.csv',"0.45MgCl2-0.33NaCl-0.22KCl",'Jiang, 2024', 750, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.38MgCl2-0.21NaCl-0.41KCl_Jiang_2024_750.0_PIM.csv',"0.38MgCl2-0.21NaCl-0.41KCl",'Jiang, 2024', 750, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.417NaCl-0.058KCl-0.525CaCl2_Wei_2022_1023.0_RIM.csv', "0.417NaCl-0.525CaCl2-0.058KCl", 'Wei, 2022', 1023, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.535NaCl-0.315MgCl2-0.15CaCl2_Wei_2022_1023.0_RIM.csv', "0.535NaCl-0.315MgCl2-0.15CaCl2", 'Wei, 2022', 1023, 3.52027, apply_savgol=True)

    # Actinides
    analyzer.add_molten_salt('ThF4_Dai_2015_1633.0_PIM.csv',"1.0ThF4",'Dai, 2015', 1633, 0, apply_savgol=True)
    analyzer.add_molten_salt('UF4_Ocádiz-Flores_2021_1357.0_PIM.csv',"1.0UF4",'Ocadiz-Flores, 2021', 1357, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.64NaCl-0.36UCl3_Andersson_2022_1250.0_AP.csv',"0.64NaCl-0.36UCl3",'Andersson, 2022', 1250, 2.5393, apply_savgol=True)
    analyzer.add_molten_salt('0.85KCl-0.15UCl3_Andersson_2024_1250.0_AP.csv',"0.85KCl-0.15UCl3",'Andersson, 2024', 1250, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.75KCl-0.25UCl3_Andersson_2024_1250.0_AP.csv',"0.75KCl-0.25UCl3",'Andersson, 2024', 1250, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.65KCl-0.35UCl3_Andersson_2024_1250.0_AP.csv',"0.65KCl-0.35UCl3",'Andersson, 2024', 1250, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.5KCl-0.5UCl3_Andersson_2024_1250.0_AP.csv',"0.5KCl-0.5UCl3",'Andersson, 2024', 1250, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.5454LiF-0.3636NaF-0.091UF4_Grizzi_2024_1473.0_AP.csv',"0.5454LiF-0.3636NaF-0.091UF4",'Grizzi, 2024', 1473, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.78NaF-0.22UF4_Zhang_2026_900.0_AP.csv',"0.78NaF-0.22UF4",'900K-AIMD-Zhang, 2026', 900, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.78NaF-0.22UF4_Zhang_2026_900.0_PIM.csv',"0.78NaF-0.22UF4",'900K-CMD-Zhang, 2026', 900, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.78NaF-0.22UF4_Zhang_2026_1000.0_PIM.csv',"0.78NaF-0.22UF4",'1000K-CMD-Zhang, 2026', 1000, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.78NaF-0.22UF4_Zhang_2026_1100.0_PIM.csv',"0.78NaF-0.22UF4",'1100K-CMD-Zhang, 2026', 1100, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.78NaF-0.22UF4_Zhang_2026_1200.0_PIM.csv',"0.78NaF-0.22UF4",'1200K-CMD-Zhang, 2026', 1200, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.57NaF-0.16KF-0.27UF4_Zhang_2026_900.0_PIM.csv',"0.57NaF-0.16KF-0.27UF4",'900K-AIMD-Zhang, 2026', 900, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.57NaF-0.16KF-0.27UF4_Zhang_2026_1000.0_PIM.csv',"0.57NaF-0.16KF-0.27UF4",'1000K-AIMD-Zhang, 2026', 1000, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.57NaF-0.16KF-0.27UF4_Zhang_2026_1100.0_PIM.csv',"0.57NaF-0.16KF-0.27UF4",'1100K-AIMD-Zhang, 2026', 1100, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.57NaF-0.16KF-0.27UF4_Zhang_2026_1200.0_PIM.csv',"0.57NaF-0.16KF-0.27UF4",'1200K-AIMD-Zhang, 2026', 1200, 0, apply_savgol=True)
    analyzer.add_molten_salt('0.63NaCl-0.37UCl3_Zhang_2026_1100.0_AP.csv',"0.63NaCl-0.37UCl3",'AIMD-Zhang, 2026', 1100, 0, apply_savgol=True)

    
    # Run
    analyzer.analyze_all()
    analyzer.plot_all()

if __name__ == "__main__":
    main()
