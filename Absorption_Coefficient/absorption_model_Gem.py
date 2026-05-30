"""
Molten Salt Absorption Coefficient Model Using the Pair Distribution Function
==============================================================================
Implements the methodology from:
    "Model for Molten Salt Absorption Coefficient Using the Pair Distribution Function"
    Jacob Numbers
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import (
    e as e_charge, epsilon_0, k as k_B, Avogadro,
    m_u, c as c_light, pi, h, k
)
from scipy import integrate
from scipy.optimize import minimize
from matplotlib.colors import Normalize, LinearSegmentedColormap
import mendeleev
import os
from typing import Dict, List, Tuple, Optional, Any

# Import configurations and registry from the compact config file
from salts_config import SaltConfig, IonPairParams, SALT_REGISTRY

# =============================================================================
# 0. IEEE PLOT FORMATTING
# =============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

c = c_light                     
kB_eV = k_B / e_charge          
hc_eV_um = 1.23984193           
WIEN_B = 2.897771955e-3         


# =============================================================================
# 1. PDF / RDF HELPER FUNCTIONS
# =============================================================================

def find_pdf_file(filename: str) -> str:
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

def load_and_clean_pdf(file_path: str) -> pd.DataFrame:
    full_path = find_pdf_file(file_path)

    with open(full_path, 'r') as f:
        first_line = f.readline().strip()
    pair_names = [name.strip() for name in first_line.split(',')
                  if name.strip() and "Unnamed" not in name]

    df = pd.read_csv(full_path, header=1)

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

    return df


# =============================================================================
# 2. FIRST-PRINCIPLES HELPER FUNCTIONS
# =============================================================================

def get_ion_mass(element_symbol: str) -> float:
    el = mendeleev.element(element_symbol)
    return el.mass * m_u

def get_reduced_mass(ion_pair: Tuple[str, str]) -> float:
    m1 = get_ion_mass(ion_pair[0])
    m2 = get_ion_mass(ion_pair[1])
    return (m1 * m2) / (m1 + m2)

def calculate_k_from_pmf(pdf_df: pd.DataFrame, ion_pair_str: str, T_pdf: float) -> float:
    r_col = f"r_{ion_pair_str}"
    rdf_col = f"rdf_{ion_pair_str}"

    pair_df = pdf_df[[r_col, rdf_col]].dropna().copy()
    pair_df = pair_df[pair_df[rdf_col] > 1e-6] 

    r_m = pair_df[r_col].values * 1e-10 
    rdf = pair_df[rdf_col].values

    V_pmf = -k_B * T_pdf * np.log(rdf)
    dV_dr = np.gradient(V_pmf, r_m)
    d2V_dr2 = np.gradient(dV_dr, r_m)

    peak_loc = pair_df[rdf_col].idxmax()
    peak_pos = pair_df.index.get_loc(peak_loc)
    k_val = d2V_dr2[peak_pos]

    return abs(k_val) if k_val < 0 else k_val

def calculate_oscillator_strengths(ion_pairs: List[IonPairParams], eps_inf_mixture: float) -> None:
    for pair in ion_pairs:
        pair.delta_eps = pair.mole_fraction * (pair.epsilon_s - pair.epsilon_inf)

def density_at_T(rho_coeffs: Tuple[float, float], T: float) -> float:
    return rho_coeffs[0] + rho_coeffs[1] * T


# =============================================================================
# 3. CORE ABSORPTION MODEL
# =============================================================================

def lorentz_dielectric(omega: np.ndarray, omega0: float, gamma: float, delta_eps: float) -> np.ndarray:
    return delta_eps * omega0**2 / ((omega0**2 - omega**2) - 1j * gamma * omega)

def alpha_vib_from_dielectric(omega: np.ndarray, eps: np.ndarray) -> np.ndarray:
    eps_r = np.real(eps)
    eps_i = np.imag(eps)
    modulus = np.sqrt(eps_r**2 + eps_i**2)
    return (omega / c) * np.sqrt(2.0) * np.sqrt(np.maximum(modulus - eps_r, 0.0))

def alpha_urbach(wl_m: np.ndarray, W0_eV: float, kappa0: float, T: float) -> np.ndarray:
    E_eV = hc_eV_um / (wl_m * 1e6) 
    E_U = kB_eV * T                 
    return kappa0 * np.exp((E_eV - W0_eV) / E_U)

def compute_alpha_total(wl_m: np.ndarray, T: float, salt: SaltConfig) -> Tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * pi * c / wl_m
    eps_total = np.full_like(omega, salt.eps_inf_mixture, dtype=complex)

    rho_T = density_at_T(salt.rho_coeffs, T)
    rho_fus = density_at_T(salt.rho_coeffs, salt.T_fus)
    density_ratio = rho_T / rho_fus if rho_fus > 0 else 1.0

    for pair in salt.ion_pairs:
        gamma_T = pair.gamma0 + pair.gamma_slope * T
        delta_eps_T = pair.delta_eps * density_ratio 
        eps_total += lorentz_dielectric(omega, pair.omega0, gamma_T, delta_eps_T)

    kappa_vib = alpha_vib_from_dielectric(omega, eps_total)
    kappa_elec = alpha_urbach(wl_m, salt.W0_eV, salt.kappa0, T)
    
    # Calculate the total absorption
    alpha_total = kappa_vib + kappa_elec
    
    # Extract the wavelength-dependent index of refraction
    n_refractive = refractive_index_from_dielectric(eps_total)

    return alpha_total, n_refractive

def refractive_index_from_dielectric(eps: np.ndarray) -> np.ndarray:
    """
    Extracts the real index of refraction n(ω) from the complex dielectric function.
    """
    eps_r = np.real(eps)
    eps_i = np.imag(eps)
    modulus = np.sqrt(eps_r**2 + eps_i**2)
    
    # n = sqrt( (|eps| + eps_r) / 2 )
    return np.sqrt((modulus + eps_r) / 2.0)


# =============================================================================
# 4. PLANCK FUNCTION AND PLANCK-MEAN ABSORPTION
# =============================================================================

def planck_spectral_radiance(wl_m: np.ndarray, T: float) -> np.ndarray:
    wl_m = np.asarray(wl_m, dtype=float)
    wl_m = np.maximum(wl_m, 1e-20)

    c1 = 2.0 * h * c**2
    c2 = h * c / k
    exponent = np.minimum(c2 / (wl_m * T), 700.0) 
    
    I_b = c1 / (wl_m**5 * (np.exp(exponent) - 1.0))
    return np.nan_to_num(I_b, nan=0.0, posinf=0.0, neginf=0.0)

def planck_mean_absorption(wl_m: np.ndarray, alpha_total: np.ndarray, T: float, range_factor: float = 20.0) -> float:
    peak_wl = WIEN_B / T
    wl_min = peak_wl / range_factor
    wl_max = peak_wl * range_factor

    mask = (wl_m >= wl_min) & (wl_m <= wl_max)
    if not np.any(mask):
        return np.nan

    wl_sub = wl_m[mask]
    alpha_sub = alpha_total[mask]
    B_sub = planck_spectral_radiance(wl_sub, T)

    numerator = integrate.trapezoid(alpha_sub * B_sub, wl_sub)
    denominator = integrate.trapezoid(B_sub, wl_sub)

    return np.nan if denominator == 0 else (numerator / denominator)

def planck_mean_refractive_index(wl_m: np.ndarray, n_total: np.ndarray, T: float, range_factor: float = 20.0) -> float:
    """Planck-mean (effective) refractive index n_eff(T)."""
    peak_wl = WIEN_B / T
    wl_min = peak_wl / range_factor
    wl_max = peak_wl * range_factor

    mask = (wl_m >= wl_min) & (wl_m <= wl_max)
    if not np.any(mask):
        return np.nan

    wl_sub = wl_m[mask]
    n_sub = n_total[mask]
    B_sub = planck_spectral_radiance(wl_sub, T)

    numerator = integrate.trapezoid(n_sub * B_sub, wl_sub)
    denominator = integrate.trapezoid(B_sub, wl_sub)

    return np.nan if denominator == 0 else (numerator / denominator)


# =============================================================================
# 5. INITIALIZATION & FITTING ROUTINES
# =============================================================================

def initialize_salt(salt: SaltConfig) -> None:
    print(f"\n{'='*60}")
    print(f"Initializing: {salt.name}")
    print(f"{'='*60}")

    pdf_df = load_and_clean_pdf(salt.pdf_file)
    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

    print(f"\n  {'Pair':<8} {'k [N/m]':>12} {'ω₀ [rad/s]':>14} {'λ_peak [µm]':>12} {'Δε':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*12} {'-'*8}")

    for pair in salt.ion_pairs:
        pair.k_N_per_m = calculate_k_from_pmf(pdf_df, pair.ion_pair_str, salt.T_pdf)
        mu = get_reduced_mass(pair.ion_pair)
        pair.omega0 = np.sqrt(pair.k_N_per_m / mu)
        lambda_peak_um = 2.0 * pi * c / pair.omega0 * 1e6

        print(f"  {pair.ion_pair_str:<8} {pair.k_N_per_m:>12.2f} "
              f"{pair.omega0:>14.3e} {lambda_peak_um:>12.1f} {pair.delta_eps:>8.3f}")
    print()

def fit_gamma_parameters(salt: SaltConfig) -> None:
    if not salt.experimental_data:
        print("  Skipping fit: No experimental data provided.")
        return
    
    # Determine fitting configuration
    fit_config = salt.experimental_fit_config or {}
    fit_mode = fit_config.get('mode', 'single')  # 'none', 'single', 'weighted_average'
    fit_target = fit_config.get('fit_target', fit_config.get('fit_parameters', 'both'))

    if fit_mode == 'none':
        print(f"  Skipping fit for {salt.name}: fitting disabled in config.")
        return

    if isinstance(fit_target, str):
        fit_target_lower = fit_target.lower()
        if fit_target_lower in ('none',):
            fit_params = []
        elif fit_target_lower in ('both', 'all'):
            fit_params = ['gamma0', 'gamma_slope']
        elif fit_target_lower in ('gamma0', 'gamma'):
            fit_params = ['gamma0']
        elif fit_target_lower == 'gamma_slope':
            fit_params = ['gamma_slope']
        else:
            print(f"  Invalid fit_target '{fit_target}'. Use 'none', 'gamma0', 'gamma_slope', or 'both'.")
            return
    elif isinstance(fit_target, (list, tuple)):
        fit_params = []
        for item in fit_target:
            item_str = str(item).lower()
            if item_str in ('gamma0', 'gamma'):
                fit_params.append('gamma0')
            elif item_str == 'gamma_slope':
                fit_params.append('gamma_slope')
        fit_params = [p for p in ['gamma0', 'gamma_slope'] if p in fit_params]
    else:
        fit_params = ['gamma0', 'gamma_slope']

    if not fit_params:
        print(f"  Skipping fit for {salt.name}: no fit parameters selected.")
        return

    fit_datasets = []
    if fit_mode == 'single':
        dataset_index = fit_config.get('dataset_index', 0)
        if dataset_index >= len(salt.experimental_data):
            print(f"  Warning: Dataset index {dataset_index} out of range. Using index 0.")
            dataset_index = 0
        
        exp_data = salt.experimental_data[dataset_index]
        if not exp_data.get('wavelength_um'):
            print(f"  Skipping fit: Selected dataset {dataset_index} has no data.")
            return
        
        exp_wl_m = np.array(exp_data['wavelength_um']) * 1e-6
        exp_alpha = np.array(exp_data['absorption_m1'])
        T_exp = exp_data.get('T_exp', salt.T_pdf)
        fit_datasets.append({
            'wavelength_um': exp_wl_m,
            'absorption_m1': exp_alpha,
            'T_exp': T_exp,
            'weight': 1.0,
            'label': f"Dataset {dataset_index}",
        })
        fit_label = f"Dataset {dataset_index}"
        T_summary = f"{T_exp:.0f}K"
        
    elif fit_mode == 'weighted_average':
        weights = fit_config.get('weights', None)
        
        valid_datasets = [exp for exp in salt.experimental_data if exp.get('wavelength_um')]
        if not valid_datasets:
            print("  Skipping fit: No datasets with data available.")
            return
        
        if weights is None or len(weights) != len(valid_datasets):
            if weights is not None and len(weights) != len(valid_datasets):
                print(f"  Warning: {len(weights)} weights provided but {len(valid_datasets)} datasets available. Using equal weighting.")
            weights = [1.0] * len(valid_datasets)

        weights = np.array(weights, dtype=float)
        if np.sum(weights) == 0:
            weights = np.ones(len(valid_datasets), dtype=float)
        weights = weights / np.sum(weights)

        for exp, w in zip(valid_datasets, weights):
            exp_wl_m = np.array(exp['wavelength_um']) * 1e-6
            exp_alpha = np.array(exp['absorption_m1'])
            T_exp = exp.get('T_exp', salt.T_pdf)
            fit_datasets.append({
                'wavelength_um': exp_wl_m,
                'absorption_m1': exp_alpha,
                'T_exp': T_exp,
                'weight': w,
                'label': exp.get('label', 'Experimental'),
            })
        fit_label = "Weighted average of all datasets"
        T_summary = ", ".join([f"{d['T_exp']:.0f}K" for d in fit_datasets])
    else:
        print(f"  Invalid fit mode '{fit_mode}'. Use 'none', 'single', or 'weighted_average'.")
        return
    
    fit_param_names = {
        'gamma0': 'γ₀',
        'gamma_slope': "γ'"
    }
    fit_target_label = ' + '.join([fit_param_names[p] for p in fit_params])
    print(f"\n--- Fitting {fit_target_label} for {salt.name} at {T_summary} ({fit_label}) ---")

    x0 = []
    original_params = []
    for pair in salt.ion_pairs:
        original_params.append((pair.gamma0, pair.gamma_slope))
        if 'gamma0' in fit_params:
            x0.append(pair.gamma0)
        if 'gamma_slope' in fit_params:
            x0.append(pair.gamma_slope)

    def objective(x):
        x_arr = np.array(x)
        if np.any(x_arr < 0):
            return 1e12 + np.sum(np.abs(x_arr[x_arr < 0])) * 1e6

        idx = 0
        for pair, (orig_gamma0, orig_gamma_slope) in zip(salt.ion_pairs, original_params):
            if 'gamma0' in fit_params:
                pair.gamma0 = x_arr[idx]
                idx += 1
            else:
                pair.gamma0 = orig_gamma0
            if 'gamma_slope' in fit_params:
                pair.gamma_slope = x_arr[idx]
                idx += 1
            else:
                pair.gamma_slope = orig_gamma_slope

        total_error = 0.0
        for dataset in fit_datasets:
            alpha_model, _ = compute_alpha_total(dataset['wavelength_um'], dataset['T_exp'], salt)
            error = np.sum((np.log10(alpha_model) - np.log10(dataset['absorption_m1']))**2)
            total_error += dataset['weight'] * error

        return total_error

    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})

    idx = 0
    for pair in salt.ion_pairs:
        if 'gamma0' in fit_params:
            pair.gamma0 = res.x[idx]
            idx += 1
        if 'gamma_slope' in fit_params:
            pair.gamma_slope = res.x[idx]
            idx += 1
        print(f"  {pair.ion_pair_str}: γ₀ = {pair.gamma0:.3e}, γ' = {pair.gamma_slope:.3e}")

    print(f"Fit Result: {'Success' if res.success else 'Failed'} (Log-MSE: {res.fun:.4f})\n")


# =============================================================================
# 6. PLOTTING FUNCTIONS
# =============================================================================

def _get_temp_colormap(T_list):
    colors = ["navy", "darkorchid", "crimson", "firebrick"]
    cmap = LinearSegmentedColormap.from_list("cool_no_yellow", colors)
    norm = Normalize(vmin=min(T_list), vmax=max(T_list))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return cmap, norm, sm


def plot_absorption_spectrum(salt: SaltConfig, wl_m: np.ndarray, T_list: np.ndarray, alpha_results: Dict[float, np.ndarray], ax: Optional[plt.Axes] = None, show_planck: bool = True, planck_T: Optional[float] = None, show_all_planck: bool = True) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    else:
        fig = ax.figure

    wl_um = wl_m * 1e6
    cmap, norm, sm = _get_temp_colormap(T_list)

    # Plot absorption coefficient curves
    for T in T_list:
        ax.semilogy(wl_um, alpha_results[T], color=cmap(norm(T)), linewidth=1.0)
        if T == max(T_list):
            ax.plot(wl_um, alpha_results[T], color=cmap(norm(T)), linewidth=1.0, label = r'$\kappa_{\gamma}(T)$')

    # Colorbar horizontally below the plot
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', aspect=40, pad=0.05)
    cbar.set_label('Temperature (K)')

    # Top-left Title placement inside plot frame
    ax.text(0.03, 0.96, salt.name, transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), zorder=10)

    if salt.ir_peaks_um:
        for wl_peak, label in salt.ir_peaks_um:
            ax.axvline(x=wl_peak, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, zorder=1)
            ax.text(wl_peak + 0.5, 0.85, label, transform=ax.get_xaxis_transform(),
                    rotation=90, fontsize=8, color='gray', va='top', zorder=2)

    if show_planck:
        ax2 = ax.twinx()
        max_I_bb = 0
        
        if show_all_planck:
            # Add a single dummy line for the legend so it doesn't populate 50 entries
            ax2.plot([], [], color='gray', linestyle='--', linewidth=1.2, label='$I_{bb,\\lambda}(T)$')
            
            # Plot the I_bb curves for the minimum and maximum temperatures
            for T in np.linspace(min(T_list), max(T_list), 5):
                I_bb = planck_spectral_radiance(wl_m, T)
                max_I_bb = max(max_I_bb, np.max(I_bb))
                
                # Plotted with the same color mapping, using a dashed line
                ax2.plot(wl_um, I_bb, linestyle=':', color=cmap(norm(T)), 
                         linewidth=1.0, alpha=0.7)
        else:
            # Plot only a single reference temperature curve
            T_planck = planck_T or salt.T_pdf
            I_bb = planck_spectral_radiance(wl_m, T_planck)
            max_I_bb = np.max(I_bb)
            ax2.plot(wl_um, I_bb, 'k--', linewidth=1.0, alpha=0.6,
                     label=f'$I_{{bb}}$({T_planck:.0f}K)')
        
        if salt.experimental_data:
            # Define colors and markers for multiple datasets
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            markers = ['o', 's', '^', 'v', 'D', '*', 'P', 'X']
            
            for i, exp in enumerate(salt.experimental_data):
                if exp.get('wavelength_um'):
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]
                    ax.scatter(exp['wavelength_um'], exp['absorption_m1'],
                        color=color, marker=marker, s=50, zorder=5,
                        edgecolors='black', linewidth=0.5, alpha=0.6,
                        label=exp.get('label', f'Experimental {i+1}'))
        
        ax2.set_yscale('log')
        # Match the exact limits of the primary axis (1e-1, 1e11) so the tick lines are consistent
        ax2.set_ylim(10**-1, (max_I_bb / 0.001))
        
        ax2.set_ylabel('$I_{bb,\\lambda}$ (W·m⁻³·sr⁻¹)', fontsize=9, alpha=1)
        
        # Mirror IEEE tick styles to secondary axis
        ax2.tick_params(direction='in', which='both')
        ax2.minorticks_on()
        
        # Merge legends to completely prevent overlapping text
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        
        lines = lines_1 + lines_2
        labels = labels_1 + labels_2
        
        # Reorder to put experimental labels last
        if salt.experimental_data:
            exp_labels = [exp.get('label', f'Experimental {i+1}') for i, exp in enumerate(salt.experimental_data) if exp.get('wavelength_um')]
            for exp_label in reversed(exp_labels):
                if exp_label in labels:
                    idx = labels.index(exp_label)
                    line = lines.pop(idx)
                    label = labels.pop(idx)
                    lines.append(line)
                    labels.append(label)
                
        ax2.legend(lines, labels, loc='upper right', 
                   fontsize=8, framealpha=0.9, edgecolor='black')
    else:

        if salt.experimental_data:
            # Define colors and markers for multiple datasets
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            markers = ['o', 's', '^', 'v', 'D', '*', 'P', 'X']
            
            for i, exp in enumerate(salt.experimental_data):
                if exp.get('wavelength_um'):
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]
                    ax.scatter(exp['wavelength_um'], exp['absorption_m1'],
                        color=color, marker=marker, s=50, zorder=5,
                        edgecolors='black', linewidth=0.5, alpha=0.6,
                        label=exp.get('label', f'Experimental {i+1}'))
        
        lines, labels = ax.get_legend_handles_labels()
        # Reorder to put experimental labels last
        if salt.experimental_data:
            exp_labels = [exp.get('label', f'Experimental {i+1}') for i, exp in enumerate(salt.experimental_data) if exp.get('wavelength_um')]
            for exp_label in reversed(exp_labels):
                if exp_label in labels:
                    idx = labels.index(exp_label)
                    line = lines.pop(idx)
                    label = labels.pop(idx)
                    lines.append(line)
                    labels.append(label)
                
        ax.legend(lines, labels, loc='upper right', fontsize=8, framealpha=0.9, edgecolor='black')

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Absorption Coefficient, $\\kappa_\\lambda$ (m⁻¹)')
    ax.set_ylim(1e-1, 1e11)
    ax.set_xlim(0, wl_um[-1] + 2)

    return fig

def plot_refractive_index(salt: SaltConfig, wl_m: np.ndarray, T_list: np.ndarray, n_results: Dict[float, np.ndarray], ax: Optional[plt.Axes] = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    else:
        fig = ax.figure

    wl_um = wl_m * 1e6
    cmap, norm, sm = _get_temp_colormap(T_list)

    # Plot refractive index curves
    for T in T_list:
        ax.plot(wl_um, n_results[T], color=cmap(norm(T)), linewidth=1.0)

    # Colorbar horizontally below the plot
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', aspect=40, pad=0.05)
    cbar.set_label('Temperature (K)')

    # Top-left Title placement inside plot frame
    ax.text(0.03, 0.96, salt.name, transform=ax.transAxes, 
            fontsize=14, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2), zorder=10)

    if salt.experimental_n:
        # Allow a single dict or list of dicts for experimental refractive-index datasets
        exp_n_data = salt.experimental_n
        if isinstance(exp_n_data, dict):
            exp_n_data = [exp_n_data]

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        markers = ['o', 's', '^', 'v', 'D', '*', 'P', 'X']
        for i, exp in enumerate(exp_n_data):
            if exp is None:
                continue
            wl_exp = exp.get('wavelength_um')
            n_exp = exp.get('n_val') or exp.get('n')
            if wl_exp and n_exp:
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                ax.scatter(wl_exp, n_exp,
                    color=color, marker=marker, s=50, zorder=5,
                    edgecolors='black', linewidth=0.5, alpha=0.6,
                    label=exp.get('label', f'Experimental n {i+1}'))

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Refractive Index, $n(\lambda)$')
    ax.set_xlim(0, wl_um[-1] + 2)
    
    # Calculate a sensible y-limit based on the high-frequency limit
    # ax.set_ylim(max(1.0, np.sqrt(salt.eps_inf_mixture) - 0.5), np.sqrt(salt.eps_inf_mixture) + 3.0)

    lines, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(lines, labels, loc='upper right', fontsize=8, framealpha=0.9, edgecolor='black')

    return fig

def plot_planck_mean_vs_temperature(salt: SaltConfig, T_array: np.ndarray, kappa_P_array: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    else:
        fig = ax.figure

    cmap, norm, _ = _get_temp_colormap(T_array)

    ax.scatter(T_array, kappa_P_array, c=T_array, cmap=cmap, norm=norm,
               edgecolors='k', s=60, alpha=0.8, zorder=3,
               label='Model $\\kappa_P(T)$')

    if salt.piecewise_fit and salt.piecewise_fit.get('enabled', False):
        T_fine = np.linspace(T_array.min(), T_array.max(), 500)
        for idx, region in enumerate(salt.piecewise_fit['regions']):
            T_lo, T_hi = region['range']
            mask = (T_fine >= T_lo) & (T_fine <= T_hi) if idx == 0 else (T_fine > T_lo) & (T_fine <= T_hi)
            T_seg = T_fine[mask]
            if T_seg.size == 0: continue
            kappa_seg = np.polyval(region['coeffs'], T_seg)
            label = 'Piecewise Fit' if idx == 0 else None
            ax.plot(T_seg, kappa_seg, 'r-', linewidth=2, label=label)

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Planck-mean $\\kappa_P$ (m⁻¹)')
    ax.set_title(f'{salt.name}', fontweight='bold')
    ax.legend(loc='best', fontsize=9)

    return fig


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

def export_properties_csv(salt: SaltConfig, T_array: np.ndarray, kappa_P_array: np.ndarray, n_eff_array: np.ndarray, wl: Optional[np.ndarray] = None, alpha_results: Optional[Dict[float, np.ndarray]] = None, n_results: Optional[Dict[float, np.ndarray]] = None, filename: str = "Salt_RHT_Properties.csv"):
    """Exports temperature-dependent and wavelength-dependent RHT properties to CSV database."""
    comp_str = "; ".join([f"{p.ion_pair_str} ({p.mole_fraction})" for p in salt.ion_pairs])
    
    # Create comma-separated lists in single strings for temperature-dependent data
    t_str = ",".join([f"{t:.1f}" for t in T_array])
    k_str = ",".join([f"{k:.4e}" for k in kappa_P_array])
    n_str = ",".join([f"{n:.5f}" for n in n_eff_array])
    
    # Calculate representative scalar values (using the mean across the temperature range)
    scalar_k = np.mean(kappa_P_array)
    scalar_n = np.mean(n_eff_array)
    
    new_row = {
        'Salt': salt.name,
        'Composition': comp_str,
        'Scalar_Absorption_m1': scalar_k,
        'Scalar_Refractive_Index': scalar_n,
        'Temperature_List_K': t_str,
        'Absorption_List_m1': k_str,
        'Refractive_Index_List': n_str
    }
    
    # Add wavelength-dependent data if provided
    if wl is not None and alpha_results is not None and n_results is not None:
        # Store wavelength list
        wl_str = ",".join([f"{w*1e6:.6e}" for w in wl])
        new_row['Wavelength_List_um'] = wl_str
        
        # Store wavelength-dependent absorption and refractive index for each temperature
        for T in sorted(T_array):
            if T in alpha_results and T in n_results:
                alpha_spec = ",".join([f"{a:.4e}" for a in alpha_results[T]])
                n_spec = ",".join([f"{n:.6f}" for n in n_results[T]])
                new_row[f'Absorption_Spectrum_T{T:.0f}K'] = alpha_spec
                new_row[f'Refractive_Index_Spectrum_T{T:.0f}K'] = n_spec
    
    # Logic to update/replace row if salt exists, or append if new
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = df[df['Salt'] != salt.name] # Drop existing row for this salt
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
        
    df.to_csv(filename, index=False)
    print(f"  Data successfully exported/updated in {filename}")

def run_analysis(salt_name: str, wl_range: Tuple[float, float] = (0.15e-6, 60e-6), n_wavelengths: int = 800, T_range: Tuple[float, float] = (450, 1500), n_temperatures: int = 50, run_fit: bool = False, experimental_fit_config: Optional[Dict[str, Any]] = None):
    
    if salt_name not in SALT_REGISTRY:
        available = ', '.join(SALT_REGISTRY.keys())
        raise ValueError(f"Unknown salt '{salt_name}'. Available: {available}")

    # Finds the folder where this Python script is saved
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Creates the folder exactly relative to the script
    output_dir = os.path.join(script_dir, 'Figures')
    os.makedirs(output_dir, exist_ok=True)
    
    salt = SALT_REGISTRY[salt_name]
    wl = np.linspace(wl_range[0], wl_range[1], n_wavelengths)
    T_list = np.linspace(T_range[0], T_range[1], n_temperatures)

    initialize_salt(salt)

    if run_fit:
        if experimental_fit_config is not None:
            salt.experimental_fit_config = experimental_fit_config
        fit_gamma_parameters(salt)

    print("Computing absorption spectra and refractive index...")
    alpha_results = {}
    n_results = {}
    for T in T_list:
        alpha_total, n_refractive = compute_alpha_total(wl, T, salt)
        alpha_results[T] = alpha_total
        n_results[T] = n_refractive
        
    print(f"  Done. {len(T_list)} temperatures computed.")

    print("Computing Planck-mean properties for RHT...")
    T_kappa_list, kappa_P_list, n_eff_list = [], [], []
    for T in T_list:
        kp = planck_mean_absorption(wl, alpha_results[T], T)
        n_eff = planck_mean_refractive_index(wl, n_results[T], T)
        
        if not np.isnan(kp) and not np.isnan(n_eff):
            T_kappa_list.append(T)
            kappa_P_list.append(kp)
            n_eff_list.append(n_eff)

    T_kappa = np.array(T_kappa_list)
    kappa_P = np.array(kappa_P_list)
    n_eff_array = np.array(n_eff_list)

    print(f"\n{'--- Planck-Mean Properties ---':^50}")
    print(f"{'Temp (K)':>12} | {'κ_P (m⁻¹)':>14} | {'n_eff':>14}")
    print(f"{'-'*12}-+-{'-'*14}-+-{'-'*14}")
    for T_val, kp_val, ne_val in zip(T_kappa, kappa_P, n_eff_array):
        print(f"{T_val:>12.1f} | {kp_val:>14.2f} | {ne_val:>14.5f}")

    print("\nUpdating Database...")
    # Calls the new CSV function (saves in the Absorption_Coefficient folder)
    csv_path = os.path.join(script_dir, 'Salt_RHT_Properties.csv')
    export_properties_csv(salt, T_kappa, kappa_P, n_eff_array, wl=wl, alpha_results=alpha_results, n_results=n_results, filename=csv_path)

    print("\nGenerating plots...")
    
    fig1 = plot_absorption_spectrum(salt, wl, T_list, alpha_results, show_all_planck=True)
    fig1.savefig(os.path.join(output_dir, f"absorption_spectrum_{salt_name}.pdf"), bbox_inches='tight')

    # NEW: Refractive Index Plot
    fig2 = plot_refractive_index(salt, wl, T_list, n_results)
    fig2.savefig(os.path.join(output_dir, f"refractive_index_{salt_name}.pdf"), bbox_inches='tight')

    if len(T_kappa) > 1:
        fig3 = plot_planck_mean_vs_temperature(salt, T_kappa, kappa_P)
        fig3.savefig(os.path.join(output_dir, f"planck_mean_{salt_name}.pdf"), bbox_inches='tight')

    plt.show()

    return alpha_results, T_kappa, kappa_P


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # You can easily change this to test the newly added configs:
    # Options: 'LiF', 'NaF', 'KF', 'NaCl', 'FLiNaK', 'FLiBe'
    SALT_NAME = 'LiF' 

    results = run_analysis(
        salt_name=SALT_NAME,
        wl_range=(0.15e-6, 80e-6),
        n_wavelengths=800,
        T_range=(450, 1500),
        n_temperatures=50,
        run_fit=False, # Set to True so the Nelder-Mead optimizer solves for gamma
        experimental_fit_config={
            'mode': 'weighted_average', # 'none', 'single', or 'weighted_average'
            'fit_target': 'both',      # 'none', 'gamma0'/'gamma', 'gamma_slope', or 'both'
            'weights': [0.5, 0.5, 0], # Only used if mode='weighted_average', assumed equal weighting if empty
            #'dataset_index': 0, # Only used if mode='single'
        }
    )