"""
Molten Salt Absorption Coefficient Model Using the Pair Distribution Function
==============================================================================
Implements the methodology from:
    "Model for Molten Salt Absorption Coefficient Using the Pair Distribution Function"
    Jacob Numbers

Key equations implemented:
    - Eq. 3:  Electronic (Urbach) absorption coefficient
    - Eq. 5:  Vibrational absorption from Lorentz oscillator model
    - Eq. 7:  Multi-oscillator dielectric function (sum over cation-anion pairs)
    - Eq. 8:  Temperature-dependent oscillator strength via density scaling
    - Eq. 9:  Oscillator strength weighting by per-component dielectric constants
    - Eq. 10: Fundamental frequency from spring constant and reduced mass
    - Eq. 11: Spring constant from second derivative of PMF at equilibrium
    - Eq. 12: Planck-mean absorption coefficient
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import (
    e as e_charge, epsilon_0, k as k_B, Avogadro,
    m_u, c as c_light, pi, h, k
)
from scipy import integrate
from matplotlib.colors import Normalize, LinearSegmentedColormap
import mendeleev
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

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

# =============================================================================
# 1. PHYSICAL CONSTANTS
# =============================================================================
c = c_light                     # Speed of light [m/s]
kB_eV = k_B / e_charge          # Boltzmann constant [eV/K]
hc_eV_um = 1.23984193           # h*c [eV·µm]
WIEN_B = 2.897771955e-3         # Wien's displacement constant [m·K]


# =============================================================================
# 2. DATA CLASSES FOR SALT DEFINITIONS
# =============================================================================

@dataclass
class IonPairParams:
    """Parameters for a single cation-anion oscillator pair."""
    ion_pair: Tuple[str, str]
    ion_pair_str: str
    epsilon_s: float
    epsilon_inf: float
    gamma0: float
    gamma_slope: float
    mole_fraction: float

    # Computed at runtime (not user-supplied)
    omega0: float = 0.0
    k_N_per_m: float = 0.0
    delta_eps: float = 0.0


@dataclass
class SaltConfig:
    """Complete configuration for a molten salt system."""
    name: str
    ion_pairs: List[IonPairParams]
    eps_inf_mixture: float
    W0_eV: float
    kappa0: float
    rho_coeffs: Tuple[float, float]     # (intercept, slope) for ρ(T) = a + b*T
    T_fus: float
    pdf_file: str
    T_pdf: float
    experimental_data: Optional[Dict[str, Any]] = None
    piecewise_fit: Optional[Dict[str, Any]] = None
    ir_peaks_um: Optional[List[Tuple[float, str]]] = None


# =============================================================================
# 3. SALT REGISTRY
# =============================================================================

SALT_REGISTRY: Dict[str, SaltConfig] = {}

def register_salt(config: SaltConfig):
    """Add a salt configuration to the global registry."""
    SALT_REGISTRY[config.name] = config

# --- LiF (100 mol%) ---
register_salt(SaltConfig(
    name='LiF',
    ion_pairs=[
        IonPairParams(
            ion_pair=('Li', 'F'),
            ion_pair_str='Li-F',
            epsilon_s=9.04,
            epsilon_inf=1.93,
            gamma0=3.6e-6,
            gamma_slope=2.4e7,
            mole_fraction=1.0,
        ),
    ],
    eps_inf_mixture=1.93,
    W0_eV=10.58,
    kappa0=1e12,
    rho_coeffs=(2370.0, -0.50),
    T_fus=1121.0,
    pdf_file='LiF_PDF.csv',
    T_pdf=1121.0,
    experimental_data={
        'wavelength_um': [],
        'absorption_m1': [],
        'label': '1160K (Barker, 1972)',
    },
    ir_peaks_um=[(16.0, 'LiF-IR'), (20.0, 'LiF-IR₂')],
))

# --- NaF (100 mol%) ---
register_salt(SaltConfig(
    name='NaF',
    ion_pairs=[
        IonPairParams(
            ion_pair=('Na', 'F'),
            ion_pair_str='Na-F',
            epsilon_s=5.07,
            epsilon_inf=1.74,
            gamma0=1.49e-6,
            gamma_slope=1.52e6,
            mole_fraction=1.0,
        ),
    ],
    eps_inf_mixture=1.74,
    W0_eV=8.82,
    kappa0=1e12,
    rho_coeffs=(2700.0, -0.59),
    T_fus=1269.0,
    pdf_file='NaF_PDF.csv',
    T_pdf=1269.0,
))

# --- KF (100 mol%) ---
register_salt(SaltConfig(
    name='KF',
    ion_pairs=[
        IonPairParams(
            ion_pair=('K', 'F'),
            ion_pair_str='K-F',
            epsilon_s=5.5,
            epsilon_inf=1.85,
            gamma0=3.08e-7,
            gamma_slope=5.28e7,
            mole_fraction=1.0,
        ),
    ],
    eps_inf_mixture=1.85,
    W0_eV=8.27,
    kappa0=1e12,
    rho_coeffs=(2640.0, -0.66),
    T_fus=1131.0,
    pdf_file='KF_PDF.csv',
    T_pdf=1131.0,
))

# --- NaCl (100 mol%) ---
register_salt(SaltConfig(
    name='NaCl',
    ion_pairs=[
        IonPairParams(
            ion_pair=('Na', 'Cl'),
            ion_pair_str='Na-Cl',
            epsilon_s=5.90,
            epsilon_inf=2.33,
            gamma0=5.73e-6,
            gamma_slope=2.22e7,
            mole_fraction=1.0,
        ),
    ],
    eps_inf_mixture=2.33,
    W0_eV=6.33,
    kappa0=1e12,
    rho_coeffs=(2140.0, -0.54),
    T_fus=1074.0,
    pdf_file='NaCl_Lu.csv',
    T_pdf=1074.0,
    experimental_data={
        'wavelength_um': [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0],
        'absorption_m1': [15, 20, 28, 40, 55, 80, 110, 160, 230, 340, 500],
        'label': '1105K (Barker, 1972)',
    },
    ir_peaks_um=[(26.0, 'NaCl-IR')],
))

# --- FLiNaK (46.5-11.5-42 mol%) ---
register_salt(SaltConfig(
    name='FLiNaK',
    ion_pairs=[
        IonPairParams(
            ion_pair=('Li', 'F'),
            ion_pair_str='Li-F',
            epsilon_s=9.04,
            epsilon_inf=1.93,
            gamma0=2.16e-6,
            gamma_slope=1.25e7,
            mole_fraction=0.465,
        ),
        IonPairParams(
            ion_pair=('Na', 'F'),
            ion_pair_str='Na-F',
            epsilon_s=5.07,
            epsilon_inf=1.74,
            gamma0=1.49e-6,
            gamma_slope=1.52e6,
            mole_fraction=0.115,
        ),
        IonPairParams(
            ion_pair=('K', 'F'),
            ion_pair_str='K-F',
            epsilon_s=5.50,
            epsilon_inf=1.85,
            gamma0=3.08e-7,
            gamma_slope=5.28e7,
            mole_fraction=0.42,
        ),
    ],
    eps_inf_mixture=1.24**2,
    W0_eV=9.64,
    kappa0=1e12,
    rho_coeffs=(2680.0, -0.685),
    T_fus=727.0,
    pdf_file='0.465LiF-0.115NaF-0.42KF_Frandsen_2020_873.0_AP.csv',
    T_pdf=873.0,
    experimental_data={
        'wavelength_um': [
            4.757, 4.900, 5.058, 5.193, 5.296, 5.447, 5.645, 5.898,
            6.160, 6.477, 6.738, 7.031, 7.372, 7.713, 7.982, 8.243,
            8.465, 8.632, 8.735, 8.877,
        ],
        'absorption_m1': [
            1.756, 2.465, 3.460, 4.775, 5.852, 7.807, 11.241, 17.030,
            25.801, 40.437, 57.732, 81.038, 120.708, 181.329, 250.253,
            325.476, 412.678, 497.293, 564.730, 663.433,
        ],
        'label': 'Endmember Interpolation (Chaleff)',
    },
    ir_peaks_um=[(16.0, 'LiF-IR'), (24.0, 'NaF-IR'), (40.0, 'KF-IR')],
    piecewise_fit={
        'enabled': True,
        'regions': [
            {
                'range': (450, 685.7),
                'coeffs': [-2.7208e-06, 7.3454e-03, -7.9090, 4.2445e+03,
                           -1.1345e+06, 1.21076e+08],
            },
            {
                'range': (685.7, 964.3),
                'coeffs': [-1.2662e-02, 3.5512e+01, -3.3751e+04, 1.0897e+07],
            },
            {
                'range': (964.3, 1500),
                'coeffs': [8.0016e-02, -2.2069e+02, 1.5646e+05],
            },
        ],
    },
))

# --- FLiBe (66-34 mol%) ---
register_salt(SaltConfig(
    name='FLiBe',
    ion_pairs=[
        IonPairParams(
            ion_pair=('Li', 'F'),
            ion_pair_str='Li-F',
            epsilon_s=9.04,
            epsilon_inf=1.93,
            gamma0=5.69e-1,
            gamma_slope=3.64e4,
            mole_fraction=0.66,
        ),
        IonPairParams(
            ion_pair=('Be', 'F'),
            ion_pair_str='Be-F',
            epsilon_s=5.07,
            epsilon_inf=1.74,
            gamma0=3.15e-5,
            gamma_slope=1.77e8,
            mole_fraction=0.34,
        ),
    ],
    eps_inf_mixture=1.24**2,
    W0_eV=7.5,
    kappa0=1e12,
    rho_coeffs=(2410.0, -0.488),
    T_fus=732.0,
    pdf_file='PDF_FLiBe_Fayfar.csv',
    T_pdf=973.0,
    experimental_data={
        'wavelength_um': [
            3.284, 3.379, 3.490, 3.569, 3.703, 3.775, 3.909,
            4.044, 4.123, 4.218, 4.234,
        ],
        'absorption_m1': [
            21.230, 26.918, 36.525, 48.728, 67.249, 92.811, 125.935,
            154.352, 192.416, 243.969, 284.190,
        ],
        'label': 'Exp. 873K (Liu)',
    },
    ir_peaks_um=[(6.0, 'BeF₂-IR'), (16.0, 'LiF-IR')],
))


# =============================================================================
# 4. PDF / RDF HELPER FUNCTIONS
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

    raise FileNotFoundError(
        f"Could not find PDF file '{filename}'. Searched:\n"
        + "\n".join(f"  - {p}" for p in search_paths)
    )

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

def get_r0_from_pdf(pdf_df: pd.DataFrame, ion_pair_str: str) -> float:
    r_col = f"r_{ion_pair_str}"
    rdf_col = f"rdf_{ion_pair_str}"

    if r_col not in pdf_df.columns or rdf_col not in pdf_df.columns:
        raise ValueError(f"Columns for pair '{ion_pair_str}' not found.")

    pair_df = pdf_df[[r_col, rdf_col]].dropna()
    if pair_df.empty:
        raise ValueError(f"No valid data for pair '{ion_pair_str}'.")

    max_idx = pair_df[rdf_col].idxmax()
    r0_angstroms = pair_df.loc[max_idx, r_col]
    return r0_angstroms * 1e-10  # Convert Å → m


# =============================================================================
# 5. FIRST-PRINCIPLES HELPER FUNCTIONS
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

    if k_val < 0:
        k_val = abs(k_val)

    return k_val

def calculate_oscillator_strengths(ion_pairs: List[IonPairParams], eps_inf_mixture: float) -> None:
    for pair in ion_pairs:
        pair.delta_eps = pair.mole_fraction * (pair.epsilon_s - pair.epsilon_inf)

def density_at_T(rho_coeffs: Tuple[float, float], T: float) -> float:
    return rho_coeffs[0] + rho_coeffs[1] * T


# =============================================================================
# 6. CORE ABSORPTION MODEL
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

def compute_alpha_total(wl_m: np.ndarray, T: float, salt: SaltConfig) -> np.ndarray:
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

    return kappa_vib + kappa_elec


# =============================================================================
# 7. PLANCK FUNCTION AND PLANCK-MEAN ABSORPTION
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
    # Increased range_factor to 20.0 to capture the long-wavelength tail of Planck's curve
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

    if denominator == 0:
        return np.nan

    return numerator / denominator


# =============================================================================
# 8. PRE-CALCULATION
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


# =============================================================================
# 9. PLOTTING FUNCTIONS
# =============================================================================

def _get_temp_colormap(T_list):
    """Custom colormap (Navy -> Purple -> Crimson -> Firebrick) with NO yellow"""
    colors = ["navy", "darkorchid", "crimson", "firebrick"]
    cmap = LinearSegmentedColormap.from_list("cool_no_yellow", colors)
    norm = Normalize(vmin=min(T_list), vmax=max(T_list))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return cmap, norm, sm


def plot_absorption_spectrum(salt: SaltConfig, wl_m: np.ndarray, T_list: np.ndarray, alpha_results: Dict[float, np.ndarray], ax: Optional[plt.Axes] = None, show_planck: bool = True, planck_T: Optional[float] = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    else:
        fig = ax.figure

    wl_um = wl_m * 1e6
    cmap, norm, sm = _get_temp_colormap(T_list)

    for T in T_list:
        ax.semilogy(wl_um, alpha_results[T], color=cmap(norm(T)), linewidth=1.0)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Temperature (K)')

    if salt.experimental_data and salt.experimental_data.get('wavelength_um'):
        exp = salt.experimental_data
        ax.scatter(exp['wavelength_um'], exp['absorption_m1'],
                   color='red', marker='o', s=50, zorder=5,
                   edgecolors='black', linewidth=0.5,
                   label=exp.get('label', 'Experimental'))

    if salt.ir_peaks_um:
        for wl_peak, label in salt.ir_peaks_um:
            ax.axvline(x=wl_peak, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax.text(wl_peak + 0.3, ax.get_ylim()[1] * 0.5, label,
                    rotation=90, fontsize=7, color='gray', va='top')

    if show_planck:
        T_planck = planck_T or salt.T_pdf
        ax2 = ax.twinx()
        I_bb = planck_spectral_radiance(wl_m, T_planck)
        ax2.plot(wl_um, I_bb, 'k--', linewidth=1.0, alpha=0.4,
                 label=f'$I_{{bb}}$({T_planck:.0f}K)')
        ax2.set_ylabel('$I_{b,\\lambda}$ (W·m⁻³·sr⁻¹)', fontsize=9, alpha=0.6)
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.7)

    ax.set_xlabel('Wavelength (µm)')
    ax.set_ylabel('Absorption Coefficient, $\\kappa_\\lambda$ (m⁻¹)')
    ax.set_title(salt.name, fontweight='bold')
    ax.set_ylim(1e-1, 1e10)
    ax.set_xlim(0, wl_um[-1] + 2)
    if salt.experimental_data and salt.experimental_data.get('wavelength_um'):
        ax.legend(loc='best', fontsize=8)

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


def plot_wavenumber_spectrum(salt: SaltConfig, wl_m: np.ndarray, T_list: np.ndarray, alpha_results: Dict[float, np.ndarray], ax: Optional[plt.Axes] = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    else:
        fig = ax.figure

    wavenumber_cm = 1.0 / (wl_m * 100) 
    cmap, norm, sm = _get_temp_colormap(T_list)

    for T in T_list:
        ax.semilogy(wavenumber_cm, alpha_results[T], color=cmap(norm(T)), linewidth=1.0)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Temperature (K)')

    for wn, label in [(14000, 'Near IR'), (4000, 'Mid IR'), (400, 'Far IR')]:
        ax.axvline(x=wn, color='gray', linestyle='--', alpha=0.5)
        ax.text(wn * 0.85, 1e-1, label, rotation=90, fontsize=8, color='gray')

    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Absorption Coefficient, $\\kappa$ (m⁻¹)')
    ax.set_title(salt.name, fontweight='bold')
    ax.invert_xaxis()

    return fig


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================

def run_analysis(salt_name: str, wl_range: Tuple[float, float] = (0.15e-6, 60e-6), n_wavelengths: int = 800, T_range: Tuple[float, float] = (450, 1500), n_temperatures: int = 50):
    
    if salt_name not in SALT_REGISTRY:
        available = ', '.join(SALT_REGISTRY.keys())
        raise ValueError(f"Unknown salt '{salt_name}'. Available: {available}")

    # Ensure the "Figure" output directory exists
    os.makedirs('Figure', exist_ok=True)
    
    salt = SALT_REGISTRY[salt_name]
    wl = np.linspace(wl_range[0], wl_range[1], n_wavelengths)
    T_list = np.linspace(T_range[0], T_range[1], n_temperatures)

    initialize_salt(salt)

    print("Computing absorption spectra...")
    alpha_results = {T: compute_alpha_total(wl, T, salt) for T in T_list}
    print(f"  Done. {len(T_list)} temperatures computed.")

    print("Computing Planck-mean absorption coefficients...")
    T_kappa_list, kappa_P_list = [], []
    for T in T_list:
        kp = planck_mean_absorption(wl, alpha_results[T], T)
        if not np.isnan(kp):
            T_kappa_list.append(T)
            kappa_P_list.append(kp)

    T_kappa = np.array(T_kappa_list)
    kappa_P = np.array(kappa_P_list)

    print(f"\n{'--- Planck-Mean Absorption Coefficients ---':^50}")
    print(f"{'Temperature (K)':>16} | {'κ_P (m⁻¹)':>14}")
    print(f"{'-'*16}-+-{'-'*14}")
    for T_val, kp_val in zip(T_kappa, kappa_P):
        print(f"{T_val:>16.1f} | {kp_val:>14.2f}")

    print("\nGenerating plots...")
    
    fig1 = plot_absorption_spectrum(salt, wl, T_list, alpha_results)
    fig1.savefig(f"Figure/absorption_spectrum_{salt_name}.png", dpi=300, bbox_inches='tight')

    fig2 = plot_wavenumber_spectrum(salt, wl, T_list, alpha_results)
    fig2.savefig(f"Figure/absorption_wavenumber_{salt_name}.png", dpi=300, bbox_inches='tight')

    if len(T_kappa) > 1:
        fig3 = plot_planck_mean_vs_temperature(salt, T_kappa, kappa_P)
        fig3.savefig(f"Figure/planck_mean_{salt_name}.png", dpi=300, bbox_inches='tight')

    plt.show()

    return alpha_results, T_kappa, kappa_P


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    SALT_NAME = 'FLiNaK'

    results = run_analysis(
        salt_name=SALT_NAME,
        wl_range=(0.15e-6, 60e-6),
        n_wavelengths=800,
        T_range=(450, 1500),
        n_temperatures=50,
    )