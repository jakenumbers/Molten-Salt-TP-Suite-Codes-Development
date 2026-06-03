"""
Molten Salt Absorption Coefficient Model Using the Pair Distribution Function
==============================================================================
Implements the methodology from:
    "Model for Molten Salt Absorption Coefficient Using the Pair Distribution
     Function" -- Jacob Numbers

The total spectral absorption coefficient is:

    kappa_total(w, T) = kappa_vib(w, T) + kappa_elec(w, T) + kappa_multi(w, T)

where:
    kappa_vib   : Lorentz oscillator model from the dielectric function
    kappa_elec  : Urbach electronic absorption tail
    kappa_multi : Multiphonon absorption (exponential tail)

Key first-principles inputs (from the pair distribution function):
    omega_0  = sqrt(k / mu)        fundamental frequency from PMF curvature
    r_0                             equilibrium separation from PDF peak
    sigma_r                         peak width -> inhomogeneous broadening

Fitted parameters:
    gamma_0, gamma'                 damping (Lorentz peak width)
    alpha_anh                       anharmonicity (multiphonon tail slope)
    C0_multi                        multiphonon prefactor (tail magnitude)

Inhomogeneous broadening (no fitted parameters):
    In a melt, each cation-anion pair oscillates in a slightly different
    local environment, producing a DISTRIBUTION of fundamental frequencies
    rather than a single omega_0.  The width of the g(r) first peak encodes
    this disorder:
        sigma_r / r_0  -->  sigma_omega / omega_0

    For the multiphonon term, the discrete-p sawtooth is smoothed with
    a Savitzky-Golay filter whose window scales with sigma_omega/omega_0.

    This introduces NO new fitted parameters -- the broadening comes
    entirely from the measured PDF.

Two-phase workflow:
    Phase 1 -- Fit salts with experimental absorption data
    Phase 2 -- Predict alpha for unmeasured salts via alpha/omega_0 vs r_0
               correlation, transfer gamma and C0 from fitted salts
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import (
    e as e_charge, epsilon_0, k as k_B, Avogadro,
    m_u, c as c_light, pi, h, k, hbar
)
from scipy import integrate
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.signal import savgol_filter
from matplotlib.colors import Normalize, LinearSegmentedColormap
import mendeleev
import os
import shutil
from typing import Dict, List, Tuple, Optional, Any

# Import configurations and registry from the config file
from salts_config import SaltConfig, IonPairParams, SALT_REGISTRY

# =============================================================================
# 0. CONSTANTS & PLOT FORMATTING
# =============================================================================
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman"],
    "mathtext.fontset":     "cm",
    "axes.labelsize":       10,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":       8,
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
})

c       = c_light                    # speed of light [m/s]
kB_eV   = k_B / e_charge             # Boltzmann constant [eV/K]
hc_eV_um = 1.23984193                # h*c in [eV*um]
WIEN_B  = 2.897771955e-3             # Wien displacement constant [m*K]

# Maximum fractional broadening sigma_omega / omega_0.
# Capped at 5% to ensure the pseudo-Voigt approximation remains valid
# and that the multiphonon Savgol window stays reasonable.
MAX_SIGMA_FRAC = 0.05


# =============================================================================
# 1. PDF / RDF HELPER FUNCTIONS
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


def load_and_clean_pdf(file_path: str) -> pd.DataFrame:
    """
    Load a partial pair distribution function CSV.

    Expected format:
        Row 0 (header):  pair names separated by commas (e.g. "Li-F, Na-F")
        Row 1 (subheader): column labels (r, g(r), r, g(r), ...)
        Rows 2+: numeric data

    Returns a DataFrame with columns named  r_<pair>, rdf_<pair>.
    """
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
    """Atomic mass in kg from mendeleev."""
    el = mendeleev.element(element_symbol)
    return el.mass * m_u


def get_reduced_mass(ion_pair: Tuple[str, str]) -> float:
    """Reduced mass mu = m1*m2 / (m1+m2) in kg."""
    m1 = get_ion_mass(ion_pair[0])
    m2 = get_ion_mass(ion_pair[1])
    return (m1 * m2) / (m1 + m2)


def calculate_k_from_pmf(pdf_df: pd.DataFrame, ion_pair_str: str,
                         T_pdf: float) -> Tuple[float, float]:
    """
    Extract the harmonic force constant k and equilibrium distance r0
    from the potential of mean force (PMF) derived from the PDF.

    The PMF is:  V_PMF(r) = -k_B * T * ln(g(r))

    The force constant is the second derivative at the g(r) peak:
        k = d^2 V_PMF / dr^2 |_{r=r0}

    Parameters
    ----------
    pdf_df       : DataFrame with columns r_<pair> [Angstrom] and rdf_<pair>
    ion_pair_str : e.g. 'Li-F'
    T_pdf        : temperature at which the PDF was obtained [K]

    Returns
    -------
    k_val       : force constant [N/m]
    r0_angstrom : equilibrium separation [Angstrom]
    """
    r_col   = f"r_{ion_pair_str}"
    rdf_col = f"rdf_{ion_pair_str}"

    pair_df = pdf_df[[r_col, rdf_col]].dropna().copy()
    pair_df = pair_df[pair_df[rdf_col] > 1e-6]     # avoid log(0)

    r_m = pair_df[r_col].values * 1e-10             # Angstrom -> m
    rdf = pair_df[rdf_col].values

    # PMF: V(r) = -kB*T*ln(g(r))
    V_pmf    = -k_B * T_pdf * np.log(rdf)
    dV_dr    = np.gradient(V_pmf, r_m)
    d2V_dr2  = np.gradient(dV_dr, r_m)

    # Peak of g(r) = minimum of V_PMF = equilibrium position
    peak_loc = pair_df[rdf_col].idxmax()
    peak_pos = pair_df.index.get_loc(peak_loc)
    k_val    = d2V_dr2[peak_pos]

    r0_angstrom = pair_df[r_col].values[peak_pos]

    # Force constant must be positive (restoring force)
    k_val = abs(k_val) if k_val < 0 else k_val
    return k_val, r0_angstrom


def calculate_sigma_r_from_pdf(pdf_df: pd.DataFrame,
                               ion_pair_str: str,
                               r0_angstrom: float) -> float:
    """
    Extract sigma_r (the width of the g(r) first peak) from the PDF.

    Uses the left-side (inner) half-width at half-maximum (HWHM) to
    avoid contamination from the asymmetric right tail (which represents
    escaped neighbors, not frequency disorder of bound pairs).

    Converts HWHM to Gaussian sigma:  sigma = HWHM / sqrt(2*ln(2))

    Caps the result at MAX_SIGMA_FRAC * r0 to prevent unphysically
    large broadening.

    Parameters
    ----------
    pdf_df       : DataFrame with r_<pair> and rdf_<pair> columns
    ion_pair_str : e.g. 'Li-F'
    r0_angstrom  : equilibrium separation [Angstrom]

    Returns
    -------
    sigma_r : standard deviation of first peak [Angstrom]
    """
    r_col   = f"r_{ion_pair_str}"
    rdf_col = f"rdf_{ion_pair_str}"

    pair_df = pdf_df[[r_col, rdf_col]].dropna().copy()
    r_arr   = pair_df[r_col].values
    rdf_arr = pair_df[rdf_col].values

    # Find the peak
    peak_idx = np.argmax(rdf_arr)
    peak_val = rdf_arr[peak_idx]

    if peak_val <= 0:
        return 0.01 * r0_angstrom  # fallback

    half_max = peak_val / 2.0

    # Find left-side half-maximum crossing (r < r0)
    left_portion = rdf_arr[:peak_idx]
    crossings = np.where(left_portion < half_max)[0]

    if len(crossings) > 0:
        # Last crossing before the peak where g(r) < half_max
        cross_idx = crossings[-1]
        # Linear interpolation between cross_idx and cross_idx+1
        r_lo = r_arr[cross_idx]
        r_hi = r_arr[cross_idx + 1]
        g_lo = rdf_arr[cross_idx]
        g_hi = rdf_arr[cross_idx + 1]
        if g_hi != g_lo:
            r_half = r_lo + (half_max - g_lo) / (g_hi - g_lo) * (r_hi - r_lo)
        else:
            r_half = r_lo
        hwhm = r_arr[peak_idx] - r_half
    else:
        # Can't find left crossing -- use a small default
        hwhm = 0.02 * r0_angstrom

    # Convert HWHM to Gaussian sigma
    sigma_r = hwhm / np.sqrt(2.0 * np.log(2.0))

    # Clamp to physical range
    sigma_r = max(sigma_r, 0.01 * r0_angstrom)
    sigma_r = min(sigma_r, MAX_SIGMA_FRAC * r0_angstrom)

    return sigma_r


def calculate_oscillator_strengths(ion_pairs: List[IonPairParams],
                                   eps_inf_mixture: float) -> None:
    """
    Compute oscillator strength for each pair:
        delta_eps_j = x_j * (eps_s,j - eps_inf,j)

    where x_j is the mole fraction of pair j.
    """
    for pair in ion_pairs:
        pair.delta_eps = pair.mole_fraction * (pair.epsilon_s - pair.epsilon_inf)


def density_at_T(rho_coeffs: Tuple[float, float], T: float) -> float:
    """Linear density model: rho(T) = a + b*T  [kg/m^3]."""
    return rho_coeffs[0] + rho_coeffs[1] * T


# =============================================================================
# 3. MULTIPHONON ABSORPTION
# =============================================================================

def bose_einstein(omega0: float, T: float) -> float:
    """
    Bose-Einstein occupation number:
        n_bar = 1 / (exp(hbar*omega0 / kB*T) - 1)
    """
    x = hbar * omega0 / (k_B * T)
    if x > 500:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)


def alpha_multiphonon(omega, omega0, T, alpha_anh, C0, sigma_omega=0.0):
    if C0 <= 0 or alpha_anh <= 0 or omega0 <= 0:
        return np.zeros_like(omega)

    n_bar = bose_einstein(omega0, T)
    ratio = omega / omega0
    p = np.maximum(np.ceil(ratio).astype(int), 2)

    # Compute in log-space to avoid overflow:
    #   log(kappa) = log(C0) + p*log(n_bar+1) - alpha*ratio
    log_kappa = (np.log(C0) + p * np.log(n_bar + 1.0) - alpha_anh * ratio)

    # Clamp to avoid overflow when exponentiating back
    log_kappa = np.minimum(log_kappa, 60.0)  # exp(60) ~ 1e26, safe

    kappa = np.exp(log_kappa)

    # Savgol smoothing based on PDF peak width
    if sigma_omega > 0 and omega0 > 0:
        frac = sigma_omega / omega0
        raw_window = int(frac * len(omega) * 0.5)
        window = max(raw_window, 5)
        if window % 2 == 0:
            window += 1
        if window < len(kappa):
            polyorder = min(3, window - 1)
            kappa = savgol_filter(kappa, window, polyorder)
            kappa = np.maximum(kappa, 0.0)

    return kappa

# =============================================================================
# 4. CORE ABSORPTION MODEL
# =============================================================================

def lorentz_dielectric(omega: np.ndarray, omega0: float, gamma: float,
                       delta_eps: float) -> np.ndarray:
    """
    Single Lorentz oscillator contribution to the dielectric function:

        eps_j(w) = delta_eps * omega0^2 / (omega0^2 - w^2 - i*gamma*w)

    This is Eq. (7) from the paper, for one cation-anion pair.
    """
    return delta_eps * omega0**2 / ((omega0**2 - omega**2) - 1j * gamma * omega)


def alpha_vib_from_dielectric(omega: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """
    Vibrational absorption coefficient from the complex dielectric function.

    From Eq. (5) of the paper:
        kappa_vib = (w/c) * sqrt(2) * sqrt(sqrt(eps'^2 + eps''^2) - eps')

    This is the standard relation between the imaginary part of the
    complex refractive index and the absorption coefficient.
    """
    eps_r = np.real(eps)
    eps_i = np.imag(eps)
    modulus = np.sqrt(eps_r**2 + eps_i**2)
    return (omega / c) * np.sqrt(2.0) * np.sqrt(np.maximum(modulus - eps_r, 0.0))


def alpha_urbach(wl_m: np.ndarray, W0_eV: float, kappa0: float,
                 T: float) -> np.ndarray:
    """
    Electronic (Urbach) absorption tail -- Eq. (3) of the paper:

        kappa_elec = kappa0 * exp((E - W0) / (kB * T))

    where E = hc/lambda is the photon energy.
    Dominates at UV/visible wavelengths, negligible in the IR for most salts.
    """
    E_eV = hc_eV_um / (wl_m * 1e6)     # photon energy [eV]
    E_U  = kB_eV * T                     # thermal energy [eV]
    return kappa0 * np.exp((E_eV - W0_eV) / E_U)


def compute_alpha_total(wl_m: np.ndarray, T: float,
                        salt: SaltConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the total absorption coefficient and refractive index.

    Total:  kappa_total = kappa_vib + kappa_elec + kappa_multi

    The vibrational part uses the summed Lorentz dielectric function
    (Eq. 7) with a density-ratio correction on the oscillator strength
    (Eq. 8) and temperature-dependent damping gamma(T) = gamma0 + gamma'*T.

    Parameters
    ----------
    wl_m : wavelength array [m]
    T    : temperature [K]
    salt : SaltConfig object with all parameters

    Returns
    -------
    alpha_total  : total absorption coefficient [m^-1]
    n_refractive : real refractive index
    """
    omega = 2.0 * pi * c / wl_m
    eps_total = np.full_like(omega, salt.eps_inf_mixture, dtype=complex)

    # Density correction: Eq. (8)
    #   delta_eps(T) = delta_eps_0 * rho(T) / rho(T_fus)
    rho_T   = density_at_T(salt.rho_coeffs, T)
    rho_fus = density_at_T(salt.rho_coeffs, salt.T_fus)
    density_ratio = rho_T / rho_fus if rho_fus > 0 else 1.0

    kappa_multi_total = np.zeros_like(omega)

    for pair in salt.ion_pairs:
        # Temperature-dependent damping: gamma(T) = gamma0 + gamma' * T
        gamma_T    = pair.gamma0 + pair.gamma_slope * T
        delta_eps_T = pair.delta_eps * density_ratio

        # Lorentz oscillator contribution to dielectric function (unchanged)
        eps_total += lorentz_dielectric(omega, pair.omega0, gamma_T, delta_eps_T)

        # Multiphonon contribution (if parameters are set)
        a_anh = getattr(pair, 'alpha_anh', 0.0)
        c0    = getattr(pair, 'C0_multi', 0.0)
        if a_anh > 0 and c0 > 0:
            # Pass sigma_omega for Savgol smoothing (0 if not set)
            sig_w = getattr(pair, 'sigma_omega', 0.0)
            kappa_multi_total += alpha_multiphonon(
                omega, pair.omega0, T, a_anh, c0, sigma_omega=sig_w)

    # Vibrational absorption from dielectric function
    kappa_vib  = alpha_vib_from_dielectric(omega, eps_total)

    # Electronic Urbach tail
    kappa_elec = alpha_urbach(wl_m, salt.W0_eV, salt.kappa0, T)

    # Total absorption: Eq. (4)
    alpha_total = kappa_vib + kappa_elec + kappa_multi_total

    # Refractive index: n = sqrt((|eps| + eps') / 2)
    n_refractive = refractive_index_from_dielectric(eps_total)

    return alpha_total, n_refractive


def refractive_index_from_dielectric(eps: np.ndarray) -> np.ndarray:
    """
    Real refractive index from complex dielectric function:
        n = sqrt( (|eps| + eps') / 2 )
    """
    eps_r   = np.real(eps)
    eps_i   = np.imag(eps)
    modulus = np.sqrt(eps_r**2 + eps_i**2)
    return np.sqrt((modulus + eps_r) / 2.0)


# =============================================================================
# 5. PLANCK FUNCTION AND PLANCK-MEAN PROPERTIES
# =============================================================================

def planck_spectral_radiance(wl_m: np.ndarray, T: float) -> np.ndarray:
    """
    Planck spectral radiance:  I_bb(lambda, T)  [W / m^3 / sr]

        I_bb = 2*h*c^2 / (lambda^5 * (exp(hc/lambda*kB*T) - 1))
    """
    wl_m = np.asarray(wl_m, dtype=float)
    wl_m = np.maximum(wl_m, 1e-20)

    c1 = 2.0 * h * c**2
    c2 = h * c / k
    exponent = np.minimum(c2 / (wl_m * T), 700.0)

    I_b = c1 / (wl_m**5 * (np.exp(exponent) - 1.0))
    return np.nan_to_num(I_b, nan=0.0, posinf=0.0, neginf=0.0)


def planck_mean_absorption(wl_m: np.ndarray, alpha_total: np.ndarray,
                           T: float, range_factor: float = 20.0) -> float:
    """
    Planck-mean absorption coefficient -- Eq. (12):

        kappa_P(T) = integral(kappa_lambda * I_bb d_lambda)
                   / integral(I_bb d_lambda)

    Integration range: Wien peak / range_factor  to  Wien peak * range_factor.
    """
    peak_wl = WIEN_B / T
    wl_min  = peak_wl / range_factor
    wl_max  = peak_wl * range_factor

    mask = (wl_m >= wl_min) & (wl_m <= wl_max)
    if not np.any(mask):
        return np.nan

    wl_sub    = wl_m[mask]
    alpha_sub = alpha_total[mask]
    B_sub     = planck_spectral_radiance(wl_sub, T)

    numerator   = integrate.trapezoid(alpha_sub * B_sub, wl_sub)
    denominator = integrate.trapezoid(B_sub, wl_sub)

    return np.nan if denominator == 0 else (numerator / denominator)


def planck_mean_refractive_index(wl_m: np.ndarray, n_total: np.ndarray,
                                 T: float, range_factor: float = 20.0) -> float:
    """Planck-mean refractive index (same weighting as kappa_P)."""
    peak_wl = WIEN_B / T
    wl_min  = peak_wl / range_factor
    wl_max  = peak_wl * range_factor

    mask = (wl_m >= wl_min) & (wl_m <= wl_max)
    if not np.any(mask):
        return np.nan

    wl_sub = wl_m[mask]
    n_sub  = n_total[mask]
    B_sub  = planck_spectral_radiance(wl_sub, T)

    numerator   = integrate.trapezoid(n_sub * B_sub, wl_sub)
    denominator = integrate.trapezoid(B_sub, wl_sub)

    return np.nan if denominator == 0 else (numerator / denominator)


# =============================================================================
# 6. INITIALIZATION & FITTING
# =============================================================================

def initialize_salt(salt: SaltConfig) -> None:
    """
    Initialize a salt: load PDF, extract omega_0, r_0, and sigma_r
    for each pair, compute oscillator strengths.
    """
    print(f"\n{'='*60}")
    print(f"Initializing: {salt.name}")
    print(f"{'='*60}")

    pdf_df = load_and_clean_pdf(salt.pdf_file)
    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

    print(f"\n  {'Pair':<8} {'k [N/m]':>12} {'w0 [rad/s]':>14} "
          f"{'lam [um]':>10} {'r0 [A]':>8} {'sig_r [A]':>9} "
          f"{'sig/r0':>7} {'d_eps':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*10} {'-'*8} {'-'*9} "
          f"{'-'*7} {'-'*8}")

    for pair in salt.ion_pairs:
        pair.k_N_per_m, pair.r0_angstrom = calculate_k_from_pmf(
            pdf_df, pair.ion_pair_str, salt.T_pdf)
        mu = get_reduced_mass(pair.ion_pair)
        pair.omega0 = np.sqrt(pair.k_N_per_m / mu)
        lam_um = 2.0 * pi * c / pair.omega0 * 1e6

        # Extract sigma_r from the PDF peak width
        pair.sigma_r = calculate_sigma_r_from_pdf(
            pdf_df, pair.ion_pair_str, pair.r0_angstrom)

        # Convert to frequency spread: sigma_omega / omega_0 ~ sigma_r / r_0
        frac = pair.sigma_r / pair.r0_angstrom if pair.r0_angstrom > 0 else 0.0
        pair.sigma_omega = frac * pair.omega0

        print(f"  {pair.ion_pair_str:<8} {pair.k_N_per_m:>12.2f} "
              f"{pair.omega0:>14.3e} {lam_um:>10.1f} "
              f"{pair.r0_angstrom:>8.3f} {pair.sigma_r:>9.4f} "
              f"{frac:>7.4f} {pair.delta_eps:>8.3f}")
    print()


def fit_gamma_parameters(salt: SaltConfig) -> None:
    """
    Fit damping and/or multiphonon parameters to experimental data.

    Configuration is read from salt.experimental_fit_config, a dict with:

        'mode' : str
            'single'           -- fit to one dataset (default)
            'weighted_average' -- fit to weighted combination of datasets
            'none'             -- skip fitting

        'fit_target' : str or list
            'damping' or 'both'     -- fit gamma0 + gamma_slope only
            'gamma0'                -- fit gamma0 only
            'gamma_slope'           -- fit gamma_slope only
            'multiphonon'           -- fit alpha_anh + C0_multi only
            'all'                   -- fit gamma0, gamma_slope, alpha_anh, C0_multi
            ['gamma_slope', 'alpha_anh', 'C0_multi']  -- explicit list

        'dataset_index' : int  (used when mode='single', default 0)

        'weights' : list of float  (used when mode='weighted_average')
            One weight per dataset with experimental data.
            If omitted or wrong length, equal weights are used.
    """
    if not salt.experimental_data:
        print("  Skipping fit: No experimental data provided.")
        return

    fit_config = salt.experimental_fit_config or {}
    fit_mode   = fit_config.get('mode', 'single')
    fit_target = fit_config.get('fit_target',
                                fit_config.get('fit_parameters', 'both'))

    if fit_mode == 'none':
        print(f"  Skipping fit for {salt.name}: fitting disabled.")
        return

    # --- Resolve which parameters to fit ---
    ALLOWED_PARAMS = ['gamma0', 'gamma_slope', 'alpha_anh', 'C0_multi']

    if isinstance(fit_target, str):
        ftl = fit_target.lower()
        FIT_MAP = {
            'none':        [],
            'both':        ['gamma0', 'gamma_slope'],
            'damping':     ['gamma0', 'gamma_slope'],
            'gamma0':      ['gamma0'],
            'gamma':       ['gamma0'],
            'gamma_slope': ['gamma_slope'],
            'multiphonon': ['alpha_anh', 'C0_multi'],
            'all':         ['gamma0', 'gamma_slope', 'alpha_anh', 'C0_multi'],
        }
        fit_params = FIT_MAP.get(ftl, [])
        if not fit_params and ftl not in FIT_MAP:
            print(f"  Invalid fit_target '{fit_target}'. "
                  f"Options: {list(FIT_MAP.keys())}")
            return
    elif isinstance(fit_target, (list, tuple)):
        fit_params = [p for p in ALLOWED_PARAMS if p in fit_target]
    else:
        fit_params = ['gamma0', 'gamma_slope']

    if not fit_params:
        print(f"  Skipping fit for {salt.name}: no parameters selected.")
        return

    # --- Build dataset list ---
    fit_datasets = []

    if fit_mode == 'single':
        dataset_index = fit_config.get('dataset_index', 0)
        valid_ds = [e for e in salt.experimental_data
                    if e.get('wavelength_um')]
        if not valid_ds:
            print("  Skipping fit: No datasets with data.")
            return
        if dataset_index >= len(valid_ds):
            dataset_index = 0
        exp = valid_ds[dataset_index]
        fit_datasets.append({
            'wavelength_m':   np.array(exp['wavelength_um']) * 1e-6,
            'absorption_m1':  np.array(exp['absorption_m1']),
            'T_exp':          exp.get('T_exp', salt.T_pdf),
            'weight':         1.0,
        })
        T_summary = f"{fit_datasets[0]['T_exp']:.0f}K"

    elif fit_mode == 'weighted_average':
        valid_ds = [e for e in salt.experimental_data
                    if e.get('wavelength_um')]
        if not valid_ds:
            print("  Skipping fit: No datasets with data.")
            return
        weights = fit_config.get('weights', None)
        if weights is None or len(weights) != len(valid_ds):
            weights = [1.0] * len(valid_ds)
        weights = np.array(weights, dtype=float)
        if np.sum(weights) == 0:
            weights = np.ones(len(valid_ds))
        weights = weights / np.sum(weights)

        for exp, w in zip(valid_ds, weights):
            if w <= 0:
                continue
            fit_datasets.append({
                'wavelength_m':  np.array(exp['wavelength_um']) * 1e-6,
                'absorption_m1': np.array(exp['absorption_m1']),
                'T_exp':         exp.get('T_exp', salt.T_pdf),
                'weight':        w,
            })
        T_summary = ", ".join([f"{d['T_exp']:.0f}K" for d in fit_datasets])
    else:
        print(f"  Invalid fit mode '{fit_mode}'. Use 'none', 'single', "
              f"or 'weighted_average'.")
        return

    if not fit_datasets:
        print("  Skipping fit: No valid datasets after filtering.")
        return

    param_labels = {
        'gamma0': 'g0', 'gamma_slope': "g'",
        'alpha_anh': 'alpha', 'C0_multi': 'C0',
    }
    label_str = ' + '.join([param_labels[p] for p in fit_params])
    print(f"\n--- Fitting {label_str} for {salt.name} at {T_summary} ---")

    # --- Build initial guess vector ---
    # One set of parameters per ion pair
    n_pairs = len(salt.ion_pairs)
    x0 = []
    original_params = []

    for pair in salt.ion_pairs:
        orig = {
            'gamma0':      pair.gamma0,
            'gamma_slope':  pair.gamma_slope,
            'alpha_anh':   getattr(pair, 'alpha_anh', 3.0),
            'C0_multi':    getattr(pair, 'C0_multi', 1e4),
        }
        original_params.append(orig)
        for p in fit_params:
            x0.append(orig[p])

    n_params_per_pair = len(fit_params)

    def objective(x):
        """Log-space MSE between model and experimental absorption."""
        x_arr = np.array(x)

        # Enforce non-negativity with soft penalty
        if np.any(x_arr < 0):
            return 1e12 + np.sum(np.abs(x_arr[x_arr < 0])) * 1e6

        # Apply parameters to pairs
        idx = 0
        for pair, orig in zip(salt.ion_pairs, original_params):
            for p in fit_params:
                setattr(pair, p, x_arr[idx])
                idx += 1
            # Keep non-fitted parameters at original values
            for p_name, p_val in orig.items():
                if p_name not in fit_params:
                    setattr(pair, p_name, p_val)

        # Evaluate fit quality across all datasets
        total_error = 0.0
        for ds in fit_datasets:
            alpha_model, _ = compute_alpha_total(
                ds['wavelength_m'], ds['T_exp'], salt)
            log_model = np.log10(np.maximum(alpha_model, 1e-30))
            log_exp   = np.log10(np.maximum(ds['absorption_m1'], 1e-30))
            error = np.sum((log_model - log_exp)**2)
            total_error += ds['weight'] * error
        return total_error

    # --- Run optimizer ---
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})

    # Apply final optimized values
    idx = 0
    for pair in salt.ion_pairs:
        for p in fit_params:
            setattr(pair, p, res.x[idx])
            idx += 1

    # Print results
    for pair in salt.ion_pairs:
        a_anh = getattr(pair, 'alpha_anh', 0.0)
        c0    = getattr(pair, 'C0_multi', 0.0)
        ratio = a_anh / pair.omega0 if pair.omega0 > 0 and a_anh > 0 else 0.0
        print(f"  {pair.ion_pair_str}: g0={pair.gamma0:.3e}  "
              f"g'={pair.gamma_slope:.3e}  "
              f"alpha={a_anh:.4f}  C0={c0:.3e}  "
              f"alpha/w0={ratio:.3e}")

    status = 'OK' if res.success else 'WARN (check convergence)'
    print(f"  Fit: {status} (Log-MSE: {res.fun:.4f})\n")


# =============================================================================
# 7. PLOTTING FUNCTIONS
# =============================================================================

def _get_temp_colormap(T_list):
    """Create a temperature colormap from navy to firebrick."""
    colors = ["navy", "darkorchid", "crimson", "firebrick"]
    cmap = LinearSegmentedColormap.from_list("cool_no_yellow", colors)
    norm = Normalize(vmin=min(T_list), vmax=max(T_list))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return cmap, norm, sm


def plot_absorption_spectrum(salt, wl_m, T_list, alpha_results,
                             ax=None, show_planck=True, planck_T=None,
                             show_all_planck=True):
    """Plot spectral absorption coefficient vs wavelength for all temperatures."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    else:
        fig = ax.figure

    wl_um = wl_m * 1e6
    cmap, norm, sm = _get_temp_colormap(T_list)

    for T in T_list:
        ax.semilogy(wl_um, alpha_results[T], color=cmap(norm(T)), linewidth=1.0)
        if T == max(T_list):
            ax.plot(wl_um, alpha_results[T], color=cmap(norm(T)),
                    linewidth=1.0, label=r'$\kappa_{\lambda}(T)$')

    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', aspect=40, pad=0.05)
    cbar.set_label('Temperature (K)')

    ax.text(0.03, 0.96, salt.name, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
            zorder=10)

    # IR peak markers
    if salt.ir_peaks_um:
        for wl_peak, label in salt.ir_peaks_um:
            ax.axvline(x=wl_peak, color='gray', linestyle='--',
                       alpha=0.5, linewidth=0.8, zorder=1)
            ax.text(wl_peak + 0.6, 0.25, label,
                    transform=ax.get_xaxis_transform(),
                    rotation=90, fontsize=9, color='gray', va='top', zorder=2)

    # Planck curves and experimental data
    if show_planck:
        ax2 = ax.twinx()
        max_I_bb = 0

        if show_all_planck:
            ax2.plot([], [], color='gray', linestyle='--', linewidth=1.2,
                     label='$I_{bb,\\lambda}(T)$')
            for T in np.linspace(min(T_list), max(T_list), 5):
                I_bb = planck_spectral_radiance(wl_m, T)
                max_I_bb = max(max_I_bb, np.max(I_bb))
                ax2.plot(wl_um, I_bb, linestyle=':', color=cmap(norm(T)),
                         linewidth=1.0, alpha=0.7)
        else:
            T_planck = planck_T or salt.T_pdf
            I_bb = planck_spectral_radiance(wl_m, T_planck)
            max_I_bb = np.max(I_bb)
            ax2.plot(wl_um, I_bb, 'k--', linewidth=1.0, alpha=0.6,
                     label=f'$I_{{bb}}$({T_planck:.0f}K)')

        # Experimental data points
        if salt.experimental_data:
            clrs = ['yellow','green','pink','red', 'blue', 'orange', 
                    'brown','gray']
            mkrs = ['o', 's', '^', 'v', 'D', '*', 'P', 'X']
            for i, exp in enumerate(salt.experimental_data):
                if exp.get('wavelength_um'):
                    ax.scatter(exp['wavelength_um'], exp['absorption_m1'],
                               color=clrs[i % len(clrs)],
                               marker=mkrs[i % len(mkrs)],
                               s=25, zorder=5, edgecolors='black',
                               linewidth=0.5, alpha=0.8,
                               label=exp.get('label', f'Exp {i+1}'))

        ax2.set_yscale('log')
        ax2.set_ylim(1e-1, max_I_bb / 0.001 if max_I_bb > 0 else 1e10)
        ax2.set_ylabel('$I_{bb,\\lambda}$ (W m$^{-3}$ sr$^{-1}$)', fontsize=9)
        ax2.tick_params(direction='in', which='both')
        ax2.minorticks_on()

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2,
                   loc='upper right', fontsize=9, framealpha=0.9,
                   edgecolor='black')
    else:
        if salt.experimental_data:
            clrs = ['red', 'blue', 'green', 'orange', 'purple']
            mkrs = ['o', 's', '^', 'v', 'D']
            for i, exp in enumerate(salt.experimental_data):
                if exp.get('wavelength_um'):
                    ax.scatter(exp['wavelength_um'], exp['absorption_m1'],
                               color=clrs[i % len(clrs)],
                               marker=mkrs[i % len(mkrs)],
                               s=25, zorder=5, edgecolors='black',
                               linewidth=0.5, alpha=0.6,
                               label=exp.get('label', f'Exp {i+1}'))
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                  edgecolor='black')

    ax.set_xlabel('Wavelength ($\\mu$m)')
    ax.set_ylabel('Absorption Coefficient, $\\kappa_\\lambda$ (m$^{-1}$)')
    ax.set_ylim(1e-1, 1e11)
    ax.set_xlim(0, wl_um[-1] + 2)
    return fig


def plot_refractive_index(salt, wl_m, T_list, n_results, ax=None):
    """Plot refractive index vs wavelength for all temperatures."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    else:
        fig = ax.figure

    wl_um = wl_m * 1e6
    cmap, norm, sm = _get_temp_colormap(T_list)

    for T in T_list:
        ax.plot(wl_um, n_results[T], color=cmap(norm(T)), linewidth=1.0)

    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', aspect=40, pad=0.05)
    cbar.set_label('Temperature (K)')

    ax.text(0.03, 0.96, salt.name, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
            zorder=10)

    # Experimental refractive index data
    if salt.experimental_n:
        exp_n_data = salt.experimental_n
        if isinstance(exp_n_data, dict):
            exp_n_data = [exp_n_data]
        # Use marker colors that are distinct from the common spectrum line
        # colors (blue/red) so experimental points are easy to distinguish.
        clrs = ['purple', 'green', 'orange', 'brown']
        mkrs = ['o', 's', '^', 'v']
        for i, exp in enumerate(exp_n_data):
            if exp is None:
                continue
            wl_exp = exp.get('wavelength_um')
            n_exp  = exp.get('n_val') or exp.get('n')
            if wl_exp and n_exp:
                ax.scatter(wl_exp, n_exp,
                           color=clrs[i % len(clrs)],
                           marker=mkrs[i % len(mkrs)],
                           s=56, zorder=5, edgecolors='black',
                           linewidth=0.5, alpha=0.65,
                           label=exp.get('label', f'Exp n {i+1}'))

    ax.set_xlabel('Wavelength ($\\mu$m)')
    ax.set_ylabel('Refractive Index, $n(\\lambda)$')
    ax.set_xlim(0, wl_um[-1] + 2)
    lines, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(lines, labels, loc='upper right', fontsize=9,
                  framealpha=0.9, edgecolor='black')
    return fig


def plot_planck_mean_vs_temperature(salt, T_array, kappa_P_array, ax=None):
    """Plot Planck-mean absorption coefficient vs temperature."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,2), constrained_layout=True)
    else:
        fig = ax.figure

    cmap, norm, _ = _get_temp_colormap(T_array)
    ax.scatter(T_array, kappa_P_array, c=T_array, cmap=cmap, norm=norm,
               edgecolors='k', s=60, alpha=0.8, zorder=3,
               label='Model $\\kappa_P(T)$')

    # Piecewise fit overlay (if configured)
    if salt.piecewise_fit and salt.piecewise_fit.get('enabled', False):
        T_fine = np.linspace(T_array.min(), T_array.max(), 500)
        for idx, region in enumerate(salt.piecewise_fit['regions']):
            T_lo, T_hi = region['range']
            mask = ((T_fine >= T_lo) & (T_fine <= T_hi) if idx == 0
                    else (T_fine > T_lo) & (T_fine <= T_hi))
            T_seg = T_fine[mask]
            if T_seg.size == 0:
                continue
            kappa_seg = np.polyval(region['coeffs'], T_seg)
            label = 'Piecewise Fit' if idx == 0 else None
            ax.plot(T_seg, kappa_seg, 'r-', linewidth=2, label=label)

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Planck-mean $\\kappa_P$ (m$^{-1}$)')
    ax.text(0.4, 0.96, salt.name, transform=ax.transAxes,
        fontsize=14, fontweight='bold', va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
        zorder=10)
    ax.legend(loc='best', fontsize=9)
    ax.set_ylim(0,20000)
    return fig


def plot_transferability(all_pair_data: List[Dict], output_dir: str):
    """
    Grouped transferability plot: alpha/omega_0 by ion pair,
    with each point labeled by source salt.
    """
    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)

    # Group by ion pair
    pair_groups = {}
    for d in all_pair_data:
        pair_groups.setdefault(d['pair'], []).append(d)

    pair_names = sorted(pair_groups.keys())
    x_positions = []
    x_labels = []
    pos = 0

    all_ratios = [d['ratio'] * 1e14 for d in all_pair_data]
    f_ratios   = [d['ratio'] * 1e14 for d in all_pair_data
                  if d['anion_family'] == 'F']

    for i, pair_name in enumerate(pair_names):
        group = pair_groups[pair_name]
        for j, d in enumerate(group):
            color = 'steelblue' if d['anion_family'] == 'F' else 'firebrick'
            ax.scatter(pos, d['ratio'] * 1e14, s=120, c=color,
                       edgecolors='black', linewidth=0.8, zorder=5)
            ax.annotate(d['salt'], (pos, d['ratio'] * 1e14),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=7, color=color)
            pos += 1
        center = pos - len(group) / 2.0 - 0.5
        x_positions.append(center)
        x_labels.append(pair_name)
        if i < len(pair_names) - 1:
            ax.axvline(x=pos - 0.5, color='lightgray', linewidth=0.8)
        pos += 1

    # Statistics
    if all_ratios:
        overall_mean = np.mean(all_ratios)
        sigma = np.std(all_ratios)
        ax.axhline(y=overall_mean, color='black', linestyle='--', linewidth=1,
                    label=f'Mean = {overall_mean:.2f}')
        ax.axhspan(overall_mean - sigma, overall_mean + sigma,
                    alpha=0.1, color='gray',
                    label=f'$\\pm 1\\sigma$ = {sigma:.2f}')
    if f_ratios:
        f_mean = np.mean(f_ratios)
        ax.axhline(y=f_mean, color='steelblue', linestyle=':', linewidth=1,
                    label=f'F$^-$ mean = {f_mean:.2f}')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontweight='bold')
    ax.set_xlabel('Ion Pair')
    ax.set_ylabel('$\\alpha/\\omega_0$ ($\\times 10^{-14}$ s)')
    # ax.set_title('Multiphonon Anharmonicity: Transferability Check')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='black')

    fig.savefig(os.path.join(output_dir, 'transferability_alpha_omega0.png'),
                bbox_inches='tight', dpi=1200)
    fig.savefig(os.path.join(output_dir, 'transferability_alpha_omega0.pdf'),
                bbox_inches='tight', dpi=1200)
    return fig


def plot_alpha_omega0_vs_r0(all_pair_data: List[Dict],
                            output_dir: str,
                            predicted_data: Optional[List[Dict]] = None,
                            filename: str = 'alpha_omega0_vs_r0.png',
                            interactive_labels: bool = False,
                            label_positions_file: Optional[str] = None,
                            save_label_positions_file: Optional[str] = None):
    """
    Scatter plot of alpha/omega_0 vs r_0 with linear regression.

    Filled markers = fitted data.
    Unfilled markers = predicted data (if provided).

    Returns (f_slope, f_intercept, f_r2, cl_intercept) for the regression
    lines, in units of 1e-14 s per Angstrom.
    """
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # Separate by anion family
    f_data  = [d for d in all_pair_data if d['anion_family'] == 'F']
    cl_data = [d for d in all_pair_data if d['anion_family'] == 'Cl']

    # --- Plot fitted points (filled circles) ---
    import json

    # Plot points and prepare labels. We create annotations which can be
    # draggable in interactive mode and can be saved/loaded from a JSON file.
    plotted_points = []  # list of (x, y, text, color, kwargs)
    for d in f_data:
        x = d['r0']
        y = d['ratio'] * 1e14
        ax.scatter(x, y, s=100, c='steelblue', edgecolors='black', linewidth=0.8, zorder=5)
        plotted_points.append((x, y, f"{d['pair']} ({d['salt']})", 'steelblue', {'fontsize': 7}))

    for d in cl_data:
        x = d['r0']
        y = d['ratio'] * 1e14
        ax.scatter(x, y, s=100, c='firebrick', edgecolors='black', linewidth=0.8, zorder=5)
        plotted_points.append((x, y, f"{d['pair']} ({d['salt']})", 'firebrick', {'fontsize': 7}))

    # --- Fluoride linear regression ---
    f_slope, f_intercept, f_r2 = 0.0, 0.0, 0.0
    cl_intercept = 0.0

    if len(f_data) >= 2:
        r0_f  = np.array([d['r0'] for d in f_data])
        rat_f = np.array([d['ratio'] * 1e14 for d in f_data])
        slope, intercept, r_val, _, _ = linregress(r0_f, rat_f)
        f_slope     = slope
        f_intercept = intercept
        f_r2        = r_val**2

        # Extend line for visualization
        r0_line = np.linspace(
            min(r0_f.min(), 1.2) - 0.2,
            max(r0_f.max(), 3.0) + 0.4, 100)
        ax.plot(r0_line, slope * r0_line + intercept, '--',
                color='steelblue', linewidth=1.2,
                label=(f'F$^-$ fit: '
                       f'$\\alpha/\\omega_0$ = '
                       f'{slope:.1f}$r_0$ {intercept:+.1f} '
                       f'($R^2$={f_r2:.3f})'))

    # --- Chloride: same slope, shifted intercept ---
    if cl_data and f_slope != 0:
        r0_cl  = np.array([d['r0'] for d in cl_data])
        rat_cl = np.array([d['ratio'] * 1e14 for d in cl_data])
        cl_intercept = np.mean(rat_cl - f_slope * r0_cl)

        r0_line_cl = np.linspace(
            min(r0_cl.min(), 2.0) - 0.3,
            max(r0_cl.max(), 3.5) + 0.4, 100)
        ax.plot(r0_line_cl, f_slope * r0_line_cl + cl_intercept, '--',
                color='firebrick', linewidth=1.2,
                label=(f'Cl$^-$ (same slope): '
                       f'$\\alpha/\\omega_0$ = {f_slope:.1f}$r_0$ '
                       f'{cl_intercept:+.1f}'))

    # --- Plot predicted points (unfilled circles) ---
    if predicted_data:
        for d in predicted_data:
            color = 'steelblue' if d['anion_family'] == 'F' else 'firebrick'
            x = d['r0']
            y = d['ratio'] * 1e14
            ax.scatter(x, y, s=120, facecolors='none', edgecolors=color,
                       linewidth=1.2, zorder=6, marker='o')
            plotted_points.append((x, y, f"{d['pair']} ({d['salt']})", color, {'fontsize': 7, 'fontstyle': 'italic'}))

    # Create annotations. If a label_positions_file is provided, use those
    # coordinates (data-space) for the text positions. Otherwise place labels
    # at small offsets from the points. Annotations are created with
    # arrowprops so the leader line is visible.
    annotations = []
    # try to load saved positions
    saved_positions = {}
    if label_positions_file:
        try:
            with open(label_positions_file, 'r') as f:
                saved_positions = json.load(f)
        except Exception:
            saved_positions = {}

    default_offsets = [(0.08, 0.06), (0.08, -0.06), (-0.08, 0.06), (-0.08, -0.06), (0.14, 0.06), (0.14, -0.06)]
    for i, (x, y, txt, color, kw) in enumerate(plotted_points):
        key = txt
        if key in saved_positions:
            tx, ty = saved_positions[key]
        else:
            dx, dy = default_offsets[i % len(default_offsets)]
            tx = x + dx
            ty = y + dy

        ann = ax.annotate(txt, xy=(x, y), xytext=(tx, ty), textcoords='data',
                          fontsize=kw.get('fontsize', 7), color=color,
                          arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                          bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
        # make annotation draggable if interactive mode is requested
        try:
            ann.set_picker(True)
            ann.draggable()
        except Exception:
            pass
        annotations.append((txt, ann))

    # Adjust y-limits to comfortably include all data points (with padding).
    try:
        y_vals = [y for (_x, y, _t, _c, _kw) in plotted_points]
        if y_vals:
            y_min = min(y_vals)
            y_max = max(y_vals)
            span = max(y_max - y_min, 1e-6)
            pad = span * 0.08
            ax.set_ylim(y_min - pad, y_max + pad)
    except Exception:
        pass

    # If interactive_labels is True, show the figure to allow manual repositioning
    # of annotations. After the window is closed save new positions if requested.
    if interactive_labels:
        print('\nInteractive label mode: drag labels to desired positions, then close the figure window to continue.')
        plt.show()
        # gather final positions
        final_positions = {}
        for key, ann in annotations:
            try:
                # get_position returns text position in data coords when textcoords='data'
                pos = ann.get_position()
                final_positions[key] = [float(pos[0]), float(pos[1])]
            except Exception:
                final_positions[key] = None
        if save_label_positions_file:
            try:
                with open(save_label_positions_file, 'w') as f:
                    json.dump(final_positions, f, indent=2)
                print(f"Saved label positions to: {save_label_positions_file}")
            except Exception as e:
                print(f"Failed to save label positions: {e}")

    # Legend entries
    ax.scatter([], [], s=100, c='gray', edgecolors='black',
               linewidth=0.8, label='Fitted')
    if predicted_data:
        ax.scatter([], [], s=100, facecolors='none', edgecolors='gray',
                   linewidth=0.8, label='Predicted')
    ax.scatter([], [], s=100, c='steelblue', edgecolors='black',
               linewidth=0.8, label='F$^-$ family')
    ax.scatter([], [], s=100, c='firebrick', edgecolors='black',
               linewidth=0.8, label='Cl$^-$ family')

    ax.set_xlabel('$r_0$ ($\\AA$)')
    ax.set_ylabel('$\\alpha/\\omega_0$ ($\\times 10^{-14}$ s)')
    # ax.set_title('Multiphonon Anharmonicity vs. Equilibrium Separation')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='black')
    ax.set_xlim(1.0, 4.0)
    ax.set_ylim(0.0, 16)

    # Save both PNG and PDF versions
    base_name = os.path.splitext(filename)[0]
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    png_path = os.path.join(output_dir, f"{base_name}.png")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=1200)
    fig.savefig(png_path, bbox_inches='tight', dpi=1200)
    print(f"  Saved: {pdf_path} and {png_path}")
    return f_slope, f_intercept, f_r2, cl_intercept


# =============================================================================
# 8. CSV EXPORT
# =============================================================================

def export_properties_csv(salt: SaltConfig, T_array: np.ndarray,
                          kappa_P_array: np.ndarray,
                          n_eff_array: np.ndarray,
                          wl: Optional[np.ndarray] = None,
                          alpha_results: Optional[Dict[float, np.ndarray]] = None,
                          n_results: Optional[Dict[float, np.ndarray]] = None,
                          filename: str = "Salt_RHT_Properties.csv"):
    """
    Export temperature-dependent and wavelength-dependent properties to CSV.

    Wavelength-dependent spectra are exported ONLY at T_fus (the computed
    temperature closest to the melting point), not at all temperatures.
    """
    comp_str = "; ".join(
        [f"{p.ion_pair_str} ({p.mole_fraction})" for p in salt.ion_pairs])

    t_str = ",".join([f"{t:.1f}" for t in T_array])
    k_str = ",".join([f"{kv:.4e}" for kv in kappa_P_array])
    n_str = ",".join([f"{n:.5f}" for n in n_eff_array])

    new_row = {
        'Salt':                    salt.name,
        'Composition':             comp_str,
        'T_fus_K':                 salt.T_fus,
        'Scalar_Absorption_m1':    np.mean(kappa_P_array),
        'Scalar_Refractive_Index': np.mean(n_eff_array),
        'Temperature_List_K':      t_str,
        'Absorption_List_m1':      k_str,
        'Refractive_Index_List':   n_str,
    }

    # Per-pair fitted/predicted parameters
    for i, pair in enumerate(salt.ion_pairs):
        prefix = f"Pair{i+1}_{pair.ion_pair_str}"
        a_anh  = getattr(pair, 'alpha_anh', 0.0)
        c0     = getattr(pair, 'C0_multi', 0.0)
        r0     = getattr(pair, 'r0_angstrom', 0.0)
        sig_r  = getattr(pair, 'sigma_r', 0.0)
        ratio  = a_anh / pair.omega0 if pair.omega0 > 0 and a_anh > 0 else 0.0

        new_row[f'{prefix}_omega0_rad_s']       = pair.omega0
        new_row[f'{prefix}_lambda_peak_um']     = (
            2.0 * pi * c / pair.omega0 * 1e6 if pair.omega0 > 0 else 0.0)
        new_row[f'{prefix}_k_N_per_m']          = pair.k_N_per_m
        new_row[f'{prefix}_r0_angstrom']        = r0
        new_row[f'{prefix}_sigma_r_angstrom']   = sig_r
        new_row[f'{prefix}_sigma_r_over_r0']    = sig_r / r0 if r0 > 0 else 0.0
        new_row[f'{prefix}_gamma0']             = pair.gamma0
        new_row[f'{prefix}_gamma_slope']        = pair.gamma_slope
        new_row[f'{prefix}_alpha_anh']          = a_anh
        new_row[f'{prefix}_C0_multi_m1']        = c0
        new_row[f'{prefix}_alpha_over_omega0_s'] = ratio

    # Wavelength-dependent data ONLY at T closest to T_fus
    if wl is not None and alpha_results is not None and n_results is not None:
        wl_str = ",".join([f"{w*1e6:.6e}" for w in wl])
        new_row['Wavelength_List_um'] = wl_str

        available_T = sorted(alpha_results.keys())
        T_fus_closest = min(available_T, key=lambda t: abs(t - salt.T_fus))

        if T_fus_closest in alpha_results and T_fus_closest in n_results:
            alpha_spec = ",".join(
                [f"{a:.4e}" for a in alpha_results[T_fus_closest]])
            n_spec = ",".join(
                [f"{n:.6f}" for n in n_results[T_fus_closest]])
            new_row[f'Absorption_Spectrum_T{T_fus_closest:.0f}K'] = alpha_spec
            new_row[f'Refractive_Index_Spectrum_T{T_fus_closest:.0f}K'] = n_spec

    # Append or update
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = df[df['Salt'] != salt.name]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(filename, index=False)
    print(f"  CSV exported: {filename}")


# =============================================================================
# 9. SINGLE-SALT ANALYSIS
# =============================================================================

def run_analysis(salt_name: str,
                 wl_range: Tuple[float, float] = (0.15e-6, 60e-6),
                 n_wavelengths: int = 800,
                 T_range: Tuple[float, float] = (450, 1500),
                 n_temperatures: int = 50,
                 run_fit: bool = False,
                 experimental_fit_config: Optional[Dict[str, Any]] = None):
    """
    Full analysis pipeline for a single salt:
        1. Initialize (load PDF, compute omega_0, r_0, sigma_r)
        2. Optionally fit gamma / multiphonon parameters
        3. Compute absorption spectra across temperature range
        4. Compute Planck-mean properties
        5. Export CSV and generate plots

    Returns a dict with the salt object and all computed results.
    """
    if salt_name not in SALT_REGISTRY:
        available = ', '.join(SALT_REGISTRY.keys())
        raise ValueError(f"Unknown salt '{salt_name}'. Available: {available}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'Figures')
    os.makedirs(output_dir, exist_ok=True)

    salt = SALT_REGISTRY[salt_name]
    wl     = np.linspace(wl_range[0], wl_range[1], n_wavelengths)
    T_list = np.linspace(T_range[0], T_range[1], n_temperatures)

    # Step 1: Initialize
    initialize_salt(salt)

    # Step 2: Fit (optional)
    if run_fit:
        if experimental_fit_config is not None:
            salt.experimental_fit_config = experimental_fit_config
        fit_gamma_parameters(salt)

    # Print parameter summary
    print(f"  --- Parameters for {salt.name} ---")
    print(f"  {'Pair':<10} {'g0':>10} {'g_slope':>10} {'alpha':>8} "
          f"{'C0 [m-1]':>12} {'a/w0 [s]':>12} {'r0 [A]':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*8} "
          f"{'-'*12} {'-'*12} {'-'*8}")
    for pair in salt.ion_pairs:
        a_anh = getattr(pair, 'alpha_anh', 0.0)
        c0    = getattr(pair, 'C0_multi', 0.0)
        r0    = getattr(pair, 'r0_angstrom', 0.0)
        ratio = a_anh / pair.omega0 if pair.omega0 > 0 and a_anh > 0 else 0.0
        print(f"  {pair.ion_pair_str:<10} {pair.gamma0:>10.3e} "
              f"{pair.gamma_slope:>10.3e} {a_anh:>8.4f} "
              f"{c0:>12.3e} {ratio:>12.3e} {r0:>8.3f}")
    print()

    # Step 3: Compute spectra
    print("Computing absorption spectra and refractive index...")
    alpha_results = {}
    n_results = {}
    for T in T_list:
        alpha_total, n_refractive = compute_alpha_total(wl, T, salt)
        alpha_results[T] = alpha_total
        n_results[T] = n_refractive
    print(f"  Done. {len(T_list)} temperatures computed.")

    # Step 4: Planck-mean properties
    print("Computing Planck-mean properties...")
    T_kappa_list, kappa_P_list, n_eff_list = [], [], []
    for T in T_list:
        kp    = planck_mean_absorption(wl, alpha_results[T], T)
        n_eff = planck_mean_refractive_index(wl, n_results[T], T)
        if not np.isnan(kp) and not np.isnan(n_eff):
            T_kappa_list.append(T)
            kappa_P_list.append(kp)
            n_eff_list.append(n_eff)

    T_kappa   = np.array(T_kappa_list)
    kappa_P   = np.array(kappa_P_list)
    n_eff_arr = np.array(n_eff_list)

    print(f"\n{'--- Planck-Mean Properties ---':^50}")
    print(f"{'Temp (K)':>12} | {'kP (m-1)':>14} | {'n_eff':>14}")
    print(f"{'-'*12}-+-{'-'*14}-+-{'-'*14}")
    for T_val, kp_val, ne_val in zip(T_kappa, kappa_P, n_eff_arr):
        print(f"{T_val:>12.1f} | {kp_val:>14.2f} | {ne_val:>14.5f}")

    # Step 5: Export and plot
    print("\nExporting CSV...")
    csv_path = os.path.join(script_dir, 'Salt_RHT_Properties.csv')
    export_properties_csv(salt, T_kappa, kappa_P, n_eff_arr,
                          wl=wl, alpha_results=alpha_results,
                          n_results=n_results, filename=csv_path)

    print("Generating plots...")
    fig1 = plot_absorption_spectrum(salt, wl, T_list, alpha_results,
                                    show_all_planck=True)
    fig1.savefig(os.path.join(output_dir,
                              f"absorption_spectrum_{salt_name}_png.png"),
                 bbox_inches='tight', dpi=1200)
    fig1.savefig(os.path.join(output_dir,
                              f"absorption_spectrum_{salt_name}.pdf"),
                 bbox_inches='tight', dpi=1200)

    fig2 = plot_refractive_index(salt, wl, T_list, n_results)
    fig2.savefig(os.path.join(output_dir,
                              f"refractive_index_{salt_name}_png.png"),
                 bbox_inches='tight', dpi=1200)
    fig2.savefig(os.path.join(output_dir,
                              f"refractive_index_{salt_name}.pdf"),
                 bbox_inches='tight', dpi=1200)

    if len(T_kappa) > 1:
        fig3 = plot_planck_mean_vs_temperature(salt, T_kappa, kappa_P)
        fig3.savefig(os.path.join(output_dir,
                                  f"planck_mean_{salt_name}_png.png"),
                     bbox_inches='tight', dpi=1200)
        fig3.savefig(os.path.join(output_dir,
                                  f"planck_mean_{salt_name}.pdf"),
                     bbox_inches='tight', dpi=1200)

    return {
        'salt': salt,
        'alpha_results': alpha_results,
        'n_results': n_results,
        'T_kappa': T_kappa,
        'kappa_P': kappa_P,
        'n_eff': n_eff_arr,
        'wl': wl,
        'T_list': T_list,
    }


# =============================================================================
# 10. BATCH ANALYSIS WITH CORRELATION-BASED PREDICTION
# =============================================================================

def _get_anion_family(ion_pair_str: str) -> str:
    """Determine anion family from pair string, e.g. 'Li-F' -> 'F'."""
    anion = ion_pair_str.split('-')[-1].strip()
    families = {'F': 'F', 'Cl': 'Cl', 'Br': 'Br', 'I': 'I'}
    return families.get(anion, 'other')


def _collect_pair_data(salt: SaltConfig, salt_name: str) -> List[Dict]:
    """Collect alpha/omega_0 vs r_0 data for all pairs with fitted alpha."""
    data = []
    for pair in salt.ion_pairs:
        a_anh = getattr(pair, 'alpha_anh', 0.0)
        c0    = getattr(pair, 'C0_multi', 0.0)
        r0    = getattr(pair, 'r0_angstrom', 0.0)
        if a_anh > 0 and pair.omega0 > 0 and r0 > 0:
            data.append({
                'salt':         salt_name,
                'pair':         pair.ion_pair_str,
                'r0':           r0,
                'omega0':       pair.omega0,
                'alpha_anh':    a_anh,
                'C0_multi':     c0,
                'gamma0':       pair.gamma0,
                'gamma_slope':  pair.gamma_slope,
                'ratio':        a_anh / pair.omega0,
                'anion_family': _get_anion_family(pair.ion_pair_str),
            })
    return data


def predict_alpha_from_correlation(r0: float, omega0: float,
                                   anion_family: str,
                                   f_slope: float, f_intercept: float,
                                   cl_intercept: float) -> float:
    """
    Predict alpha from the alpha/omega_0 vs r_0 linear correlation.

    For fluorides:  alpha/omega_0 = f_slope * r0 + f_intercept  [x1e-14 s]
    For chlorides:  same slope, shifted intercept from NaCl data point.

    Returns alpha (dimensionless).
    """
    if anion_family == 'Cl':
        ratio_1e14 = f_slope * r0 + cl_intercept
    else:
        ratio_1e14 = f_slope * r0 + f_intercept

    ratio = ratio_1e14 * 1e-14      # convert from 1e-14 units to [s]
    alpha_pred = ratio * omega0      # alpha = (alpha/omega_0) * omega_0
    return max(alpha_pred, 0.1)      # floor to avoid non-physical values


def predict_missing_parameters(pair: IonPairParams,
                               all_pair_data: List[Dict],
                               f_slope: float, f_intercept: float,
                               cl_intercept: float,
                               force_predict: bool = False) -> Dict[str, str]:
    """
    Fill in missing parameters for a predicted salt's ion pair.

    Priority order for each parameter:
        1. Config value (if > 0, keep it)
        2. Same ion pair average from fitted salts
        3. Same anion family average from fitted salts
        4. Global default

    Returns a dict mapping parameter name -> source description (for logging).
    """
    r0     = getattr(pair, 'r0_angstrom', 0.0)
    family = _get_anion_family(pair.ion_pair_str)
    sources = {}

    # --- alpha_anh: predict from correlation ---
    # If force_predict is True (predicting unmeasured salts), treat config
    # values as missing so the correlation is always used.
    a_anh = getattr(pair, 'alpha_anh', 0.0) if not force_predict else 0.0
    if a_anh <= 0 and r0 > 0 and pair.omega0 > 0:
        pair.alpha_anh = predict_alpha_from_correlation(
            r0, pair.omega0, family, f_slope, f_intercept, cl_intercept)
        sources['alpha_anh'] = f'r0 correlation ({family}- family)'
    elif a_anh > 0:
        sources['alpha_anh'] = 'config'
    else:
        pair.alpha_anh = 3.0
        sources['alpha_anh'] = 'default (3.0)'

    # --- C0_multi ---
    c0 = getattr(pair, 'C0_multi', 0.0) if not force_predict else 0.0
    if c0 <= 0:
        # Try same ion pair from fitted data
        same_pair = [d['C0_multi'] for d in all_pair_data
                     if d['pair'] == pair.ion_pair_str and d['C0_multi'] > 0]
        if same_pair:
            pair.C0_multi = np.mean(same_pair)
            sources['C0_multi'] = f'same-pair avg ({len(same_pair)} salts)'
        else:
            # Fallback: same anion family
            same_fam = [d['C0_multi'] for d in all_pair_data
                        if d['anion_family'] == family and d['C0_multi'] > 0]
            if same_fam:
                pair.C0_multi = np.mean(same_fam)
                sources['C0_multi'] = f'{family}- family avg'
            else:
                pair.C0_multi = 1e4
                sources['C0_multi'] = 'default (1e4)'
    else:
        sources['C0_multi'] = 'config'

    # --- gamma0 ---
    if pair.gamma0 <= 0:
        same_pair = [d['gamma0'] for d in all_pair_data
                     if d['pair'] == pair.ion_pair_str and d['gamma0'] > 0]
        if same_pair:
            pair.gamma0 = np.mean(same_pair)
            sources['gamma0'] = f'same-pair avg ({len(same_pair)} salts)'
        else:
            same_fam = [d['gamma0'] for d in all_pair_data
                        if d['anion_family'] == family and d['gamma0'] > 0]
            if same_fam:
                pair.gamma0 = np.mean(same_fam)
                sources['gamma0'] = f'{family}- family avg'
            else:
                pair.gamma0 = 1e-6
                sources['gamma0'] = 'default (1e-6)'
    else:
        sources['gamma0'] = 'config'

    # --- gamma_slope ---
    if pair.gamma_slope <= 0:
        same_pair = [d['gamma_slope'] for d in all_pair_data
                     if d['pair'] == pair.ion_pair_str and d['gamma_slope'] > 0]
        if same_pair:
            pair.gamma_slope = np.mean(same_pair)
            sources['gamma_slope'] = f'same-pair avg ({len(same_pair)} salts)'
        else:
            same_fam = [d['gamma_slope'] for d in all_pair_data
                        if d['anion_family'] == family and d['gamma_slope'] > 0]
            if same_fam:
                pair.gamma_slope = np.mean(same_fam)
                sources['gamma_slope'] = f'{family}- family avg'
            else:
                pair.gamma_slope = 1e7
                sources['gamma_slope'] = 'default (1e7)'
    else:
        sources['gamma_slope'] = 'config'

    return sources


def build_correlation(all_pair_data: List[Dict], output_dir: str):
    """
    Build the alpha/omega_0 vs r_0 correlation from fitted data.
    Generates the correlation plot and returns regression parameters.

    Returns
    -------
    f_slope      : fluoride regression slope [1e-14 s / Angstrom]
    f_intercept  : fluoride regression intercept [1e-14 s]
    f_r2         : fluoride R^2
    cl_intercept : chloride intercept (same slope as fluoride)
    """
    f_slope, f_intercept, f_r2, cl_intercept = plot_alpha_omega0_vs_r0(
        all_pair_data, output_dir, filename='alpha_omega0_vs_r0_fitted.png')

    print(f"\n{'='*60}")
    print(f"  CORRELATION RESULTS")
    print(f"{'='*60}")
    print(f"  F- family:  a/w0 = {f_slope:.3f} * r0 + ({f_intercept:.3f})"
          f"  [x1e-14 s]  R2 = {f_r2:.4f}")
    if cl_intercept != 0:
        print(f"  Cl- family: a/w0 = {f_slope:.3f} * r0 + ({cl_intercept:.3f})"
              f"  [x1e-14 s]  (same slope, shifted)")
    print()

    return f_slope, f_intercept, f_r2, cl_intercept


def run_batch_analysis(
        salt_configs: List[Dict],
        default_wl_range: Tuple[float, float] = (0.15e-6, 80e-6),
        default_n_wavelengths: int = 800,
        default_T_range: Tuple[float, float] = (450, 1500),
        default_n_temperatures: int = 50,
        default_run_fit: bool = True,
        default_fit_config: Optional[Dict[str, Any]] = None,
        prediction_configs: Optional[List[Dict]] = None):
    """
    Two-phase batch analysis:

    Phase 1 -- Fit salts with experimental data.
    Phase 2 -- Predict parameters for unmeasured salts using the
              alpha/omega_0 vs r_0 correlation from Phase 1.

    Parameters
    ----------
    salt_configs : list of dicts
        Each dict must have 'name' (str). Optional overrides:
            'wl_range'       : (float, float) wavelength range [m]
            'n_wavelengths'  : int
            'T_range'        : (float, float) temperature range [K]
            'n_temperatures' : int
            'run_fit'        : bool
            'fit_config'     : dict (see fit_gamma_parameters docstring)

    prediction_configs : list of dicts (same format, but these salts
        will have alpha/C0/gamma predicted from the correlation).
        Optional overrides same as salt_configs.

    default_* : fallback values when not specified per-salt.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'Figures')
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================================
    # PHASE 1: FIT SALTS WITH EXPERIMENTAL DATA
    # =====================================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: FITTING SALTS WITH EXPERIMENTAL DATA")
    print("=" * 70)

    batch_results = {}
    all_pair_data = []

    for cfg in salt_configs:
        name = cfg['name']
        print(f"\n>>> Processing: {name}")
        try:
            result = run_analysis(
                salt_name=name,
                wl_range=cfg.get('wl_range', default_wl_range),
                n_wavelengths=cfg.get('n_wavelengths', default_n_wavelengths),
                T_range=cfg.get('T_range', default_T_range),
                n_temperatures=cfg.get('n_temperatures', default_n_temperatures),
                run_fit=cfg.get('run_fit', default_run_fit),
                experimental_fit_config=cfg.get('fit_config', default_fit_config),
            )
            batch_results[name] = result
            all_pair_data.extend(_collect_pair_data(result['salt'], name))
        except Exception as exc:
            print(f"  ERROR processing {name}: {exc}")
            import traceback
            traceback.print_exc()

    if not all_pair_data:
        print("\nNo fitted pair data collected. Cannot build correlation.")
        return batch_results

    # Transferability plot (fitted data only)
    plot_transferability(all_pair_data, output_dir)

    # Build correlation
    f_slope, f_intercept, f_r2, cl_intercept = build_correlation(
        all_pair_data, output_dir)

    # =====================================================================
    # PHASE 2: PREDICT SALTS USING CORRELATION
    # =====================================================================
    if not prediction_configs:
        plt.show()
        return batch_results

    print("\n" + "=" * 70)
    print("  PHASE 2: PREDICTING SALTS USING CORRELATION")
    print("=" * 70)

    predicted_pair_data = []

    for cfg in prediction_configs:
        name = cfg['name']
        print(f"\n>>> Predicting: {name}")

        if name not in SALT_REGISTRY:
            print(f"  ERROR: '{name}' not in SALT_REGISTRY. Skipping.")
            continue

        salt = SALT_REGISTRY[name]

        # Initialize to get omega_0 and r_0 from PDF
        try:
            initialize_salt(salt)
            calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
        except Exception as exc:
            print(f"  ERROR initializing {name}: {exc}")
            continue

        # Predict missing parameters for each pair
        print(f"\n  --- Predicted Parameters for {name} ---")
        print(f"  {'Pair':<10} {'Param':<14} {'Value':>12}   {'Source'}")
        print(f"  {'-'*10} {'-'*14} {'-'*12}   {'-'*30}")

        for pair in salt.ion_pairs:
            sources = predict_missing_parameters(
                pair, all_pair_data,
                f_slope, f_intercept, cl_intercept,
                force_predict=True)

            for param_name, source in sources.items():
                val = getattr(pair, param_name, 0.0)
                print(f"  {pair.ion_pair_str:<10} {param_name:<14} "
                      f"{val:>12.3e}   {source}")

            # Collect for the combined plot
            r0 = getattr(pair, 'r0_angstrom', 0.0)
            a_anh = getattr(pair, 'alpha_anh', 0.0)
            if r0 > 0 and pair.omega0 > 0 and a_anh > 0:
                predicted_pair_data.append({
                    'salt':         name,
                    'pair':         pair.ion_pair_str,
                    'r0':           r0,
                    'omega0':       pair.omega0,
                    'alpha_anh':    a_anh,
                    'C0_multi':     getattr(pair, 'C0_multi', 0.0),
                    'ratio':        a_anh / pair.omega0,
                    'anion_family': _get_anion_family(pair.ion_pair_str),
                })

        # Compute spectra with predicted parameters
        wl_range = cfg.get('wl_range', default_wl_range)
        n_wl     = cfg.get('n_wavelengths', default_n_wavelengths)
        T_range  = cfg.get('T_range', default_T_range)
        n_T      = cfg.get('n_temperatures', default_n_temperatures)

        wl     = np.linspace(wl_range[0], wl_range[1], n_wl)
        T_list = np.linspace(T_range[0], T_range[1], n_T)

        print(f"\n  Computing predicted spectra for {name}...")
        alpha_results = {}
        n_results = {}
        for T in T_list:
            alpha_total, n_refractive = compute_alpha_total(wl, T, salt)
            alpha_results[T] = alpha_total
            n_results[T] = n_refractive

        T_kappa_list, kappa_P_list, n_eff_list = [], [], []
        for T in T_list:
            kp    = planck_mean_absorption(wl, alpha_results[T], T)
            n_eff = planck_mean_refractive_index(wl, n_results[T], T)
            if not np.isnan(kp) and not np.isnan(n_eff):
                T_kappa_list.append(T)
                kappa_P_list.append(kp)
                n_eff_list.append(n_eff)

        T_kappa   = np.array(T_kappa_list)
        kappa_P   = np.array(kappa_P_list)
        n_eff_arr = np.array(n_eff_list)

        # Export CSV
        csv_path = os.path.join(script_dir, 'Salt_RHT_Properties.csv')
        export_properties_csv(salt, T_kappa, kappa_P, n_eff_arr,
                              wl=wl, alpha_results=alpha_results,
                              n_results=n_results, filename=csv_path)

        # Plots
        fig1 = plot_absorption_spectrum(salt, wl, T_list, alpha_results,
                                        show_all_planck=True)
        fig1.savefig(os.path.join(output_dir,
                                  f"absorption_spectrum_{name}_predicted.png"),
                     bbox_inches='tight', dpi=1200)

        fig2 = plot_refractive_index(salt, wl, T_list, n_results)
        fig2.savefig(os.path.join(output_dir,
                                  f"refractive_index_{name}_predicted.png"),
                     bbox_inches='tight')

        if len(T_kappa) > 1:
            fig3 = plot_planck_mean_vs_temperature(salt, T_kappa, kappa_P)
            fig3.savefig(os.path.join(output_dir,
                                      f"planck_mean_{name}_predicted.png"),
                         bbox_inches='tight', dpi=1200)

        batch_results[name + '_predicted'] = {
            'salt': salt,
            'alpha_results': alpha_results,
            'n_results': n_results,
            'T_kappa': T_kappa,
            'kappa_P': kappa_P,
            'n_eff': n_eff_arr,
            'wl': wl,
            'T_list': T_list,
        }

    # =====================================================================
    # Combined correlation plot (fitted + predicted)
    # =====================================================================
    plot_alpha_omega0_vs_r0(
        all_pair_data, output_dir,
        predicted_data=predicted_pair_data,
        filename='alpha_omega0_vs_r0_with_predictions.png')

    # Print prediction summary
    print(f"\n{'='*60}")
    print(f"  PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Salt':<12} {'Pair':<10} {'r0 [A]':>8} "
          f"{'alpha':>8} {'C0 [m-1]':>12} {'a/w0 [s]':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*12}")
    for d in predicted_pair_data:
        print(f"  {d['salt']:<12} {d['pair']:<10} {d['r0']:>8.3f} "
              f"{d['alpha_anh']:>8.4f} {d['C0_multi']:>12.3e} "
              f"{d['ratio']:>12.3e}")

    plt.show()
    return batch_results


# =============================================================================
# ENTRY POINT
# =============================================================================
# Configuration guide:
#
# FITTED_SALTS -- salts with experimental absorption data to fit against.
#   Required keys:
#     'name'       : str -- must match a name in SALT_REGISTRY
#   Optional keys (override defaults):
#     'wl_range'       : (float, float) -- wavelength range in meters
#     'n_wavelengths'  : int
#     'T_range'        : (float, float) -- temperature range in Kelvin
#     'n_temperatures' : int
#     'run_fit'        : bool -- whether to run the optimizer
#     'fit_config'     : dict with keys:
#         'mode'          : 'single' | 'weighted_average' | 'none'
#         'fit_target'    : 'damping' | 'multiphonon' | 'all' | list
#                           list example: ['gamma_slope', 'alpha_anh', 'C0_multi']
#         'dataset_index' : int (for mode='single')
#         'weights'       : list of float (for mode='weighted_average')
#
# PREDICTED_SALTS -- salts without experimental data; alpha and C0 will
#   be predicted from the correlation built in Phase 1.
#   Same optional keys as FITTED_SALTS (except 'run_fit' and 'fit_config'
#   are not used -- parameters come from the correlation).

if __name__ == "__main__":

    # -----------------------------------------------------------------
    # PHASE 1: Salts with experimental data to fit
    # -----------------------------------------------------------------
    FITTED_SALTS = [
        {
            'name': 'LiF',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (1121, 1600),
            'run_fit': True,
            'fit_config': {
                'mode': 'weighted_average',
                'fit_target': ['gamma_slope', 'alpha_anh', 'C0_multi'],
                'weights': [0.5, 0.5, 0],
            },
        },
        {
            'name': 'NaCl',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (1073.8, 1600),
            'run_fit': True,
            'fit_config': {
                'mode': 'single',
                'dataset_index': 0,
                'fit_target': ['gamma_slope', 'alpha_anh', 'C0_multi'],
            },
        },
        {
            'name': 'FLiNaK',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (737, 1300),
            'run_fit': True,
            'fit_config': {
                'mode': 'single',
                'dataset_index': 0,
                'fit_target': ['gamma_slope', 'alpha_anh', 'C0_multi'],
            },
        },
        {
            'name': 'FLiBe',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (732, 1300),
            'run_fit': True,
            'fit_config': {
                'mode': 'single',
                'dataset_index': 0,
                'fit_target': ['gamma_slope', 'alpha_anh', 'C0_multi'],
            },
        },
    ]

    # -----------------------------------------------------------------
    # PHASE 2: Salts to predict using the correlation
    # -----------------------------------------------------------------
    PREDICTED_SALTS = [
        {
            'name': 'NaF',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (1268, 1600),
            'n_temperatures': 50,
        },
        {
            'name': 'KF',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (1131, 1600),
            'n_temperatures': 50,
        },
        {
            'name': 'LiCl',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (878, 1600),
            'n_temperatures': 50,
        },
        {
            'name': 'KCl',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (1042.7, 1600),
            'n_temperatures': 50,
        },
        {
            'name': 'NaF-UF$_4$',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (900, 1600),
            'n_temperatures': 50,
        },
        {
            'name': 'FLiNa+UF$_4$',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (737, 1600),
            'n_temperatures': 50,
        },
        # {
        #     'name': 'FLiNaK_UF4',
        #     'wl_range': (0.15e-6, 160e-6),
        #     'T_range': (763, 1600),
        #     'n_temperatures': 50,
        # },
        {
            'name': 'NaCl-UCl$_3$',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (829, 1600),
            'n_temperatures': 50,
        },
        {
            'name': 'NaCl-KCl',
            'wl_range': (0.15e-6, 80e-6),
            'T_range': (829, 1600),
            'n_temperatures': 50,
        },
    ]

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------
    results = run_batch_analysis(
        salt_configs=FITTED_SALTS,
        prediction_configs=PREDICTED_SALTS,
        default_wl_range=(0.15e-6, 80e-6),
        default_n_wavelengths=800,
        default_T_range=(450, 1500),
        default_n_temperatures=50,
    )
