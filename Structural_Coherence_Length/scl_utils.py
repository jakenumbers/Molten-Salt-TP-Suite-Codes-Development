"""
scl_utils.py — Shared utility functions for the SCL analysis pipeline.

Used by both pdf_prepare.py and scl_analysis.py.
"""
import os
import re
import numpy as np
from scipy.signal import find_peaks
from mendeleev import element


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
    """Standardize ion pair strings to cation-anion order (e.g., 'LiCl-KCl' -> 'Cl-Li')."""
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


def format_composition_with_subscripts(composition_str):
    """Format composition string for plots (e.g., UCl3 -> UCl₃)."""
    def replace_with_subscript(match):
        el = match.group(1)
        num = match.group(2)
        if num:
            subs = ''.join(['₀₁₂₃₄₅₆₇₈₉'[int(d)] for d in num])
            return f"{el}{subs}"
        return el
    parts = composition_str.split('-')
    formatted = [re.sub(r'([A-Z][a-z]?)(\d*)', replace_with_subscript, p) for p in parts]
    return '-'.join(formatted)


def estimate_fwhm(x, y):
    """Estimate FWHM of the first peak in g(r) via half-maximum interpolation.

    Returns:
        fwhm in same units as x, or None if no peak found.
    """
    peaks, _ = find_peaks(y, prominence=0.05 * np.max(y) if np.max(y) > 0 else 0, distance=10)
    if len(peaks) == 0:
        return None
    p_idx = peaks[0]
    half_max = y[p_idx] / 2.0

    left = np.where(y[:p_idx] <= half_max)[0]
    x_left = x[left[-1]] if len(left) > 0 else x[0]

    right = np.where(y[p_idx:] <= half_max)[0]
    x_right = x[p_idx + right[0]] if len(right) > 0 else x[-1]

    return x_right - x_left


def parse_composition(comp_str):
    """Parse composition string into (fractions, ion_counts, sorted_comp_str).

    Example: '0.5NaCl-0.5KCl' -> ({NaCl: 0.5, KCl: 0.5}, {NaCl: {Na:1,Cl:1}, ...}, '0.5NaCl-0.5KCl')
    """
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
