"""
pdf_prepare.py — Prepare raw partial PDF CSV files for SCL analysis.

Reads raw PDF CSVs (with columns: pair_name, unit, r1, g1, r2, g2, ...),
standardises ion pair names, sorts/cleans, extends zeros at r=0,
applies optional FWHM-adaptive Savgol smoothing, interpolates all pairs
onto a common grid, and writes prepared CSVs into Prepared_PDF_CSV/.

Usage:
    python pdf_prepare.py          (runs all entries defined in main())
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from scl_utils import (
    get_scl_dir,
    standardize_ion_pair,
    parse_composition,
    estimate_fwhm,
)

# ==========================================
# Configuration
# ==========================================
COMMON_GRID_POINTS = 2000


# ==========================================
# Core preparation logic
# ==========================================

def _find_raw_csv(filename):
    """Locate the raw PDF CSV in expected directories."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    candidates = [
        os.path.join(repo_root, "PDF_Analysis", "PDF_CSV", filename),
        os.path.join(script_dir, "PDF_CSV", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    print(f"Error: Raw file {filename} not found in:")
    for c in candidates:
        print(f"  - {c}")
    return None


def _determine_savgol_params(x, y, user_window=None, user_polyorder=None):
    """Auto-determine Savgol parameters from FWHM with optional user overrides.

    Returns (window_length, polyorder) or None if smoothing should be skipped.
    """
    auto_wl = None
    auto_po = 3

    fwhm = estimate_fwhm(x, y)
    if fwhm is not None and len(x) > 1:
        dx = x[1] - x[0]
        if dx > 0:
            auto_wl = int(round(fwhm / (2 * dx)))
            auto_wl = max(auto_wl, 5)
            if auto_wl % 2 == 0:
                auto_wl += 1
            print(f"      Auto Savgol: FWHM={fwhm:.4f} A, dx={dx:.5f} A -> wl={auto_wl}")

    wl = user_window if user_window is not None else auto_wl
    po = user_polyorder if user_polyorder is not None else auto_po

    if wl is None:
        print(f"      Savgol skipped: could not determine window (no peak found)")
        return None

    if wl < 5:
        wl = 5
    if wl % 2 == 0:
        wl += 1
    if po >= wl:
        po = wl - 1
    po = max(po, 1)

    override = user_window is not None or user_polyorder is not None
    tag = "user override" if override else "auto"
    print(f"      Savgol ({tag}): wl={wl}, po={po}")
    return (wl, po)


def prepare_pdf(
    raw_filename,
    comp_str,
    source,
    temp,
    apply_savgol=False,
    savgol_window_length=None,
    savgol_polyorder=None,
    grid_points=COMMON_GRID_POINTS,
):
    """Read a raw PDF CSV, clean/standardise, optionally smooth, grid, and save.

    Parameters
    ----------
    raw_filename : str
        Filename of raw CSV in PDF_CSV/.
    comp_str : str
        Composition string, e.g. '0.5NaCl-0.5KCl'.
    source : str
        Literature source tag.
    temp : float
        Temperature in K.
    apply_savgol : bool
        Whether to apply Savgol smoothing.
    savgol_window_length, savgol_polyorder : int or None
        User overrides; None = auto from FWHM.
    grid_points : int
        Number of points on the common interpolation grid.

    Returns
    -------
    output_path : str or None
        Path to the written prepared CSV, or None on failure.
    """
    path = _find_raw_csv(raw_filename)
    if path is None:
        return None

    fractions, ion_counts, comp = parse_composition(comp_str)
    print(f"\n  Preparing: {comp}  ({source}, {temp}K)")
    print(f"    Raw file: {os.path.basename(path)}")

    df = pd.read_csv(path, header=None)

    # ------------------------------------------------------------------
    # 1. Read each ion-pair column, clean, standardise
    # ------------------------------------------------------------------
    pair_data = {}  # name -> (x, y)
    for i in range(0, df.shape[1], 2):
        raw_name = df.iloc[0, i]
        if pd.isna(raw_name):
            continue

        name = standardize_ion_pair(raw_name)
        x = df.iloc[2:, i].dropna().astype(float).values
        y = df.iloc[2:, i + 1].dropna().astype(float).values

        # Sort by r
        idx = np.argsort(x)
        x, y = x[idx], y[idx]

        # Remove duplicate r values
        x, u_idx = np.unique(x, return_index=True)
        y = y[u_idx]

        # Zero-extend at r = 0
        if x[0] > 0.1:
            x_ext = np.linspace(0, x[0], int(x[0] * 20))
            x = np.concatenate([x_ext[:-1], x])
            y = np.concatenate([np.zeros(len(x_ext) - 1), y])

        # ----------------------------------------------------------
        # 2. Adaptive Savgol smoothing (per pair)
        # ----------------------------------------------------------
        if apply_savgol and len(y) > 5:
            print(f"    Savgol for {name}:")
            params = _determine_savgol_params(x, y, savgol_window_length, savgol_polyorder)
            if params is not None:
                wl, po = params
                nonzero_idx = np.where(y > 0.001)[0]
                if len(nonzero_idx) > wl:
                    start = nonzero_idx[0]
                    y[start:] = savgol_filter(y[start:], window_length=wl, polyorder=po, mode='nearest')

        # Clip negative values
        y = np.maximum(y, 0)
        pair_data[name] = (x, y)

    if not pair_data:
        print(f"    WARNING: no ion pairs found in {raw_filename}")
        return None

    # ------------------------------------------------------------------
    # 3. Interpolate all pairs onto a common grid
    # ------------------------------------------------------------------
    max_x = min(np.max(x_arr) for x_arr, _ in pair_data.values())
    x_grid = np.linspace(0, max_x, grid_points)

    grid_df = pd.DataFrame({'r (A)': x_grid})
    for name, (x, y) in sorted(pair_data.items()):
        spline = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
        grid_df[name] = spline(x_grid)

    # ------------------------------------------------------------------
    # 4. Write prepared CSV
    # ------------------------------------------------------------------
    out_dir = os.path.join(get_scl_dir(), 'Prepared_PDF_CSV')
    os.makedirs(out_dir, exist_ok=True)

    safe_source = ''.join(c if c.isalnum() else '_' for c in source.split(',')[0].strip())
    out_name = f"{comp.replace('-', '_')}_{safe_source}_{int(temp)}K.csv"
    out_path = os.path.join(out_dir, out_name)

    # Write metadata as comment-style header rows
    with open(out_path, 'w', newline='') as f:
        f.write(f"# composition,{comp}\n")
        f.write(f"# source,{source}\n")
        f.write(f"# temperature_K,{temp}\n")
        f.write(f"# raw_file,{raw_filename}\n")
        f.write(f"# savgol,{apply_savgol}\n")
        if savgol_window_length is not None:
            f.write(f"# savgol_window_length,{savgol_window_length}\n")
        if savgol_polyorder is not None:
            f.write(f"# savgol_polyorder,{savgol_polyorder}\n")
        f.write(f"# grid_points,{grid_points}\n")

    grid_df.to_csv(out_path, index=False, mode='a')
    print(f"    -> Saved: {out_path}")
    return out_path


# ==========================================
# Batch runner
# ==========================================

class PDFPreparer:
    """Batch prepare multiple raw PDF files."""

    def __init__(self):
        self.jobs = []

    def add(self, raw_filename, comp_str, source, temp,
            apply_savgol=False, savgol_window_length=None, savgol_polyorder=None):
        self.jobs.append(dict(
            raw_filename=raw_filename,
            comp_str=comp_str,
            source=source,
            temp=temp,
            apply_savgol=apply_savgol,
            savgol_window_length=savgol_window_length,
            savgol_polyorder=savgol_polyorder,
        ))

    def run(self):
        print(f"=== PDF Preparation: {len(self.jobs)} file(s) ===")
        results = []
        for job in self.jobs:
            out = prepare_pdf(**job)
            results.append(out)
        n_ok = sum(1 for r in results if r is not None)
        print(f"\n=== Done: {n_ok}/{len(self.jobs)} prepared successfully ===")
        return results


# ==========================================
# Main — define all salts to prepare
# ==========================================

def main():
    prep = PDFPreparer()

    # Unary Salts
    prep.add('LiF_Walz_2019_1121.0_PIM.csv', "1.0LiF", 'Walz, 2019', 1121, apply_savgol=True)
    prep.add('NaF_Walz_2019_1266.0_PIM.csv', "1.0NaF", 'Walz, 2019', 1266, apply_savgol=True)
    prep.add('KF_Walz_2019_1131.0_PIM.csv', "1.0KF", 'Walz, 2019', 1131, apply_savgol=True)
    prep.add('LiCl_Walz_2019_878.0_PIM.csv', "1.0LiCl", 'Walz, 2019', 878, apply_savgol=True)
    prep.add('NaCl_Lu_2021_1200.0_PIM.csv', "1.0NaCl", 'Lu, 2021', 1200, apply_savgol=True)
    prep.add('KCl_Walz_2019_1043.0_PIM.csv', "1.0KCl", 'Walz, 2019', 1043, apply_savgol=True)
    prep.add('MgCl2_Roy_2021_1073.0_AP.csv', "1.0MgCl2", 'Roy, 2021', 1073, apply_savgol=True)
    prep.add('CaCl2_Bu_2021_1100.0_AP.csv', "1.0CaCl2", 'Bu, 2021', 1100, apply_savgol=True)
    prep.add('SrCl2_McGreevy_1987_1198.0_Exp.csv', "1.0SrCl2", 'McGreevy, 1987', 1198, apply_savgol=True)

    # Mixtures
    prep.add('0.6LiF-0.4NaF_Grizzi_2024_1473.0_AP.csv', "0.6LiF-0.4NaF", 'Grizzi, 2024', 1473, apply_savgol=True)
    prep.add('0.5LiF-0.5BeF2_Sun_2024_900.0_AP.csv', "0.5LiF-0.5BeF2", 'Sun, 2024', 900, apply_savgol=True)
    prep.add('0.66LiF-0.34BeF2_Fayfar_2024_973.0_AP.csv', "0.66LiF-0.34BeF2", 'Fayfar, 2024', 973, apply_savgol=True)
    prep.add('0.5LiCl-0.5KCl_Jiang_2016_727.0_RIM.csv', "0.5LiCl-0.5KCl", 'Jiang, 2016', 727, apply_savgol=True)
    prep.add('0.637LiCl-0.363KCl_Jiang_2016_750.0_RIM.csv', "0.637LiCl-0.363KCl", 'Jiang, 2016', 750, apply_savgol=True)
    prep.add('0.5NaCl-0.5KCl_Manga_2014_1100.0_RIM.csv', "0.5NaCl-0.5KCl", 'Manga, 2014', 1100, apply_savgol=True)
    prep.add('0.7LiCl-0.3CaCl2_Liang_2024_1073.0_RIM.csv', "0.7LiCl-0.3CaCl2", 'Liang, 2024', 1073, apply_savgol=True)
    prep.add('0.4903NaCl-0.5097CaCl2_Wei_2022_1023.0_RIM.csv', "0.4903NaCl-0.5097CaCl2", 'Wei, 2022', 1023, apply_savgol=True)
    prep.add('0.718KCl-0.282CaCl2_Wei_2022_1300.0_RIM.csv', "0.718KCl-0.282CaCl2", 'Wei, 2022', 1300, apply_savgol=True)
    prep.add('0.465LiF-0.115NaF-0.42KF_Frandsen_2020_873.0_AP.csv', "0.465LiF-0.115NaF-0.42KF", 'Frandsen, 2020', 873, apply_savgol=True)
    prep.add('0.345NaF-0.065MgF2-0.59KF_Solano_2021_1073.0_AP.csv', "0.345NaF-0.59KF-0.065MgF2", 'Solano, 2021', 1073, apply_savgol=True)
    prep.add('0.45MgCl2-0.33NaCl-0.22KCl_Jiang_2024_750.0_PIM.csv', "0.45MgCl2-0.33NaCl-0.22KCl", 'Jiang, 2024', 750, apply_savgol=True)
    prep.add('0.38MgCl2-0.21NaCl-0.41KCl_Jiang_2024_750.0_PIM.csv', "0.38MgCl2-0.21NaCl-0.41KCl", 'Jiang, 2024', 750, apply_savgol=True)
    prep.add('0.417NaCl-0.058KCl-0.525CaCl2_Wei_2022_1023.0_RIM.csv', "0.417NaCl-0.525CaCl2-0.058KCl", 'Wei, 2022', 1023, apply_savgol=True)
    prep.add('0.535NaCl-0.315MgCl2-0.15CaCl2_Wei_2022_1023.0_RIM.csv', "0.535NaCl-0.315MgCl2-0.15CaCl2", 'Wei, 2022', 1023, apply_savgol=True)

    # Actinides
    prep.add('ThF4_Dai_2015_1633.0_PIM.csv', "1.0ThF4", 'Dai, 2015', 1633, apply_savgol=True)
    prep.add('UF4_OcadizFlores_2021_1357.0_PIM.csv', "1.0UF4", 'OcadizFlores, 2021', 1357, apply_savgol=True)
    prep.add('0.64NaCl-0.36UCl3_Andersson_2022_1250.0_AP.csv', "0.64NaCl-0.36UCl3", 'Andersson, 2022', 1250, apply_savgol=True)
    prep.add('0.85KCl-0.15UCl3_Andersson_2024_1250.0_AP.csv', "0.85KCl-0.15UCl3", 'Andersson, 2024', 1250, apply_savgol=True)
    prep.add('0.75KCl-0.25UCl3_Andersson_2024_1250.0_AP.csv', "0.75KCl-0.25UCl3", 'Andersson, 2024', 1250, apply_savgol=True)
    prep.add('0.65KCl-0.35UCl3_Andersson_2024_1250.0_AP.csv', "0.65KCl-0.35UCl3", 'Andersson, 2024', 1250, apply_savgol=True)
    prep.add('0.5KCl-0.5UCl3_Andersson_2024_1250.0_AP.csv', "0.5KCl-0.5UCl3", 'Andersson, 2024', 1250, apply_savgol=True)
    prep.add('0.66LiF-0.34BeF2_Yin_2025_973.0_AP.csv', "0.66LiF-0.34BeF2", 'Yin, 2025', 973, apply_savgol=True)
    prep.add('0.625LiF-0.3125BeF2-0.0625ThF4_Yin_2025_973.0_AP.csv', "0.625LiF-0.3125BeF2-0.0625ThF4", 'Yin, 2025', 973, apply_savgol=True)
    prep.add('0.60LiF-0.30BeF2-0.10ThF4_Yin_2025_973.0_AP.csv', "0.60LiF-0.30BeF2-0.10ThF4", 'Yin, 2025', 973, apply_savgol=True)
    prep.add('0.5455LiF-0.2727BeF2-0.1818ThF4_Yin_2025_973.0_AP.csv', "0.5455LiF-0.2727BeF2-0.1818ThF4", 'Yin, 2025', 973, apply_savgol=True)
    prep.add('0.5454LiF-0.3636NaF-0.091UF4_Grizzi_2024_1473.0_AP.csv', "0.5454LiF-0.3636NaF-0.091UF4", 'Grizzi, 2024', 1473, apply_savgol=True)
    prep.add('0.78NaF-0.22UF4_Zhang_2026_900.0_AP.csv', "0.78NaF-0.22UF4", '900K-AIMD-Zhang, 2026', 900, apply_savgol=True)
    prep.add('0.78NaF-0.22UF4_Zhang_2026_900.0_PIM.csv', "0.78NaF-0.22UF4", '900K-CMD-Zhang, 2026', 900, apply_savgol=True)
    prep.add('0.78NaF-0.22UF4_Zhang_2026_1000.0_PIM.csv', "0.78NaF-0.22UF4", '1000K-CMD-Zhang, 2026', 1000, apply_savgol=True)
    prep.add('0.78NaF-0.22UF4_Zhang_2026_1100.0_PIM.csv', "0.78NaF-0.22UF4", '1100K-CMD-Zhang, 2026', 1100, apply_savgol=True)
    prep.add('0.78NaF-0.22UF4_Zhang_2026_1200.0_PIM.csv', "0.78NaF-0.22UF4", '1200K-CMD-Zhang, 2026', 1200, apply_savgol=True)
    prep.add('0.57NaF-0.16KF-0.27UF4_Zhang_2026_900.0_PIM.csv', "0.57NaF-0.16KF-0.27UF4", '900K-AIMD-Zhang, 2026', 900, apply_savgol=True)
    prep.add('0.57NaF-0.16KF-0.27UF4_Zhang_2026_1000.0_PIM.csv', "0.57NaF-0.16KF-0.27UF4", '1000K-AIMD-Zhang, 2026', 1000, apply_savgol=True)
    prep.add('0.57NaF-0.16KF-0.27UF4_Zhang_2026_1100.0_PIM.csv', "0.57NaF-0.16KF-0.27UF4", '1100K-AIMD-Zhang, 2026', 1100, apply_savgol=True)
    prep.add('0.57NaF-0.16KF-0.27UF4_Zhang_2026_1200.0_PIM.csv', "0.57NaF-0.16KF-0.27UF4", '1200K-AIMD-Zhang, 2026', 1200, apply_savgol=True)
    prep.add('0.63NaCl-0.37UCl3_Zhang_2026_1100.0_AP.csv', "0.63NaCl-0.37UCl3", 'AIMD-Zhang, 2026', 1100, apply_savgol=True)

    prep.run()


if __name__ == "__main__":
    main()
