import os
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import re
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from TC_models import functionlibrary

# Import the subscript formatting function from SCL_calc.py
import sys
import os

# Add Structural_Coherence_Length folder to path
scl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Structural_Coherence_Length')
if scl_dir not in sys.path:
    sys.path.insert(0, scl_dir)
from SCL_calc import format_composition_with_subscripts


def _load_data():
    # Get paths relative to this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # All data files in REFERENCE_PROPERTIES folder
    tc_compound_data_path = os.path.join(parent_dir, 'REFERENCE_PROPERTIES', 'TC_compound_data.xlsx')
    tc_measurement_data_path = os.path.join(parent_dir, 'REFERENCE_PROPERTIES', 'TC_measurement_data.xlsx')
    mstdb_path = os.path.join(parent_dir, 'REFERENCE_PROPERTIES', 'MSTDB.csv')
    
    # SCL results in Structural_Coherence_Length folder
    scl_results_path = os.path.join(parent_dir, 'Structural_Coherence_Length', 'SCL_results.csv')
    
    TC_C_df = pd.read_excel(tc_compound_data_path)
    MSTDB_df = pd.read_csv(mstdb_path)
    TC_Measurement_df = pd.read_excel(tc_measurement_data_path)
    # Match TC_calc source of SCL data
    SCL_PDF_df = pd.read_csv(scl_results_path, encoding='latin-1')
    return TC_C_df, MSTDB_df, SCL_PDF_df, TC_Measurement_df


def _parse_composition(comp: Union[str, Dict[str, float], List[Tuple[str, float]]]) -> Tuple[List[str], List[float], str]:
    """
    Normalize composition to (compounds, mol_fracs [0-1]) and a display label like '0.6NaCl-0.4KCl'.
    Accepts:
    - '0.6NaCl-0.4KCl' (string)
    - {'NaCl':0.6,'KCl':0.4} (dict)
    - [('NaCl',0.6),('KCl',0.4)] (list of tuples)
    """
    if isinstance(comp, str):
        pairs = []
        for token in comp.split('-'):
            token = token.strip()
            # split leading float and trailing salt
            i = 0
            while i < len(token) and (token[i].isdigit() or token[i] in '.'):  # collect numeric prefix
                i += 1
            frac = float(token[:i]) if i > 0 else 0.0
            salt = token[i:]
            if not salt:
                raise ValueError(f"Invalid composition token: {token}")
            pairs.append((salt, frac))
    elif isinstance(comp, dict):
        pairs = list(comp.items())
    else:
        pairs = list(comp)

    # Function to get cation atomic number for sorting
    def get_cation_atomic_number(salt):
        try:
            from mendeleev import element
            import re

            # Extract cation from salt (e.g., 'NaCl' -> 'Na', 'LiF' -> 'Li')
            # Look for element symbol followed by optional number
            elements = re.findall(r'([A-Z][a-z]?)(\d*)', salt)
            if elements:
                cation = elements[0][0]  # First element is typically the cation
                el = element(cation)
                return el.atomic_number
        except:
            pass
        return 999  # Default high number for unknown elements

    # Sort by cation atomic number instead of alphabetically
    pairs.sort(key=lambda x: get_cation_atomic_number(x[0]))
    compounds = [p[0] for p in pairs]
    fracs = [float(p[1]) for p in pairs]
    # If provided in mol%, convert to fraction if any value > 1
    if any(f > 1.0 for f in fracs):
        fracs = [f * 0.01 for f in fracs]
    label = '-'.join([f"{f}{c}" for c, f in zip(compounds, fracs)])
    # For unary salts, omit the 1.0 from the label
    if len(compounds) == 1 and fracs[0] == 1.0:
        label = compounds[0]
    return compounds, fracs, label


def _select_scl_row(SCL_PDF_df: pd.DataFrame, scl_composition_with_source: Optional[str]) -> Optional[pd.Series]:
    """
    scl_composition_with_source like '1.0NaCl (NIST, 2023)'. Extract comp and source and match a row.
    Robustly matches using named columns if available: 'Composition' and 'Source'.
    Returns the first matching row as Series or None.
    """
    if not scl_composition_with_source:
        return None

    # Extract composition and source text from the label
    if '(' in scl_composition_with_source and ')' in scl_composition_with_source:
        comp_str = scl_composition_with_source.split(' (', 1)[0].strip()
        source = scl_composition_with_source.split('(', 1)[1].rsplit(')', 1)[0].strip()
    else:
        comp_str = scl_composition_with_source.strip()
        source = None

    # Prefer named columns if present
    comp_col = None
    src_col = None
    for c in SCL_PDF_df.columns:
        if str(c).strip().lower() == 'composition':
            comp_col = c
        if str(c).strip().lower() == 'source':
            src_col = c

    df = SCL_PDF_df.copy()
    # Normalize whitespace for robust matching
    if comp_col is not None:
        df[comp_col] = df[comp_col].astype(str).str.strip()
    else:
        # Fallback to first column
        first_col = df.columns[0]
        df[first_col] = df[first_col].astype(str).str.strip()
        comp_col = first_col
    if source is not None:
        if src_col is not None:
            df[src_col] = df[src_col].astype(str).str.strip()
        else:
            # Fallback to second column if available
            if len(df.columns) > 1:
                second_col = df.columns[1]
                df[second_col] = df[second_col].astype(str).str.strip()
                src_col = second_col

    # Helper: parse a composition label like '0.41KCl-0.38MgCl2-0.21NaCl' into a canonical dict {salt: frac}
    def _parse_comp_label(label: str) -> Dict[str, float]:
        parts = []
        for token in str(label).split('-'):
            token = token.strip()
            i = 0
            while i < len(token) and (token[i].isdigit() or token[i] in '.eE+-'):
                # allow scientific notation tokens
                if token[i] == ' ':
                    break
                i += 1
            try:
                frac = float(token[:i]) if i > 0 else float('nan')
            except Exception:
                frac = float('nan')
            salt = token[i:].strip()
            if salt:
                parts.append((salt, frac))
        # Build dict, tolerate small numeric noise
        d: Dict[str, float] = {}
        for salt, frac in parts:
            d[salt] = float(frac)
        return d

    target_comp = _parse_comp_label(comp_str)

    # Build a boolean mask for composition equality ignoring order, with small tolerance on fractions
    def _row_matches_comp(row_label: str) -> bool:
        row_comp = _parse_comp_label(row_label)
        if set(row_comp.keys()) != set(target_comp.keys()):
            return False
        for k in row_comp:
            a = row_comp[k]
            b = target_comp[k]
            # handle NaN gracefully
            if not (pd.notna(a) and pd.notna(b)):
                return False
            if abs(a - b) > 1e-6:
                return False
        return True

    # Vectorized mask over the dataframe's composition column
    comp_mask = df[comp_col].astype(str).map(_row_matches_comp)

    # Primary: exact match on both composition (order-insensitive) and source when provided
    if source is not None:
        if src_col is None:
            print(f"SCL match error: Source provided ('{source}') but no 'Source' column found in SCL file. Label='{scl_composition_with_source}'")
            return None
        rows = df[comp_mask & (df[src_col] == source)]
        if not rows.empty:
            return rows.iloc[0]
        avail = df.loc[comp_mask, src_col].unique().tolist()
        print(f"SCL exact match not found for Composition~='{comp_str}' (order-insensitive), Source='{source}'. Available sources for this composition: {avail}")
        return None

    # Secondary: match on composition only (order-insensitive) when no source provided
    rows = df[comp_mask]
    return rows.iloc[0] if not rows.empty else None


def _experimental_tc_at_melt_from_measurements(
    TC_Measurement_df: pd.DataFrame,
    sources: List[str],
    melting_temp: float,
) -> Tuple[List[str], List[float], Optional[float], Optional[float], List[float]]:
    """
    For each measurement source in TC_Measurement_df['Source'], linear-fit (TC vs T) and evaluate at melting_temp.
    Returns (sources_used, tc_at_melt_list, avg_tc_at_melt, global_min_measured_temp, tc_at_min_temp_list)
    """
    tc_at_melt_vals: List[float] = []
    tc_at_min_temp_vals: List[float] = []
    sources_used: List[str] = []
    min_temps: List[float] = []
    linear_fits: List[np.poly1d] = []

    for src in sources:
        rows = TC_Measurement_df[TC_Measurement_df['Source'] == src]
        if rows.empty:
            continue
        T = rows.iloc[:, 3].astype(float).values
        TC = rows.iloc[:, 4].astype(float).values
        if len(T) < 2:
            continue
        coeffs = np.polyfit(T, TC, 1)
        linear_fit = np.poly1d(coeffs)
        tc_at_melt = float(linear_fit(melting_temp))
        tc_at_melt_vals.append(tc_at_melt)
        sources_used.append(src)
        min_temps.append(float(np.min(T)))
        linear_fits.append(linear_fit)

    avg_tc_at_melt = float(np.mean(tc_at_melt_vals)) if tc_at_melt_vals else None
    global_min_meas_temp = float(np.min(min_temps)) if min_temps else None
    if global_min_meas_temp is not None:
        for fit in linear_fits:
            tc_at_min_temp_vals.append(float(fit(global_min_meas_temp)))
    return sources_used, tc_at_melt_vals, avg_tc_at_melt, global_min_meas_temp, tc_at_min_temp_vals


def _mstdb_lines_at_range(MSTDB_df: pd.DataFrame, requested_formulas: List[str], temp_range: Tuple[float, float]):
    """
    Recreate GUI's MSTDB selections using GUI's derived TC_only_df['Formula'] strings if provided.
    The caller should pass the exact formula strings (e.g., '1.0NaCl (Ref)') as in GUI's menu.
    Here we best-effort search in MSTDB to build A+B lines and first point is at melt temp.
    Returns list of dicts {label, T, TC, reference}
    """
    out = []
    if not requested_formulas:
        return out

    # Build a light version of TC_only_df mapping from MSTDB_df
    for _, row in MSTDB_df.iterrows():
        try:
            conductivity_A = pd.to_numeric(str(row['Thermal Conductivity (W/m K):  A + B*T(K)']), errors='coerce')
            conductivity_B = pd.to_numeric(str(row.iloc[row.index.get_loc('Thermal Conductivity (W/m K):  A + B*T(K)') + 1]), errors='coerce')
            if pd.isna(conductivity_A):
                continue
            A = float(conductivity_A)
            B = float(conductivity_B) if not pd.isna(conductivity_B) else 0.0

            df_compounds = str(row['Formula']).split('-')
            comp_str = str(row['Composition (Mole %)'])
            if '-' not in comp_str:
                comp_percents = [1.0]
                formula_label = df_compounds[0]
            else:
                comp_percents = list(map(float, comp_str.split('-')))
                formula_label = '-'.join([f"{c}{s}" for c, s in zip(comp_percents, df_compounds)])

            reference = str(row.iloc[row.index.get_loc('Thermal Conductivity (W/m K):  A + B*T(K)') + 4])
            melt_T = str(row['Melting T (K)'])

            # The GUI shows just the composition + reference as a display label; match by composition label only.
            if formula_label not in requested_formulas:
                # also allow matching prefix before any appended source the caller might pass
                if not any(str(req).startswith(formula_label) for req in requested_formulas):
                    continue

            T = np.linspace(temp_range[0], temp_range[1], 100)
            TC = A + B * T
            out.append({
                'label': formula_label,
                'reference': reference,
                'T': T,
                'TC': TC,
                'melt_tc_first_point': float(TC[0]) if len(TC) else None,
                'melt_T': melt_T,
            })
        except Exception:
            continue

    return out


def plot_tc_cli(
    composition: Union[str, Dict[str, float], List[Tuple[str, float]]],
    temp_range: Tuple[float, float],
    methods: List[str],
    scl_composition_with_source: Optional[str] = None,
    measurement_sources: Optional[List[str]] = None,
    mstdb_formulas: Optional[List[str]] = None,
    use_available_data: bool = True,
    save_results_csv: bool = False,
    show_plot: bool = False,
    output_dir: str = None,
    show_diff_comp_in_legend: bool = True,
    export_plot_data_csv: bool = False,
    *,
    existing_fig: Optional[plt.Figure] = None,
    existing_ax: Optional[plt.Axes] = None,
    shared_palette: Optional[List[Any]] = None,
    palette_offset: int = 0,
    show_composition_annotation: bool = True,
    append_composition_to_model_label: bool = False,
    finalize_plot: bool = True,
    figure_name_override: Optional[str] = None,
    legend_loc: Optional[str] = None,
    composition_label_override: Optional[str] = None,
    series_marker: Optional[str] = None,
    series_markevery: Optional[int] = None,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    ax_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Non-GUI plotting function mirroring TC_calc.CreateGraphs behavior.

    Params
    - composition: composition as string/dict/list. Fractions can be 0-1 or 0-100 (%).
    - temp_range: (T_melt, T_max). The first value is treated as melting temperature.
    - methods: list of method names as from TC_models.functionlibrary().keys().
    - scl_composition_with_source: e.g., '1.0NaCl (NIST, 2023)'. Used by present models.
    - measurement_sources: list of 'Source' names from TC_measurement_data.xlsx.
    - mstdb_formulas: list of GUI-like composition strings to draw MSTDB A+B*T lines.
    - use_available_data: if True, pass expon=1 as in GUI when radio DATA is selected; else expon=0.
    - save_results_csv: if True, append results to TC_calc_results.csv in the same directory.
    - show_plot: whether to display the plot window.
    - output_dir: directory to save the figure.
    - show_diff_comp_in_legend: if True, append experimental composition in legend when it differs from the modeled salt; if False, never append composition.
    - export_plot_data_csv: if True, export temperature and thermal conductivity data to CSV for replotting.
    """
    TC_C_df, MSTDB_df, SCL_PDF_df, TC_Measurement_df = _load_data()
    
    # Set default output_dir to TC_plots within Thermal_Conductivity folder
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TC_plots')

    compounds, mol_fracs, comp_label = _parse_composition(composition)
    T_melt = float(temp_range[0])

    # Sort compounds and fracs alphabetically (already sorted by _parse_composition)
    selected_items_sorted = compounds
    compound_values = mol_fracs

    fig = existing_fig
    ax = existing_ax

    if fig is None or ax is None:
        # Apply publication-style matplotlib settings to match GUI
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'Times New Roman'
        plt.rcParams['font.size'] = 13#12
        plt.rcParams['axes.labelsize'] = 15#14
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['xtick.labelsize'] = 13#12
        plt.rcParams['ytick.labelsize'] = 13#12
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.major.width'] = 1.5
        plt.rcParams['ytick.major.width'] = 1.5
        # Legend and small text/annotation sizing (~0.85x of tick labels)
        _tick_size = 13#12
        _small_text = int(round(0.85 * _tick_size))
        plt.rcParams['legend.frameon'] = False
        plt.rcParams['legend.fontsize'] = _small_text

        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111)

    ax.set_xlabel('Temperature [K]')
    # Use middle dots and superscripts via mathtext
    ax.set_ylabel('k [W·m$^{-1}$·K$^{-1}$]')

    # Set title with properly formatted chemical composition
    formatted_title = format_composition_with_subscripts(comp_label)
    # ax.set_title(f'{formatted_title}')  # Commented out to use text instead

    # Add composition as text in the top left corner (no border)
    if show_composition_annotation:
        ax.text(0.05, 0.95, formatted_title, transform=ax.transAxes, fontsize=15,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    estimation_methods_dict = functionlibrary()

    # Track compositions for legend labels
    model_compositions = set()
    experimental_sources = set()

    # Define consistent colors for models
    model_colors = {
        'KTM': '#377eb8',      # Blue
        'SCM': '#e41a1c',      # Red  
        'PGM': '#4daf4a'       # Green
    }
    
    # Colors and markers for experimental data points
    num_colors = len(methods) + (len(mstdb_formulas) if mstdb_formulas else 0) + (len(measurement_sources) if measurement_sources else 0)
    palette_local = shared_palette or sns.color_palette('deep', max(num_colors, 1))
    color_i = palette_offset

    # Markers for experimental data points - cycle through different shapes
    exp_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+']  # 10 different marker shapes
    marker_i = 0

    # SCL row selection
    scl_row = _select_scl_row(SCL_PDF_df, scl_composition_with_source)
    # If we found an SCL row, restrict the SCL dataframe to just that row so models that
    # internally search by composition will necessarily use the intended Source.
    SCL_PDF_df_for_models = SCL_PDF_df
    if scl_row is not None:
        try:
            SCL_PDF_df_for_models = pd.DataFrame([scl_row])
        except Exception:
            SCL_PDF_df_for_models = SCL_PDF_df
    # Debug: log which SCL source was matched for this config
    try:
        if scl_row is not None:
            # Try named column, else second column
            src_val = scl_row.get('Source', None)
            if src_val is None and len(scl_row.index) > 1:
                src_val = scl_row.iloc[1]
            print(f"SCL matched for {scl_composition_with_source}: Source=\"{src_val}\"")
        else:
            print(f"SCL matched for {scl_composition_with_source}: None (no exact match)")
    except Exception:
        pass

    # Prepare optional SCL CSV swap to force internal CSV readers to use the matched row
    scl_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Structural_Coherence_Length', 'SCL_results.csv')
    scl_backup_path = None
    if scl_row is not None and os.path.exists(scl_csv_path):
        try:
            scl_backup_path = scl_csv_path + '.bak'
            # Backup original only once per invocation (overwrite if leftover)
            shutil.copy2(scl_csv_path, scl_backup_path)
            # Write a one-row CSV with the matched SCL row
            pd.DataFrame([scl_row]).to_csv(scl_csv_path, index=False)
            print(f"Temporarily swapped SCL_results.csv to matched row for {scl_composition_with_source}")
        except Exception as e:
            print(f"Warning: failed to swap SCL_results.csv: {e}")

    # Aggregations like GUI
    model_results_store: Dict[str, Tuple[np.ndarray, Union[np.ndarray, Dict[str, np.ndarray]]]] = {}
    model_predictions_at_melt: Dict[str, float] = {}
    model_predictions_at_min_temp: Dict[str, float] = {}
    min_measured_temp: Optional[float] = None
    additional_outputs: Dict[str, Union[float, List[float], np.ndarray]] = {}
    
    # Store plot data for CSV export
    plot_data_for_csv: List[Dict[str, Any]] = []

    try:
        # Run models
        for method in methods:
            if method not in estimation_methods_dict:
                print(f"Warning: method '{method}' not available. Skipping.")
                continue
            run = estimation_methods_dict[method]

            # Always call models with the (possibly filtered) SCL dataframe so they pick the intended Source
            T, out = run(
                TC_C_df,
                MSTDB_df,
                SCL_PDF_df_for_models,
                selected_items_sorted,
                compound_values,
                temp_range,
                0, 0, 0, 0,
                1 if use_available_data else 0
            )

            # Handle dictionary outputs
            if isinstance(out, dict):
                lambda_mix_T = out['thermal_conductivity']
                if method in ['SCM', 'SCM, Mix Data']:
                    additional_outputs[f'{method}_specific_heat_m'] = out.get('specific_heat_m', '')
                    additional_outputs[f'{method}_specific_heat_prime'] = out.get('specific_heat_prime', '')
                    additional_outputs[f'{method}_sound_velocity_m'] = out.get('sound_velocity_m', '')
                    additional_outputs[f'{method}_sound_velocity_prime'] = out.get('sound_velocity_prime', '')
            else:
                lambda_mix_T = out

            model_results_store[method] = (np.array(T), np.array(lambda_mix_T) if not isinstance(lambda_mix_T, dict) else lambda_mix_T)

            # Prediction at melt temp
            if len(T) > 0:
                closest_idx = int(np.argmin(np.abs(np.array(T) - T_melt)))
                if isinstance(lambda_mix_T, dict):
                    model_predictions_at_melt[method] = float(lambda_mix_T['thermal_conductivity'][closest_idx])
                else:
                    model_predictions_at_melt[method] = float(np.array(lambda_mix_T)[closest_idx])
            else:
                model_predictions_at_melt[method] = ''

            # Plot
            model_label = method
            if append_composition_to_model_label:
                model_label = f"{model_label} ({formatted_title})"
            if composition_label_override is not None:
                model_label = composition_label_override

            # Determine base model and if it's Mix Data
            base_model = None
            is_mix_data = 'Mix Data' in method
            if 'KTM' in method:
                base_model = 'KTM'
            elif 'SCM' in method:
                base_model = 'SCM'
            elif 'PGM' in method:
                base_model = 'PGM'
            
            # Special handling for multi-composition plots: use different colors per composition
            is_multi_composition_plot = composition_label_override is not None
            if is_multi_composition_plot:
                # For multi-composition plots, use palette colors instead of consistent model colors
                model_color = palette_local[color_i % len(palette_local)]
                color_i += 1
                line_style = '-'  # Always solid for multi-composition plots
            else:
                # Regular single-composition plot logic
                # Check if both regular and Mix Data versions of this specific model family are being plotted
                both_versions_plotted = False
                if base_model:
                    # Look for both versions of this model family in methods list
                    if base_model == 'KTM':
                        regular_exists = any('KTM' in method and 'Mix Data' not in method for method in methods)
                        mix_exists = any('KTM, Mix Data' in method for method in methods)
                    elif base_model == 'SCM':
                        regular_exists = any('SCM' in method and 'Mix Data' not in method for method in methods)
                        mix_exists = any('SCM, Mix Data' in method for method in methods)
                    elif base_model == 'PGM':
                        regular_exists = any('PGM' in method and 'Mix Data' not in method for method in methods)
                        mix_exists = any('PGM, Mix Data' in method for method in methods)
                    else:
                        regular_exists = False
                        mix_exists = False
                    both_versions_plotted = regular_exists and mix_exists
                
                # Use consistent color for model, fallback to palette if not recognized
                if base_model and base_model in model_colors:
                    model_color = model_colors[base_model]
                else:
                    model_color = palette_local[color_i % len(palette_local)]
                    color_i += 1

                # Determine line style: if both versions of this model family are plotted, Mix Data is solid, regular is dashed
                if both_versions_plotted:
                    line_style = '-' if is_mix_data else '--'
                else:
                    line_style = '-'  # solid line when only one version of this model family is plotted

            marker = series_marker
            markevery = None
            if marker:
                if series_markevery is not None and series_markevery > 0:
                    markevery = series_markevery
                elif len(T) > 0:
                    markevery = max(1, len(T) // 12)

            plot_kwargs = {
                'label': model_label,
                'color': model_color,
                'linestyle': line_style,
                'linewidth': 2.0
            }
            if marker:
                plot_kwargs['marker'] = marker
                if markevery:
                    plot_kwargs['markevery'] = markevery

            if isinstance(lambda_mix_T, dict):
                ax.plot(T, lambda_mix_T['thermal_conductivity'], **plot_kwargs)
            else:
                ax.plot(T, lambda_mix_T, **plot_kwargs)

            # Collect data for CSV export
            if export_plot_data_csv:
                tc_data = lambda_mix_T['thermal_conductivity'] if isinstance(lambda_mix_T, dict) else lambda_mix_T
                for temp, tc_val in zip(T, tc_data):
                    plot_data_for_csv.append({
                        'Composition': composition,
                        'Method': method,
                        'Temperature_K': temp,
                        'Thermal_Conductivity_W_mK': tc_val
                    })

            # Track model compositions for legend logic
            model_compositions.add(comp_label)
            color_i += 1
    finally:
        # Restore original SCL CSV if we swapped it
        if scl_backup_path and os.path.exists(scl_backup_path):
            try:
                shutil.move(scl_backup_path, scl_csv_path)
                print("Restored original SCL_results.csv")
            except Exception as e:
                print(f"Warning: failed to restore SCL_results.csv: {e}")

    # MSTDB-TP lines (optional)
    mstdb_records = _mstdb_lines_at_range(MSTDB_df, mstdb_formulas or [], temp_range)
    for rec in mstdb_records:
        # Extract just the author name and year from reference for cleaner legend
        ref_parts = rec['reference'].split(',')
        if len(ref_parts) >= 2:
            author_year = f"{ref_parts[0].strip()}, {ref_parts[1].strip()}"
        else:
            author_year = rec['reference']

        ax.plot(rec['T'], rec['TC'], label=f"MSTDB: {author_year}", linestyle='--', color=palette_local[color_i % len(palette_local)] if palette_local else None)
        color_i += 1

    # Experimental measurement data -> linear fit and TC at melt
    experimental_at_melt_vals: List[float] = []
    experimental_at_min_temp_vals: List[float] = []
    exp_ref_list: List[str] = []
    min_measured_temp: Optional[float] = None
    if measurement_sources:
        used_srcs, tc_vals, avg_tc, global_min, _ = _experimental_tc_at_melt_from_measurements(TC_Measurement_df, measurement_sources, T_melt)
        
        if tc_vals:
            experimental_at_melt_vals = tc_vals
            min_measured_temp = global_min
            # Calculate experimental values at min measured temp
            if min_measured_temp is not None:
                used_srcs, tc_vals, _, global_min, _ = _experimental_tc_at_melt_from_measurements(TC_Measurement_df, measurement_sources, T_melt)
                experimental_at_min_temp_vals = [float(fit(min_measured_temp)) for fit in [np.poly1d(np.polyfit(TC_Measurement_df[TC_Measurement_df['Source'] == src].iloc[:, 3].astype(float).values, TC_Measurement_df[TC_Measurement_df['Source'] == src].iloc[:, 4].astype(float).values, 1)) for src in used_srcs]]
        
        # Use high-contrast colors suitable for professional papers, avoiding model colors (red, blue, green)
        exp_colors = ['#9467bd', '#ff7f0e', '#4daf4a', '#17becf','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',  '#ff9896', '#c5b0d5']
        exp_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+']

        for i, src in enumerate(used_srcs):
            rows = TC_Measurement_df[TC_Measurement_df['Source'] == src]
            T_exp, TC_exp = rows.iloc[:, 3].astype(float).values, rows.iloc[:, 4].astype(float).values
            coeffs = np.polyfit(T_exp, TC_exp, 1)
            fit = np.poly1d(coeffs)
            x_fit = np.linspace(min(T_exp), max(T_exp), 100)
            
            # Parse experimental source label
            md_prefix = ""
            author_name = ""
            year = ""
            exp_composition_str = ""
            
            if '(' in src and ')' in src:
                # Split composition and (author, year) parts
                parts = src.split('(', 1)
                exp_composition_str = parts[0].strip()

                # Format the composition string with subscripts for the legend
                if exp_composition_str:
                    exp_composition_str = format_composition_with_subscripts(exp_composition_str)

                # Extract (author, year) part
                author_year_part = parts[1].rstrip(')').strip()
                if ',' in author_year_part:
                    author_name = author_year_part.split(',')[0].strip()
                    # Look for year in the remaining part
                    year_match = re.search(r'(\d{4})', author_year_part)
                    if year_match:
                        year = f", {year_match.group(1)}"
                else:
                    author_name = author_year_part.strip()
                # MD prefix detection inside the (author, year; MD) section
                if 'MD' in author_year_part:
                    md_prefix = "MD: "
            else:
                # Fallback if format is different
                exp_composition_str = src.strip()
                if 'MD' in src:
                    md_prefix = "MD: "

            # Base label (without composition), with optional MD prefix
            base_label = f"{md_prefix}{author_name}{year}".strip()
            
            # Use larger symbols with distinct colors and black edges for cleaner look
            label = base_label
            ax.scatter(T_exp, TC_exp, label=label, color=exp_colors[i % len(exp_colors)], 
                      marker=exp_markers[i % len(exp_markers)], s=80, edgecolors='black', 
                      linewidths=0.6, zorder=5, alpha=0.5)
            ax.plot(x_fit, fit(x_fit), color=exp_colors[i % len(exp_colors)], 
                   linestyle=':', linewidth=2.0, alpha=1, zorder=4)

            experimental_sources.add(comp_label)
            exp_ref_list.append(src)
            marker_i += 1

        # Model predictions at minimum measured temperature across all measurement sets
        if min_measured_temp is not None:
            for method in methods:
                if method in model_results_store:
                    T_arr, lam = model_results_store[method]
                    if len(T_arr) > 0:
                        idx = int(np.argmin(np.abs(np.array(T_arr) - min_measured_temp)))
                        if isinstance(lam, dict):
                            model_predictions_at_min_temp[method] = float(lam['thermal_conductivity'][idx])
                        else:
                            model_predictions_at_min_temp[method] = float(np.array(lam)[idx])
                    else:
                        model_predictions_at_min_temp[method] = ''
                else:
                    model_predictions_at_min_temp[method] = ''
    fig_path: Optional[str] = None
    if finalize_plot:
        save_label = figure_name_override or comp_label
        fig_path = _finalize_plot(
            fig,
            ax,
            output_dir,
            save_label,
            show_plot=show_plot,
            legend_loc=legend_loc,
            legend_kwargs=legend_kwargs,
            ax_kwargs=ax_kwargs,
        )

    # Build melt_row for CSV (GUI-compatible)
    all_model_names = list(functionlibrary().keys())
    model_cols_at_melt = {name: '' for name in all_model_names}
    model_cols_at_min = {name: '' for name in all_model_names}
    for k, v in model_predictions_at_melt.items():
        if k in model_cols_at_melt:
            model_cols_at_melt[k] = v
    for k, v in model_predictions_at_min_temp.items():
        if k in model_cols_at_min:
            model_cols_at_min[k] = v

    melt_row = {
        'composition': comp_label,
        'pdf_source': (scl_composition_with_source.split('(', 1)[1].rsplit(')', 1)[0].strip() if scl_composition_with_source and '(' in scl_composition_with_source else ''),
        'melting_temp': T_melt,
        'min_measured_temp': min_measured_temp if min_measured_temp is not None else '',
        'tc_experimental': (float(np.mean(experimental_at_melt_vals)) if experimental_at_melt_vals else ''),
        'tc_experimental_at_min_temp': (
            float(np.mean(experimental_at_min_temp_vals))
            if experimental_at_min_temp_vals and min_measured_temp is not None
            else ''
        ),
        'exp_reference': ','.join(exp_ref_list) if exp_ref_list else '',
        'tc_mstdb': (float(np.mean([rec['melt_tc_first_point'] for rec in mstdb_records if rec['melt_tc_first_point'] is not None])) if mstdb_records else ''),
        'mstdb_reference': ','.join({rec['reference'] for rec in mstdb_records}) if mstdb_records else ''
    }

    # Additional outputs
    for key, val in additional_outputs.items():
        melt_row[key] = val

    # Add model columns
    for name in all_model_names:
        melt_row[f"{name}_at_melt"] = model_cols_at_melt[name]
    for name in all_model_names:
        melt_row[f"{name}_at_min_temp"] = model_cols_at_min[name]

    # Optionally save CSV with GUI header compatibility
    if save_results_csv:
        _save_results_to_csv([melt_row])

    # Export plot data to CSV if requested
    if export_plot_data_csv and plot_data_for_csv:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame
        plot_df = pd.DataFrame(plot_data_for_csv)
        
        # For each method, append data to the corresponding CSV file
        for method in plot_df['Method'].unique():
            method_data = plot_df[plot_df['Method'] == method]
            
            # Create compact format: one row per salt with comma-separated values
            compact_data = []
            
            for unique_label in method_data['Composition'].unique():
                label_data = method_data[method_data['Composition'] == unique_label]
                
                # Get temperatures and thermal conductivity values
                temps = label_data['Temperature_K'].values
                tc_values = label_data['Thermal_Conductivity_W_mK'].values
                
                # Convert to comma-separated strings
                temp_str = ','.join([f"{t:.6f}" for t in temps])
                tc_str = ','.join([f"{tc:.6f}" for tc in tc_values])
                
                compact_data.append({
                    'Salt': unique_label,
                    'Temperatures_K': temp_str,
                    'Thermal_Conductivity_W_mK': tc_str
                })
            
            # Create DataFrame for compact format
            compact_df = pd.DataFrame(compact_data)
            
            # Sanitize method name for filename (replace spaces and special characters)
            safe_method = method.replace(' ', '_').replace('-', '_').replace('.', '_')
            csv_filename = f"{safe_method}_thermal_conductivity_data.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(csv_path)
            
            # Append to CSV (write header only if file doesn't exist)
            compact_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
            print(f"Thermal conductivity data for '{method}' appended to: {csv_path}")

    return {
        'figure_path': fig_path,
        'melt_row': melt_row,
        'fig': fig,
        'ax': ax,
        'next_color_index': color_i,
    }


def _finalize_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    output_dir: str,
    save_label: str,
    *,
    show_plot: bool = False,
    legend_loc: Optional[str] = None,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    x_bounds: Optional[Tuple[float, float]] = None,
    ax_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Finalize axes styling, add axis padding/legend, and persist figure."""

    fig.tight_layout()
    
    # First, handle axis limits and ticks if provided
    if ax_kwargs:
        ticks_at_limits = bool(ax_kwargs.get('ticks_at_limits', False))
        # Apply axis limits first
        if 'xlim' in ax_kwargs:
            x_min, x_max = ax_kwargs['xlim']
            ax.set_xlim(x_min, x_max)
        if 'ylim' in ax_kwargs:
            y_min, y_max = ax_kwargs['ylim']
            ax.set_ylim(y_min, y_max)
            
        # Apply ticks if specified
        if 'xticks' in ax_kwargs:
            xticks_val = ax_kwargs['xticks']
            if isinstance(xticks_val, int):
                cur_xlim = ax.get_xlim()
                xticks_val = np.linspace(cur_xlim[0], cur_xlim[1], max(2, xticks_val))
            ax.xaxis.set_major_locator(mticker.FixedLocator(xticks_val))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        elif ticks_at_limits and 'xlim' in ax_kwargs:
            xticks_val = [x_min, x_max]
            ax.xaxis.set_major_locator(mticker.FixedLocator(xticks_val))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        if 'yticks' in ax_kwargs:
            yticks_val = ax_kwargs['yticks']
            if isinstance(yticks_val, int):
                cur_ylim = ax.get_ylim()
                yticks_val = np.linspace(cur_ylim[0], cur_ylim[1], max(2, yticks_val))
            ax.yaxis.set_major_locator(mticker.FixedLocator(yticks_val))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        elif ticks_at_limits and 'ylim' in ax_kwargs:
            yticks_val = [y_min, y_max]
            ax.yaxis.set_major_locator(mticker.FixedLocator(yticks_val))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            
        # If limits were set via ax_kwargs, we're done with auto-scaling
        if 'xlim' in ax_kwargs and 'ylim' in ax_kwargs:
            pass  # Skip auto-scaling since we've set explicit limits
        else:
            # Only auto-scale the axes that weren't explicitly set
            ax.relim()
            ax.autoscale_view()
    else:
        # Default behavior if no ax_kwargs provided
        ax.relim()
        ax.autoscale_view()

    # Only apply padding if we're not using explicit limits from ax_kwargs
    if not ax_kwargs or 'xlim' not in ax_kwargs:
        x_min, x_max = ax.dataLim.intervalx
        if x_bounds is not None and all(np.isfinite(val) for val in x_bounds):
            if not np.isfinite(x_min) or not np.isfinite(x_max):
                x_min, x_max = x_bounds
            else:
                x_min = min(x_min, x_bounds[0])
                x_max = max(x_max, x_bounds[1])

        if np.isfinite(x_min) and np.isfinite(x_max):
            x_range = x_max - x_min
            if not np.isfinite(x_range) or x_range == 0:
                x_range = max(abs(x_min), 1.0)
            x_pad = max(x_range * 0.05, 5.0)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if not ax_kwargs or 'ylim' not in ax_kwargs:
        y_min, y_max = ax.dataLim.intervaly
        if np.isfinite(y_min) and np.isfinite(y_max):
            y_range = y_max - y_min
            if not np.isfinite(y_range) or y_range == 0:
                y_range = max(abs(y_min), 0.5)
            y_pad = max(y_range * 0.1, 0.1)
            ax.set_ylim(y_min - y_pad, y_max + y_pad*2)

    legend_params: Dict[str, Any] = {
        'facecolor': 'white',
        'frameon': True,
        'framealpha': 0.7,
        'edgecolor': 'none',
        'borderpad': 0.5,
        'borderaxespad': 0.5,
        'handletextpad': 0.5,
        'columnspacing': 0.8,
        'handlelength': 1.5,
        'labelspacing': 0.3,
        'fontsize': 13.5,
    }
    if legend_kwargs:
        legend_params.update(legend_kwargs)
    if legend_loc is not None:
        legend_params['loc'] = legend_loc
    elif 'loc' not in legend_params:
        legend_params['loc'] = 'upper right'

    legend = ax.legend(**legend_params)
    if legend and legend.get_frame():
        legend.get_frame().set_linewidth(0)

    os.makedirs(output_dir, exist_ok=True)
    sanitized_label = save_label.replace(' ', '_')
    fig_path = os.path.join(output_dir, f"TC_{sanitized_label}.png")
    fig.savefig(fig_path, bbox_inches='tight', dpi=1600)
    fig_path_pdf = os.path.join(output_dir, f"TC_{sanitized_label}.pdf")
    fig.savefig(fig_path_pdf, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig_path


def plot_multi_composition_cli(
    composition_configs: List[Dict[str, Any]],
    temp_range: Optional[Tuple[float, float]] = None,
    methods: Optional[List[str]] = None,
    scl_composition_with_source: Optional[str] = None,
    measurement_sources: Optional[List[str]] = None,
    mstdb_formulas: Optional[List[str]] = None,
    use_available_data: bool = True,
    save_results_csv: bool = False,
    show_plot: bool = False,
    output_dir: str = None,
    show_diff_comp_in_legend: bool = True,
    export_plot_data_csv: bool = False,
    figure_label: Optional[str] = None,
    legend_loc: Optional[str] = None,
    ax_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Overlay multiple compositions on a single comparison plot.

    Each entry in ``composition_configs`` can override any of the standard ``plot_tc_cli``
    parameters (e.g., methods, temp_range, measurement_sources, etc.). Values provided to
    this function serve as defaults when not specified per composition.
    """

    if not composition_configs:
        raise ValueError("composition_configs must contain at least one composition entry")
    
    # Set default output_dir to TC_plots within Thermal_Conductivity folder
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TC_plots')

    # Determine a palette large enough for all series across compositions
    total_colors = 0
    for cfg in composition_configs:
        comp = cfg['composition']
        comp_compounds, _, _ = _parse_composition(comp)
        desired_method = 'SCM' if len(comp_compounds) == 1 else 'SCM, Mix Data'
        cfg_methods = [desired_method]
        cfg_measurements = cfg.get('measurement_sources', measurement_sources) or []
        cfg_mstdb = cfg.get('mstdb_formulas', mstdb_formulas) or []
        total_colors += len(cfg_methods) + len(cfg_measurements) + len(cfg_mstdb)
    palette = sns.color_palette('deep', max(total_colors, 1))

    fig: Optional[plt.Figure] = None
    ax: Optional[plt.Axes] = None
    color_index = 0
    melt_rows: List[Dict[str, Any]] = []
    comp_labels: List[str] = []
    temp_bounds: List[Tuple[float, float]] = []

    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+']

    for idx, cfg in enumerate(composition_configs):
        comp = cfg['composition']
        comp_compounds, comp_fracs, normalized_label = _parse_composition(comp)
        
        # Use custom label from configuration if provided, otherwise use formatted composition
        if 'label' in cfg:
            formatted_label = cfg['label']
        else:
            formatted_label = format_composition_with_subscripts(normalized_label)
        desired_method = 'SCM' if len(comp_compounds) == 1 else 'SCM, Mix Data'
        comp_methods = [desired_method]
        if not comp_methods:
            raise ValueError(f"Methods must be provided for composition '{comp}'")
        comp_temp_range = cfg.get('temp_range', temp_range)
        if not comp_temp_range:
            raise ValueError(f"Temperature range must be provided for composition '{comp}'")

        comp_result = plot_tc_cli(
            composition=comp,
            temp_range=comp_temp_range,
            methods=comp_methods,
            scl_composition_with_source=cfg.get('scl_composition_with_source', scl_composition_with_source),
            measurement_sources=cfg.get('measurement_sources', measurement_sources),
            mstdb_formulas=cfg.get('mstdb_formulas', mstdb_formulas),
            use_available_data=cfg.get('use_available_data', use_available_data),
            save_results_csv=cfg.get('save_results_csv', save_results_csv),
            show_plot=False,
            output_dir=output_dir,
            show_diff_comp_in_legend=cfg.get('show_diff_comp_in_legend', show_diff_comp_in_legend),
            export_plot_data_csv=cfg.get('export_plot_data_csv', export_plot_data_csv),
            existing_fig=fig,
            existing_ax=ax,
            shared_palette=palette,
            palette_offset=color_index,
            show_composition_annotation=False,
            append_composition_to_model_label=False,
            finalize_plot=False,
            figure_name_override=figure_label,
            legend_loc=legend_loc,
            composition_label_override=formatted_label,
            series_marker=marker_cycle[idx % len(marker_cycle)],
            series_markevery=cfg.get('series_markevery', 12),
            ax_kwargs=ax_kwargs,
        )

        fig = comp_result['fig']
        ax = comp_result['ax']
        color_index = comp_result['next_color_index']
        melt_rows.append(comp_result['melt_row'])
        comp_labels.append(comp_result['melt_row']['composition'])
        temp_bounds.append((float(comp_temp_range[0]), float(comp_temp_range[1])))

    assert fig is not None and ax is not None  # for type checkers

    # Harmonize x-limits using provided temperature ranges (unless explicitly specified)
    if temp_bounds:
        global_min = min(bound[0] for bound in temp_bounds)
        global_max = max(bound[1] for bound in temp_bounds)
        if not ax_kwargs or 'xlim' not in ax_kwargs:
            ax.set_xlim(global_min, global_max)

    save_label = figure_label or "__".join(comp_labels)
    legend_loc_to_use = legend_loc or 'upper left'
    legend_kwargs_final: Dict[str, Any] = {}

    fig_path = _finalize_plot(
        fig,
        ax,
        output_dir,
        save_label,
        show_plot=show_plot,
        legend_loc=legend_loc_to_use,
        legend_kwargs=legend_kwargs_final or None,
        x_bounds=(global_min, global_max),
        ax_kwargs=ax_kwargs,
    )

    return {
        'figure_path': fig_path,
        'melt_rows': melt_rows,
    }


def run_many(configs: List[Dict]):
    """
    Convenience to run many configurations in code. Each config maps directly to plot_tc_cli params.
    """
    results = []
    for cfg in configs:
        if 'compositions' in cfg:
            res = plot_multi_composition_cli(
                composition_configs=cfg['compositions'],
                temp_range=cfg.get('temp_range'),
                methods=cfg.get('methods'),
                scl_composition_with_source=cfg.get('scl_composition_with_source'),
                measurement_sources=cfg.get('measurement_sources'),
                mstdb_formulas=cfg.get('mstdb_formulas'),
                use_available_data=cfg.get('use_available_data', True),
                save_results_csv=cfg.get('save_results_csv', False),
                show_plot=cfg.get('show_plot', False),
                output_dir=cfg.get('output_dir'),
                show_diff_comp_in_legend=cfg.get('show_diff_comp_in_legend', True),
                export_plot_data_csv=cfg.get('export_plot_data_csv', False),
                figure_label=cfg.get('figure_label'),
                legend_loc=cfg.get('legend_loc'),
                ax_kwargs=cfg.get('ax_kwargs'),
            )
        else:
            res = plot_tc_cli(
                composition=cfg['composition'],
                temp_range=cfg['temp_range'],
                methods=cfg['methods'],
                scl_composition_with_source=cfg.get('scl_composition_with_source'),
                measurement_sources=cfg.get('measurement_sources'),
                mstdb_formulas=cfg.get('mstdb_formulas'),
                use_available_data=cfg.get('use_available_data', True),
                save_results_csv=cfg.get('save_results_csv', False),
                show_plot=cfg.get('show_plot', False),
                output_dir=cfg.get('output_dir'),
                show_diff_comp_in_legend=cfg.get('show_diff_comp_in_legend', True),
                export_plot_data_csv=cfg.get('export_plot_data_csv', False),
                figure_name_override=cfg.get('figure_label'),
                legend_loc=cfg.get('legend_loc'),
                ax_kwargs=cfg.get('ax_kwargs'),
            )
        results.append(res)
    return results


def _save_results_to_csv(melt_results: List[Dict]):
    """Append results to the same CSV structure as TC_calc.save_results_to_csv()."""
    import csv

    # Save results to Thermal_Conductivity folder
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TC_calc_results.csv')
    model_columns = list(functionlibrary().keys())

    # GUI header mapping
    model_columns_at_melt = [f"{col} at Melt Temp (W/m-K)" for col in model_columns]
    model_columns_at_min = [f"{col} at Min Meas Temp (W/m-K)" for col in model_columns]

    specific_heat_columns = [
        'GECM c_m (J/kg-K)',
        'GECM c\' (J/kg-K²)',
        'GECM_Mix c_m (J/kg-K)',
        'GECM_Mix c\' (J/kg-K²)'
    ]
    sound_velocity_columns = [
        'GECM v_m (m/s)',
        'GECM v\' (m/s/K)',
        'GECM_Mix v_m (m/s)',
        'GECM_Mix v\' (m/s/K)'
    ]

    header = [
        'Salt Composition', 'PDF Source', 'Melting Temp (K)', 'Min Measured Temp (K)'
    ] + model_columns_at_melt + model_columns_at_min + [
        'TC Experimental (W/m-K)', 'TC Experimental at Min Temp (W/m-K)', 'Exp. Reference', 'TC MSTDB (W/m-K)', 'MSTDB Reference'
    ] + specific_heat_columns + sound_velocity_columns

    rows = []
    for res in melt_results:
        row = [
            res.get('composition', ''),
            res.get('pdf_source', ''),
            res.get('melting_temp', ''),
            res.get('min_measured_temp', ''),
        ]
        for model in model_columns:
            row.append(res.get(f"{model}_at_melt", ''))
        for model in model_columns:
            row.append(res.get(f"{model}_at_min_temp", ''))
        row += [
            res.get('tc_experimental', ''),
            res.get('tc_experimental_at_min_temp', ''),
            res.get('exp_reference', ''),
            res.get('tc_mstdb', ''),
            res.get('mstdb_reference', ''),
        ]
        # Present Model fields map to GECM columns in GUI file
        row.extend([
            res.get('Present Model_specific_heat_m', ''),
            res.get('Present Model_specific_heat_prime', ''),
            res.get('Present Model, Mix Data_specific_heat_m', ''),
            res.get('Present Model, Mix Data_specific_heat_prime', ''),
        ])
        row.extend([
            res.get('Present Model_sound_velocity_m', ''),
            res.get('Present Model_sound_velocity_prime', ''),
            res.get('Present Model, Mix Data_sound_velocity_m', ''),
            res.get('Present Model, Mix Data_sound_velocity_prime', ''),
        ])
        rows.append(row)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)
