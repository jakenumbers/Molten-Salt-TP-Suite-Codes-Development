import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ase.io import iread
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def main():
    # -------------------------------------------------------------
    # Shared SCL Trajectory Analyzer
    # Usage: python trajectory.py --system Output/0.5NaCl-0.5KCl_930K
    # Runs the SCL analysis inside the given system folder, and
    # dumps the SCL plot into Output/Plots/<system_folder>/
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Shared SCL Trajectory Analyzer (runs on a system folder)")
    parser.add_argument('--system', type=str, required=True,
                        help='Path to the system folder, e.g. Output/0.5NaCl-0.5KCl_930K')
    parser.add_argument('--traj', type=str, default=None,
                        help='Path to trajectory file (default: auto-detect main .extxyz in system folder)')
    parser.add_argument('--comp', type=str, default=None,
                        help='System name for the output file (default: auto-detect from folder name)')
    parser.add_argument('--temp', type=float, default=None,
                        help='Temperature of the system in K (default: auto-detect from folder name)')
    parser.add_argument('--bph', action='store_true', default=False,
                        help='Enable computed phonon transfer factor b_PH (default: off, b_PH = 0.0)')
    args = parser.parse_args()

    # -------------------------------------------------------------
    # RESOLVE SYSTEM FOLDER
    # -------------------------------------------------------------
    SYSTEM_DIR = os.path.abspath(args.system)
    if not os.path.isdir(SYSTEM_DIR):
        raise ValueError(f"System folder '{SYSTEM_DIR}' does not exist.")

    # Run inside the system folder so all data I/O is local
    os.chdir(SYSTEM_DIR)

    # -------------------------------------------------------------
    # AUTO-DETECT SYSTEM INFO FROM FOLDER NAME (Output/[salt_index]/)
    # -------------------------------------------------------------
    folder_name = os.path.basename(SYSTEM_DIR)  # e.g. "0.5NaCl-0.5KCl_1130K"

    if args.comp is None and args.temp is None:
        head, _, tail = folder_name.rpartition('_')
        if head and tail.endswith('K'):
            args.comp = head
            args.temp = float(tail[:-1])
            print(f"Auto-detected composition '{args.comp}' and temperature {args.temp:.1f} K "
                  f"from folder '{folder_name}'")
        else:
            raise ValueError(
                f"Could not auto-detect composition/temperature from folder name '{folder_name}'. "
                "Please provide --comp and --temp explicitly.")

    if args.traj is None:
        # Prefer the main production trajectory (exclude NVE seed trajectories)
        candidates = sorted(glob.glob('*.extxyz'))
        main_traj = [f for f in candidates if '_NVE_seed_' not in f]
        if not main_traj:
            main_traj = candidates
        if not main_traj:
            raise ValueError(f"No .extxyz trajectory files found in '{SYSTEM_DIR}'.")
        args.traj = main_traj[0]
        print(f"Auto-detected trajectory file '{args.traj}'.")

    SYSTEM_NAME = args.comp
    TEMP_K = int(args.temp)
    TRAJ_FILE = args.traj

    # -------------------------------------------------------------
    # RESOLVE OUTPUT PLOTS FOLDER (Output/Plots/<folder_name>/)
    # -------------------------------------------------------------
    plots_dir = os.path.join(os.path.dirname(SYSTEM_DIR), 'Plots', folder_name)
    os.makedirs(plots_dir, exist_ok=True)

    # =========================================================
    # PHASE 1: READ TRAJECTORY & EXTRACT TOPOLOGY
    # =========================================================
    print(f"Reading {TRAJ_FILE} into memory...")
    raw_traj = []
    try:
        for frame in iread(TRAJ_FILE):
            raw_traj.append(frame)
    except Exception as e:
        print(f"Warning: Skipping malformed frame in {TRAJ_FILE}: {e}")

    if len(raw_traj) == 0:
        raise ValueError("Trajectory file is empty or invalid.")

    # Filter to frames with consistent atom count (handles NVE seed files that
    # may contain frames from melt/equilibration stages with different sizes)
    frame0 = raw_traj[0]
    total_atoms = len(frame0)
    raw_traj = [f for f in raw_traj if len(f) == total_atoms]
    if len(raw_traj) == 0:
        raise ValueError(f"No valid {total_atoms}-atom frames found in '{TRAJ_FILE}'.")
    symbols = frame0.get_chemical_symbols()
    atom_counts = {el: list(symbols).count(el) for el in set(symbols)}
    elements = list(atom_counts.keys())
    sys_cations = [el for el in elements if el != 'Cl']

    print("\n# System Topology Detected:")
    for el, count in atom_counts.items():
        print(f"n_{el.lower()} = {count}")
    print(f"total_atoms = {total_atoms}")

    n_frames = len(raw_traj)
    boxes = np.zeros((n_frames, 6), dtype=np.float32)
    coords = np.zeros((n_frames, total_atoms, 3), dtype=np.float32)
    for i, frame in enumerate(raw_traj):
        coords[i] = frame.get_positions()
        boxes[i] = frame.cell.cellpar()

    exact_volume = boxes[0][0] * boxes[0][1] * boxes[0][2]

    # Load into MDAnalysis
    u = mda.Universe.empty(total_atoms, trajectory=True)
    u.add_TopologyAttr('name', symbols)
    u.load_new(coords, format="Memory", dimensions=boxes)

    atom_groups = {el: u.select_atoms(f'name {el}') for el in elements}
    plot_range = (0.0, 10.5)

    def get_weighted_gr(rdf_obj, N_g1, N_g2, weight, is_self=False):
        counts = rdf_obj.results.count
        r_inner = np.linspace(plot_range[0], plot_range[1], 201)[:-1]
        r_outer = np.linspace(plot_range[0], plot_range[1], 201)[1:]
        shell_volumes = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)
        density = (N_g2 - 1) / exact_volume if is_self else N_g2 / exact_volume
        expected = n_frames * N_g1 * density * shell_volumes
        with np.errstate(divide='ignore', invalid='ignore'):
            gr = np.where(expected > 0, counts / expected, 0.0)
        return rdf_obj.results.bins, gr * weight, gr

    # =========================================================
    # PHASE 2: CALCULATE RDF SPLINES
    # =========================================================
    rel_conc = {el: count / total_atoms for el, count in atom_counts.items()}

    def mock_standardize(el1, el2):
        if el1 != 'Cl' and el2 == 'Cl':
            return f"{el1}-{el2}"
        if el1 == 'Cl' and el2 != 'Cl':
            return f"{el2}-{el1}"
        return '-'.join(sorted([el1, el2]))

    raw_weights = {}
    for el1, c1 in rel_conc.items():
        for el2, c2 in rel_conc.items():
            pair = mock_standardize(el1, el2)
            raw_weights[pair] = c1 * c2

    total_w = sum(raw_weights.values())
    W_dict = {k: v / total_w for k, v in raw_weights.items()}

    print("Calculating all pairwise RDFs...")
    splines = {}
    pdf_data = {}

    for el1 in elements:
        for el2 in elements:
            pair_name = mock_standardize(el1, el2)
            if pair_name not in splines:
                is_self = (el1 == el2)
                ex_block = (1, 1) if is_self else None
                rdf_obj = rdf.InterRDF(atom_groups[el1], atom_groups[el2], nbins=200, range=plot_range, exclusion_block=ex_block)
                rdf_obj.run()

                weight = W_dict[pair_name]
                x_vals, wg_gr, raw_gr = get_weighted_gr(rdf_obj, atom_counts[el1], atom_counts[el2], weight, is_self)

                splines[pair_name] = interp1d(x_vals, wg_gr, kind='linear', bounds_error=False, fill_value=0)
                pdf_data[pair_name] = {'x': x_vals, 'y': wg_gr, 'weight': weight, 'y_raw': raw_gr}

    # =========================================================
    # PHASE 3: SCL CALCULATION
    # =========================================================
    print("\n--- SCL CALCULATION ---")
    ca_pairs = [pair for pair in pdf_data.keys() if 'Cl' in pair and pair != 'Cl-Cl']
    sum_ca_weights = sum(pdf_data[pair]['weight'] for pair in ca_pairs)

    total_weighted_scl = 0
    total_weight_norm = 0
    plot_S_data = {}

    x_spacing = x_vals[1] - x_vals[0]
    min_dist = int(0.5 / x_spacing)

    for pair in ca_pairs:
        print(f"\nAnalyzing Pair: {pair}")
        y_grid = pdf_data[pair]['y']

        # Find Peaks
        peaks, _ = find_peaks(y_grid, prominence=0.1 * np.max(y_grid) if np.max(y_grid) > 0 else 0, distance=min_dist, width=2)
        p_idx = peaks[0] if len(peaks) > 0 else np.argmax(y_grid)
        r_peak = x_vals[p_idx]
        g_peak = y_grid[p_idx]

        # Find Minima
        y_after = y_grid[p_idx:]
        mins, _ = find_peaks(-y_after, prominence=0.01 * np.max(y_grid), distance=min_dist)
        if len(mins) > 0:
            m_idx = p_idx + mins[0]
        else:
            search_end = min(len(x_vals) - 1, int(p_idx + (2.0 / x_spacing)))
            m_idx = p_idx + np.argmin(y_grid[p_idx:search_end])
        g_min = y_grid[m_idx]

        print(f"  Transfer Step (r_peak): {r_peak:.3f} A")
        print(f"  Peak Height: {g_peak:.3f} | Minimum Depth: {g_min:.3f}")

        delta_r = r_peak
        transfer_points = np.arange(delta_r, 10.5, delta_r)

        b_KF = 1.0 - (g_peak - g_min) / g_peak if g_peak > 1e-6 else 1.0
        b_KF = np.clip(b_KF, 0, 1)

        # Phonon transfer factor: computed or hardcoded to 0.0
        if args.bph:
            b_PH = 1.0 - (pdf_data[pair]['weight'] / sum_ca_weights) if sum_ca_weights > 0 else 1.0
            b_PH = np.clip(b_PH, 0, 1)
        else:
            b_PH = 0.0

        cation = pair.split('-')[0]
        cc_name = mock_standardize(cation, cation)
        has_cc = cc_name in splines

        beta_vals = []
        for m, r_m in enumerate(transfer_points, 1):
            # g_tot includes ALL pairs containing this specific cation
            g_tot = 0
            for p_name, func in splines.items():
                if cation in p_name.split('-'):
                    g_tot += func(r_m)

            g_ideal = 0
            if m % 2 != 0:
                g_ideal = splines[pair](r_m)
            else:
                if has_cc:
                    g_ideal = splines[cc_name](r_m)
                else:
                    g_ideal = pdf_data[cc_name]['weight']

            b_NI = 1.0 - (g_ideal / g_tot) if g_tot > 1e-6 else 1.0
            b_NI = np.clip(b_NI, 0, 1)

            if b_KF == 1 or b_PH == 1 or b_NI == 1:
                beta = float('inf')
            else:
                beta = (b_KF / (1 - b_KF)) + (b_PH / (1 - b_PH)) + (b_NI / (1 - b_NI))
            beta_vals.append(beta)

        S_discrete = [1.0]
        int_beta = 0
        for beta in beta_vals:
            if beta == float('inf'):
                int_beta = -float('inf')
            else:
                int_beta -= beta * delta_r
            S_discrete.append(np.exp(int_beta))

        scl_pair = delta_r * sum(S_discrete[:-1])
        RTE = pdf_data[pair]['weight'] / sum_ca_weights if sum_ca_weights > 0 else 0
        total_weighted_scl += scl_pair * RTE
        total_weight_norm += RTE
        print(f"  Individual SCL: {scl_pair:.3f} A (Weight: {RTE:.3f})")

        # Map S(r) for plotting
        S_y_grid = np.zeros_like(x_vals)
        curr_s_idx = 0
        for i, x in enumerate(x_vals):
            if curr_s_idx < len(transfer_points) and x >= transfer_points[curr_s_idx]:
                curr_s_idx += 1
            S_y_grid[i] = S_discrete[curr_s_idx] if curr_s_idx < len(S_discrete) else S_discrete[-1]
        plot_S_data[pair] = {'S_y': S_y_grid, 'transfer_points': transfer_points}

    avg_SCL = total_weighted_scl / total_weight_norm if total_weight_norm > 0 else 0
    print(f"\nFinal Average SCL: {avg_SCL:.4f} A")

    # =========================================================
    # PHASE 4: SAVE SCALAR OUTPUT (in system folder)
    # =========================================================
    bph_label = "bPH_on" if args.bph else "bPH_off"
    output_filename = f"scl_{args.comp}_{int(args.temp)}K_{bph_label}.txt"

    with open(output_filename, "w") as f:
        f.write(str(avg_SCL))
    print(f"Scalar saved successfully to {output_filename}")

    # =========================================================
    # PHASE 5: PLOTTING (saved to Output/Plots/<folder>/)
    # =========================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = cm.tab10(np.linspace(0, 1, len(sys_cations)))
    color_map = {sys_cations[i]: colors[i] for i in range(len(sys_cations))}

    # Top Plot: Weighted G(r)
    for pair in ca_pairs:
        cat = pair.split('-')[0]
        ax1.plot(x_vals, pdf_data[pair]['y'], label=f'Weighted {pair}', color=color_map[cat], linewidth=2)
        ax1.axhline(y=pdf_data[pair]['weight'], color=color_map[cat], linestyle=':', alpha=0.4)

    ax1.set_ylabel(r'Weighted $G(r)$')
    ax1.set_title(rf'{SYSTEM_NAME} MLIP Structure (Avg SCL = {avg_SCL:.2f} $\AA$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    max_y = max([max(pdf_data[pair]['y']) for pair in ca_pairs])
    ax1.set_ylim(0, max_y * 1.2)

    # Bottom Plot: S(r) probability
    for pair in ca_pairs:
        cat = pair.split('-')[0]
        ax2.plot(x_vals, plot_S_data[pair]['S_y'], label=rf'$S(r)$ {pair}', color=color_map[cat], linewidth=2, drawstyle='steps-post')
        ax2.axvline(x=avg_SCL, color='green', linestyle='-.', linewidth=2, label=rf'Avg $\ell_{{sc}} = {avg_SCL:.2f} \AA$')

    ax2.set_xlabel(r'Distance $r$ ($\AA$)')
    ax2.set_ylabel('Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10.5)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plot_filename = f'{SYSTEM_NAME}_{TEMP_K}K_SCL_Results.png'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    print(f"Plots saved to '{plot_path}'.")

    # =========================================================
    # PHASE 6: EXPORT ALL PAIRWISE PDF DATA AS CSV (in system folder)
    # =========================================================
    all_pairs_sorted = sorted(pdf_data.keys())
    n_rows = len(x_vals)

    sep = ","  # comma separator

    def write_pdf_csv(filename, data_key):
        with open(filename, "w", newline='') as f:
            # Row 1: pair names as spanning headers (one per 2-column block)
            cols = []
            for pair in all_pairs_sorted:
                cols.append(pair)
                cols.append("")
            f.write(sep.join(cols) + "\n")
            # Row 2: column sub-headers
            cols = []
            for _ in all_pairs_sorted:
                cols.append("R (A)")
                cols.append("RDF")
            f.write(sep.join(cols) + "\n")
            # Data rows
            for i in range(n_rows):
                cols = []
                for pair in all_pairs_sorted:
                    cols.append(f"{pdf_data[pair]['x'][i]:.9f}")
                    cols.append(f"{pdf_data[pair][data_key][i]:.9f}")
                f.write(sep.join(cols) + "\n")

    csv_weighted = f'{SYSTEM_NAME}_{TEMP_K}K_weighted_PDF.csv'
    write_pdf_csv(csv_weighted, 'y')
    print(f"Weighted PDF data saved to '{csv_weighted}'.")

    csv_unweighted = f'{SYSTEM_NAME}_{TEMP_K}K_unweighted_PDF.csv'
    write_pdf_csv(csv_unweighted, 'y_raw')
    print(f"Unweighted PDF data saved to '{csv_unweighted}'.")


if __name__ == "__main__":
    main()