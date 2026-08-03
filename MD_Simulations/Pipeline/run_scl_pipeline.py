import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase import units
from ase.build import bulk
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from ase.data import atomic_masses, atomic_numbers


import MDAnalysis as mda
from MDAnalysis.analysis import rdf
from ase.io import iread
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# =========================================================
# SYSTEM PARAMETERS & AUTO-SCALING LOGIC
# =========================================================
MONOVALENT = ['Li', 'Na', 'K', 'Rb', 'Cs']
DIVALENT = ['Mg', 'Ca', 'Sr', 'Ba', 'Zn']
TETRAVALENT = ['Zr'] 

def parse_composition_and_scale(comp_str):
    """Parses standard composition strings and dynamically scales to the best lattice."""
    components = comp_str.split('-')
    num_components = len(components) # Count the number of salts
    
    fractions = {}
    has_cs = False
    has_divalent = False
    
    for comp in components:
        match = re.match(r"([0-9.]+)([A-Z][a-z]?)([A-Z][a-z]?\d?)", comp)
        if not match:
            raise ValueError(f"Invalid format: {comp}. Must be like '0.417NaCl'.")
            
        frac = float(match.group(1))
        cation = match.group(2)
        fractions[cation] = frac
        
        if cation == 'Cs':
            has_cs = True
        if cation in DIVALENT:
            has_divalent = True
        
    tot_frac = sum(fractions.values())
    fractions = {k: v / tot_frac for k, v in fractions.items()}
    
    # DYNAMIC SCALING LOGIC
    if has_divalent or 'Zr' in fractions:
        structure_type = 'fluorite'
        supercell_dim = 8 if num_components >= 5 else 3
        total_cation_sites = 4 * (supercell_dim ** 3) # 4 cations per unit cell
    elif has_cs:
        structure_type = 'cesiumchloride'
        # assuming the only time Cs is in the salt will be when the huge salts are being simulated
        supercell_dim = 11 if num_components >= 4 else 5 
        total_cation_sites = 1 * (supercell_dim ** 3)
    else:
        structure_type = 'rocksalt'
        supercell_dim = 4 if num_components >= 4 else 3
        total_cation_sites = 4 * (supercell_dim ** 3) # 4 cations per unit cell
        
    atom_counts = {k: int(round(v * total_cation_sites)) for k, v in fractions.items()}
    
    diff = total_cation_sites - sum(atom_counts.values())
    if diff != 0:
        largest_cation = max(fractions, key=fractions.get)
        atom_counts[largest_cation] += diff
        
    n_cl = 0
    for cation, count in atom_counts.items():
        if cation in TETRAVALENT:
            n_cl += count * 4
        elif cation in DIVALENT:
            n_cl += count * 2
        elif cation in MONOVALENT:
            n_cl += count * 1
            
    atom_counts['Cl'] = n_cl
    
    # Return the new supercell_dim variable so the main script can use it!
    return atom_counts, structure_type, supercell_dim

def main():
    parser = argparse.ArgumentParser(description="Universal MACE MD and SCL Calculator for Chlorides")
    parser.add_argument('--comp', type=str, required=True, help='Composition e.g., "0.417NaCl-0.058KCl-0.525CaCl2"')
    parser.add_argument('--temp', type=float, required=True, help='Target production temperature in K')
    parser.add_argument('--density', type=float, required=True, help='Initial density guess in g/cm^3')
    parser.add_argument('--model', type=str, default='SuperSalt-swa.model', help='Path to MACE model')
    args = parser.parse_args()

    # Parse and scale atoms 
    atom_counts, structure_type, supercell_dim = parse_composition_and_scale(args.comp)
    
    elements = list(atom_counts.keys())
    sys_cations = [el for el in elements if el != 'Cl']
    total_atoms = sum(atom_counts.values())
    
    TEMP_K = args.temp
    SYSTEM_NAME = args.comp
    TRAJ_FILE = f'{SYSTEM_NAME}_{TEMP_K}K.extxyz'

    print("# =========================================================")
    print("# SYSTEM PARAMETERS")
    print("# =========================================================")
    print(f"TEMP_K = {TEMP_K}")
    print(f"SYSTEM_NAME = '{SYSTEM_NAME}'")
    print(f"TRAJ_FILE = '{TRAJ_FILE}'")
    print("\n# Atom Counts")
    for el, count in atom_counts.items():
        print(f"n_{el.lower()} = {count}")
    print(f"total_atoms = {total_atoms}")
    print(f"Structure Base = {structure_type.capitalize()}")

    # =========================================================
    # PHASE 0-3: MACE MD SIMULATION
    # =========================================================
    print("\n# =========================================================")
    print("# PHASE 0-3: MACE MD SIMULATION")
    print("# =========================================================")
    calc = MACECalculator(model_paths=args.model, device='cuda')

    if structure_type == 'fluorite':
        atoms = bulk('CaF2', crystalstructure='fluorite', a=5.46, cubic=True)
        atoms = atoms * (supercell_dim, supercell_dim, supercell_dim)
        base_cation_symbol = 'Ca'
        base_anion_symbol = 'F'
    elif structure_type == 'cesiumchloride':
        atoms = bulk('CsCl', crystalstructure='cesiumchloride', a=4.12, cubic=True)
        atoms = atoms * (supercell_dim, supercell_dim, supercell_dim)
        base_cation_symbol = 'Cs'
        base_anion_symbol = 'Cl'
    else:
        atoms = bulk('NaCl', crystalstructure='rocksalt', a=5.64, cubic=True)
        atoms = atoms * (supercell_dim, supercell_dim, supercell_dim)
        base_cation_symbol = 'Na'
        base_anion_symbol = 'Cl'

    base_cat_indices = [atom.index for atom in atoms if atom.symbol == base_cation_symbol]
    base_an_indices = [atom.index for atom in atoms if atom.symbol == base_anion_symbol]

    print("# Assign Cations")
    np.random.seed(42)
    np.random.shuffle(base_cat_indices)
    
    current_idx = 0
    for cation in sys_cations:
        count = atom_counts[cation]
        indices = base_cat_indices[current_idx : current_idx + count]
        for i in indices: 
            atoms[i].symbol = cation
        current_idx += count

    print(f"# Assign {atom_counts['Cl']} Cl and create vacancies")
    np.random.shuffle(base_an_indices)
    cl_indices = base_an_indices[:atom_counts['Cl']]
    delete_indices = base_an_indices[atom_counts['Cl']:]

    for i in cl_indices: 
        atoms[i].symbol = 'Cl'
    del atoms[delete_indices]

    print(f"# Scale to initial density ({args.density} g/cm3)")
    total_mass_g_mol = sum([atom_counts[el] * atomic_masses[atomic_numbers[el]] for el in elements])
    volume_cm3 = (total_mass_g_mol / 6.022e23) / args.density 
    liquid_box_length = (volume_cm3 * 1e24) ** (1/3)
    atoms.set_cell([liquid_box_length, liquid_box_length, liquid_box_length], scale_atoms=True)

    atoms.calc = calc
    if os.path.exists(TRAJ_FILE): os.remove(TRAJ_FILE)

    print("\n--- PHASE 0: GEOMETRY OPTIMIZATION ---", flush=True)
    opt = FIRE(atoms)
    opt.run(fmax=0.5, steps=5000)
    # =========================================================
    # OPTIONAL: MASS SCALING FOR LIGHT ATOMS (Prevents NL Thrashing)
    # =========================================================
    masses = atoms.get_masses()
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Li':
            masses[i] = 22.990  # Set Li mass to Na mass
    atoms.set_masses(masses)

    print("\n--- PHASE 1: 2500K MELT ---", flush=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=2500)
    Stationary(atoms)
    dyn_melt = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=2500, friction=0.01)
    dyn_melt.run(20000)

    # 3. CRITICAL: Restore True Mass for Production Dynamics
    print("Restoring true Lithium mass (6.94 amu) for valid dynamic calculation...", flush=True)
    true_masses = atoms.get_masses()
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Li':
            true_masses[i] = 6.941
    atoms.set_masses(true_masses)

    print(f"\n--- PHASE 2: {TEMP_K}K NPT EQUILIBRATION ---", flush=True)
    p_ext = 1.01325 * units.bar 
    dyn_npt = NPT(atoms, timestep=1.0 * units.fs, temperature_K=TEMP_K, externalstress=p_ext,
                  ttime=25.0 * units.fs, pfactor=2e6 * units.GPa * (units.fs**2))
    dyn_npt.run(25000)

    eq_vol = atoms.get_volume()
    eq_box = eq_vol ** (1/3)
    atoms.set_cell([eq_box, eq_box, eq_box], scale_atoms=True)

    print(f"\n--- PHASE 3: {TEMP_K}K NVT PRODUCTION FOR RDF (Box Size: {eq_box:.2f} Å) ---", flush=True)
    dyn_nvt = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=TEMP_K, friction=0.01)

    def write_frame(): write(TRAJ_FILE, atoms, append=True)
    dyn_nvt.attach(write_frame, interval=100) 
    dyn_nvt.run(100000) 

    # =========================================================
    # PHASE 4: SCL ALGORITHM (EXACT MATCH TO ORIGINAL)
    # =========================================================
    print("\nMD Complete! Starting SCL Analysis via MDAnalysis...", flush=True)
    print(f"Reading {TRAJ_FILE} into memory...")
    
    raw_traj = []
    try:
        for frame in iread(TRAJ_FILE):
            raw_traj.append(frame)
    except Exception as e:
        print(f"File read error: {e}")

    traj = [frame for frame in raw_traj if len(frame) == total_atoms]
    if len(traj) == 0:
        raise ValueError(f"No valid {total_atoms}-atom frames found.")

    n_frames = len(traj)
    boxes = np.zeros((n_frames, 6), dtype=np.float32)
    coords = np.zeros((n_frames, total_atoms, 3), dtype=np.float32)
    for i, frame in enumerate(traj):
        coords[i] = frame.get_positions()
        boxes[i] = frame.cell.cellpar()

    exact_volume = boxes[0][0] * boxes[0][1] * boxes[0][2]

    u = mda.Universe.empty(total_atoms, trajectory=True)
    u.add_TopologyAttr('name', traj[0].get_chemical_symbols())
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
        return rdf_obj.results.bins, gr * weight

    # Molar Fraction Weighting Scheme
    rel_conc = {el: atom_counts[el]/total_atoms for el in elements}

    def mock_standardize(el1, el2):
        if el1 != 'Cl' and el2 == 'Cl': return f"{el1}-{el2}"
        if el1 == 'Cl' and el2 != 'Cl': return f"{el2}-{el1}"
        return '-'.join(sorted([el1, el2]))

    raw_weights = {}
    for el1, c1 in rel_conc.items():
        for el2, c2 in rel_conc.items():
            pair = mock_standardize(el1, el2)
            raw_weights[pair] = c1 * c2 
            
    total_w = sum(raw_weights.values())
    W_dict = {k: v/total_w for k, v in raw_weights.items()}

    # Calculate ALL combinations dynamically
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
                x_vals, wg_gr = get_weighted_gr(rdf_obj, atom_counts[el1], atom_counts[el2], weight, is_self)
                
                splines[pair_name] = interp1d(x_vals, wg_gr, kind='linear', bounds_error=False, fill_value=0)
                pdf_data[pair_name] = {'x': x_vals, 'y': wg_gr, 'weight': weight}

    # =========================================================
    # PHASE 5: MULTI-PAIR SCL CALCULATION
    # =========================================================
    print("\n--- SCL CALCULATION ---")
    ca_pairs = [pair for pair in pdf_data.keys() if 'Cl' in pair and pair != 'Cl-Cl']
    sum_ca_weights = sum(pdf_data[pair]['weight'] for pair in ca_pairs)

    total_weighted_scl = 0
    total_weight_norm = 0
    plot_S_data = {}
    cutoffs = {}

    x_spacing = x_vals[1] - x_vals[0]
    min_dist = int(0.5 / x_spacing)

    for pair in ca_pairs:
        print(f"\nAnalyzing Pair: {pair}")
        y_grid = pdf_data[pair]['y']
        
        # Find Peaks
        peaks, _ = find_peaks(y_grid, prominence=0.1*np.max(y_grid) if np.max(y_grid) > 0 else 0, distance=min_dist, width=2)
        p_idx = peaks[0] if len(peaks) > 0 else np.argmax(y_grid)
        r_peak = x_vals[p_idx]
        g_peak = y_grid[p_idx]
        
        # Find Minima
        y_after = y_grid[p_idx:]
        mins, _ = find_peaks(-y_after, prominence=0.01*np.max(y_grid), distance=min_dist)
        if len(mins) > 0:
            m_idx = p_idx + mins[0]
        else:
            search_end = min(len(x_vals)-1, int(p_idx + (2.0/x_spacing)))
            m_idx = p_idx + np.argmin(y_grid[p_idx:search_end])
        g_min = y_grid[m_idx]

        r_min = x_vals[m_idx]
        cation = pair.split('-')[0]
        cutoffs[cation] = r_min

        print(f"  Transfer Step (r_peak): {r_peak:.3f} A")
        print(f"  First Minimum (Cutoff): {r_min:.3f} A")
        print(f"  Peak Height: {g_peak:.3f} | Minimum Depth: {g_min:.3f}")

        delta_r = r_peak
        transfer_points = np.arange(delta_r, 10.5, delta_r)

        b_KF = 1.0 - (g_peak - g_min) / g_peak if g_peak > 1e-6 else 1.0
        b_KF = np.clip(b_KF, 0, 1)

        b_PH = 1.0 - (pdf_data[pair]['weight'] / sum_ca_weights) if sum_ca_weights > 0 else 1.0
        b_PH = np.clip(b_PH, 0, 1)

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
    print(f"\nAverage SCL: {avg_SCL:.4f} A")

    # =========================================================
    # PHASE 5.5: STRUCTURAL DISTRIBUTIONS (CND & BAD)
    # =========================================================
    print("\n--- PHASE 5.5: EXTRACTING CND & BAD ---")
    cnd_records = {cat: [] for cat in cutoffs.keys()}
    bad_records = {cat: [] for cat in cutoffs.keys()}
    
    # Subsample the trajectory to save time (e.g., last 200 frames)
    sample_traj = traj[-200:] 
    
    print(f"Analyzing local geometry over {len(sample_traj)} frames...")
    
    for frame in sample_traj:
        symbols = np.array(frame.get_chemical_symbols())
        
        for cat, rcut in cutoffs.items():
            # Use ASE's optimized neighbor_list to find all pairs within the dynamic cutoff
            i, j = neighbor_list('ij', frame, rcut)
            
            cat_indices = np.where(symbols == cat)[0]
            
            for cat_idx in cat_indices:
                # Find all neighbors for this specific cation
                neighbors = j[i == cat_idx]
                
                # Filter for only Cl atoms
                cl_neighbors = neighbors[symbols[neighbors] == 'Cl']
                cn = len(cl_neighbors)
                cnd_records[cat].append(cn)
                
                # Calculate angles if Coordination Number >= 2
                if cn >= 2:
                    for idx1 in range(cn):
                        for idx2 in range(idx1 + 1, cn):
                            # get_angle(atom1, vertex, atom2, mic=True) accounts for periodic boundaries
                            angle = frame.get_angle(cl_neighbors[idx1], cat_idx, cl_neighbors[idx2], mic=True)
                            bad_records[cat].append(angle)

    # Export CND to CSV
    cnd_filename = f"CND_{args.comp}_{int(args.temp)}K.csv"
    with open(cnd_filename, "w") as f:
        f.write("Cation,CN\n")
        for cat, cns in cnd_records.items():
            for cn in cns:
                f.write(f"{cat},{cn}\n")
                
    # Export BAD to CSV
    bad_filename = f"BAD_{args.comp}_{int(args.temp)}K.csv"
    with open(bad_filename, "w") as f:
        f.write("Cation,Angle_deg\n")
        for cat, angles in bad_records.items():
            for ang in angles:
                f.write(f"{cat},{ang:.2f}\n")
                
    print(f"Saved raw CND data to {cnd_filename}")
    print(f"Saved raw BAD data to {bad_filename}")

    # =========================================================
    # PHASE 6: PLOTTING
    # =========================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(sys_cations)))
    color_map = {sys_cations[i]: colors[i] for i in range(len(sys_cations))}

    # Top Plot
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

    # Bottom Plot
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
    plt.savefig(plot_filename, dpi=300)
    # Save purely the scalar value to a text file for the dependency pipeline
    output_filename = f"scl_{args.comp}_{int(args.temp)}K.txt"
    with open(output_filename, "w") as f:
        f.write(str(avg_SCL))
    print(f"Pipeline data successfully saved to {output_filename}", flush=True)
    print(f"Plots saved to '{plot_filename}'.")

if __name__ == "__main__":
    main()
