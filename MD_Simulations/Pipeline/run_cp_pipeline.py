import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from mace.calculators import MACECalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from ase.data import atomic_masses, atomic_numbers

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
        supercell_dim = 8 if num_components >= 4 else 3
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
    parser = argparse.ArgumentParser(description="Universal MACE Cp and Thermal Expansion Calculator")
    parser.add_argument('--comp', type=str, required=True, help='Composition e.g., "0.417NaCl-0.058KCl-0.525CaCl2"')
    parser.add_argument('--temp', type=float, required=True, help='Target Melting/Anchor Temperature in K')
    parser.add_argument('--density', type=float, required=True, help='Initial density guess in g/cm^3')
    parser.add_argument('--model', type=str, default='SuperSalt-swa.model', help='Path to MACE model')
    args = parser.parse_args()

    atom_counts, structure_type, supercell_dim = parse_composition_and_scale(args.comp)
    elements = list(atom_counts.keys())
    sys_cations = [el for el in elements if el != 'Cl']
    total_atoms = sum(atom_counts.values())
    
    ANCHOR_TEMP_K = args.temp
    SYSTEM_NAME = args.comp.replace('.', '') # Clean filename
    CSV_FILE = f"{SYSTEM_NAME}_{args.temp}K_Enthalpy_vs_T.csv"

    print("# =========================================================")
    print("# SYSTEM PARAMETERS")
    print("# =========================================================")
    print(f"ANCHOR_TEMP_K = {ANCHOR_TEMP_K}")
    print(f"SYSTEM_NAME = '{SYSTEM_NAME}'")
    print("\n# Atom Counts")
    for el, count in atom_counts.items():
        print(f"n_{el.lower()} = {count}")
    print(f"total_atoms = {total_atoms}")
    print(f"Structure Base = {structure_type.capitalize()}")

    # =========================================================
    # PHASE 0-2: MACE MD SIMULATION
    # =========================================================
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

    np.random.seed(42)
    np.random.shuffle(base_cat_indices)
    
    current_idx = 0
    for cation in sys_cations:
        count = atom_counts[cation]
        indices = base_cat_indices[current_idx : current_idx + count]
        for i in indices: atoms[i].symbol = cation
        current_idx += count

    np.random.shuffle(base_an_indices)
    cl_indices = base_an_indices[:atom_counts['Cl']]
    delete_indices = base_an_indices[atom_counts['Cl']:]

    for i in cl_indices: atoms[i].symbol = 'Cl'
    del atoms[delete_indices]

    total_mass_g_mol = sum([atom_counts[el] * atomic_masses[atomic_numbers[el]] for el in elements])
    volume_cm3 = (total_mass_g_mol / 6.022e23) / args.density 
    liquid_box_length = (volume_cm3 * 1e24) ** (1/3)
    atoms.set_cell([liquid_box_length, liquid_box_length, liquid_box_length], scale_atoms=True)

    atoms.calc = calc
    
    with open(CSV_FILE, "w") as f:
        f.write("Temperature_K,Enthalpy_eV_per_atom,Volume_A3\n")

    print("\n--- PHASE 0: GEOMETRY OPTIMIZATION ---", flush=True)
    atoms.rattle(stdev=0.1)
    opt = FIRE(atoms)
    opt.run(fmax=1.0, steps=1000)

    # MASS SCALING FOR LIGHT ATOMS (Prevents NL Thrashing)
    masses = atoms.get_masses()
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Li':
            masses[i] = 22.990 
    atoms.set_masses(masses)

    print("\n--- PHASE 1: 2500K MELT ---", flush=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=2500)
    Stationary(atoms)
    dyn_melt = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=2500, friction=0.01)
    dyn_melt.run(10000)

    # 3. CRITICAL: Restore True Mass for Production Dynamics
    print("Restoring true Lithium mass (6.94 amu) for valid dynamic calculation...", flush=True)
    true_masses = atoms.get_masses()
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Li':
            true_masses[i] = 6.941
    atoms.set_masses(true_masses)

    # =========================================================
    # DYNAMIC TEMPERATURE SWEEP LOGIC
    # =========================================================
    MAX_SAFE_TEMP = 1550.0  # Safe upper limit to prevent boiling/vaporization
    
    # Check if a standard 500 K jump exceeds our safe limit
    if (ANCHOR_TEMP_K + 500.0) <= MAX_SAFE_TEMP:
        # We have plenty of room. Use standard 100 K steps.
        temperatures = [ANCHOR_TEMP_K + (i * 100) for i in range(5, -1, -1)]
        p_ext = 1.01325 * units.bar 
    else:
        # Squeeze the 6 points into whatever safe window we have left
        window = MAX_SAFE_TEMP - ANCHOR_TEMP_K
        
        # Failsafe: If the anchor itself is above the max safe temp!
        if window <= 0:
            raise ValueError(f"Anchor temp ({ANCHOR_TEMP_K} K) is near or above the boiling limit!")
            
        step_size = window / 5.0
        
        # Generate the temperatures and round to 1 decimal place for clean files
        temperatures = [round(ANCHOR_TEMP_K + (i * step_size), 1) for i in range(5, -1, -1)]
        p_ext = 1.01325 * units.bar 
        
        print(f"\nWARNING: Standard sweep exceeds safe boiling limit.")
        print(f"Dynamically squeezed step size to {step_size:.1f} K to fit 6 points.")

    dyn_npt = NPT(atoms, timestep=1.0 * units.fs, temperature_K=temperatures[0], externalstress=p_ext,
                  ttime=25.0 * units.fs, pfactor=2e6 * units.GPa * (units.fs**2))

    for T in temperatures:
        print(f"\n--- SWEEP: {T}K ---", flush=True)
        dyn_npt.set_temperature(temperature_K=T)

        print(f"Equilibrating at {T}K for 20 ps...", flush=True)
        dyn_npt.run(20000) 

        print(f"Production at {T}K for 50 ps...", flush=True)
        enthalpies = []
        volumes = []

        def record_thermo():
            e_tot = atoms.get_total_energy()
            vol = atoms.get_volume()
            enthalpies.append(e_tot + (p_ext * vol))
            volumes.append(vol)

        dyn_npt.attach(record_thermo, interval=100)
        dyn_npt.run(50000)
        dyn_npt.observers = [] 

        mean_h_per_atom = np.mean(enthalpies) / total_atoms
        mean_v = np.mean(volumes)
        
        with open(CSV_FILE, "a") as f:
            f.write(f"{T},{mean_h_per_atom},{mean_v}\n")

    print("\nMD Sweep Complete! Starting Analysis...", flush=True)

    # =========================================================
    # PHASE 3: THERMODYNAMIC ANALYSIS & PLOTTING
    # =========================================================
    df = pd.read_csv(CSV_FILE)
    T_vals = df['Temperature_K'].values
    H_eV_atom = df['Enthalpy_eV_per_atom'].values
    V_A3 = df['Volume_A3'].values

    eV_to_J = 1.602176634e-19
    N_A = 6.02214076e23

    # Handle non-integer formula units by calculating per mole of atoms
    H_J_mol_atoms = H_eV_atom * eV_to_J * N_A 
    V_m3 = V_A3 * 1e-30
    V_molar_atoms = (V_m3 / total_atoms) * N_A

    # Enthalpy & Volumetric Heat Capacity
    poly_coeffs = np.polyfit(T_vals, H_J_mol_atoms, 2)
    a_H, b_H, c_H = poly_coeffs
    Cp_molar_vals = 2 * a_H * T_vals + b_H
    Cp_vol_vals = Cp_molar_vals / V_molar_atoms

    cp_line_coeffs = np.polyfit(T_vals, Cp_vol_vals, 1)
    a_cp, b_cp = cp_line_coeffs

    T_smooth = np.linspace(min(T_vals), max(T_vals), 100)
    H_smooth = np.polyval(poly_coeffs, T_smooth)
    Cp_vol_smooth = a_cp * T_smooth + b_cp

    # Thermal Expansion Calculation
    vol_coeffs = np.polyfit(T_vals, V_A3, 1)
    a_V, b_V = vol_coeffs
    alpha_MT = a_V / (a_V * ANCHOR_TEMP_K + b_V)
    
    Cp_MT = a_cp * ANCHOR_TEMP_K + b_cp

    # Plotting
    eq_label = rf'$C_p(T) = {a_cp:.2f}T + {b_cp:.2e}$ J/(m$^3\cdot$K)'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    ax1.plot(T_vals, H_J_mol_atoms / 1000, 'ko', label='MD Data', markersize=8) 
    ax1.plot(T_smooth, H_smooth / 1000, 'b-', label='Quadratic Fit')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Enthalpy (kJ/mol of atoms)')
    ax1.set_title(f'Molten {args.comp} Enthalpy vs. T')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(T_smooth, Cp_vol_smooth, 'r-', linewidth=2.5, label=eq_label)
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel(r'$C_p$ [J/(m$^3\cdot$K)]')
    ax2.set_title('Volumetric Heat Capacity')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plot_filename = f'{SYSTEM_NAME}_Heat_Capacity_Results.png'
    plt.savefig(plot_filename, dpi=300)

    print(f"\nPlots saved to '{plot_filename}'.")
    print("\n# =========================================================")
    print("# FINAL THERMODYNAMIC PROPERTIES")
    print("# =========================================================")
    print(f"Target Anchor Temperature (Tm): {ANCHOR_TEMP_K} K")
    print(f"Volumetric Cp at {ANCHOR_TEMP_K} K: {Cp_MT:.2e} J/(m^3*K)")
    print(f"Volumetric Thermal Expansion (\u03B1) at {ANCHOR_TEMP_K} K: {alpha_MT:.5e} K^-1")

    # =========================================================
    # NEW PIPELINE OUTPUT LOGIC
    # =========================================================
    # Create a completely unique filename using variables already in your script
    output_filename = f"cp_{args.comp}_{int(ANCHOR_TEMP_K)}K.txt"
    
    # Write the two values separated by a comma
    with open(output_filename, "w") as f:
        f.write(f"{Cp_MT},{alpha_MT}")
        
    print(f"Pipeline data successfully saved to {output_filename}", flush=True)
    print("Job perfectly finished!", flush=True)

if __name__ == "__main__":
    main()
