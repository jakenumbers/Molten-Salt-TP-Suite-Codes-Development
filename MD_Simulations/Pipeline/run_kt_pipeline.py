import os
import re
import argparse
import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Universal MACE Bulk Modulus Calculator")
    parser.add_argument('--comp', type=str, required=True, help='Composition e.g., "0.5NaCl-0.5KCl"')
    parser.add_argument('--temp', type=float, required=True, help='Target Temperature in K')
    parser.add_argument('--density', type=float, required=True, help='Initial density guess in g/cm^3')
    parser.add_argument('--model', type=str, default='SuperSalt-swa.model', help='Path to MACE model')
    # Defaulting seed to 42 for Kt since we don't need multiple statistical trials
    parser.add_argument('--seed', type=int, default=42, help='Random seed for thermal velocities')
    args = parser.parse_args()

    atom_counts, structure_type, supercell_dim = parse_composition_and_scale(args.comp)
    elements = list(atom_counts.keys())
    sys_cations = [el for el in elements if el != 'Cl']
    total_atoms = sum(atom_counts.values())
    
    TARGET_TEMP = args.temp
    SYSTEM_NAME = args.comp.replace('.', '')
    RANDOM_SEED = args.seed

    print("# =========================================================")
    print(f"# INITIALIZING {SYSTEM_NAME} BULK MODULUS (Kt)")
    print("# =========================================================")
    print(f"Target Temp: {TARGET_TEMP} K | Density: {args.density} g/cm^3")
    print(f"Structure Base: {structure_type.capitalize()} | Total Atoms: {total_atoms}")

    # =========================================================
    # 1. SETUP & BUILD
    # =========================================================
    np.random.seed(42) # Hardcoded configuration seed
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

    # =========================================================
    # 2. OPTIMIZATION & MELT
    # =========================================================
    np.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    print("\n--- PHASE 0: GEOMETRY OPTIMIZATION ---", flush=True)
    atoms.rattle(stdev=0.1, rng=rng)
    opt = FIRE(atoms) # Removed the maxstep restriction
    opt.run(fmax=1.0, steps=1000)

    has_lithium = 'Li' in elements
    has_zinc = 'Zn' in elements
    
    if has_lithium:
        print("Lithium detected: Temporarily scaling Li mass to Na mass for stability...", flush=True)
        masses = atoms.get_masses()
        for i, atom in enumerate(atoms):
            if atom.symbol == 'Li':
                masses[i] = 22.990 
        atoms.set_masses(masses)

    MELT_TEMP = 2500
    MELT_TIMESTEP = 0.5 if has_lithium else 1.0
    MELT_STEPS = int(10000 / MELT_TIMESTEP)

    print(f"\n--- PHASE 1: {MELT_TEMP}K MELT (Langevin NVT) ---", flush=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=MELT_TEMP, rng=rng)
    Stationary(atoms) # ADD THIS LINE
    dyn_melt = Langevin(atoms, timestep=MELT_TIMESTEP * units.fs, temperature_K=MELT_TEMP, friction=0.01)
    dyn_melt.run(MELT_STEPS)

    if has_lithium:
        print("Restoring true Lithium mass (6.94 amu) for valid dynamic calculation...", flush=True)
        true_masses = atoms.get_masses()
        for i, atom in enumerate(atoms):
            if atom.symbol == 'Li':
                true_masses[i] = 6.941
        atoms.set_masses(true_masses)

    # =========================================================
    # 3. PRESSURE SWEEP (NPT) FOR BULK MODULUS
    # =========================================================
    csv_file = f"Vol_vs_Press_{SYSTEM_NAME}_{TARGET_TEMP}K.csv"
    with open(csv_file, "w") as f:
        f.write("Pressure_bar,Mean_Volume_A3\n")

    pressures_bar = [1.01325, 1000.0, 2000.0, 3000.0]

    # Bring down to target temperature at 1 bar
    print(f"\n--- PHASE 2: COOLING TO TARGET TEMP ({TARGET_TEMP} K) ---", flush=True)
    dyn_npt = NPT(atoms, timestep=1.0 * units.fs, temperature_K=TARGET_TEMP, externalstress=pressures_bar[0] * units.bar,
                  ttime=25.0 * units.fs, pfactor=2e6 * units.GPa * (units.fs**2))
    dyn_npt.run(20000) # 20 ps equilibration

    for i, P in enumerate(pressures_bar):
        print(f"\n--- PRESSURE SWEEP: {P} bar ---", flush=True)
        
        dyn_npt = NPT(atoms, timestep=1.0 * units.fs, temperature_K=TARGET_TEMP, externalstress=P * units.bar,
                      ttime=25.0 * units.fs, pfactor=2e6 * units.GPa * (units.fs**2))

        # Only equilibrate if it's NOT the first step
        if i > 0:
            print(f"Equilibrating at {P} bar for 20 ps...", flush=True)
            dyn_npt.run(20000)
        else:
            print(f"System already equilibrated to {P} bar during cooling phase. Skipping equilibration...", flush=True)

        print(f"Production at {P} bar for 50 ps...", flush=True)
        volumes = []

        def record_vol():
            volumes.append(atoms.get_volume())

        dyn_npt.attach(record_vol, interval=100)
        dyn_npt.run(50000)
        dyn_npt.observers = [] # Clear observer for the next pressure step

        mean_v = np.mean(volumes)
        
        with open(csv_file, "a") as f:
            f.write(f"{P},{mean_v}\n")

    print("\nMD Pressure Sweep Complete! Starting Analysis...", flush=True)

    # =========================================================
    # 4. BULK MODULUS CALCULATION & OUTPUT
    # =========================================================
    df = pd.read_csv(csv_file)
    P_bar = df['Pressure_bar'].values
    V_A3 = df['Mean_Volume_A3'].values

    # Convert to standard SI units
    P_Pa = P_bar * 1e5
    V_m3 = V_A3 * 1e-30

    # # Fit Volume vs Pressure to a linear equation: V(P) = a*P + b
    # poly_coeffs = np.polyfit(P_Pa, V_m3, 1)
    # dV_dP = poly_coeffs[0] 
    # V_0 = V_m3[0] 

    # # Isothermal Bulk Modulus (K_T = -V * dP/dV)
    # K_T_Pa = -V_0 / dV_dP

    # Fit Volume vs Pressure to a quadratic equation: V(P) = a*P^2 + b*P + c
    poly_coeffs = np.polyfit(P_Pa, V_m3, 2)
    # The derivative dV/dP evaluated at P = 0 is just the linear coefficient 'b'
    dV_dP_at_1bar = poly_coeffs[1] 
    V_0 = poly_coeffs[2] # The y-intercept is the theoretical V_0 at 0 bar

    # Isothermal Bulk Modulus (K_T = -V * dP/dV)
    K_T_Pa = -V_0 / dV_dP_at_1bar
    K_T_GPa = K_T_Pa / 1e9

    print(f"\n--- RESULTS ---", flush=True)
    print(f"Isothermal Bulk Modulus (K_T) at {TARGET_TEMP} K: {K_T_GPa:.3f} GPa", flush=True)
    
    # Save purely the scalar value to a text file for the dependency pipeline
    output_filename = f"Kt_{args.comp}_{int(args.temp)}K.txt"
    with open(output_filename, "w") as f:
        f.write(str(K_T_GPa))
        
    print(f"Result saved to {output_filename}", flush=True)
    print("Job perfectly finished!", flush=True)

if __name__ == "__main__":
    main()
