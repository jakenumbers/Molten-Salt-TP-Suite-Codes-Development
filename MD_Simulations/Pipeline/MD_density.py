import os
import re
import argparse
import numpy as np
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
    """Parses composition strings and scales lattice size based on complexity."""
    components = comp_str.split('-')
    num_components = len(components)
    
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
        total_cation_sites = 4 * (supercell_dim ** 3) 
    elif has_cs:
        structure_type = 'cesiumchloride'
        supercell_dim = 11 if num_components >= 4 else 5 
        total_cation_sites = 1 * (supercell_dim ** 3) 
    else:
        structure_type = 'rocksalt'
        supercell_dim = 4 if num_components >= 4 else 3
        total_cation_sites = 4 * (supercell_dim ** 3) 
        
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
    return atom_counts, structure_type, supercell_dim

def main():
    parser = argparse.ArgumentParser(description="MACE Molten Salt Density Finder")
    parser.add_argument('--comp', type=str, required=True, help='Composition e.g., "4NaCl-5ZrCl4"')
    parser.add_argument('--temp', type=float, required=True, help='Target Temperature in K')
    parser.add_argument('--model', type=str, default='SuperSalt-swa.model', help='Path to MACE model')
    args = parser.parse_args()

    atom_counts, structure_type, supercell_dim = parse_composition_and_scale(args.comp)
    elements = list(atom_counts.keys())
    sys_cations = [el for el in elements if el != 'Cl']
    total_atoms = sum(atom_counts.values())
    
    TARGET_TEMP = args.temp
    # Clean the filename by removing decimals, if any
    CLEAN_COMP = args.comp.replace('.', '') 
    
    print(f"--- DENSITY FINDER: {args.comp} at {TARGET_TEMP} K ---", flush=True)

    # =========================================================
    # BUILD STRUCTURE WITH SAFE GENERIC DENSITY
    # =========================================================
    calc = MACECalculator(model_paths=args.model, device='cuda')

    if structure_type == 'fluorite':
        atoms = bulk('CaF2', crystalstructure='fluorite', a=5.46, cubic=True)
    elif structure_type == 'cesiumchloride':
        atoms = bulk('CsCl', crystalstructure='cesiumchloride', a=4.12, cubic=True)
    else:
        atoms = bulk('NaCl', crystalstructure='rocksalt', a=5.64, cubic=True)
        
    atoms = atoms * (supercell_dim, supercell_dim, supercell_dim)

    base_cat_indices = [atom.index for atom in atoms if atom.symbol in ['Ca', 'Cs', 'Na']]
    base_an_indices = [atom.index for atom in atoms if atom.symbol in ['F', 'Cl']]

    np.random.seed(42)
    np.random.shuffle(base_cat_indices)
    
    current_idx = 0
    for cation in sys_cations:
        count = atom_counts[cation]
        for i in base_cat_indices[current_idx : current_idx + count]: 
            atoms[i].symbol = cation
        current_idx += count

    np.random.shuffle(base_an_indices)
    cl_indices = base_an_indices[:atom_counts['Cl']]
    delete_indices = base_an_indices[atom_counts['Cl']:]

    for i in cl_indices: atoms[i].symbol = 'Cl'
    del atoms[delete_indices]

    # Use a generic starting guess of 2.0 g/cm^3
    total_mass_g_mol = sum([atom_counts[el] * atomic_masses[atomic_numbers[el]] for el in elements])
    safe_starting_density = 2.0 
    volume_cm3 = (total_mass_g_mol / 6.02214076e23) / safe_starting_density 
    liquid_box_length = (volume_cm3 * 1e24) ** (1/3)
    atoms.set_cell([liquid_box_length, liquid_box_length, liquid_box_length], scale_atoms=True)
    atoms.calc = calc

    # Optimize and mass-scale Li
    atoms.rattle(stdev=0.1)
    opt = FIRE(atoms)
    opt.run(fmax=1.0, steps=1000)

    masses = atoms.get_masses()
    for i, atom in enumerate(atoms):
        if atom.symbol == 'Li': masses[i] = 22.990 
    atoms.set_masses(masses)

    # =========================================================
    # MELT AND EQUILIBRATE TO FIND TRUE DENSITY
    # =========================================================
    print("Melting at 2500K...", flush=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=2500)
    Stationary(atoms)
    dyn_melt = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=2500, friction=0.01)
    dyn_melt.run(5000)

    print(f"Equilibrating NPT volume at {TARGET_TEMP}K...", flush=True)
    p_ext = 1.01325 * units.bar 
    dyn_npt = NPT(atoms, timestep=1.0 * units.fs, temperature_K=TARGET_TEMP, externalstress=p_ext,
                  ttime=25.0 * units.fs, pfactor=2e6 * units.GPa * (units.fs**2))
    
    # 20 ps equilibration to let the barostat find the right volume
    dyn_npt.run(20000)

    print("Collecting volume data...", flush=True)
    volumes = []
    def record_vol(): volumes.append(atoms.get_volume())
    dyn_npt.attach(record_vol, interval=100)
    
    # 10 ps production to average out the barostat noise
    dyn_npt.run(10000) 

    # =========================================================
    # CALCULATE AND OUTPUT DENSITY
    # =========================================================
    mean_v_A3 = np.mean(volumes)
    mean_v_cm3 = mean_v_A3 * 1e-24
    mass_grams = total_mass_g_mol / 6.02214076e23
    
    true_density = mass_grams / mean_v_cm3

    print(f"\n--- RESULTS ---")
    print(f"Calculated Density: {true_density:.4f} g/cm^3")

    # Save exactly as your bash script expects it
    output_filename = f"density_{args.comp}_{int(TARGET_TEMP)}K.txt"
    with open(output_filename, "w") as f:
        f.write(f"{true_density:.4f}")
        
    print(f"Density successfully written to {output_filename}", flush=True)

if __name__ == "__main__":
    main()