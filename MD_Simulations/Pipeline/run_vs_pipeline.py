from asyncio.log import logger
import os
import re
import argparse
import numpy as np
import pandas as pd
from ase import units
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.bussi import Bussi
from ase.md.verlet import VelocityVerlet
from ase.io import write
import os
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

class TransportLogger:
    def __init__(self, dyn, atoms, species_list):
        self.dyn = dyn
        self.atoms = atoms
        self.species_list = species_list
        self.data_buffer = [] # Store data in memory

    def __call__(self):
        step = self.dyn.get_number_of_steps()
        vol = self.atoms.get_volume()
        stress = self.atoms.get_stress(include_ideal_gas=True, voigt=True)
        p_yz, p_xz, p_xy = stress[3], stress[4], stress[5]
        
        velocities = self.atoms.get_velocities()
        masses = self.atoms.get_masses()
        symbols = np.array(self.atoms.get_chemical_symbols())
        
        row_data = [step, vol, p_yz, p_xz, p_xy]
        
        for sp in self.species_list:
            mask = (symbols == sp)
            v_sp = velocities[mask]
            m_sp = masses[mask][:, np.newaxis]
            j_vector = np.sum(m_sp * v_sp, axis=0)
            row_data.extend(j_vector)
            
        self.data_buffer.append(row_data)

    def write_to_csv(self, filename):
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Create header
            header = ["Step", "Volume_A3", "P_yz_eV_A3", "P_xz_eV_A3", "P_xy_eV_A3"]
            for sp in self.species_list:
                header.extend([f"J_{sp}_x", f"J_{sp}_y", f"J_{sp}_z"])
            writer.writerow(header)
            writer.writerows(self.data_buffer)

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
    parser = argparse.ArgumentParser(description="Universal MACE Speed of Sound Calculator")
    parser.add_argument('--comp', type=str, required=True, help='Composition e.g., "0.5NaCl-0.5KCl"')
    parser.add_argument('--temp', type=float, required=True, help='Target Temperature in K')
    parser.add_argument('--density', type=float, required=True, help='Initial density guess in g/cm^3')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for thermal velocities')
    parser.add_argument('--model', type=str, default='SuperSalt-swa.model', help='Path to MACE model')
    args = parser.parse_args()

    atom_counts, structure_type, supercell_dim = parse_composition_and_scale(args.comp)
    elements = list(atom_counts.keys())
    sys_cations = [el for el in elements if el != 'Cl']
    total_atoms = sum(atom_counts.values())
    
    TARGET_TEMP = args.temp
    SYSTEM_NAME = args.comp.replace('.', '')
    RANDOM_SEED = args.seed

    print("# =========================================================")
    print(f"# INITIALIZING {SYSTEM_NAME} Vs ARRAY (SEED {RANDOM_SEED})")
    print("# =========================================================")
    print(f"Target Temp: {TARGET_TEMP} K | Density: {args.density} g/cm^3")
    print(f"Structure Base: {structure_type.capitalize()} | Total Atoms: {total_atoms}")

    # =========================================================
    # 1. SETUP & BUILD (CONFIGURATIONAL SEED = 42)
    # =========================================================
    # HARDCODED seed so the physical lattice is identical for all array tasks
    np.random.seed(42)

    calc = MACECalculator(model_paths=args.model, device='cuda')

    if structure_type == 'cesiumchloride':
        atoms = bulk('CsCl', crystalstructure='cesiumchloride', a=4.12, cubic=True)
        atoms = atoms * (supercell_dim, supercell_dim, supercell_dim)
        base_cation_symbol = 'Cs'
        base_anion_symbol = 'Cl'
    elif structure_type == 'fluorite':
        atoms = bulk('CaF2', crystalstructure='fluorite', a=5.46, cubic=True)
        atoms = atoms * (supercell_dim, supercell_dim, supercell_dim)
        base_cation_symbol = 'Ca'
        base_anion_symbol = 'F'
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
    # 2. OPTIMIZATION & MELT (THERMAL SEED ACTIVATED)
    # =========================================================
    # Switch to the unique array seed for the MD velocities
    np.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    print("\n--- PHASE 0: GEOMETRY OPTIMIZATION ---", flush=True)
    atoms.rattle(stdev=0.1, rng=rng)
    opt = FIRE(atoms) # Removed the maxstep restriction
    opt.run(fmax=1.0, steps=1000)

    # =========================================================
    # DYNAMIC MELT PROTOCOL (Li & Zn SAFETY)
    # =========================================================
    has_lithium = 'Li' in elements
    has_zinc = 'Zn' in elements
    
    # 1. Mass Scaling for Melt (Prevents thrashing for Li)
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

    # 3. CRITICAL: Restore True Mass for Production Dynamics
    if has_lithium:
        print("Restoring true Lithium mass (6.94 amu) for valid dynamic calculation...", flush=True)
        true_masses = atoms.get_masses()
        for i, atom in enumerate(atoms):
            if atom.symbol == 'Li':
                true_masses[i] = 6.941
        atoms.set_masses(true_masses)

    # =========================================================
    # 3. EQUILIBRATION & PRODUCTION (OPTIMIZED)
    # =========================================================
    print(f"\n--- PHASE 2: {TARGET_TEMP}K EQUILIBRATION (Bussi CSVR) ---", flush=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TARGET_TEMP, rng=rng)
    dyn_equil = Bussi(atoms, timestep=1.0 * units.fs, temperature_K=TARGET_TEMP, taut=25.0 * units.fs)
    dyn_equil.run(20000) # 20 ps

    # --- NEW: NVE BURN-IN (NO RECORDING) ---
    print(f"\n--- PHASE 3a: {TARGET_TEMP}K NVE BURN-IN (80 ps) ---", flush=True)
    dyn_nve_burn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    dyn_nve_burn.run(80000) # 80 ps of silent equilibration

    # --- NEW: NVE PRODUCTION (RECORDING) ---
    print(f"\n--- PHASE 3b: {TARGET_TEMP}K NVE PRODUCTION (70 ps) ---", flush=True)
    dyn_nve_prod = VelocityVerlet(atoms, timestep=1.0 * units.fs)

    eV_to_J = 1.602176634e-19
    A_to_m = 1e-10
    amu_to_kg = 1.66053906660e-27
    ASE_velocity_to_ms = np.sqrt(eV_to_J / amu_to_kg)
    ASE_force_to_N = eV_to_J / A_to_m

    masses_kg = atoms.get_masses() * amu_to_kg
    V_m3 = atoms.get_volume() * (A_to_m**3)
    rho_kg_m3 = np.sum(masses_kg) / V_m3

    Lx_m, Ly_m, Lz_m = atoms.cell.cellpar()[:3] * A_to_m
    kx, ky, kz = 2*np.pi/Lx_m, 2*np.pi/Ly_m, 2*np.pi/Lz_m

    stresses_Pa, J_vals, Jdot_vals, T_vals = [], [], [], []

    raw_velocities = []

    # Identify unique species in the melt (e.g., ['Na', 'K', 'Cl'])
    unique_species = list(set(atoms.get_chemical_symbols()))
    transport_csv = f"Transport_{args.comp}_{int(args.temp)}K.csv"

    # Initialize the Logger
    logger = TransportLogger(dyn_nve_prod, atoms, species_list=unique_species)
    # Attach the logger to run every 5 steps
    dyn_nve_prod.attach(logger, interval=5)

    def record_dynamics():

        vel_ms = atoms.get_velocities() * ASE_velocity_to_ms
        raw_velocities.append(atoms.get_velocities())
        pos_m = atoms.get_positions() * A_to_m
        forces_N = atoms.get_forces() * ASE_force_to_N
        
        virial_stress_Pa = atoms.get_stress() * (eV_to_J / A_to_m**3)
        kin_xx_Pa = -np.sum(masses_kg * vel_ms[:, 0]**2) / V_m3
        kin_yy_Pa = -np.sum(masses_kg * vel_ms[:, 1]**2) / V_m3
        kin_zz_Pa = -np.sum(masses_kg * vel_ms[:, 2]**2) / V_m3
        
        stresses_Pa.append([virial_stress_Pa[0] + kin_xx_Pa, 
                            virial_stress_Pa[1] + kin_yy_Pa, 
                            virial_stress_Pa[2] + kin_zz_Pa])
        
        px = np.exp(-1j * kx * pos_m[:, 0])
        py = np.exp(-1j * ky * pos_m[:, 1])
        pz = np.exp(-1j * kz * pos_m[:, 2])
        
        Jx = np.sum(masses_kg * vel_ms[:, 0] * px)
        Jy = np.sum(masses_kg * vel_ms[:, 1] * py)
        Jz = np.sum(masses_kg * vel_ms[:, 2] * pz)
        
        Jdotx = np.sum((forces_N[:, 0] - 1j * kx * masses_kg * vel_ms[:, 0]**2) * px)
        Jdoty = np.sum((forces_N[:, 1] - 1j * ky * masses_kg * vel_ms[:, 1]**2) * py)
        Jdotz = np.sum((forces_N[:, 2] - 1j * kz * masses_kg * vel_ms[:, 2]**2) * pz)
        
        J_vals.append([np.abs(Jx)**2, np.abs(Jy)**2, np.abs(Jz)**2])
        Jdot_vals.append([np.abs(Jdotx)**2, np.abs(Jdoty)**2, np.abs(Jdotz)**2])
        T_vals.append(atoms.get_temperature())

    def calculate_vdos(velocities, dt_ps):
        """
        Transforms a velocity time series into a normalized VDOS spectrum.
        velocities: numpy array of shape (n_frames, n_atoms, 3)
        dt_ps: Time step between recorded frames in picoseconds
        """
        n_frames = velocities.shape[0]
        
        # 1. Apply a Hanning window to prevent spectral leakage (optional but recommended for MD)
        window = np.hanning(n_frames)[:, np.newaxis, np.newaxis]
        windowed_vel = velocities * window
        
        # 2. Fourier Transform the velocity time series along the time axis (axis=0)
        v_fft = np.fft.rfft(windowed_vel, axis=0)
        
        # 3. Element-wise product with the complex conjugate (The Convolution Theorem shortcut)
        power_spectrum = np.real(v_fft * np.conjugate(v_fft))
        
        # 4. Average across all atoms of this species (axis=1) and all x,y,z components (axis=2)
        vdos = np.mean(power_spectrum, axis=(1, 2))
        
        # 5. Extract frequencies (d is the sample spacing in picoseconds -> yields THz)
        freqs_THz = np.fft.rfftfreq(n_frames, d=dt_ps)
        
        # Convert THz to rad/ps (standard unit for your SCM formalisms)
        freqs_rad_ps = freqs_THz * 2 * np.pi
        
        # 6. Normalize the area under the curve to 1.0
        vdos_norm = vdos / np.trapezoid(vdos, freqs_rad_ps)
        
        return freqs_rad_ps, vdos_norm

    dyn_nve_prod.attach(record_dynamics, interval=10)

    # 1. Define the trajectory file path
    out_dir = f"../{args.comp}_{int(args.temp)}K"
    os.makedirs(out_dir, exist_ok=True) # <-- THIS CREATES THE FOLDER
    traj_filename = f"{args.comp}_{args.temp}K_NVE_seed_{args.seed}.extxyz"

    # 2. Make sure we delete any old trajectory from previous failed runs
    if os.path.exists(traj_filename):
        os.remove(traj_filename)

    # 3. Create a custom observer to dump the frame WITH velocities
    def dump_trajectory():
        # append=True adds the frame to the file instead of overwriting
        write(traj_filename, atoms, format="extxyz", append=True)

    # 4. Attach the observer to run EVERY 5 STEPS (5 fs)
    dyn_nve_prod.attach(dump_trajectory, interval=5)

    dyn_nve_prod.run(70000) # 70 ps of pure, usable data collection

    logger.write_to_csv(transport_csv)

    print("NVE Production complete.")

    # =========================================================
    # 4. CALCULATION
    # =========================================================
    print("\n--- PHASE 4: SPEED OF SOUND CALCULATION ---", flush=True)

    # No more slicing needed! All data is post-equilibration.
    converged_T_vals = T_vals
    converged_stresses = np.array(stresses_Pa)
    converged_J = np.array(J_vals)
    converged_Jdot = np.array(Jdot_vals)

    T_avg = np.mean(converged_T_vals)
    var_stress_avg = np.mean([np.var(converged_stresses[:, 0]), 
                              np.var(converged_stresses[:, 1]), 
                              np.var(converged_stresses[:, 2])])

    cinf2_x = (1 / kx**2) * (np.mean(converged_Jdot[:, 0]) / np.mean(converged_J[:, 0]))
    cinf2_y = (1 / ky**2) * (np.mean(converged_Jdot[:, 1]) / np.mean(converged_J[:, 1]))
    cinf2_z = (1 / kz**2) * (np.mean(converged_Jdot[:, 2]) / np.mean(converged_J[:, 2]))
    cinf2_avg = np.mean([cinf2_x, cinf2_y, cinf2_z])

    c_inf = np.sqrt(cinf2_avg)
    kB = 1.380649e-23
    fluct_term = (V_m3 / (rho_kg_m3 * kB * T_avg)) * var_stress_avg
    c_s2 = cinf2_avg - fluct_term
    c_s = np.sqrt(c_s2) if c_s2 > 0 else float('nan')

    print(f"Adiabatic Speed of Sound (c_s): {c_s:.2f} m/s", flush=True)

    output_filename = f"Vs_{args.comp}_seed_{RANDOM_SEED}_{int(args.temp)}K.txt"
    with open(output_filename, "w") as f:
        f.write(str(c_s))
    print(f"Result saved to {output_filename}", flush=True)

    print("\n--- PHASE 5: VDOS CALCULATION AND MODE OVERLAP ---", flush=True)

    # Convert recorded velocities to a numpy array: Shape (n_frames, total_atoms, 3)
    vel_array = np.array(raw_velocities)
    dt_record = 0.01 # 10 fs = 0.01 ps

    # 1. Dynamically calculate VDOS for EVERY cation species in the system
    vdos_results = {}
    freqs = None
    
    for cation in sys_cations:
        print(f"Extracting VDOS for {cation}...", flush=True)
        cation_indices = [atom.index for atom in atoms if atom.symbol == cation]
        vel_cation = vel_array[:, cation_indices, :]
        
        f, vdos = calculate_vdos(vel_cation, dt_record)
        if freqs is None:
            freqs = f # Frequencies are the same for all, just save it once
        vdos_results[cation] = vdos

    # 2. Save the dynamic VDOS curves to a single CSV
    # Construct header dynamically: "Freq_rad_ps, VDOS_Na, VDOS_K, ..."
    header_str = "Freq_rad_ps," + ",".join([f"VDOS_{cat}" for cat in sys_cations])
    data_columns = [freqs] + [vdos_results[cat] for cat in sys_cations]
    
    csv_filename = f"VDOS_{args.comp}_{int(args.temp)}K.csv"
    np.savetxt(csv_filename, 
               np.column_stack(data_columns), 
               header=header_str, 
               delimiter=",", comments="")
    print(f"VDOS curves saved to {csv_filename}")

    # 3. Calculate Overlap Integrals (b_PH) for all cation pairs
    import itertools
    
    if len(sys_cations) == 1:
        # Unary salt (e.g., pure NaCl). No cation-cation scattering occurs.
        overlap_fraction = 1.0 
        print(f"Unary salt detected. b_PH set to 1.0")
    else:
        # Binary or multi-component salt. Calculate pairwise overlap.
        overlaps = []
        for cat1, cat2 in itertools.combinations(sys_cations, 2):
            overlap_integrand = np.minimum(vdos_results[cat1], vdos_results[cat2])
            pair_overlap = np.trapezoid(overlap_integrand, freqs)
            overlaps.append(pair_overlap)
            print(f"VDOS Mode Matching ({cat1} and {cat2}): {pair_overlap * 100:.2f}%")
        
        # If there are 3+ cations, we average the overlaps. For 2 cations, it's just the one value.
        overlap_fraction = np.mean(overlaps) 
        
    # 4. Save the b_PH text file for the aggregate script
    output_vdos_filename = f"b_PH_{args.comp}_seed_{RANDOM_SEED}_{int(args.temp)}K.txt"
    with open(output_vdos_filename, "w") as f:
        f.write(str(overlap_fraction))
    print(f"Phonon Transfer Factor (b_PH) saved to {output_vdos_filename}", flush=True)

if __name__ == "__main__":
    main()