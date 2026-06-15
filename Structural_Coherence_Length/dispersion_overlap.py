import numpy as np
import matplotlib.pyplot as plt

def calculate_dispersion(k, m_c, m_a, a, num_points=200):
    """
    Calculates the acoustic and optic dispersion branches for a 1D diatomic chain.
    """
    # Wavevector q ranges from 0 to pi/a (edge of the first Brillouin zone)
    q = np.linspace(0, np.pi / a, num_points)
    
    # Calculate the coupled mass terms
    mass_sum = (1 / m_c) + (1 / m_a)
    mass_prod = m_c * m_a
    
    # Evaluate the core dispersion equation
    term1 = k * mass_sum
    term2 = k * np.sqrt((mass_sum)**2 - (4 * np.sin(q * a)**2) / mass_prod)
    
    # Extract the two branches
    omega_acoustic = np.sqrt(term1 - term2)
    omega_optic = np.sqrt(term1 + term2)
    
    return q, omega_acoustic, omega_optic

# Defined scenarios using approximate atomic masses (amu), 
# effective bond lengths (Å), and relative effective force constants (k).
scenarios = [
    {
        "label": "LiF vs BeF$_2$ (High Stiffness Mismatch)",
        "chain1": {"name": "LiF",  "k": 1-0.001, "m_c": 6.94, "m_a": 19.0, "a": 1.86018},
        "chain2": {"name": "BeF2", "k": 1-0.170, "m_c": 9.01, "m_a": 19.0, "a": 1.53366}
    },
    {
        "label": "LiF vs NaF (Mass & Size Offset)",
        "chain1": {"name": "LiF", "k": 1-0.108, "m_c": 6.94,  "m_a": 19.0, "a": 1.81931},
        "chain2": {"name": "NaF", "k": 1-0.217, "m_c": 22.99, "m_a": 19.0, "a": 2.19106}
    },
    {
        "label": "NaCl vs KCl (Ideal Chloride Baseline)",
        "chain1": {"name": "NaCl", "k": 1-0.088, "m_c": 22.99, "m_a": 35.45, "a": 2.57565},
        "chain2": {"name": "KCl",  "k": 1-0.161, "m_c": 39.10, "m_a": 35.45, "a": 2.96275}
    },
    {
        "label": "NaCl vs UCl$_3$ (Extreme Heavy Cation)",
        "chain1": {"name": "NaCl", "k": 1-0.216,  "m_c": 22.99, "m_a": 35.45, "a": 2.65778},
        "chain2": {"name": "UCl3", "k": 1-0.044, "m_c": 238.0, "m_a": 35.45, "a": 2.77544}
    },
    {
        "label": "NaCl vs CaCl$_2$ (Valence Shift)",
        "chain1": {"name": "NaCl",  "k": 1-0.192,  "m_c": 22.99, "m_a": 35.45, "a": 2.68712},
        "chain2": {"name": "CaCl2", "k": 1-0.033, "m_c": 40.08, "m_a": 35.45, "a": 2.67223}
    },
    {
        "label": "NaCl vs MgCl$_2$ (Similar Mass, Stiff Bond)",
        "chain1": {"name": "NaCl",  "k": 1-0.157,  "m_c": 22.99, "m_a": 35.45, "a": 2.63635},
        "chain2": {"name": "MgCl2", "k": 1-0.01, "m_c": 24.31, "m_a": 35.45, "a": 2.2883}
    }
]

# Initialize the figure with a 2x3 grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), constrained_layout=True)
axes = axes.flatten()

# Find the absolute maximum frequency across all chains to lock the y-axis
global_max_omega = 0
for sc in scenarios:
    for chain_key in ["chain1", "chain2"]:
        c = sc[chain_key]
        _, _, w_op = calculate_dispersion(c["k"], c["m_c"], c["m_a"], c["a"])
        if max(w_op) > global_max_omega:
            global_max_omega = max(w_op)

# Loop through scenarios and plot
for i, sc in enumerate(scenarios):
    ax = axes[i]
    c1, c2 = sc["chain1"], sc["chain2"]
    
    # Calculate Chain 1
    q1, w_ac1, w_op1 = calculate_dispersion(c1["k"], c1["m_c"], c1["m_a"], c1["a"])
    # Calculate Chain 2
    q2, w_ac2, w_op2 = calculate_dispersion(c2["k"], c2["m_c"], c2["m_a"], c2["a"])
    
    # Plot Chain 1 (Blue)
    ax.plot(q1, w_ac1, color='#1f77b4', linestyle='-',  linewidth=2.5, label=f'{c1["name"]} Acoustic')
    ax.plot(q1, w_op1, color='#1f77b4', linestyle='--', linewidth=2.5, label=f'{c1["name"]} Optic')
    
    # Plot Chain 2 (Red)
    ax.plot(q2, w_ac2, color='#d62728', linestyle='-',  linewidth=2.5, label=f'{c2["name"]} Acoustic')
    ax.plot(q2, w_op2, color='#d62728', linestyle='--', linewidth=2.5, label=f'{c2["name"]} Optic')
    
    # Shade allowed frequency bands to visualize transmission vs reflection
    ax.axhspan(0, np.max(w_ac1), color='#1f77b4', alpha=0.1)
    ax.axhspan(np.min(w_op1), np.max(w_op1), color='#1f77b4', alpha=0.1)
    
    ax.axhspan(0, np.max(w_ac2), color='#d62728', alpha=0.1)
    ax.axhspan(np.min(w_op2), np.max(w_op2), color='#d62728', alpha=0.1)
    
    # Styling and labeling
    ax.set_title(sc["label"], fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Wavevector, $q$ (rad/Å)', fontsize=11)
    if i % 3 == 0:
        ax.set_ylabel('Frequency, $\omega$ (rad/ps)', fontsize=11)
    
    # Set limits
    max_q = max(np.pi / c1["a"], np.pi / c2["a"])
    ax.set_xlim(0, max_q)
    ax.set_ylim(0, global_max_omega * 1.05)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Structural Coherence: Diatomic Dispersion Overlap in Salt Mixtures', fontsize=18, fontweight='bold')

plt.show()