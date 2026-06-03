"""Parameter sensitivity analysis for the absorption model.

Varies the following parameters and reports the change in:
 - peak wavelength (x position) of the absorption spectrum
 - peak amplitude (y position)
 - Planck-mean absorption coefficient

This script uses `LiF` as a representative salt and the model
implemented in `abs_model_claude_V2.py`.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from abs_model_claude_V2 import (
    SALT_REGISTRY, initialize_salt, calculate_oscillator_strengths,
    compute_alpha_total, planck_mean_absorption
)


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'Figures', 'sensitivity')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def metric_peak(wl_um, alpha):
    """Return peak wavelength (um) and peak amplitude for a spectrum."""
    idx = np.nanargmax(alpha)
    return wl_um[idx], alpha[idx]


def run_sensitivity(salt_name='LiF', T=1173.0):
    salt = SALT_REGISTRY.get(salt_name)
    if salt is None:
        raise KeyError(f"Salt '{salt_name}' not found in registry")

    # Initialize once to extract omega0, r0, sigma, etc.
    initialize_salt(salt)

    # Wavelength grid [m]
    wl_m = np.linspace(2e-6, 40e-6, 5000)
    wl_um = wl_m * 1e6

    # Baseline
    alpha_base, n_base = compute_alpha_total(wl_m, T, salt)
    base_peak_x, base_peak_y = metric_peak(wl_um, alpha_base)
    base_planck = planck_mean_absorption(wl_m, alpha_base, T)

    print(f"Baseline: peak @ {base_peak_x:.3f} um, amp={base_peak_y:.3e}, kP={base_planck:.3e}")

    # Parameters to sweep and target values
    # For pair-specific params (epsilon_s, epsilon_inf, gamma0, gamma_slope)
    # LiF has a single ion pair; we sweep that pair's values.
    pair = salt.ion_pairs[0]

    factors = np.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])

    sweep_defs = [
        ('epsilon_s', 'pair', pair.epsilon_s),
        ('epsilon_inf', 'pair', pair.epsilon_inf),
        ('alpha_anh', 'pair', getattr(pair, 'alpha_anh', 3.0)),
        ('C0_multi', 'pair', getattr(pair, 'C0_multi', 1e4)),
        ('gamma0', 'pair', pair.gamma0),
        ('gamma_slope', 'pair', pair.gamma_slope),
        ('eps_inf_mixture', 'salt', salt.eps_inf_mixture),
        ('W0_eV', 'salt', salt.W0_eV),
        ('kappa0', 'salt', salt.kappa0),
    ]

    results = {}

    for name, scope, orig in sweep_defs:
        xs = []
        ys = []
        kPs = []
        for f in factors:
            # Apply change
            if scope == 'pair':
                setattr(pair, name, orig * f)
                # Recompute oscillator strengths if eps values changed
                if name in ('epsilon_s', 'epsilon_inf'):
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
            else:
                setattr(salt, name, orig * f)
                if name == 'eps_inf_mixture':
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

            alpha, _ = compute_alpha_total(wl_m, T, salt)
            px, py = metric_peak(wl_um, alpha)
            kP = planck_mean_absorption(wl_m, alpha, T)

            xs.append(px)
            ys.append(py)
            kPs.append(kP)

        # Restore original values
        if scope == 'pair':
            setattr(pair, name, orig)
            if name in ('epsilon_s', 'epsilon_inf'):
                calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
        else:
            setattr(salt, name, orig)
            if name == 'eps_inf_mixture':
                calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

        results[name] = {
            'factors': factors,
            'peak_x_um': np.array(xs),
            'peak_y': np.array(ys),
            'kP': np.array(kPs),
            'orig': orig,
        }

        # Plot results for this parameter
        fig, axs = plt.subplots(3, 1, figsize=(6, 5), constrained_layout=True)
        axs[0].plot(factors, results[name]['peak_x_um'], 'o-')
        axs[0].axhline(base_peak_x, color='k', linestyle='--', alpha=0.6)
        axs[0].set_ylabel('Peak $\\lambda$ (um)')
        axs[0].set_xscale('log')

        axs[1].plot(factors, results[name]['peak_y'], 'o-')
        axs[1].axhline(base_peak_y, color='k', linestyle='--', alpha=0.6)
        axs[1].set_ylabel('Peak amplitude (m$^{-1}$)')
        axs[1].set_yscale('log')
        axs[1].set_xscale('log')

        axs[2].plot(factors, results[name]['kP'], 'o-')
        axs[2].axhline(base_planck, color='k', linestyle='--', alpha=0.6)
        axs[2].set_ylabel('Planck-mean $\\kappa_P$ (m$^{-1}$)')
        axs[2].set_xscale('log')
        axs[2].set_xlabel('Multiplier applied to baseline')

        title = f"Sensitivity: {name} (baseline={orig:.3g})"
        fig.suptitle(title)
        outpath = os.path.join(OUTPUT_DIR, f'{salt.name}_sensitivity_{name}.png')
        fig.savefig(outpath, dpi=1200)
        print(f"Wrote: {outpath}")

    # Also make a small diagnostic plot showing baseline vs a few perturbed spectra
    sample_factors = [0.8, 1.0, 1.2]
    fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
    for f in sample_factors:
        # vary gamma_slope as an example
        pair.gamma_slope = pair.gamma_slope * f
        calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
        alpha, _ = compute_alpha_total(wl_m, T, salt)
        ax.semilogy(wl_um, alpha, label=f'gamma_slope x{f}')
        # restore gamma_slope for next iteration
        pair.gamma_slope = pair.gamma_slope / f

    ax.set_xlim(0, 30)
    ax.set_ylim(1e-1, 1e11)
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Absorption (m$^{-1}$)')
    ax.legend()
    outpath = os.path.join(OUTPUT_DIR, f'{salt.name}_spectra_examples.png')
    fig.savefig(outpath, dpi=1200)
    print(f"Wrote: {outpath}")


if __name__ == '__main__':
    run_sensitivity('LiF', T=1173.0)
    # Also produce percent-change in Planck-mean vs Temperature for +20% perturbations
    def plot_planck_pct_change(salt_name='LiF', perturb=0.20,
                               T_array=None, wl_range=(0.15e-6, 40e-6)):
        salt = SALT_REGISTRY.get(salt_name)
        if salt is None:
            raise KeyError(f"Salt '{salt_name}' not found in registry")

        initialize_salt(salt)
        pair = salt.ion_pairs[0]

        if T_array is None:
            T_array = np.linspace(900.0, 1400.0, 51)

        wl_m = np.linspace(wl_range[0], wl_range[1], 3000)

        # Baseline kP over temperatures
        kP_base = np.array([planck_mean_absorption(wl_m, compute_alpha_total(wl_m, T, salt)[0], T)
                             for T in T_array])

        params = [
            ('epsilon_s', 'pair'),
            ('epsilon_inf', 'pair'),
            ('alpha_anh', 'pair'),
            ('C0_multi', 'pair'),
            ('gamma0', 'pair'),
            ('gamma_slope', 'pair'),
            ('eps_inf_mixture', 'salt'),
            ('W0_eV', 'salt'),
            ('kappa0', 'salt'),
        ]

        # LaTeX labels for legend
        label_map = {
            'epsilon_s': r'$\epsilon_s$',
            'epsilon_inf': r'$\epsilon_\infty$',
            'alpha_anh': r'$\alpha_{anh}$',
            'C0_multi': r'$C_{0,\mathrm{multi}}$',
            'gamma0': r'$\gamma_0$',
            'gamma_slope': r"$\gamma\,'$",
            'eps_inf_mixture': r'$\epsilon_{\infty,\mathrm{mix}}$',
            'W0_eV': r'$W_0\,(\mathrm{eV})$',
            'kappa0': r'$\kappa_0$',
        }

        pct_changes = {}
        for name, scope in params:
            # store original
            if scope == 'pair':
                orig = getattr(pair, name)
                setattr(pair, name, orig * (1.0 + perturb))
                if name in ('epsilon_s', 'epsilon_inf'):
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
            else:
                orig = getattr(salt, name)
                setattr(salt, name, orig * (1.0 + perturb))
                if name == 'eps_inf_mixture':
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

            kP_pert = np.array([planck_mean_absorption(wl_m, compute_alpha_total(wl_m, T, salt)[0], T)
                                for T in T_array])

            # compute percent change safely (avoid divide by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                pct = 100.0 * (kP_pert - kP_base) / np.where(kP_base == 0, np.nan, kP_base)

            pct_changes[name] = pct

            # restore original
            if scope == 'pair':
                setattr(pair, name, orig)
                if name in ('epsilon_s', 'epsilon_inf'):
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
            else:
                setattr(salt, name, orig)
                if name == 'eps_inf_mixture':
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

        # Plot all on one figure
        fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
        cmap = plt.get_cmap('tab10')
        for i, (name, _) in enumerate(params):
            ax.plot(T_array, pct_changes[name], label=label_map.get(name, name), color=cmap(i % 10))

        ax.axhline(0.0, color='k', linewidth=0.6)
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Percent change in $\kappa_P$ (%)')
        #ax.set_title(f"Planck-mean percent change for +{int(perturb*100)}% parameter perturbation ({salt_name})")
        ax.legend(fontsize=8)
        outpath = os.path.join(OUTPUT_DIR, f'{salt_name}_planck_pct_change_plus{int(perturb*100)}.png')
        fig.savefig(outpath, dpi=1200)
        print(f"Wrote: {outpath}")

    # (defer running until we loop over selected salts below)
    def plot_planck_absolute(salt_name='LiF', perturb=0.20,
                             T_array=None, wl_range=(0.15e-6, 40e-6)):
        """Plot absolute Planck-mean absorption (kappa_P) vs T for +perturb on parameters.

        Uses distinct line styles and spaced markers; legend shows LaTeX symbols.
        """
        salt = SALT_REGISTRY.get(salt_name)
        if salt is None:
            raise KeyError(f"Salt '{salt_name}' not found in registry")

        initialize_salt(salt)
        pair = salt.ion_pairs[0]

        if T_array is None:
            T_array = np.linspace(900.0, 1400.0, 51)

        wl_m = np.linspace(wl_range[0], wl_range[1], 3000)

        # Baseline kP over temperatures
        kP_base = np.array([planck_mean_absorption(wl_m, compute_alpha_total(wl_m, T, salt)[0], T)
                             for T in T_array])

        params = [
            ('epsilon_s', 'pair'),
            ('epsilon_inf', 'pair'),
            ('alpha_anh', 'pair'),
            ('C0_multi', 'pair'),
            ('gamma0', 'pair'),
            ('gamma_slope', 'pair'),
            ('eps_inf_mixture', 'salt'),
            ('W0_eV', 'salt'),
            ('kappa0', 'salt'),
        ]

        # LaTeX labels for legend
        label_map = {
            'epsilon_s': r'$\epsilon_s$',
            'epsilon_inf': r'$\epsilon_\infty$',
            'alpha_anh': r'$\alpha_{anh}$',
            'C0_multi': r'$C_{0,\mathrm{multi}}$',
            'gamma0': r'$\gamma_0$',
            'gamma_slope': r"$\gamma\,'$",
            'eps_inf_mixture': r'$\epsilon_{\infty,\mathrm{mix}}$',
            'W0_eV': r'$W_0\,(\mathrm{eV})$',
            'kappa0': r'$\kappa_0$',
        }

        linestyles = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,1)), (0, (1,1))]
        markers = ['o', 's', '^', 'v', 'D', 'P', 'X']

        fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)

        # Plot baseline
        ax.plot(T_array, kP_base, color='k', linewidth=1.5, label='Baseline', zorder=10)

        for i, (name, scope) in enumerate(params):
            # apply perturbation +20%
            if scope == 'pair':
                orig = getattr(pair, name)
                setattr(pair, name, orig * (1.0 + perturb))
                if name in ('epsilon_s', 'epsilon_inf'):
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
            else:
                orig = getattr(salt, name)
                setattr(salt, name, orig * (1.0 + perturb))
                if name == 'eps_inf_mixture':
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

            kP_pert = np.array([planck_mean_absorption(wl_m, compute_alpha_total(wl_m, T, salt)[0], T)
                                for T in T_array])

            # restore original
            if scope == 'pair':
                setattr(pair, name, orig)
                if name in ('epsilon_s', 'epsilon_inf'):
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
            else:
                setattr(salt, name, orig)
                if name == 'eps_inf_mixture':
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

            ls = linestyles[i % len(linestyles)]
            mk = markers[i % len(markers)]
            # spread markers: markevery chooses spacing based on index
            markevery = max(1, len(T_array) // 8 + i)
            label = label_map.get(name, name)
            ax.plot(T_array, kP_pert, linestyle=ls, marker=mk, markevery=markevery,
                    linewidth=1.1, markersize=5, label=label, alpha=0.9)

        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Planck-mean $\kappa_P$ (m$^{-1}$)')
        ax.set_yscale('log')
        #ax.set_title(f"Planck-mean $\kappa_P$ for +{int(perturb*100)}% parameter perturbation ({salt_name})")
        ax.legend(fontsize=9, ncol=2)
        outpath = os.path.join(OUTPUT_DIR, f'{salt_name}_planck_absolute_plus{int(perturb*100)}.png')
        fig.savefig(outpath, dpi=1200)
        print(f"Wrote: {outpath}")

    # (defer running until we loop over selected salts below)
    def plot_spectra_perturbations(salt_name='LiF', perturb=0.20,
                                   T=1173.0, wl_range=(0.15e-6, 40e-6)):
        """Plot raw absorption spectra for +perturb on each parameter at a single T.

        Each parameter's perturbed spectrum is plotted in a distinct color/style
        with spaced markers for readability.
        """
        salt = SALT_REGISTRY.get(salt_name)
        if salt is None:
            raise KeyError(f"Salt '{salt_name}' not found in registry")

        initialize_salt(salt)
        pair = salt.ion_pairs[0]

        wl_m = np.linspace(wl_range[0], wl_range[1], 5000)
        wl_um = wl_m * 1e6

        alpha_base = compute_alpha_total(wl_m, T, salt)[0]

        params = [
            ('epsilon_s', 'pair'),
            ('epsilon_inf', 'pair'),
            ('alpha_anh', 'pair'),
            ('C0_multi', 'pair'),
            ('gamma0', 'pair'),
            ('gamma_slope', 'pair'),
            ('eps_inf_mixture', 'salt'),
            ('W0_eV', 'salt'),
            ('kappa0', 'salt'),
        ]

        label_map = {
            'epsilon_s': r'$\epsilon_s$',
            'epsilon_inf': r'$\epsilon_\infty$',
            'alpha_anh': r'$\alpha_{anh}$',
            'C0_multi': r'$C_{0,\mathrm{multi}}$',
            'gamma0': r'$\gamma_0$',
            'gamma_slope': r"$\gamma\,'$",
            'eps_inf_mixture': r'$\epsilon_{\infty,\mathrm{mix}}$',
            'W0_eV': r'$W_0\,(\mathrm{eV})$',
            'kappa0': r'$\kappa_0$',
        }

        cmap = plt.get_cmap('tab10')
        linestyles = ['-', '--', '-.', ':', (0, (3,1,1,1)), (0, (5,1)), (0, (1,1))]
        markers = ['o', 's', '^', 'v', 'D', 'P', 'X']

        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        ax.semilogy(wl_um, alpha_base, color='k', linewidth=1.4, label='Baseline', zorder=10)

        for i, (name, scope) in enumerate(params):
            # apply perturbation
            if scope == 'pair':
                orig = getattr(pair, name)
                setattr(pair, name, orig * (1.0 + perturb))
                if name in ('epsilon_s', 'epsilon_inf'):
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
            else:
                orig = getattr(salt, name)
                setattr(salt, name, orig * (1.0 + perturb))
                if name == 'eps_inf_mixture':
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

            alpha_pert = compute_alpha_total(wl_m, T, salt)[0]

            # restore
            if scope == 'pair':
                setattr(pair, name, orig)
                if name in ('epsilon_s', 'epsilon_inf'):
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)
            else:
                setattr(salt, name, orig)
                if name == 'eps_inf_mixture':
                    calculate_oscillator_strengths(salt.ion_pairs, salt.eps_inf_mixture)

            color = cmap(i % 10)
            ls = linestyles[i % len(linestyles)]
            mk = markers[i % len(markers)]
            markevery = max(1, len(wl_um) // 80 + i)
            ax.semilogy(wl_um, alpha_pert, linestyle=ls, color=color,
                        marker=mk, markevery=markevery, markersize=4,
                        linewidth=1.0, label=label_map.get(name, name), alpha=0.9)

        ax.set_xlim(0, 30)
        ax.set_ylim(1e-1, 1e11)
        ax.set_xlabel('Wavelength ($\mu$m)')
        ax.set_ylabel('Absorption Coefficient $\kappa_\lambda$ (m$^{-1}$)')
        #ax.set_title(f'Absorption spectra (+{int(perturb*100)}% perturbations) at {T:.0f}K')
        ax.legend(fontsize=9, ncol=2)
        outpath = os.path.join(OUTPUT_DIR, f'{salt.name}_absorption_spectra_plus{int(perturb*100)}.png')
        fig.savefig(outpath, dpi=1200)
        print(f"Wrote: {outpath}")

    # Run analyses for multiple salts
    salts_to_run = ['LiF', 'FLiNaK']
    for sname in salts_to_run:
        try:
            print(f"\n=== Running sensitivity for {sname} ===")
            run_sensitivity(sname, T=1173.0)
            plot_planck_pct_change(sname, perturb=0.20)
            plot_planck_absolute(sname, perturb=0.20)
            plot_spectra_perturbations(sname, perturb=0.20, T=1173.0)
        except Exception as e:
            print(f"Error running for {sname}: {e}")
