import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Use header=1 to read below the top-level classification categories
filepath = r'Structural_Coherence_Length\Diagnostics\Data for FLiNaK comparison.xlsx'
df = pd.read_excel(filepath, header=1)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 12,
    'axes.labelsize': 13, 'axes.labelweight': 'bold', 'axes.linewidth': 1.5,
    'xtick.labelsize': 11, 'ytick.labelsize': 12,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 1.75, 'ytick.major.width': 1.75,
    'legend.frameon': False, 'legend.fontsize': 11,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Helvetica',
    'mathtext.it': 'Helvetica:italic', 'mathtext.bf': 'Helvetica:bold',
})

fig, ax = plt.subplots(figsize=(9, 5.5))

keys = list(df.columns)
x_all, y_all = [], []

def plot_and_store(x, y, **kwargs):
    ax.plot(x, y, **kwargs)
    x_all.extend(x.dropna().tolist())
    y_all.extend(y.dropna().tolist())

# --- 1. Experimental Group (Solid lines, diverse markers, warm/natural colors) ---
# plot_and_store(df[keys[4]].dropna(), df[keys[5]].dropna(), label="Exp Ishii et al.", color='#d62728', marker="v", ls="", lw=1.5)
plot_and_store(df[keys[6]].dropna(), df[keys[7]].dropna(), label="Exp Gallagher et al.", color="#121d9a", marker="d", ls="", lw=1.5)

# Round robin data with +- 15% error bars
x_rr = df[keys[8]].dropna()
y_rr = df[keys[9]].dropna()
y_err = y_rr * 0.15
ax.errorbar(x_rr, y_rr, yerr=y_err, label="Exp Munro et al.", color='#8c564b', marker="X", ls="", lw=1.5, capsize=4, capthick=1.5)
x_all.extend(x_rr.tolist())
y_all.extend(y_rr.tolist())
y_all.extend((y_rr + y_err).tolist())
y_all.extend((y_rr - y_err).tolist())

# Merritt et al. (Moved from MD to Experimental, changed style to solid line)
plot_and_store(df[keys[10]].dropna(), df[keys[11]].dropna(), label="Exp Merritt et al.", color='#e377c2', marker="o", ls="", lw=1.5)


# --- 2. MD Group (Dashed lines, geometric markers, cool colors) ---
plot_and_store(df[keys[12]].dropna(), df[keys[13]].dropna(), label="MD Gallagher", markerfacecolor='none', color='#17becf', marker="s", ls="", lw=1.5)
plot_and_store(df[keys[14]].dropna(), df[keys[15]].dropna(), label="MD Ishii", markerfacecolor='none', color="#c91e1e", marker="^", ls="", lw=1.5)


# --- 3. Models Group (Dotted/Dash-dot lines, NO markers, bright colors) ---
plot_and_store(df[keys[0]].dropna(), df[keys[1]].dropna(), label="Model SCM with Exp PDF", color='#ff7f0e', marker="", ls=":", lw=2.5)
plot_and_store(df[keys[2]].dropna(), df[keys[3]].dropna(), label="Model KTM", color='#2ca02c', marker="", ls="-.", lw=2.5)


ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Thermal Conductivity (W/m·K)")

# --- Axis Limits ---
x_min = np.floor(min(x_all) / 100) * 100
x_max = np.ceil(max(x_all) / 100) * 105
y_min = np.floor(min(y_all) / 0.1) * 0.1
y_max = np.ceil(max(y_all) / 0.1) * 0.12

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


# --- Legend Grouping ---
handles, labels = ax.get_legend_handles_labels()
def get_h(name): 
    return [h for h, l in zip(handles, labels) if l == name][0]

blank = Line2D([], [], linestyle='')

new_handles = [
    get_h("Exp Gallagher et al."), get_h("Exp Munro et al."), get_h("Exp Merritt et al."), #blank, get_h("Exp Ishii et al."), get_h("Exp Gallagher et al."), get_h("Exp Munro et al."), get_h("Exp Merritt et al."),
    get_h("MD Gallagher"), get_h("MD Ishii"),
    get_h("Model SCM with Exp PDF"), get_h("Model KTM")
]

new_labels = [
    "(Exp) Gallagher et al.", "(Exp) RR correlation", "(Exp) Merritt et al.", #r"$\bf{Experimental}$", "(Exp) Ishii et al.", "(Exp) Gallagher et al.", "(Exp) RR correlation", "(Exp) Merritt et al.",
    "(MD) Gallagher", "(MD) Ishii",
    "SCM (w/ PDF from MD)", "KTM"
]

ax.legend(new_handles, new_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=1)

# Add "FLiNaK" label to the top left corner
ax.text(0.08, 0.9, 'FLiNaK', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

plt.savefig("flinak_tc", bbox_inches="tight")
plt.show()