import py_compile
import subprocess
import sys
import os

# Use Python 3.12 which has all the required packages (ase, MDAnalysis, seaborn, etc.)
PYTHON = r'C:\Users\qcn\AppData\Local\Programs\Python\Python312\python.exe'
if not os.path.exists(PYTHON):
    PYTHON = sys.executable  # Fallback

# -------------------------------------------------------------
# AUTO-DISCOVER SYSTEMS FROM THE Output FOLDER
# Every subfolder of Output/ is a system (e.g. 0.5NaCl-0.5KCl_930K)
# -------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output')

systems = []
if os.path.isdir(OUTPUT_DIR):
    for entry in sorted(os.listdir(OUTPUT_DIR)):
        full_path = os.path.join(OUTPUT_DIR, entry)
        # Only include directories (systems); skip the Plots folder itself
        if os.path.isdir(full_path) and entry != 'Plots':
            systems.append(full_path)
else:
    print(f"WARNING: Output folder not found at '{OUTPUT_DIR}'. No systems to process.")

# Shared scripts location
SHARED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Shared_Scripts')
TRAJ_SCRIPT = os.path.join(SHARED_DIR, 'trajectory.py')
PLOT_SCRIPT = os.path.join(SHARED_DIR, 'plotting.py')

# Create the Plots folder in Output
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'Plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"Using Python: {PYTHON}\n")
print(f"Auto-discovered {len(systems)} system(s) in '{OUTPUT_DIR}':")
for s in systems:
    print(f"  - {s}")
print()

errors = []

# =========================================================
# PHASE 1: SYNTAX CHECK ALL SCRIPTS
# =========================================================
print("=" * 70)
print("PHASE 1: SYNTAX CHECK")
print("=" * 70)

for script in [TRAJ_SCRIPT, PLOT_SCRIPT]:
    try:
        py_compile.compile(script, doraise=True)
        print(f"OK (compile): {script}")
    except Exception as e:
        errors.append(f"{script}: {e}")
        print(f"FAIL (compile): {script}: {e}")

# =========================================================
# PHASE 2: RUN TRAJECTORY SCRIPTS (SCL ANALYSIS)
# =========================================================
print("\n" + "=" * 70)
print("PHASE 2: RUN shared trajectory.py (SCL analysis, bPH off + on)")
print("=" * 70)

for f in systems:
    for extra_args in ([], ['--bph']):
        label = 'bPH_off' if not extra_args else 'bPH_on'
        try:
            print(f"\n--- Running {TRAJ_SCRIPT} --system {f} [{label}] ---")
            result = subprocess.run(
                [PYTHON, TRAJ_SCRIPT, '--system', f] + extra_args,
                capture_output=True, text=True, timeout=7200
            )
            # Print last 15 lines of stdout
            out_lines = result.stdout.strip().splitlines()
            for line in out_lines[-15:]:
                print(f"  {line}")
            if result.returncode != 0:
                err_tail = result.stderr.strip().splitlines()[-5:]
                for line in err_tail:
                    print(f"  STDERR: {line}")
                errors.append(f"{TRAJ_SCRIPT} --system {f} [{label}] exited with code {result.returncode}")
                print(f"  FAIL: {TRAJ_SCRIPT} --system {f} [{label}]")
            else:
                print(f"  OK: {TRAJ_SCRIPT} --system {f} [{label}]")
        except subprocess.TimeoutExpired:
            errors.append(f"{TRAJ_SCRIPT} --system {f} [{label}] timed out")
            print(f"  TIMEOUT: {TRAJ_SCRIPT} --system {f} [{label}]")
        except Exception as e:
            errors.append(f"{TRAJ_SCRIPT} --system {f} [{label}]: {e}")
            print(f"  ERROR: {TRAJ_SCRIPT} --system {f} [{label}]: {e}")

# =========================================================
# PHASE 3: RUN PLOTTING SCRIPTS
# =========================================================
print("\n" + "=" * 70)
print("PHASE 3: RUN shared plotting.py")
print("=" * 70)

for f in systems:
    try:
        print(f"\n--- Running {PLOT_SCRIPT} --system {f} ---")
        result = subprocess.run(
            [PYTHON, PLOT_SCRIPT, '--system', f],
            capture_output=True, text=True, timeout=7200
        )
        out_lines = result.stdout.strip().splitlines()
        for line in out_lines[-15:]:
            print(f"  {line}")
        if result.returncode != 0:
            err_tail = result.stderr.strip().splitlines()[-5:]
            for line in err_tail:
                print(f"  STDERR: {line}")
            errors.append(f"{PLOT_SCRIPT} --system {f} exited with code {result.returncode}")
            print(f"  FAIL: {PLOT_SCRIPT} --system {f}")
        else:
            print(f"  OK: {PLOT_SCRIPT} --system {f}")
    except subprocess.TimeoutExpired:
        errors.append(f"{PLOT_SCRIPT} --system {f} timed out")
        print(f"  TIMEOUT: {PLOT_SCRIPT} --system {f}")
    except Exception as e:
        errors.append(f"{PLOT_SCRIPT} --system {f}: {e}")
        print(f"  ERROR: {PLOT_SCRIPT} --system {f}: {e}")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "=" * 70)
if errors:
    print(f"{len(errors)} error(s) occurred:")
    for e in errors:
        print(f"  - {e}")
    raise SystemExit(1)
else:
    print("All scripts compiled and executed successfully!")