import py_compile
import subprocess
import sys
import os

# Use Python 3.12 which has all the required packages (ase, MDAnalysis, seaborn, etc.)
PYTHON = r'C:\Users\qcn\AppData\Local\Programs\Python\Python312\python.exe'
if not os.path.exists(PYTHON):
    PYTHON = sys.executable  # Fallback

systems = [
    'Output/0.5LiCl-0.5KCl_738K',
    'Output/0.5LiCl-0.5KCl_938K',
    'Output/0.5NaCl-0.5KCl_930K',
    'Output/0.5NaCl-0.5KCl_1130K',
    'Output/0.68KCl-0.32MgCl2_703.0K',
    'Output/0.25LiCl-0.75KCl_934K',
    'Output/0.43KCl-0.57MgCl2_750K',
    'Output/0.56NaCl-0.44MgCl2_750K',
]

print(f"Using Python: {PYTHON}\n")

errors = []

# =========================================================
# PHASE 1: SYNTAX CHECK ALL SCRIPTS
# =========================================================
print("=" * 70)
print("PHASE 1: SYNTAX CHECK")
print("=" * 70)

for f in systems:
    for script in ['trajectory.py', 'plotting.py']:
        path = os.path.join(f, script)
        try:
            py_compile.compile(path, doraise=True)
            print(f"OK (compile): {path}")
        except Exception as e:
            errors.append(f"{path}: {e}")
            print(f"FAIL (compile): {path}: {e}")

# =========================================================
# PHASE 2: RUN TRAJECTORY SCRIPTS (SCL ANALYSIS)
# =========================================================
print("\n" + "=" * 70)
print("PHASE 2: RUN trajectory.py (SCL analysis, bPH off + on)")
print("=" * 70)

for f in systems:
    script = os.path.join(f, 'trajectory.py')
    for extra_args in ([], ['--bph']):
        label = 'bPH_off' if not extra_args else 'bPH_on'
        try:
            print(f"\n--- Running {script} [{label}] ---")
            result = subprocess.run(
                [PYTHON, script] + extra_args,
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
                errors.append(f"{script} [{label}] exited with code {result.returncode}")
                print(f"  FAIL: {script} [{label}]")
            else:
                print(f"  OK: {script} [{label}]")
        except subprocess.TimeoutExpired:
            errors.append(f"{script} [{label}] timed out")
            print(f"  TIMEOUT: {script} [{label}]")
        except Exception as e:
            errors.append(f"{script} [{label}]: {e}")
            print(f"  ERROR: {script} [{label}]: {e}")

# =========================================================
# PHASE 3: RUN PLOTTING SCRIPTS
# =========================================================
print("\n" + "=" * 70)
print("PHASE 3: RUN plotting.py")
print("=" * 70)

for f in systems:
    script = os.path.join(f, 'plotting.py')
    try:
        print(f"\n--- Running {script} ---")
        result = subprocess.run(
            [PYTHON, script],
            capture_output=True, text=True, timeout=7200
        )
        out_lines = result.stdout.strip().splitlines()
        for line in out_lines[-15:]:
            print(f"  {line}")
        if result.returncode != 0:
            err_tail = result.stderr.strip().splitlines()[-5:]
            for line in err_tail:
                print(f"  STDERR: {line}")
            errors.append(f"{script} exited with code {result.returncode}")
            print(f"  FAIL: {script}")
        else:
            print(f"  OK: {script}")
    except subprocess.TimeoutExpired:
        errors.append(f"{script} timed out")
        print(f"  TIMEOUT: {script}")
    except Exception as e:
        errors.append(f"{script}: {e}")
        print(f"  ERROR: {script}: {e}")

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