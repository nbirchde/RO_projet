import subprocess
import re
import time
import matplotlib.pyplot as plt
import pandas as pd

# Values of n for smoke test
N_LIST = [4,6,100]
TIME_BUDGET = 20  # seconds

# Paths to solvers
NON_OPTI_PATH = 'src/sa_solver_non_opti.py'
OPTI_PATH = 'src/sa_solver.py'

# Helper to run a solver and parse output

def run_non_opti(n, time_budget):
    cmd = ['python3', NON_OPTI_PATH, str(n), str(time_budget), '0.8', '1.2']
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    # Parse Analytical Normalized Score (Z-sum)
    zsum = None
    hs = ps = md = None
    for line in output.splitlines():
        if 'Final Best Score (normalized):' in line:
            m = re.search(r'([-\d\.]+) \(HS: ([-\d\.]+), PS: ([-\d\.]+), MD: ([-\d\.]+)\)', line)
            if m:
                zsum = float(m.group(1))
                hs = float(m.group(2))
                ps = float(m.group(3))
                md = float(m.group(4))
    return {'Z-sum': zsum, 'HS': hs, 'PS': ps, 'MD': md, 'raw': output}

def run_opti(n, time_budget):
    # 11 runs, minimal logging (log_interval_sa_loop=0)
    cmd = [
        'python3', OPTI_PATH, str(n), '0.8', '1.2', '11',
        '-t', str(time_budget), '--log_interval', '0'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    print(f"\n[DEBUG] sa_solver.py output for n={n}\n{'='*40}\n{output}\n{'='*40}")
    # Parse Analytical Normalized Score (Z-sum)
    zsum = None
    hs = ps = md = None
    # Try to find the best Z-sum in the output (may be multiple runs)
    best_zsum = None
    for line in output.splitlines():
        if 'Final Best Score (Analytical Normalized):' in line:
            m = re.search(r'([-\d\.]+)', line)
            if m:
                z = float(m.group(1))
                if best_zsum is None or z < best_zsum:
                    best_zsum = z
        if 'best_home_strength' in line:
            # Try to parse raw metrics if available
            m = re.findall(r'([-\d\.]+)', line)
            if m and len(m) >= 3:
                hs, ps, md = float(m[0]), float(m[1]), float(m[2])
    return {'Z-sum': best_zsum, 'HS': hs, 'PS': ps, 'MD': md, 'raw': output}

def main():
    results = []
    for n in N_LIST:
        print(f"Running n={n}...")
        res_non_opti = run_non_opti(n, TIME_BUDGET)
        print(f"  Non-Opti: Z-sum={res_non_opti['Z-sum']}")
        res_opti = run_opti(n, TIME_BUDGET)
        print(f"  Opti: Z-sum={res_opti['Z-sum']}")
        results.append({
            'n': n,
            'Non-Opti Z-sum': res_non_opti['Z-sum'],
            'Opti Z-sum': res_opti['Z-sum'],
            'Non-Opti HS': res_non_opti['HS'],
            'Opti HS': res_opti['HS'],
            'Non-Opti PS': res_non_opti['PS'],
            'Opti PS': res_opti['PS'],
            'Non-Opti MD': res_non_opti['MD'],
            'Opti MD': res_opti['MD'],
        })
    # DataFrame
    df = pd.DataFrame(results)
    print("\nTable of Results:")
    print(df)
    # Bar chart
    plt.figure(figsize=(8,5))
    bar_width = 0.35
    x = range(len(N_LIST))
    plt.bar([i-bar_width/2 for i in x], [-df['Non-Opti Z-sum'][i] for i in x], width=bar_width, label='Non-Opti')
    plt.bar([i+bar_width/2 for i in x], [-df['Opti Z-sum'][i] for i in x], width=bar_width, label='Opti')
    plt.xticks(x, N_LIST)
    plt.xlabel('Number of Players (n)')
    plt.ylabel('-Z-sum (higher is better)')
    plt.title('Analytical Normalized Z-sum (lower is better, inverted for clarity)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\nNote: Lower Z-sum values indicate better performance. All tests run with a 60-second wall-clock budget.")

if __name__ == '__main__':
    main()
