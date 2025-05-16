import subprocess
import re
import time
import matplotlib.pyplot as plt
import pandas as pd

# Values of n for smoke test
N_LIST = [4,6,8,25,250,500,1000]
TIME_BUDGET = 60  # seconds

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
        # Updated to match the new output format in sa_solver_non_opti.py
        if 'Final Best Score (Analytical Normalized):' in line:
            # The regex needs to match the format "Score (Analytical Normalized): X.XXXX (HS: Y.YYYY, PS: Z.ZZZZ, MD: W.WWWW)"
            m = re.search(r'Analytical Normalized\): ([-\d\.]+) \(HS: ([-\d\.]+), PS: ([-\d\.]+), MD: ([-\d\.]+)\)', line)
            if m:
                zsum = float(m.group(1))
                hs = float(m.group(2))
                ps = float(m.group(3))
                md = float(m.group(4))
            # Also capture the raw metrics line which is printed separately now
        if 'best_home_strength:' in line:
             m = re.findall(r'([-\d\.]+)', line)
             if m and len(m) >= 3:
                 # Ensure we assign to the correct variables
                 hs = float(m[0]) # best_home_strength
                 ps = float(m[1]) # best_pen_seq
                 md = float(m[2]) # best_max_dev

    return {'Z-sum': zsum, 'HS': hs, 'PS': ps, 'MD': md, 'raw': output}

def run_opti(n, time_budget):
    # 11 runs, minimal logging (log_interval_sa_loop=0)
    cmd = [
        'python3', OPTI_PATH, str(n), '0.8', '1.2', '10',
        '-t', str(time_budget), '--log_interval', '0'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
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
    
    # Save raw metrics to CSV file for use in scientific report
    csv_filename = 'sa_algorithms_comparison_results.csv'
    df.to_csv(csv_filename, index=False, float_format='%.4f')
    print(f"\nSaved raw metrics to '{csv_filename}' for use in your report")
    
    # Create another CSV with a more report-friendly format
    # This rearranges data for easier table creation in scientific reports
    report_data = []
    for i, row in df.iterrows():
        n = row['n']
        report_data.append({
            'n': n,
            'Non-Opti HS': row['Non-Opti HS'],
            'Opti HS': row['Opti HS'],
            'HS Improvement (%)': (1 - row['Opti HS'] / row['Non-Opti HS']) * 100 if row['Non-Opti HS'] != 0 else 0,
            'Non-Opti PS': row['Non-Opti PS'],
            'Opti PS': row['Opti PS'],
            'PS Improvement (%)': (1 - row['Opti PS'] / row['Non-Opti PS']) * 100 if row['Non-Opti PS'] != 0 else 0,
            'Non-Opti MD': row['Non-Opti MD'],
            'Opti MD': row['Opti MD'],
            'MD Improvement (%)': (1 - row['Opti MD'] / row['Non-Opti MD']) * 100 if row['Non-Opti MD'] != 0 else 0,
        })
    
    report_df = pd.DataFrame(report_data)
    report_csv = 'sa_algorithms_report_table.csv'
    report_df.to_csv(report_csv, index=False, float_format='%.2f')
    print(f"Saved report-friendly metrics with improvement percentages to '{report_csv}'")
    # Set up publication-quality plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Create a figure with multiple subplots with more space between them
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    fig.suptitle('Evolution of Raw Metrics by Number of Players', fontsize=14)
    
    # Convert n to numeric for line plots
    df['n'] = pd.to_numeric(df['n'])
    
    # Create darker and more distinguishable colors with markers
    non_opti_style = {'color': '#0066CC', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 6}
    opti_style = {'color': '#CC0000', 'marker': 's', 'linestyle': '-', 'linewidth': 2, 'markersize': 6}
    
    # First row: Home Strength - Linear and Log Scale
    # Linear scale (left)
    axes[0, 0].plot(df['n'], df['Non-Opti HS'], label='Non-Opti', **non_opti_style)
    axes[0, 0].plot(df['n'], df['Opti HS'], label='Opti', **opti_style)
    axes[0, 0].set_xlabel('Number of Players (n)')
    axes[0, 0].set_ylabel('Raw Home Strength')
    axes[0, 0].set_title('Home Strength (Linear Scale)')
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Log scale (right)
    axes[0, 1].plot(df['n'], df['Non-Opti HS'], label='Non-Opti', **non_opti_style)
    axes[0, 1].plot(df['n'], df['Opti HS'], label='Opti', **opti_style)
    axes[0, 1].set_xlabel('Number of Players (n)')
    axes[0, 1].set_ylabel('Raw Home Strength (log scale)')
    axes[0, 1].set_title('Home Strength (Log Scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Second row: Penalty Sequence - Linear and Log Scale
    # Linear scale (left)
    axes[1, 0].plot(df['n'], df['Non-Opti PS'], label='Non-Opti', **non_opti_style)
    axes[1, 0].plot(df['n'], df['Opti PS'], label='Opti', **opti_style)
    axes[1, 0].set_xlabel('Number of Players (n)')
    axes[1, 0].set_ylabel('Raw Total Penalty Sequence')
    axes[1, 0].set_title('Total Penalty Sequence (Linear Scale)')
    axes[1, 0].legend(loc='upper left')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Log scale (right)
    axes[1, 1].plot(df['n'], df['Non-Opti PS'], label='Non-Opti', **non_opti_style)
    axes[1, 1].plot(df['n'], df['Opti PS'], label='Opti', **opti_style)
    axes[1, 1].set_xlabel('Number of Players (n)')
    axes[1, 1].set_ylabel('Raw Total Penalty Sequence (log scale)')
    axes[1, 1].set_title('Total Penalty Sequence (Log Scale)')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend(loc='upper left')
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Third row: Max Deviation - Linear and Log Scale
    # Linear scale (left)
    axes[2, 0].plot(df['n'], df['Non-Opti MD'], label='Non-Opti', **non_opti_style)
    axes[2, 0].plot(df['n'], df['Opti MD'], label='Opti', **opti_style)
    axes[2, 0].set_xlabel('Number of Players (n)')
    axes[2, 0].set_ylabel('Raw Max Deviation')
    axes[2, 0].set_title('Max Deviation (Linear Scale)')
    axes[2, 0].legend(loc='upper left')
    axes[2, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Log scale (right)
    axes[2, 1].plot(df['n'], df['Non-Opti MD'], label='Non-Opti', **non_opti_style)
    axes[2, 1].plot(df['n'], df['Opti MD'], label='Opti', **opti_style)
    axes[2, 1].set_xlabel('Number of Players (n)')
    axes[2, 1].set_ylabel('Raw Max Deviation (log scale)')
    axes[2, 1].set_title('Max Deviation (Log Scale)')
    axes[2, 1].set_yscale('log')
    axes[2, 1].legend(loc='upper left')
    axes[2, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure as a high-quality PNG
    plt.savefig('raw_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved raw metrics comparison plot to 'raw_metrics_comparison.png'")
    plt.show()
    
    # Create a relative performance plot - ratio of Opti/Non-Opti for each metric
    plt.figure(figsize=(10, 6))
    
    # Calculate ratios (Opti/Non-Opti) - lower is better for all metrics
    ratios = pd.DataFrame({
        'n': df['n'],
        'HS Ratio': df['Opti HS'] / df['Non-Opti HS'],
        'PS Ratio': df['Opti PS'] / df['Non-Opti PS'],
        'MD Ratio': df['Opti MD'] / df['Non-Opti MD']
    })
    
    # Plot ratios with better styling (values below 1 mean Opti is better)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    plt.plot(ratios['n'], ratios['HS Ratio'], color='#008000', marker='o', 
             linestyle='-', linewidth=2, markersize=6, label='Home Strength')
    plt.plot(ratios['n'], ratios['PS Ratio'], color='#800080', marker='s', 
             linestyle='-', linewidth=2, markersize=6, label='Penalty Sequence')
    plt.plot(ratios['n'], ratios['MD Ratio'], color='#008080', marker='^', 
             linestyle='-', linewidth=2, markersize=6, label='Max Deviation')
    
    plt.xlabel('Number of Players (n)', fontsize=12)
    plt.ylabel('Ratio (Opti/Non-Opti)', fontsize=12)
    plt.title('Relative Performance: Optimized vs. Non-Optimized', fontsize=14)
    plt.figtext(0.5, 0.01, 'Values below 1 indicate the optimized version performs better', 
                ha='center', fontsize=10, fontstyle='italic')
    
    plt.legend(loc='best', frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log')  # Log scale for x to better visualize across different n
    
    # Save the ratio plot as a high-quality PNG
    plt.savefig('performance_ratio_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved performance ratio plot to 'performance_ratio_comparison.png'")
    plt.tight_layout()
    plt.show()

    print("\nNote: Lower values for all raw metrics indicate better performance.")
    print("The ratio plot shows Opti/Non-Opti - values below 1 mean the optimized version performs better.")
    print("All tests ran with a " + str(TIME_BUDGET) + "-second wall-clock budget per n.")

if __name__ == '__main__':
    main()
