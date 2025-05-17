import subprocess
import re
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# Values of n for benchmark
N_LIST = [5, 25, 100, 250, 500, 700, 1000, 1250]
TIME_BUDGET = 60  # seconds

# Calculate theoretical worst-case values for each metric
def calculate_worst_case_hs(n):
    """Theoretical maximum for HomeStrength = n(n-1)(n+1)/6"""
    return n * (n - 1) * (n + 1) / 6.0

def calculate_worst_case_ps(n):
    """Theoretical maximum for Penalty Sequence = n(n-2)"""
    return n * (n - 2)

def calculate_worst_case_md(n):
    """Theoretical maximum for Max Deviation = (n-1)/2"""
    return (n - 1) / 2.0

# Paths to solvers
NON_OPTI_PATH = 'src/sa_solver_non_opti.py'
OPTI_PATH = 'src/sa_solver.py'

# Helper to run a solver and parse output

def run_non_opti(n, time_budget):
    print(f"    Running non-optimized solver for n={n}...")
    
    try:
        # For non-optimized solver, use the correct module syntax
        cmd = ['python3', '-m', 'src.sa_solver_non_opti', str(n), str(time_budget), '0.8', '1.2']
            
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
    except Exception as e:
        print(f"    Error running non-optimized solver: {e}")
        zsum = None
        hs = ps = md = None
        output = str(e)

    return {'Z-sum': zsum, 'HS': hs, 'PS': ps, 'MD': md, 'raw': output}

def run_opti(n, time_budget):
    print(f"    Running optimized solver for n={n}...")
    
    try:
        # Command syntax should follow: python3 -m src.sa_solver <n_players> [-t <time_budget>] [<alpha> <beta>] [<runs>]
        cmd = [
            'python3', '-m', 'src.sa_solver', str(n),
            '-t', str(time_budget),
            '0.8', '1.2', '9'  # alpha, beta, runs (9 parallel runs)
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
    except Exception as e:
        print(f"    Error running optimized solver: {e}")
        best_zsum = None
        hs = ps = md = None
        output = str(e)
        
    return {'Z-sum': best_zsum, 'HS': hs, 'PS': ps, 'MD': md, 'raw': output}

def main():
    results = []
    
    print(f"Starting benchmark with time budget of {TIME_BUDGET} seconds per algorithm per n value")
    print(f"Testing for n values: {N_LIST}")
    
    for n in N_LIST:
        print(f"\nRunning n={n}...")
        
        # Calculate theoretical worst-case metrics for this n
        worst_hs = calculate_worst_case_hs(n)
        worst_ps = calculate_worst_case_ps(n)
        worst_md = calculate_worst_case_md(n)
        
        print(f"  Theoretical worst-case values for n={n}:")
        print(f"  - HomeStrength: {worst_hs:.2f}")
        print(f"  - Penalty Sequence: {worst_ps:.2f}")
        print(f"  - Max Deviation: {worst_md:.2f}")
        
        # Calculate analytical factors (mean and std dev) for this n
        mu_hs = n * (n - 1) * (n + 1) / 12.0
        sigma_hs = n * math.sqrt(n**2 - 1) / (4 * math.sqrt(3))
        
        mu_ps = n * (n - 2) / 2.0
        sigma_ps = math.sqrt(n * (n - 2)) / 2.0 if n > 2 else 1.0
        
        if n > 1:
            mu_md = 0.5 * math.sqrt(n-1) * math.sqrt(2 * math.log(n))
            sigma_md = 0.5 * math.sqrt(n-1) * math.pi / math.sqrt(12 * math.log(n)) if n > 1 else 1.0
        else:
            mu_md = 0.0
            sigma_md = 1.0
            
        # Calculate z-scores for worst-case scenarios
        z_worst_hs = (worst_hs - mu_hs) / sigma_hs
        z_worst_ps = (worst_ps - mu_ps) / sigma_ps
        z_worst_md = (worst_md - mu_md) / sigma_md
        
        print(f"  Z-scores of worst-case scenarios:")
        print(f"  - Z-score HS: {z_worst_hs:.2f}")
        print(f"  - Z-score PS: {z_worst_ps:.2f}")
        print(f"  - Z-score MD: {z_worst_md:.2f}")
        
        # Run algorithms
        res_non_opti = run_non_opti(n, TIME_BUDGET)
        print(f"  Non-Opti: Z-sum={res_non_opti['Z-sum']}")
        res_opti = run_opti(n, TIME_BUDGET)
        print(f"  Opti: Z-sum={res_opti['Z-sum']}")
        
        # Convert None values to NaN to avoid errors
        non_opti_z = float('nan') if res_non_opti['Z-sum'] is None else res_non_opti['Z-sum']
        opti_z = float('nan') if res_opti['Z-sum'] is None else res_opti['Z-sum']
        
        results.append({
            'n': n,
            'Non-Opti Z-sum': non_opti_z,
            'Opti Z-sum': opti_z,
            'Non-Opti HS': res_non_opti['HS'],
            'Opti HS': res_opti['HS'],
            'Non-Opti PS': res_non_opti['PS'],
            'Opti PS': res_opti['PS'],
            'Non-Opti MD': res_non_opti['MD'],
            'Opti MD': res_opti['MD'],
            'Worst-case HS': worst_hs,
            'Worst-case PS': worst_ps,
            'Worst-case MD': worst_md,
            'Z-score Worst HS': z_worst_hs,
            'Z-score Worst PS': z_worst_ps,
            'Z-score Worst MD': z_worst_md,
        })
    # DataFrame
    df = pd.DataFrame(results)
    print("\nTable of Results:")
    print(df)
    
    # Save raw metrics to CSV file for use in scientific report
    csv_filename = 'sa_algorithms_comparison_results.csv'
    df.to_csv(csv_filename, index=False, float_format='%.4f')
    print(f"\nSaved raw metrics to '{csv_filename}' for use in your report")
    
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
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
    fig.suptitle('Evolution of Raw Metrics by Number of Players', fontsize=14)
    
    # Convert n to numeric for line plots
    df['n'] = pd.to_numeric(df['n'])
    
    # Create darker and more distinguishable colors with markers
    # Use drawstyle='steps-post' to avoid interpolation between points
    non_opti_style = {'color': '#0066CC', 'marker': 'o', 'linestyle': '-', 'linewidth': 2, 'markersize': 6, 'drawstyle': 'default'}
    opti_style = {'color': '#CC0000', 'marker': 's', 'linestyle': '-', 'linewidth': 2, 'markersize': 6, 'drawstyle': 'default'}
    worst_case_style = {'color': '#008800', 'marker': '^', 'linestyle': '--', 'linewidth': 2, 'markersize': 6, 'drawstyle': 'default'}
    
    # First row: Home Strength - Linear and Log Scale
    # Linear scale (left)
    axes[0, 0].plot(df['n'], df['Non-Opti HS'], label='Non-Opti', **non_opti_style)
    axes[0, 0].plot(df['n'], df['Opti HS'], label='Opti', **opti_style)
    axes[0, 0].plot(df['n'], df['Worst-case HS'], label='Worst Case', **worst_case_style)
    axes[0, 0].set_xlabel('Number of Players (n)')
    axes[0, 0].set_ylabel('Raw Home Strength')
    axes[0, 0].set_title('Home Strength (Linear Scale)')
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Log scale (right)
    axes[0, 1].plot(df['n'], df['Non-Opti HS'], label='Non-Opti', **non_opti_style)
    axes[0, 1].plot(df['n'], df['Opti HS'], label='Opti', **opti_style)
    axes[0, 1].plot(df['n'], df['Worst-case HS'], label='Worst Case', **worst_case_style)
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
    axes[1, 0].plot(df['n'], df['Worst-case PS'], label='Worst Case', **worst_case_style)
    axes[1, 0].set_xlabel('Number of Players (n)')
    axes[1, 0].set_ylabel('Raw Total Penalty Sequence')
    axes[1, 0].set_title('Total Penalty Sequence (Linear Scale)')
    axes[1, 0].legend(loc='upper left')
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Log scale (right)
    axes[1, 1].plot(df['n'], df['Non-Opti PS'], label='Non-Opti', **non_opti_style)
    axes[1, 1].plot(df['n'], df['Opti PS'], label='Opti', **opti_style)
    axes[1, 1].plot(df['n'], df['Worst-case PS'], label='Worst Case', **worst_case_style)
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
    axes[2, 0].plot(df['n'], df['Worst-case MD'], label='Worst Case', **worst_case_style)
    axes[2, 0].set_xlabel('Number of Players (n)')
    axes[2, 0].set_ylabel('Raw Max Deviation')
    axes[2, 0].set_title('Max Deviation (Linear Scale)')
    axes[2, 0].legend(loc='upper left')
    axes[2, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Log scale (right)
    axes[2, 1].plot(df['n'], df['Non-Opti MD'], label='Non-Opti', **non_opti_style)
    axes[2, 1].plot(df['n'], df['Opti MD'], label='Opti', **opti_style)
    axes[2, 1].plot(df['n'], df['Worst-case MD'], label='Worst Case', **worst_case_style)
    axes[2, 1].set_xlabel('Number of Players (n)')
    axes[2, 1].set_ylabel('Raw Max Deviation (log scale)')
    axes[2, 1].set_title('Max Deviation (Log Scale)')
    axes[2, 1].set_yscale('log')
    axes[2, 1].legend(loc='upper left')
    axes[2, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure as a high-quality PNG
    plt.savefig('raw_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved raw metrics comparison plot to 'raw_metrics_comparison.png'")
    plt.close(fig)  # Close the figure to avoid displaying it twice
    
    # Create Z-Score Bar Chart
    fig_z, ax_z = plt.subplots(figsize=(14, 8))
    
    # Prepare data for bar chart
    x = np.arange(len(N_LIST))
    bar_width = 0.2
    
    # Get Z-scores for each n value and invert them (multiply by -1) so higher is better
    non_opti_z_scores = pd.Series(df['Non-Opti Z-sum']) * -1
    opti_z_scores = pd.Series(df['Opti Z-sum']) * -1
    
    # Replace NaN with appropriate values for visualization
    for i, n in enumerate(N_LIST):
        if pd.isna(non_opti_z_scores.iloc[i]):
            print(f"Warning: Non-optimized Z-score for n={n} is missing. Using placeholder.")
            non_opti_z_scores.iloc[i] = 0  # Use 0 as placeholder
        if pd.isna(opti_z_scores.iloc[i]):
            print(f"Warning: Optimized Z-score for n={n} is missing. Using placeholder.")
            opti_z_scores.iloc[i] = 0  # Use 0 as placeholder
    
    # Plot non-optimized Z-scores
    bars1 = ax_z.bar(x - bar_width/2, non_opti_z_scores, bar_width, label='Non-Optimized', color='#0066CC')
    
    # Plot optimized Z-scores
    bars2 = ax_z.bar(x + bar_width/2, opti_z_scores, bar_width, label='Optimized', color='#CC0000')
    
    # Customize chart
    ax_z.set_xlabel('Number of Players (n)', fontsize=12)
    ax_z.set_ylabel('Z-Score (Inverted, Higher is Better)', fontsize=12)
    ax_z.set_title('Z-Score Comparison: Non-Optimized vs. Optimized', fontsize=14)
    ax_z.set_xticks(x)
    ax_z.set_xticklabels(N_LIST)
    ax_z.legend()
    ax_z.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add explanation about inverted Z-scores
    plt.figtext(0.5, 0.01, 'Z-scores have been inverted (multiplied by -1) so that higher values indicate better performance',
                ha='center', fontsize=10, fontstyle='italic')
    
    # Add value labels above each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):  # Only add label if the value is not NaN
                if height > 100:  # For large values, truncate to fit better
                    label_text = f'{height:.0f}'
                elif height > 10:
                    label_text = f'{height:.1f}'
                else:
                    label_text = f'{height:.2f}'
                ax_z.annotate(label_text,
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig('z_score_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved Z-score comparison bar chart to 'z_score_comparison.png'")
    plt.close(fig_z)  # Close the figure
    
    print("\nNote: Lower values for all raw metrics indicate better performance.")
    print("The Z-score chart shows inverted Z-scores, so higher values are better.")
    print("All tests ran with a " + str(TIME_BUDGET) + "-second wall-clock budget per n.")

if __name__ == '__main__':
    main()
