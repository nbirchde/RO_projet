import subprocess
import numpy as np
import pandas as pd
import re
import sys
import os
from tqdm import tqdm

# --- Configuration ---
N_PLAYERS = 6
ITERATIONS = 10000
RUNS_PER_COMBINATION = 4 # Number of parallel SA chains
ALPHA_VALUES = np.arange(0.0, 2.01, 0.1) # 0.0 to 2.0 inclusive, step 0.1
BETA_VALUES = np.arange(0.0, 2.01, 0.1)  # 0.0 to 2.0 inclusive, step 0.1
OUTPUT_CSV = "calibration_results_n6_dense.csv" # New output file name
SA_SOLVER_SCRIPT = os.path.join("src", "sa_solver.py")
# --- End Configuration ---

# Regex to find the normalized values following specific headers
# Using lookbehind assertions to ensure we get the value after the correct header
# Added flexibility for potential variations in spacing or log level names
REGEX_NORM_HS = re.compile(r"Home Strength:.*?Normalized:\s*(\d+\.\d+)", re.DOTALL)
REGEX_NORM_PS = re.compile(r"Total Penalty Sequence \(Breaks\):.*?Normalized:\s*(\d+\.\d+)", re.DOTALL)
REGEX_NORM_MD = re.compile(r"Max Deviation \(from ideal home games\):.*?Normalized:\s*(\d+\.\d+)", re.DOTALL)

results = []

print(f"Starting calibration for n={N_PLAYERS}, iterations={ITERATIONS}, runs={RUNS_PER_COMBINATION}")
print(f"Alpha range: {ALPHA_VALUES.min()} to {ALPHA_VALUES.max()} (step {ALPHA_VALUES[1]-ALPHA_VALUES[0]:.1f})")
print(f"Beta range: {BETA_VALUES.min()} to {BETA_VALUES.max()} (step {BETA_VALUES[1]-BETA_VALUES[0]:.1f})")
print(f"Total combinations: {len(ALPHA_VALUES) * len(BETA_VALUES)}")
print(f"Output file: {OUTPUT_CSV}")
print("-" * 30)

# Use tqdm for progress bar
total_combinations = len(ALPHA_VALUES) * len(BETA_VALUES)
pbar = tqdm(total=total_combinations, desc="Running SA combinations")

for alpha in ALPHA_VALUES:
    for beta in BETA_VALUES:
        alpha_str = f"{alpha:.2f}"
        beta_str = f"{beta:.2f}"
        command = [
            "python3", # Use python3 explicitly
            SA_SOLVER_SCRIPT,
            str(N_PLAYERS),
            str(ITERATIONS),
            alpha_str,
            beta_str,
            str(RUNS_PER_COMBINATION)
        ]

        try:
            # Execute the command, capture stderr (where logs go), decode, timeout after 60s
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False, # Don't raise exception on non-zero exit code
                timeout=60 # Add a timeout per run
            )
            log_output = process.stderr

            # Parse the output
            match_hs = REGEX_NORM_HS.search(log_output)
            match_ps = REGEX_NORM_PS.search(log_output)
            match_md = REGEX_NORM_MD.search(log_output)

            if match_hs and match_ps and match_md:
                norm_hs = float(match_hs.group(1))
                norm_ps = float(match_ps.group(1))
                norm_md = float(match_md.group(1))
                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "norm_hs": norm_hs,
                    "norm_ps": norm_ps,
                    "norm_md": norm_md,
                    "total_norm_score": norm_hs + alpha * norm_ps + beta * norm_md, # Calculate total score
                    "status": "success"
                })
            else:
                print(f"\nWarning: Could not parse output for alpha={alpha_str}, beta={beta_str}. Log tail:")
                print(log_output[-500:]) # Print last 500 chars for debugging
                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "norm_hs": np.nan,
                    "norm_ps": np.nan,
                    "norm_md": np.nan,
                    "total_norm_score": np.nan, # Add NaN for total score on error
                    "status": "parse_error"
                })

        except subprocess.TimeoutExpired:
            print(f"\nWarning: Timeout expired for alpha={alpha_str}, beta={beta_str}")
            results.append({
                "alpha": alpha,
                "beta": beta,
                "norm_hs": np.nan,
                "norm_ps": np.nan,
                "norm_md": np.nan,
                "total_norm_score": np.nan, # Add NaN for total score on timeout
                "status": "timeout"
            })
        except Exception as e:
            print(f"\nError running alpha={alpha_str}, beta={beta_str}: {e}")
            results.append({
                "alpha": alpha,
                "beta": beta,
                "norm_hs": np.nan,
                "norm_ps": np.nan,
                "norm_md": np.nan,
                "total_norm_score": np.nan, # Add NaN for total score on error
                "status": "execution_error"
            })
        finally:
             pbar.update(1) # Update progress bar

pbar.close()
print("-" * 30)

# Create DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')

print(f"Calibration finished. Results saved to {OUTPUT_CSV}")
print(df.status.value_counts())

# Check if there were errors
if df['status'].ne('success').any():
    print("\nSome runs encountered errors or timeouts. Check the CSV and log output.")
else:
    print("\nAll runs completed successfully.")
