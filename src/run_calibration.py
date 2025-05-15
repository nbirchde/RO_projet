import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
import time # For seeding SA runs uniquely
import multiprocessing # Added for freeze_support

# Import necessary functions from your project
from src.sa_solver import solve_sa, solve_sa_parallel
from src.metrics import get_all_fairness_metrics # To get theoretically normalized metrics for final reporting and Pareto plotting
# Removed import of get_or_calculate_normalization_factors

# --- Logging Configuration ---
# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
N_PLAYERS = 100
ITERATIONS = 1000000  # Iterations for each SA run
RUNS_PER_COMBINATION = 6 # Number of parallel SA chains for each (alpha, beta)
ALPHA_VALUES = np.arange(0.5, 1.51, 0.1) # 0.5 to 1.5 inclusive, step 0.1
BETA_VALUES = np.arange(0.5, 1.51, 0.1)  # 0.5 to 1.5 inclusive, step 0.1
OUTPUT_CSV = f"calibration_results_n{N_PLAYERS}_analytical_norm.csv" # Updated for analytical normalization
# Removed empirical normalization configuration
SA_BASE_SEED = 1000 # Base seed for SA runs, will be incremented for each combination

# --- End Configuration ---

if __name__ == '__main__':
    multiprocessing.freeze_support() # Added to fix multiprocessing runtime error

    results = []

    log.info(f"Starting calibration for n={N_PLAYERS}, iterations={ITERATIONS}, runs_per_combination={RUNS_PER_COMBINATION}")
    log.info(f"Alpha range: {ALPHA_VALUES.min():.2f} to {ALPHA_VALUES.max():.2f} (step {ALPHA_VALUES[1]-ALPHA_VALUES[0]:.1f})")
    log.info(f"Beta range: {BETA_VALUES.min():.2f} to {BETA_VALUES.max():.2f} (step {BETA_VALUES[1]-BETA_VALUES[0]:.1f})")
    total_combinations = len(ALPHA_VALUES) * len(BETA_VALUES)
    log.info(f"Total (alpha, beta) combinations: {total_combinations}")
    log.info(f"Output file: {OUTPUT_CSV}")
    log.info("-" * 30)

    # Removed empirical normalization calculation section

    # Use tqdm for progress bar
    pbar = tqdm(total=total_combinations, desc="Running SA combinations")

    current_sa_run_seed = SA_BASE_SEED

    for alpha_raw in ALPHA_VALUES:
        for beta_raw in BETA_VALUES:
            alpha = round(alpha_raw, 2) # Ensure consistent float representation
            beta = round(beta_raw, 2)
            
            # These are the parameters SA will use for its objective function
            # Pass n instead of empirical factors
            sa_kwargs = {
                'alpha_pen_seq': alpha,
                'beta_obj': beta,
                'n_players': N_PLAYERS, # Pass n to the solver
                'log_interval_sa_loop': 0, # Suppress verbose SA loop logging during calibration
            }

            try:
                log.debug(f"Running SA for alpha={alpha:.2f}, beta={beta:.2f} with seed {current_sa_run_seed}")
                if RUNS_PER_COMBINATION > 1:
                    # solve_sa_parallel returns: best_schedule, best_score (analytical), raw_metrics, analytical_metrics
                    best_schedule_list, analytical_obj_score, raw_metrics, analytical_metrics = solve_sa_parallel(
                        n=N_PLAYERS,
                        iterations=ITERATIONS,
                        runs=RUNS_PER_COMBINATION,
                        seed=current_sa_run_seed,
                        **sa_kwargs
                    )
                else:
                    # solve_sa returns: best_schedule, best_score (analytical), raw_metrics, analytical_metrics
                    best_schedule_list, analytical_obj_score, raw_metrics, analytical_metrics = solve_sa(
                        n=N_PLAYERS,
                        iterations=ITERATIONS,
                        seed=current_sa_run_seed,
                        **sa_kwargs
                    )

                current_sa_run_seed += RUNS_PER_COMBINATION # Increment seed for next (alpha,beta) pair

                if not best_schedule_list:
                    log.warning(f"SA returned no schedule for alpha={alpha:.2f}, beta={beta:.2f}. Recording as failure.")
                    results.append({
                        "alpha": alpha, "beta": beta,
                        "anal_norm_hs": np.nan, "anal_norm_ps": np.nan, "anal_norm_md": np.nan,
                        "total_analytical_score": np.nan, # This is the score SA optimized
                        "raw_hs": np.nan, "raw_ps": np.nan, "raw_md": np.nan,
                        "status": "sa_returned_empty"
                    })
                    pbar.update(1)
                    continue

                # Raw metrics from the SA solver
                raw_hs, raw_ps, raw_md = raw_metrics

                # Analytical normalized metrics (z-scores) from the SA solver
                anal_norm_hs, anal_norm_ps, anal_norm_md = analytical_metrics

                # The total_analytical_score is what SA optimized, based on analytical norms + current alpha/beta
                total_analytical_score = analytical_obj_score # Already returned by solver

                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "anal_norm_hs": anal_norm_hs, # Analytically normalized HS (for Pareto plot)
                    "anal_norm_ps": anal_norm_ps, # Analytically normalized PS (for Pareto plot)
                    "anal_norm_md": anal_norm_md, # Analytically normalized MD (for Pareto plot)
                    "total_analytical_score": total_analytical_score, # Score SA optimized (based on analytical norms + current alpha/beta)
                    "raw_hs": raw_hs, # Raw HS value from SA's best
                    "raw_ps": raw_ps, # Raw PS value from SA's best
                    "raw_md": raw_md, # Raw MD value from SA's best
                    "status": "success"
                })
                log.debug(f"Success for alpha={alpha:.2f}, beta={beta:.2f}. Analytical Score: {analytical_obj_score:.4f}. Anal Norm HS: {anal_norm_hs:.4f}, PS: {anal_norm_ps:.4f}, MD: {anal_norm_md:.4f}")

            except Exception as e:
                log.error(f"Error running SA for alpha={alpha:.2f}, beta={beta:.2f}: {e}", exc_info=True)
                results.append({
                    "alpha": alpha, "beta": beta,
                    "anal_norm_hs": np.nan, "anal_norm_ps": np.nan, "anal_norm_md": np.nan,
                    "total_analytical_score": np.nan,
                    "raw_hs": np.nan, "raw_ps": np.nan, "raw_md": np.nan,
                    "status": "execution_error"
                })
            finally:
                 pbar.update(1)

    pbar.close()
    log.info("-" * 30)

    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        # Define column order for clarity
        column_order = [
            "alpha", "beta", "status", 
            "anal_norm_hs", "anal_norm_ps", "anal_norm_md", # Analytically normalized for Pareto
            "total_analytical_score", 
            "raw_hs", "raw_ps", "raw_md"
        ]
        # Reorder if all columns are present, otherwise use existing order
        df_columns = [col for col in column_order if col in df.columns]
        df = df[df_columns]
        
        df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
        log.info(f"Calibration finished. Results saved to {OUTPUT_CSV}")
        log.info("Status counts:\\n" + str(df.status.value_counts(dropna=False)))
        
        successful_runs = df[df['status'] == 'success'].shape[0]
        log.info(f"Number of successful runs: {successful_runs}/{total_combinations}")

        if successful_runs < total_combinations:
            log.warning("Some runs encountered errors or did not complete successfully. Check the CSV and log output.")
        else:
            log.info("All combinations recorded a \'success\' status.")
    else:
        log.warning("No results were generated during calibration.")

    log.info("Calibration script completed.")
