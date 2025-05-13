import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
import time # For seeding SA runs uniquely
import multiprocessing # Added for freeze_support

# Import necessary functions from your project
from src.sa_solver import get_empirical_normalization_factors, solve_sa, solve_sa_parallel
from src.metrics import get_all_fairness_metrics # To get theoretically normalized metrics for final reporting and Pareto plotting

# --- Logging Configuration ---
# Configure logging to show info level messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
N_PLAYERS = 200
ITERATIONS = 50000  # Iterations for each SA run
RUNS_PER_COMBINATION = 6 # Number of parallel SA chains for each (alpha, beta)
ALPHA_VALUES = np.arange(0.5, 1.51, 0.1) # 0.5 to 1.5 inclusive, step 0.1
BETA_VALUES = np.arange(0.5, 1.51, 0.1)  # 0.5 to 1.5 inclusive, step 0.1
OUTPUT_CSV = f"calibration_results_n{N_PLAYERS}_empirical_norm_v2_median_subtracted.csv" # UPDATED for new normalization
NUM_EMPIRICAL_SAMPLES = 200 # Number of samples for empirical normalization
EMPIRICAL_FACTORS_SEED = 42 # Seed for generating empirical factors (for reproducibility)
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

    # 1. Calculate Empirical Normalization Factors (once)
    log.info(f"Calculating empirical normalization factors using {NUM_EMPIRICAL_SAMPLES} samples with seed {EMPIRICAL_FACTORS_SEED} for n={N_PLAYERS}...")
    # The get_empirical_normalization_factors function in sa_solver already logs its progress.
    med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md = get_empirical_normalization_factors(
        N_PLAYERS, num_samples=NUM_EMPIRICAL_SAMPLES, seed=EMPIRICAL_FACTORS_SEED
    )
    # Log the obtained factors from this script as well for clarity
    log.info(f"Successfully obtained empirical factors for n={N_PLAYERS}:")
    log.info(f"  HS: Median={med_hs:.4f}, Sigma={sigma_hs:.4f}")
    log.info(f"  PS: Median={med_ps:.4f}, Sigma={sigma_ps:.4f}")
    log.info(f"  MD: Median={med_md:.4f}, Sigma={sigma_md:.4f}")
    log.info("-" * 30)

    # Use tqdm for progress bar
    pbar = tqdm(total=total_combinations, desc="Running SA combinations")

    current_sa_run_seed = SA_BASE_SEED

    for alpha_raw in ALPHA_VALUES:
        for beta_raw in BETA_VALUES:
            alpha = round(alpha_raw, 2) # Ensure consistent float representation
            beta = round(beta_raw, 2)
            
            # These are the parameters SA will use for its objective function
            sa_kwargs = {
                'alpha_pen_seq': alpha,
                'beta_obj': beta,
                'med_hs': med_hs,
                'sigma_hs': sigma_hs,
                'med_ps': med_ps,
                'sigma_ps': sigma_ps,
                'med_md': med_md,
                'sigma_md': sigma_md,
                'log_interval_sa_loop': 0, # Suppress verbose SA loop logging during calibration
                'num_empirical_samples': NUM_EMPIRICAL_SAMPLES # Pass for completeness, though factors are pre-calculated
            }

            try:
                log.debug(f"Running SA for alpha={alpha:.2f}, beta={beta:.2f} with seed {current_sa_run_seed}")
                if RUNS_PER_COMBINATION > 1:
                    # solve_sa_parallel returns: best_schedule, best_score (empirical), best_metrics (raw)
                    best_schedule_list, empirical_obj_score, raw_metrics = solve_sa_parallel(
                        n=N_PLAYERS, 
                        iterations=ITERATIONS, 
                        runs=RUNS_PER_COMBINATION, 
                        seed=current_sa_run_seed, 
                        **sa_kwargs
                    )
                else:
                    # solve_sa returns: best_schedule, best_score (empirical), best_metrics (raw)
                    best_schedule_list, empirical_obj_score, raw_metrics = solve_sa(
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
                        "norm_hs": np.nan, "norm_ps": np.nan, "norm_md": np.nan,
                        "total_norm_score": np.nan, "empirical_score": np.nan, # empirical_score is what SA optimized
                        "raw_hs": np.nan, "raw_ps": np.nan, "raw_md": np.nan,
                        "status": "sa_returned_empty"
                    })
                    pbar.update(1)
                    continue

                # Raw metrics from the SA solver (what it found as best for its empirical objective)
                raw_hs, raw_ps, raw_md = raw_metrics

                # Calculate empirically normalized metrics (raw_metric - median_metric) / sigma_metric)
                # These are the values SA's objective function components are based on
                # Handle potential division by zero if sigma is zero, though unlikely for these metrics
                emp_norm_hs = (raw_hs - med_hs) / sigma_hs if sigma_hs != 0 else np.nan
                emp_norm_ps = (raw_ps - med_ps) / sigma_ps if sigma_ps != 0 else np.nan
                emp_norm_md = (raw_md - med_md) / sigma_md if sigma_md != 0 else np.nan

                # For reporting and verification, get metrics normalized by THEORETICAL maximums
                # This uses the project's standard get_all_fairness_metrics
                metrics_for_reporting = get_all_fairness_metrics(best_schedule_list, N_PLAYERS)
                
                reported_norm_hs = metrics_for_reporting.get('normalized_home_strength', np.nan)
                reported_norm_ps = metrics_for_reporting.get('normalized_total_penalty_sequence', np.nan)
                reported_norm_md = metrics_for_reporting.get('normalized_max_deviation', np.nan)

                # This is the score based on theoretical normalization, for comparison/plotting
                # It uses the alpha and beta of the current iteration
                total_theoretical_norm_score = reported_norm_hs + alpha * reported_norm_ps + beta * reported_norm_md
                
                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "norm_hs": reported_norm_hs, # Theoretically normalized HS (for verification/other reports)
                    "norm_ps": reported_norm_ps, # Theoretically normalized PS (for verification/other reports)
                    "norm_md": reported_norm_md, # Theoretically normalized MD (for verification/other reports)
                    "total_norm_score": total_theoretical_norm_score, # Score based on theoretical norms + current alpha/beta
                    "empirical_score": empirical_obj_score, # Score SA optimized (based on empirical sigmas + current alpha/beta)
                    "emp_norm_hs": emp_norm_hs, # Empirically normalized HS (for Pareto plot)
                    "emp_norm_ps": emp_norm_ps, # Empirically normalized PS (for Pareto plot)
                    "emp_norm_md": emp_norm_md, # Empirically normalized MD (for Pareto plot)
                    "raw_hs": raw_hs, # Raw HS value from SA's best
                    "raw_ps": raw_ps, # Raw PS value from SA's best
                    "raw_md": raw_md, # Raw MD value from SA's best
                    "status": "success"
                })
                log.debug(f"Success for alpha={alpha:.2f}, beta={beta:.2f}. Empirical Score: {empirical_obj_score:.4f}. Reported HS_n: {reported_norm_hs:.4f}, PS_n: {reported_norm_ps:.4f}, MD_n: {reported_norm_md:.4f}")

            except Exception as e:
                log.error(f"Error running SA for alpha={alpha:.2f}, beta={beta:.2f}: {e}", exc_info=True)
                results.append({
                    "alpha": alpha, "beta": beta,
                    "norm_hs": np.nan, "norm_ps": np.nan, "norm_md": np.nan,
                    "total_norm_score": np.nan, "empirical_score": np.nan,
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
            "emp_norm_hs", "emp_norm_ps", "emp_norm_md", # Empirically normalized for Pareto
            "norm_hs", "norm_ps", "norm_md", "total_norm_score", # Theoretically normalized for other reporting
            "empirical_score", 
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
