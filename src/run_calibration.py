import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
import time
import multiprocessing

from src.sa_solver import solve_sa, solve_sa_parallel
from src.metrics import get_all_fairness_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log = logging.getLogger(__name__)

N_PLAYERS = 300
ITERATIONS = 1000000
RUNS_PER_COMBINATION = 8
ALPHA_VALUES = np.arange(0.5, 1.51, 0.1)
BETA_VALUES = np.arange(0.5, 1.51, 0.1)
OUTPUT_CSV = f"calibration_results_n{N_PLAYERS}_analytical_norm.csv"
SA_BASE_SEED = 1000

if __name__ == '__main__':
    multiprocessing.freeze_support()

    results = []

    log.info(f"Starting calibration for n={N_PLAYERS}, iterations={ITERATIONS}, runs_per_combination={RUNS_PER_COMBINATION}")
    log.info(f"Alpha range: {ALPHA_VALUES.min():.2f} to {ALPHA_VALUES.max():.2f} (step {ALPHA_VALUES[1]-ALPHA_VALUES[0]:.1f})")
    log.info(f"Beta range: {BETA_VALUES.min():.2f} to {BETA_VALUES.max():.2f} (step {BETA_VALUES[1]-BETA_VALUES[0]:.1f})")
    total_combinations = len(ALPHA_VALUES) * len(BETA_VALUES)
    log.info(f"Total (alpha, beta) combinations: {total_combinations}")
    log.info(f"Output file: {OUTPUT_CSV}")
    log.info("-" * 30)

    pbar = tqdm(total=total_combinations, desc="Running SA combinations")

    current_sa_run_seed = SA_BASE_SEED

    for alpha_raw in ALPHA_VALUES:
        for beta_raw in BETA_VALUES:
            alpha = round(alpha_raw, 2)
            beta = round(beta_raw, 2)

            sa_kwargs = {
                'alpha_pen_seq': alpha,
                'beta_obj': beta,
                'n_players': N_PLAYERS,
                'log_interval_sa_loop': 0,
            }

            try:
                log.debug(f"Running SA for alpha={alpha:.2f}, beta={beta:.2f} with seed {current_sa_run_seed}")
                if RUNS_PER_COMBINATION > 1:
                    best_schedule_list, analytical_obj_score, raw_metrics, analytical_metrics, scaled_obj_score, scaled_metrics = solve_sa_parallel(
                        n=N_PLAYERS,
                        iterations=ITERATIONS,
                        runs=RUNS_PER_COMBINATION,
                        seed=current_sa_run_seed,
                        **sa_kwargs
                    )
                else:
                    best_schedule_list, analytical_obj_score, raw_metrics, analytical_metrics, scaled_obj_score, scaled_metrics = solve_sa(
                        n=N_PLAYERS,
                        iterations=ITERATIONS,
                        seed=current_sa_run_seed,
                        **sa_kwargs
                    )

                current_sa_run_seed += RUNS_PER_COMBINATION

                if not best_schedule_list:
                    log.warning(f"SA returned no schedule for alpha={alpha:.2f}, beta={beta:.2f}. Recording as failure.")
                    results.append({
                        "alpha": alpha, "beta": beta,
                        "anal_norm_hs": np.nan, "anal_norm_ps": np.nan, "anal_norm_md": np.nan,
                        "total_analytical_score": np.nan,
                        "scaled_hs": np.nan, "scaled_ps": np.nan, "scaled_md": np.nan,
                        "total_scaled_score": np.nan,
                        "raw_hs": np.nan, "raw_ps": np.nan, "raw_md": np.nan,
                        "status": "sa_returned_empty"
                    })
                    pbar.update(1)
                    continue

                raw_hs, raw_ps, raw_md = raw_metrics
                anal_norm_hs, anal_norm_ps, anal_norm_md = analytical_metrics
                scaled_hs, scaled_ps, scaled_md = scaled_metrics
                total_analytical_score = analytical_obj_score
                total_scaled_score = scaled_obj_score

                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "anal_norm_hs": anal_norm_hs,
                    "anal_norm_ps": anal_norm_ps,
                    "anal_norm_md": anal_norm_md,
                    "total_analytical_score": total_analytical_score,
                    "scaled_hs": scaled_hs,
                    "scaled_ps": scaled_ps,
                    "scaled_md": scaled_md,
                    "total_scaled_score": total_scaled_score,
                    "raw_hs": raw_hs,
                    "raw_ps": raw_ps,
                    "raw_md": raw_md,
                    "status": "success"
                })
                log.debug(f"Success for alpha={alpha:.2f}, beta={beta:.2f}. Analytical Score: {analytical_obj_score:.4f}. Scaled Score: {scaled_obj_score:.4f}. Anal Norm HS: {anal_norm_hs:.4f}, PS: {anal_norm_ps:.4f}, MD: {anal_norm_md:.4f}. Scaled HS: {scaled_hs:.4f}, PS: {scaled_ps:.4f}, MD: {scaled_md:.4f}")

            except Exception as e:
                log.error(f"Error running SA for alpha={alpha:.2f}, beta={beta:.2f}: {e}", exc_info=True)
                results.append({
                    "alpha": alpha, "beta": beta,
                    "anal_norm_hs": np.nan, "anal_norm_ps": np.nan, "anal_norm_md": np.nan,
                    "total_analytical_score": np.nan,
                    "scaled_hs": np.nan, "scaled_ps": np.nan, "scaled_md": np.nan,
                    "total_scaled_score": np.nan,
                    "raw_hs": np.nan, "raw_ps": np.nan, "raw_md": np.nan,
                    "status": "execution_error"
                })
            finally:
                 pbar.update(1)

    pbar.close()
    log.info("-" * 30)

    if results:
        df = pd.DataFrame(results)
        column_order = [
            "alpha", "beta", "status",
            "anal_norm_hs", "anal_norm_ps", "anal_norm_md",
            "total_analytical_score",
            "scaled_hs", "scaled_ps", "scaled_md",
            "total_scaled_score",
            "raw_hs", "raw_ps", "raw_md"
        ]
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
