import os
import sys
import time # Add time for measuring execution
import multiprocessing # Import multiprocessing
import logging # Ensure logging is imported
try:
    import numba # Import Numba at the top level
except ImportError:
    numba = None # Define numba as None if import fails
    print("Numba not found, proceeding without Numba thread configuration.")


# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import sa_solver
import config # Import config to potentially access ALPHA, BETA if needed directly

# Configure logging for the profiler script itself if needed, or rely on sa_solver's
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_sa_experiments():
    log.info("Starting SA solver experiments...")

    n_arg = 1000 # Set n_arg to 1000 for large instance test
    
    # Parameter sets to test
    # Format: (iterations, initial_temp, alpha_pen_seq, beta_obj, description, log_interval)
    parameter_sets = [
        (1000000, 1.0, None, None, "1M iters, T0=1.0, n=1000, log_interval=100k", 100000), # For n=1000 test
    ]

    results = []

    for i, (iters, temp, alpha, beta, desc, log_interval) in enumerate(parameter_sets): # New loop
        log.info(f"--- Experiment {i+1}/{len(parameter_sets)}: {desc} ---")
        start_time = time.time()
        
        if numba: # Check if numba was imported successfully
            # Attempt to set Numba threads - user requested 11
            # Numba defaults to a number of threads equal to the number of CPUs.
            # We can try to set it, but it might be overridden or managed internally by Numba.
            num_threads_to_use = 11 
            actual_cpu_count = multiprocessing.cpu_count()
            if num_threads_to_use > actual_cpu_count:
                log.warning(f"Requested {num_threads_to_use} threads, but only {actual_cpu_count} are available. Numba will likely use {actual_cpu_count}.")
                # Numba will cap at actual_cpu_count anyway, no need to explicitly set num_threads_to_use to actual_cpu_count here for set_num_threads
            
            try:
                # Only set if we want to override Numba's default for the jitted functions
                # For many typical JIT-compiled loops, Numba parallel=True implies using all available threads.
                # Explicitly setting might be useful if we want *fewer* than all cores.
                # Given the user wants 11 (out of likely 12), this is a slight reduction from default.
                numba.set_num_threads(num_threads_to_use) 
                log.info(f"Attempted to set Numba threads to {num_threads_to_use}. Actual threads used by Numba: {numba.get_num_threads()}.")
            except Exception as e:
                log.error(f"Error setting Numba thread count: {e}. Numba will use its default.")
        else:
            log.info("Numba not available. Skipping Numba thread configuration. SA solver will run without Numba JIT if Numba is not in its environment either.")

        best_schedule, best_score, best_metrics = sa_solver.solve_sa(
            n=n_arg,
            iterations=iters,
            initial_temp=temp,
            alpha_pen_seq=alpha, # Uses config default if None
            beta_obj=beta,       # Uses config default if None
            seed=42 + i, # Vary seed slightly for different runs
            log_interval_sa_loop=log_interval # Pass the log interval
        )
        end_time = time.time()
        duration = end_time - start_time

        log.info(f"Finished Experiment {i+1} in {duration:.2f} seconds.")
        log.info(f"  Description: {desc}")
        log.info(f"  Best Unnormalized Score: {best_score:.4f}")
        log.info(f"  Best Metrics (HS, PenSeq, MaxDev): {best_metrics}")
        # log.info(f"  Best Schedule: {best_schedule}") # Can be very verbose
        results.append({
            "description": desc,
            "iterations": iters,
            "initial_temp": temp,
            "alpha_pen_seq": alpha if alpha is not None else config.ALPHA,
            "beta_obj": beta if beta is not None else config.BETA,
            "score": best_score,
            "metrics": best_metrics,
            "duration": duration
        })
        log.info("--------------------------------------------------\\n")

    log.info("=== SA Solver Experiment Summary ===")
    for res in results:
        log.info(f"Desc: {res['description']}, Score: {res['score']:.4f}, Metrics: {res['metrics']}, Duration: {res['duration']:.2f}s")
    
    # Ensure exact solver part is commented out for n=1000 run
    # if n_arg == 6: 
    #     log.info("\\\\n=== Running Exact Solver for n=6 for Comparison ===")
    #     try:
    #         from exact_model import solve_exact 
    #         exact_start_time = time.time()
    #         exact_results = solve_exact(n=n_arg, alpha_pen_seq=config.ALPHA, beta=config.BETA, time_limit=60) 
    #         exact_duration = time.time() - exact_start_time
    #         if exact_results and exact_results.get("status") == "Optimal":
    #             log.info(f"Exact Solver (n=6) completed in {exact_duration:.2f}s.")
    #             log.info(f"  Optimal Objective Value: {exact_results.get('objective_value'):.4f}")
    #             exact_metrics = exact_results.get('metrics', {})
    #             log.info(f"  Metrics (HS, PenSeq, MaxDev): ({exact_metrics.get('home_strength')}, {exact_metrics.get('penalites_sequence')}, {exact_metrics.get('max_deviation')})")
    #         else:
    #             log.warning(f"Exact Solver (n=6) did not find an optimal solution. Status: {exact_results.get('status')}")
    #     except ImportError:
    #         log.error("Could not import `exact_model.solve_exact`. Ensure it's in the Python path (e.g., src/).")
    #     except Exception as e:
    #         log.error(f"Error running exact_model for n=6: {e}")

if __name__ == "__main__":
    print("Profiler script started...")
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        import sa_solver # Re-check import after path manipulation
        import config
        # from exact_model import solve_exact # Not needed for n=1000 run
    except ImportError as e:
        log.error(f"Failed to import modules. Ensure 'src' is in sys.path. Error: {e}")
        exit(1)
        
    run_sa_experiments()
