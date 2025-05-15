#!/usr/bin/env python3
"""
Simulated Annealing (SA) heuristic for the fair round-robin scheduling problem.

This script implements an SA algorithm to find high-quality, equitable
round-robin schedules by minimizing a weighted sum of fairness metrics
using empirical normalization.

Usage:
    python src/sa_solver_non_opti.py [n_players] [iterations] [alpha_pen_seq] [beta]

Args:
    n_players (int, optional): Number of players (must be even). Defaults to 6.
    iterations (int, optional): Number of SA iterations. Defaults to 10000.
    alpha_pen_seq (float, optional): Weight for the Pénalité de séquence objective term. Defaults to 1.0.
    beta (float, optional): Weight for the Max Deviation objective term. Defaults to 1.0.
"""
import sys
import os
import random
import math
import copy
import numpy as np
import os # Add os import

# Add the project root directory to sys.path
current_dir_sa_non_opti = os.path.dirname(os.path.abspath(__file__))
project_root_sa_non_opti = os.path.abspath(os.path.join(current_dir_sa_non_opti, os.pardir))
if project_root_sa_non_opti not in sys.path:
    sys.path.insert(0, project_root_sa_non_opti)

# Import necessary functions for raw metric calculation
from src.metrics import calculate_home_strength, calculate_raw_max_deviation, calculate_raw_total_penalty_sequence
# Import necessary functions from normalization_manager
from src.normalization_manager import calculate_normalized_score, calculate_analytical_factors # Updated import
# Import schedule generator utility
from src.schedule_utils import initial_schedule
from src import config # Import the configuration

def compute_metrics(schedule, n):
    """
    Computes the fairness metrics for a given schedule.

    Metrics:
        - Delta HomeStrength: Sum of opponent indices (1-based) for games where player 'i' is home vs 'j'.
        - Pénalité de séquence: Total count of consecutive home or away games for any player.
        - Max Deviation: Maximum absolute deviation of the number of home games
          per player from the ideal average ((n-1)/2).

    Args:
        schedule (list): The schedule, a list of rounds, each a list of (home, away) tuples (1-based indices).
        n (int): The number of players.

    Returns:
        tuple: (delta_strength, penalites_sequence, max_deviation)
    """

    raw_home_strength = calculate_home_strength(schedule, n)
    penalites_sequence = calculate_raw_total_penalty_sequence(schedule, n)
    max_dev = calculate_raw_max_deviation(schedule, n)

    return raw_home_strength, penalites_sequence, max_dev


def neighbor(schedule, n):
    """
    Generates a neighboring schedule by flipping the home/away assignment
    of a randomly selected match in a randomly selected round.

    Args:
        schedule (list): The current schedule (with 1-based indices).
        n (int): The number of players (must be even for this implementation).

    Returns:
        list: A new schedule representing a neighbor of the input schedule (with 1-based indices).

    Raises:
        ValueError: If n is odd, as the standard schedule structure assumes even n.
        IndexError: If the schedule is malformed (e.g., empty rounds).
    """
    new_sched = copy.deepcopy(schedule)
    if n % 2 != 0:
        # This simple neighbor assumes the structure derived from even n
        raise ValueError("Neighbor generation requires even n for this schedule structure.")

    num_rounds = n - 1
    if num_rounds <= 0: return new_sched # Handle n=0 or n=2 case

    # Select a non-empty round and a match within it
    try:
        # Filter out potentially empty rounds if initial_schedule produced them for odd n
        valid_round_indices = [r for r, rnd in enumerate(new_sched) if rnd]
        if not valid_round_indices: return new_sched # No matches to flip

        rnd_idx = random.choice(valid_round_indices)
        round_len = len(new_sched[rnd_idx])
        if round_len == 0: return new_sched # Should not happen with filtering, but safe check

        match_idx = random.randrange(round_len)

        # Flip home/away
        home, away = new_sched[rnd_idx][match_idx]
        new_sched[rnd_idx][match_idx] = (away, home)
    except IndexError as e:
        print(f"Warning: IndexError during neighbor generation (schedule might be malformed?). Error: {e}")
        # Return original schedule if error occurs
        return schedule
    except Exception as e:
         print(f"Warning: Unexpected error during neighbor generation: {e}")
         return schedule

    return new_sched


def solve_sa(n, iterations=10000, initial_temp=1.0, cooling_rate=0.95, alpha_pen_seq=None, beta_obj=None, seed=42):
    """
    Solves the fair round-robin problem using Simulated Annealing.

    Args:
        n (int): Number of players (must be even).
        iterations (int, optional): Number of SA iterations. Defaults to 10000.
        initial_temp (float, optional): Initial temperature. Defaults to 1.0.
        cooling_rate (float, optional): Temp. cooling rate. Defaults to 0.95.
        alpha_pen_seq (float, optional): Weight for Pénalité de séquence objective. Defaults to 1.0.
        beta_obj (float, optional): Weight for Max Deviation objective. Defaults to 1.0.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (best_schedule, best_norm_score, raw_metrics, analytical_metrics) where
               raw_metrics is (raw_home_strength, raw_penalites_sequence, raw_max_deviation) and
               analytical_metrics is (anal_norm_hs, anal_norm_ps, anal_norm_md).

    Raises:
        ValueError: If n is odd.
    """
    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA
    random.seed(seed)
    if n % 2 != 0:
        # The current neighbor function relies on the structure for even n
        raise ValueError("SA solver currently requires an even number of players (n).")

    current_sched = initial_schedule(n)
    if not current_sched: # Handle case where initial schedule is empty (e.g., n=0)
        # Return default values for failure case
        return [], float('inf'), (float('inf'), float('inf'), float('inf')), (float('inf'), float('inf'), float('inf'))

    # --- Normalization Setup ---
    # Analytical factors are calculated within calculate_normalized_score

    # --- Initialization ---
    c_home_strength, c_penalites_sequence, c_max_dev = compute_metrics(current_sched, n)

    # Calculate initial normalized score using the centralized function
    current_norm_score, current_anal_hs, current_anal_ps, current_anal_md = calculate_normalized_score(
        c_home_strength, c_penalites_sequence, c_max_dev,
        alpha_pen_seq, beta_obj, n # Pass n
    )
    current_unnorm_score = c_home_strength + alpha_pen_seq * c_penalites_sequence + beta_obj * c_max_dev # For reporting

    best_sched = copy.deepcopy(current_sched)
    best_norm_score = current_norm_score
    best_unnorm_score = current_unnorm_score # Keep track of best unnormalized score too
    best_raw_metrics = (c_home_strength, c_penalites_sequence, c_max_dev)
    best_analytical_metrics = (current_anal_hs, current_anal_ps, current_anal_md)


    T = initial_temp

    print(f"Initial Score (Analytical Normalized): {current_norm_score:.4f} (Raw HS: {c_home_strength}, Raw PS: {c_penalites_sequence}, Raw MD: {c_max_dev:.2f})")

    for it in range(iterations):
        candidate_sched = neighbor(current_sched, n)
        cand_home_strength, cand_penalites_sequence, cand_max_dev = compute_metrics(candidate_sched, n)

        # Calculate candidate normalized score using the centralized function
        candidate_norm_score, cand_anal_hs, cand_anal_ps, cand_anal_md = calculate_normalized_score(
            cand_home_strength, cand_penalites_sequence, cand_max_dev,
            alpha_pen_seq, beta_obj, n # Pass n
        )
        candidate_unnorm_score = cand_home_strength + alpha_pen_seq * cand_penalites_sequence + beta_obj * cand_max_dev # For reporting

        # Acceptance criterion (using NORMALIZED scores)
        delta_norm_score = candidate_norm_score - current_norm_score
        if delta_norm_score < 0 or random.random() < math.exp(-delta_norm_score / T):
            current_sched = candidate_sched
            current_norm_score = candidate_norm_score
            current_unnorm_score = candidate_unnorm_score # Update reported score
            current_anal_hs, current_anal_ps, current_anal_md = cand_anal_hs, cand_anal_ps, cand_anal_md # Update analytical norms

            # Update best found solution (based on NORMALIZED score)
            if current_norm_score < best_norm_score:
                best_sched = copy.deepcopy(current_sched)
                best_norm_score = current_norm_score
                best_unnorm_score = candidate_unnorm_score # Store corresponding unnormalized score
                best_raw_metrics = (cand_home_strength, cand_penalites_sequence, cand_max_dev)
                best_analytical_metrics = (cand_anal_hs, cand_anal_ps, cand_anal_md)


        # Cool down temperature
        T *= cooling_rate
        if T < 1e-5: # Prevent T from becoming too small
             T = 1e-5

        if it % (iterations // 10) == 0:
             # Report the analytical normalized scores and the overall normalized score
             print(f"Iter {it}/{iterations}, Temp: {T:.4f}, Current Norm Score: {current_norm_score:.4f}, Best Norm Score: {best_norm_score:.4f}")
             print(f"  Current Anal Norms: (HS: {current_anal_hs:.4f}, PS: {current_anal_ps:.4f}, MD: {current_anal_md:.4f})")


    final_raw_hs, final_raw_ps, final_raw_md = best_raw_metrics
    final_anal_hs, final_anal_ps, final_anal_md = best_analytical_metrics

    # Return the best schedule, the analytical normalized score, raw metrics, and analytical metrics
    print(f"Final Best Score (Analytical Normalized): {best_norm_score:.4f} (Raw HS: {final_raw_hs}, Raw PS: {final_raw_ps}, Raw MD: {final_raw_md:.2f})")
    print(f"Final Best Analytical Norms: (HS: {final_anal_hs:.4f}, PS: {final_anal_ps:.4f}, MD: {final_anal_md:.4f})")

    return best_sched, best_norm_score, best_raw_metrics, best_analytical_metrics


def main():
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    iters_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    alpha_pen_seq_arg = float(sys.argv[3]) if len(sys.argv) > 3 else config.ALPHA
    beta_arg = float(sys.argv[4]) if len(sys.argv) > 4 else config.BETA

    if n_arg % 2 != 0:
        print(f"Error: Number of players (n={n_arg}) must be even.")
        sys.exit(1)

    print(f"Running SA (Non-Optimized) for n={n_arg}, iterations={iters_arg}, alpha_pen_seq={alpha_pen_seq_arg}, beta={beta_arg}")

    # solve_sa now returns best_schedule, best_norm_score, raw_metrics, analytical_metrics
    best_schedule, best_norm_score, raw_metrics, analytical_metrics = solve_sa(
        n_arg,
        iterations=iters_arg,
        alpha_pen_seq=alpha_pen_seq_arg, # Pass explicitly
        beta_obj=beta_arg # Pass explicitly
    )

    # Unpack raw and analytical metrics
    final_raw_hs, final_raw_ps, final_raw_md = raw_metrics
    final_anal_hs, final_anal_ps, final_anal_md = analytical_metrics

    # Report the analytical normalized score and detailed metrics
    print(f"\n--- Best SA Schedule (Analytical Normalized Score: {best_norm_score:.4f}) ---")
    print(f"Raw Metrics: HomeStrength={final_raw_hs}, Total Pénalités Séquence={final_raw_ps}, Max Deviation={final_raw_md:.2f}")
    print(f"Analytical Normalized Metrics (Z-Scores): HS={final_anal_hs:.4f}, PS={final_anal_ps:.4f}, MD={final_anal_md:.4f}")

    for r, rnd in enumerate(best_schedule):
        # Format matches for readability (using 1-based indices)
        match_strs = [f"{h}v{a}(H)" for h, a in rnd]
        print(f"Round {r+1}: {', '.join(match_strs)}") # Round numbers are naturally 1-based for display

if __name__ == '__main__':
    main()
