#!/usr/bin/env python3
"""
Simulated Annealing (SA) heuristic for the fair round-robin scheduling problem.

This script implements an SA algorithm to find high-quality, equitable
round-robin schedules by minimizing the same objective function used in the
exact model: Z = Delta_HomeStrength + alpha_pen_seq * Pénalité_de_séquence + beta * Max_Deviation.

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
import numpy as np # For calculating standard deviation if needed later, using max deviation for now

# Add the project root directory to sys.path to enable importing modules from src
# This allows running the script directly from the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

# Now import modules directly
from metrics import calculate_home_strength, normalize_home_strength, normalize_total_pen_seq, normalize_max_dev
import config # Import the configuration

def initial_schedule(n):
    """
    Generates an initial round-robin schedule using the circle method.

    Assigns home/away arbitrarily in the first instance. Handles odd 'n'
    by adding a dummy player (None) for bye rounds, although the main
    solver logic assumes even 'n'.

    Args:
        n (int): Number of players.

    Returns:
        list: A list of rounds, where each round is a list of
              (home_player, away_player) tuples (using 1-based indexing).
    """
    players = list(range(1, n + 1)) # Use 1-based indexing
    original_n = n
    if n % 2:
        # If n is odd, add a dummy player for the circle method logic
        players.append(None)
        n_effective = n + 1 # Use n_effective for circle method calculation
    else:
        n_effective = n

    half = n_effective // 2
    schedule = []
    for r in range(n_effective - 1): # Iterate through rounds needed for n_effective players
        round_pairs = []
        for i in range(half):
            p1 = players[i]
            p2 = players[n_effective - 1 - i]
            # Only add the match if both players are not the dummy player
            if p1 is not None and p2 is not None:
                # Assign home arbitrarily (e.g., p1 is home)
                round_pairs.append((p1, p2))
        if round_pairs: # Only add non-empty rounds (relevant if n was odd)
             schedule.append(round_pairs)
        # Rotate players for the next round (excluding the fixed player 1)
        # Note: The circle method traditionally fixes player 0.
        # If we use 1-based indexing, we fix player 1.
        fixed_player = players[0]
        rotated_part = [players[-1]] + players[1:-1]
        players = [fixed_player] + rotated_part

    # Ensure the schedule has the correct number of rounds for the original n
    return schedule[:original_n - 1]


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
    players = list(range(1, n + 1)) # Use 1-based indexing
    if not schedule: # Handle empty schedule
        return 0, 0, 0
    rounds = len(schedule)
    if rounds == 0:
        return 0, 0, 0
    ideal_home_games = (n - 1) / 2

    # Use the new central function to calculate raw home strength
    raw_home_strength = calculate_home_strength(schedule, n)
    
    home_games = {i: 0 for i in players} # Use 1-based keys, still needed for max_dev and pen_seq
    sequences_dom_ext = {i: [] for i in players} # For penalty calculation, 1-based keys

    for r, rnd in enumerate(schedule):
        for home, away in rnd:
            home_games[home] += 1
            sequences_dom_ext[home].append('H')
            sequences_dom_ext[away].append('A')

    # Pénalité de séquence: consecutive same H/A
    penalites_sequence = 0
    for i in players: # Iterate over 1-based player list
        for r in range(rounds - 1):
            # Check sequence for player i
            if sequences_dom_ext[i][r] == sequences_dom_ext[i][r+1]:
                penalites_sequence += 1

    # Max Deviation (Proxy for StdDevAdvantage)
    max_dev = 0
    if ideal_home_games > 0: # Avoid division by zero if n=0 or n=1 (edge cases)
        for i in players: # Iterate over 1-based player list
            deviation = abs(home_games[i] - ideal_home_games)
            if deviation > max_dev:
                max_dev = deviation

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
        tuple: (best_schedule, best_score, final_metrics) where final_metrics
               is a tuple (delta_strength, penalites_sequence, max_deviation).

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
        return [], 0, (0, 0, 0)

    # --- Normalization Setup ---
    # Corrected S_max denominator for HomeStrength
    max_home_strength_approx = (n * (n - 1) * (n + 1) / 6.0) if n >= 2 else 1.0
    max_penalites_sequence_approx = n * (n - 2.0) if n > 2 else 1.0
    max_maxdev_approx = (n - 1.0) / 2.0 if n > 1 else 1.0
    
    max_home_strength_approx = max(max_home_strength_approx, 1.0)
    max_penalites_sequence_approx = max(max_penalites_sequence_approx, 1.0)
    max_maxdev_approx = max(max_maxdev_approx, 1.0)

    # --- Initialization ---
    c_home_strength, c_penalites_sequence, c_max_dev = compute_metrics(current_sched, n)
    
    # Use imported normalization functions
    norm_c_home_strength = normalize_home_strength(c_home_strength, n)
    norm_c_pen_seq = normalize_total_pen_seq(c_penalites_sequence, n)
    norm_c_max_dev = normalize_max_dev(c_max_dev, n)
    
    current_norm_score = norm_c_home_strength + alpha_pen_seq * norm_c_pen_seq + beta_obj * norm_c_max_dev
    current_unnorm_score = c_home_strength + alpha_pen_seq * c_penalites_sequence + beta_obj * c_max_dev # For reporting

    best_sched = copy.deepcopy(current_sched)
    best_norm_score = current_norm_score
    best_unnorm_score = current_unnorm_score # Keep track of best unnormalized score too
    best_metrics = (c_home_strength, c_penalites_sequence, c_max_dev)

    T = initial_temp

    print(f"Initial Score (Unnormalized): {current_unnorm_score:.2f} (HomeStrength: {c_home_strength}, PénSeq: {c_penalites_sequence}, MaxDev: {c_max_dev:.2f})")

    for it in range(iterations):
        candidate_sched = neighbor(current_sched, n)
        cand_home_strength, cand_penalites_sequence, cand_max_dev = compute_metrics(candidate_sched, n)
        
        # Use imported normalization functions for candidate
        norm_cand_home_strength = normalize_home_strength(cand_home_strength, n)
        norm_cand_pen_seq = normalize_total_pen_seq(cand_penalites_sequence, n)
        norm_cand_max_dev = normalize_max_dev(cand_max_dev, n)
        
        candidate_norm_score = norm_cand_home_strength + alpha_pen_seq * norm_cand_pen_seq + beta_obj * norm_cand_max_dev
        candidate_unnorm_score = cand_home_strength + alpha_pen_seq * cand_penalites_sequence + beta_obj * cand_max_dev # For reporting

        # Acceptance criterion (using NORMALIZED scores)
        delta_norm_score = candidate_norm_score - current_norm_score
        if delta_norm_score < 0 or random.random() < math.exp(-delta_norm_score / T):
            current_sched = candidate_sched
            current_norm_score = candidate_norm_score
            current_unnorm_score = candidate_unnorm_score # Update reported score
            # Update best found solution (based on NORMALIZED score)
            if current_norm_score < best_norm_score:
                best_sched = copy.deepcopy(current_sched)
                best_norm_score = current_norm_score
                best_unnorm_score = candidate_unnorm_score # Store corresponding unnormalized score
                best_metrics = (cand_home_strength, cand_penalites_sequence, cand_max_dev)

        # Cool down temperature
        T *= cooling_rate
        if T < 1e-5: # Prevent T from becoming too small
             T = 1e-5

        if it % (iterations // 10) == 0:
             # Report the unnormalized scores for better user understanding
             print(f"Iter {it}/{iterations}, Temp: {T:.4f}, Current Score: {current_unnorm_score:.2f}, Best Score: {best_unnorm_score:.2f}")


    final_home_strength, final_penalites_sequence, final_max_dev = best_metrics
    # Return the best unnormalized score and corresponding schedule/metrics
    print(f"Final Best Score (Unnormalized): {best_unnorm_score:.2f} (HomeStrength: {final_home_strength}, PénSeq: {final_penalites_sequence}, MaxDev: {final_max_dev:.2f})")
    return best_sched, best_unnorm_score, best_metrics


def main():
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    iters_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    alpha_pen_seq_arg = float(sys.argv[3]) if len(sys.argv) > 3 else config.ALPHA
    beta_arg = float(sys.argv[4]) if len(sys.argv) > 4 else config.BETA

    print(f"Running SA for n={n_arg}, iterations={iters_arg}, alpha_pen_seq={alpha_pen_seq_arg}, beta={beta_arg}")

    best_schedule, best_score, (final_home_strength, final_penalites_sequence, final_max_dev) = solve_sa(
        n_arg,
        iterations=iters_arg,
        alpha_pen_seq=alpha_pen_seq_arg, # Pass explicitly
        beta_obj=beta_arg # Pass explicitly
    )

    # Report the unnormalized score in the final summary
    print(f"\n--- Best SA Schedule (Score: {best_score:.2f}) ---")
    print(f"Metrics: HomeStrength={final_home_strength}, Total Pénalités Séquence={final_penalites_sequence}, Max Deviation={final_max_dev:.2f}")
    for r, rnd in enumerate(best_schedule):
        # Format matches for readability (using 1-based indices)
        match_strs = [f"{h}v{a}(H)" for h, a in rnd]
        print(f"Round {r+1}: {', '.join(match_strs)}") # Round numbers are naturally 1-based for display

if __name__ == '__main__':
    main()
