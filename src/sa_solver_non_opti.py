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
import time

# Add the project root directory to sys.path to enable importing modules from src
# This allows running the script directly from the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

from src.metrics import calculate_home_strength, calculate_raw_max_deviation, calculate_raw_total_penalty_sequence
from src.normalization_manager import calculate_normalized_score
import src.config as config # Import the configuration

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
    half = n // 2
    schedule = []
    for r in range(n - 1): # Iterate through rounds needed for n players
        round_pairs = []
        for i in range(half):
            p1 = players[i]
            p2 = players[n - 1 - i] # if i = 0, n - 1 - i = 5 (last player)
            round_pairs.append((p2, p1))       

        schedule.append(round_pairs)
          
        # Rotate players for the next round (excluding the fixed player 1 (players[0]))
        rotated_part = [players[n - 1]] + players[1:n - 1] #[2,3,4,5,6] -> [6,2,3,4,5]
        players = [players[0]] + rotated_part #[1,6,2,3,4,5]

    return schedule # example: [[(2,1), (3,4)], [(4,5), (6,3)], [(5,2), (1,6)], [(1,4), (2,3)], [(3,5), (6,1)]]


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
        tuple: (obj_hs, obj_ps, obj_md) normalized metrics
    """
    
    home_strength = calculate_home_strength(schedule, n)
    penalites_sequence = calculate_raw_total_penalty_sequence(schedule, n)
    max_dev = calculate_raw_max_deviation(schedule, n)

    # Use the normalization manager to get normalized z-scores
    # We only need the z-scores (obj_hs, obj_ps, obj_md), not the total score or scaled scores
    _, obj_hs, obj_ps, obj_md, _, _, _, _ = calculate_normalized_score(
        home_strength, penalites_sequence, max_dev, 
        config.ALPHA, config.BETA, n
    )
    
    return obj_hs, obj_ps, obj_md

def calculate_norm_score(obj_hs, obj_ps, obj_md, alpha_pen_seq, beta_obj):
    """
    Calculate the normalized score based on the given z-score metrics and weights.

    Args:
        obj_hs (float): Normalized home strength z-score.
        obj_ps (float): Normalized penalty sequence z-score.
        obj_md (float): Normalized maximum deviation z-score.
        alpha_pen_seq (float): Weight for the penalty sequence.
        beta_obj (float): Weight for the maximum deviation.

    Returns:
        float: The weighted sum of z-scores.
    """
    # Simple weighted sum of z-scores (based on how calculate_normalized_score returns total_normalized_score)
    return obj_hs + alpha_pen_seq * obj_ps + beta_obj * obj_md


def neighbor(schedule, n, prob_flip=0.95, prob_swap_rounds=0, prob_swap_players=0.05):
    """
    Generates a neighboring schedule by either:
    - flipping the home/away assignment of a randomly selected match.
    - swapping two entire rounds (days) in the schedule.
    - swapping two players throughout the entire schedule.

    The choice of move is made based on the provided probabilities.

    Args:
        schedule (list): The current schedule (with 1-based indices).
        n (int): The even number of players.

    Returns:
        list: A new schedule representing a neighbor of the input schedule.

    """
    if n == 0 : return schedule # Handle case where n=0 (no players)
    new_sched = copy.deepcopy(schedule)
    num_rounds = n - 1
    round_len = n//2

    # Ensure the probabilities sum to 1
    total_prob = prob_flip + prob_swap_rounds + prob_swap_players
    if total_prob != 1:
        prob_flip /= total_prob
        prob_swap_rounds /= total_prob
        prob_swap_players /= total_prob



    # Randomly choose a move type based on the given probabilities
    move_type = None
    rand_val = random.random()

    if rand_val < prob_flip:
        move_type = "flip_home_away"
    elif rand_val < prob_flip + prob_swap_rounds:
        move_type = "swap_rounds"
    else:
        move_type = "swap_players"


    if move_type == "flip_home_away":
        # Assumes schedule is not empty and rounds are not empty.
        # And that n is the number of actual players (1 to n).
        round_idx = random.randrange(num_rounds)
        match_idx = random.randrange(round_len)
        home, away = new_sched[round_idx][match_idx]
        new_sched[round_idx][match_idx] = (away, home)
                

    elif move_type == "swap_rounds":
        if num_rounds >= 2:
            idx1, idx2 = random.sample(range(num_rounds), 2)
            new_sched[idx1], new_sched[idx2] = new_sched[idx2], new_sched[idx1]

    elif move_type == "swap_players":
        player_to_swap1, player_to_swap2 = random.sample(range(1, n + 1), 2)

        for round_idx in range(num_rounds):
            for match_idx in range(round_len):
                home, away = new_sched[round_idx][match_idx]
                
                new_home = home
                new_away = away

                if home == player_to_swap1:
                    new_home = player_to_swap2
                elif home == player_to_swap2:
                    new_home = player_to_swap1
                
                if away == player_to_swap1:
                    new_away = player_to_swap2
                elif away == player_to_swap2:
                    new_away = player_to_swap1
                
                if (new_home, new_away) != (home, away):
                    new_sched[round_idx][match_idx] = (new_home, new_away)
    
    return new_sched



def solve_sa(n, iterations=10000, initial_temp=1.0, cooling_rate=0.95, alpha_pen_seq=None, beta_obj=None, seed=51):
    """
    Solves the fair round-robin problem using Simulated Annealing.

    Args:
        n (int): Even number of players.
        iterations (int, optional): Number of SA iterations. Defaults to 10000.
        initial_temp (float, optional): Initial temperature. Defaults to 1.0.
        cooling_rate (float, optional): Temp. cooling rate. Defaults to 0.95.
        alpha_pen_seq (float, optional): Weight for Pénalité de séquence objective. Defaults to 1.0.
        beta_obj (float, optional): Weight for Max Deviation objective. Defaults to 1.0.
        seed (int, optional): Random seed for reproducibility. Defaults to 51.

    Returns:
        tuple: (best_schedule, best_score, final_metrics) where final_metrics
               is a tuple (obj_hs, obj_ps, obj_md).
    """
    random.seed(seed)

    # Use default values from config if not specified
    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA

    # 1. Generate the initial solution
    current_sched = initial_schedule(n)

    # 2. Evaluate the initial solution
    # Calculate normalized metrics (z-scores)
    obj_c_home_strength, obj_c_pen_seq, obj_c_max_dev = compute_metrics(current_sched, n)
    
    # Calculate the normalized score (used for acceptance decisions)
    current_norm_score = calculate_norm_score(obj_c_home_strength, obj_c_pen_seq, obj_c_max_dev, alpha_pen_seq, beta_obj)
    
    # 3. Initialize the best solution found
    best_sched = copy.deepcopy(current_sched)
    best_norm_score = current_norm_score
    best_metrics = (obj_c_home_strength, obj_c_pen_seq, obj_c_max_dev)

    # 4. Initialize temperature
    T = initial_temp

    print(f"Initial Score (normalized): {current_norm_score:.2f} (HS: {obj_c_home_strength:.2f}, PS: {obj_c_pen_seq:.2f}, MD: {obj_c_max_dev:.2f})")

    # 5. Main Simulated Annealing loop
    for it in range(iterations):
        # a. Generate a neighbor
        candidate_sched = neighbor(current_sched, n)
        
        # b. Evaluate the neighbor
        obj_cand_home_strength, obj_cand_pen_seq, obj_cand_max_dev = compute_metrics(candidate_sched, n)
        candidate_norm_score = calculate_norm_score(obj_cand_home_strength, obj_cand_pen_seq, obj_cand_max_dev, alpha_pen_seq, beta_obj)

        # c. Decide whether to accept the neighbor (Metropolis criterion)
        delta_norm_score = candidate_norm_score - current_norm_score
        
        if delta_norm_score < 0: # If neighbor is better, always accept
            accept = True
        else: # Otherwise, accept with a probability dependent on T and degradation
            probability = math.exp(-delta_norm_score / T)
            accept = random.random() < probability

        if accept:
            current_sched = candidate_sched
            current_norm_score = candidate_norm_score
            
            # d. Update the best solution if necessary
            if current_norm_score < best_norm_score:
                best_sched = copy.deepcopy(current_sched)
                best_norm_score = current_norm_score
                best_metrics = (obj_cand_home_strength, obj_cand_pen_seq, obj_cand_max_dev)

        # e. Cool the temperature
        T *= cooling_rate
        if T < 1e-5: # Prevent T from becoming too small
             T = 1e-5

        # Periodic display
        if it % (iterations // 10) == 0:
             print(f"Iter {it}/{iterations}, Temp: {T:.4f}, Current: {current_norm_score:.2f}, Best: {best_norm_score:.2f}")

    final_home_strength, final_penalites_sequence, final_max_dev = best_metrics
    print(f"Final Best Score (normalized): {best_norm_score:.2f} (HS: {final_home_strength:.2f}, PS: {final_penalites_sequence:.2f}, MD: {final_max_dev:.2f})")
    
    return best_sched, best_norm_score, best_metrics


def main():
    start_time = time.time()

    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    if n_arg % 2 != 0:
        raise ValueError(f"Number of players (n_arg={n_arg}) must be even for this solver.")
    
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

    # Report the normalized score in the final summary
    print(f"\n--- Best SA Schedule (Score: {best_score:.2f}) ---")
    print(f"Metrics: HomeStrength={final_home_strength:.4f}, Total Pénalités Séquence={final_penalites_sequence:.4f}, Max Deviation={final_max_dev:.4f}")
    
    # Uncomment this to display the full schedule
    print("\nFinal Schedule:")
    for r, rnd in enumerate(best_schedule):
        # Format matches for readability (using 1-based indices)
        match_strs = [f"{h}-{a}" for h, a in rnd]
        print(f"Round {r+1}: {', '.join(match_strs)}") # Round numbers are naturally 1-based for display
    
    print("\nTime used: {:.2f} seconds".format(time.time() - start_time))
if __name__ == '__main__':
    main()