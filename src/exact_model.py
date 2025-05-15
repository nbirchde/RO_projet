#!/usr/bin/env python3
"""
Exact MILP model for fair round-robin scheduling using PuLP.

This script defines and solves a Mixed-Integer Linear Program (MILP)
to find an optimal round-robin schedule for 'n' players that minimizes
a weighted sum of fairness metrics:
1. Delta HomeStrength: Sum of strength differences for home games (using 1-based player indices).
2. Pénalité de séquence: Number of consecutive home or away games for any player.
3. Max Deviation: Maximum deviation of home games per player from the average.

Usage:
    python src/exact_model.py [n_players] [alpha_pen_seq] [beta] [time_limit_seconds]

Args:
    n_players (int, optional): Number of players (must be even). Defaults to 6.
    alpha_pen_seq (float, optional): Weight for the Pénalité de séquence objective term. Defaults to 1.0.
    beta (float, optional): Weight for the Max Deviation objective term. Defaults to 1.0.
    time_limit_seconds (int, optional): Time limit for the solver in seconds. Defaults to None (no limit).
"""
import sys
import itertools
import pulp
import math
import csv
import numpy as np # For np.arange
import os # Add os import

# Add the project root directory to sys.path
current_dir_exact = os.path.dirname(os.path.abspath(__file__))
project_root_exact = os.path.abspath(os.path.join(current_dir_exact, os.pardir))
if project_root_exact not in sys.path:
    sys.path.insert(0, project_root_exact)

# Import necessary functions from normalization_manager
from src.normalization_manager import get_or_calculate_normalization_factors, calculate_normalized_score
from src import config # Import the configuration

def solve_exact(n, alpha_pen_seq=None, beta=None, time_limit=None):
    """
    Defines and solves the fair round-robin scheduling MILP.

    Args:
        n (int): Number of players (must be even).
        alpha_pen_seq (float, optional): Weight for the Pénalité de séquence objective term. Defaults to 1.0.
        beta (float, optional): Weight for the Max Deviation objective term. Defaults to 1.0.

    Returns:
        dict: A dictionary containing the solution status, objective value,
              schedule string (using 1-based player indices), and calculated fairness metrics, e.g.,
              {
                  "status": "Optimal",
                  "objective_value": 39.5,
                  "schedule_str": "R1: 1v2(H), ... | R2: ...",
                  "metrics": {
                      "home_strength": 35.0,
                      "penalites_sequence": 4.0,
                      "max_deviation": 0.5
                  }
              }
              Returns status 'Not Solved' or other if no optimal solution found.

    Raises:
        ValueError: If n is not an even number.
        pulp.PulpSolverError: If the solver encounters an error.
    """
    if n % 2 != 0:
        raise ValueError("Number of players (n) must be even.")

    # Use configured alpha and beta if not provided
    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta is None:
        beta = config.BETA

    players = list(range(1, n + 1)) # Use 1-based indexing
    rounds = list(range(n-1)) # Keep rounds 0-based (0 to n-2)
    round_pairs = list(range(n-2)) # Rounds for penalty calculation (0 to n-3)
    prob = pulp.LpProblem("fair_round_robin", pulp.LpMinimize)

    # --- Decision Variables ---
    # x[i][j][r] = 1 if player i (1-based) at home vs j (1-based) in round r (0-based)
    x = pulp.LpVariable.dicts('x', (players, players, rounds), cat='Binary')

    # --- Auxiliary Variables for Fairness ---
    # is_home[i][r] = 1 if player i (1-based) plays at home in round r (0-based)
    is_home = pulp.LpVariable.dicts('is_home', (players, rounds), cat='Binary')
    # pen_seq[i][r] = 1 if player i (1-based) has a penalty (consecutive H or A) between rounds r and r+1
    pen_seq = pulp.LpVariable.dicts('pen_seq', (players, round_pairs), cat='Binary')
    # H_i = Total home games for player i (1-based)
    H_i = pulp.LpVariable.dicts('H_i', players, lowBound=0, cat='Integer')
    # MaxDev = Maximum absolute deviation of H_i from the average
    MaxDev = pulp.LpVariable('MaxDev', lowBound=0, cat='Continuous')

    # --- Objective Function ---
    # Define terms for the objective function
    # New HomeStrength: sum max(0, rank_opponent - rank_home_player) * x_home_vs_opponent_in_round
    # This is equivalent to summing (j-i)*x[i][j][r] only when j > i.
    # Player indices i, j are 1-based.
    home_strength_term = pulp.lpSum(
        (j - i) * x[i][j][r]
        for i in players for j in players if j > i for r in rounds # Sum only where j > i
    )
    total_penalites_sequence_term = pulp.lpSum(pen_seq[i][r] for i in players for r in round_pairs)
    max_deviation_term = MaxDev

    # Get empirical normalization factors
    med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md = get_or_calculate_normalization_factors(n)

    # Define the normalized objective function using empirical factors
    # Minimize (raw_hs - med_hs)/sigma_hs + alpha * (raw_ps - med_ps)/sigma_ps + beta * (raw_md - med_md)/sigma_md
    # Note: raw_hs, raw_ps, raw_md are represented by the pulp variables home_strength_term, total_penalites_sequence_term, MaxDev
    prob += (home_strength_term - med_hs) / sigma_hs + \
            alpha_pen_seq * (total_penalites_sequence_term - med_ps) / sigma_ps + \
            beta * (max_deviation_term - med_md) / sigma_md
            
    # --- Constraints ---
    # Basic Round Robin Constraints
    # Each pair meets exactly once (implicitly handled by symmetry and below)
    # One match per player (1-based) per round (0-based)
    for i in players:
        for r in rounds:
            # Sum of i hosting j + sum of j hosting i must be 1
            prob += pulp.lpSum(x[i][j][r] for j in players if i != j) + \
                    pulp.lpSum(x[j][i][r] for j in players if i != j) == 1

    # Symmetry: For each game (i,j) in round r, exactly one is home
    # Symmetry: Ensure each pair plays exactly once over the tournament
    # The constraint x[i][j][r] + x[j][i][r] <= 1 is redundant given other constraints.
    for i, j in itertools.permutations(players, 2):
         if i < j: # Avoid double counting pairs (using 1-based indices)
             prob += pulp.lpSum(x[i][j][r] + x[j][i][r] for r in rounds) == 1


    # Fairness Constraints
    # Link is_home to x variables
    for i in players:
        for r in rounds:
            prob += is_home[i][r] == pulp.lpSum(x[i][j][r] for j in players if i != j)

    # Link penalties pen_seq[i][r] to is_home variables
    for i in players:
        for r in round_pairs: # Iterate up to the second to last round (0 to n-3)
            # Penalty if H-H: pen_seq >= home[r] + home[r+1] - 1
            prob += pen_seq[i][r] >= is_home[i][r] + is_home[i][r+1] - 1
            # Penalty if A-A: pen_seq >= away[r] + away[r+1] - 1
            # where away = 1 - home
            prob += pen_seq[i][r] >= (1 - is_home[i][r]) + (1 - is_home[i][r+1]) - 1
            # Ensure pen_seq[i][r] is not forced to 1 unnecessarily (redundant for minimization?)
            # pen_seq[i][r] <= 1 (implicit)

    # Link H_i (total home games) to is_home
    ideal_home_games = (n - 1) / 2
    for i in players:
        prob += H_i[i] == pulp.lpSum(is_home[i][r] for r in rounds)

    # Link MaxDev to H_i deviations
    for i in players:
        prob += MaxDev >= H_i[i] - ideal_home_games
        prob += MaxDev >= ideal_home_games - H_i[i]

    # --- Solve ---
    solver_options = {}
    if time_limit is not None:
        solver_options['timeLimit'] = time_limit
        
    # Use msg=True if running directly and want solver logs, False for cleaner library use.
    # For this script's direct execution, let's keep msg=True to see solver progress.
    show_solver_msg = __name__ == '__main__' # Show messages only when run as script
    solver = pulp.PULP_CBC_CMD(msg=show_solver_msg, **solver_options)
    
    print(f"Solving with CBC. Time limit: {time_limit if time_limit is not None else 'None'} seconds.")
    prob.solve(solver)

    # --- Process Results ---
    results = {
        "status": pulp.LpStatus[prob.status],
        "schedule_str": None,
        "metrics": None,
        "objective_value": None
    }

    if prob.status == pulp.LpStatusOptimal:
        results["objective_value"] = pulp.value(prob.objective)

        # Extract schedule into a more structured format if needed, or just string
        schedule_dict = {r: [] for r in rounds} # 0-based round keys
        schedule_lines = []
        for r in rounds: # Iterate 0-based rounds
            round_matches = []
            for i, j in itertools.permutations(players, 2): # Iterate 1-based players
                 if i < j: # Process each pair once
                    if pulp.value(x[i][j][r]) == 1:
                        round_matches.append(f"{i}v{j}(H)") # Use 1-based i, j
                    elif pulp.value(x[j][i][r]) == 1:
                         round_matches.append(f"{i}v{j}(A)") # Use 1-based i, j; i is Away
            schedule_dict[r] = round_matches
            schedule_lines.append(f"R{r+1}: {', '.join(round_matches)}") # Display round as 1-based
        results["schedule_str"] = " | ".join(schedule_lines) # Pipe separated string for CSV

        # Calculate final metrics
        # Use the value of the objective term for the new HomeStrength
        final_home_strength = pulp.value(home_strength_term)
        final_penalites_sequence = sum(pulp.value(pen_seq[i][r]) for i in players for r in round_pairs)
        final_max_dev = pulp.value(MaxDev)
        results["metrics"] = {
            "home_strength": final_home_strength, # This is now the new raw HS
            "penalites_sequence": final_penalites_sequence,
            "max_deviation": final_max_dev
        }

        # --- Optional: Print summary if run directly ---
        if __name__ == '__main__':
            print(f"Status: {results['status']}")
            print(f"Objective Value: {results['objective_value']:.2f}")
            print("\n--- Schedule (1-based player indices) ---")
            for line in schedule_lines:
                print(line.replace("R", "Round "))
            print("\n--- Fairness Metrics ---")
            print(f"Raw HomeStrength: {results['metrics']['home_strength']:.2f}")
            print(f"Total Pénalités Séquence: {results['metrics']['penalites_sequence']:.0f}")
            print(f"Max Deviation: {results['metrics']['max_deviation']:.2f}")
            print("\nHome Games per Player (1-based):")
            for i_player in players: # Iterate 1-based players
                h_count = pulp.value(H_i[i_player])
                dev = h_count - ideal_home_games
                print(f"  Player {i_player}: {h_count:.0f} (Deviation: {dev:+.1f})")
            
            print("\nPlayer H/A Sequences (1-based player indices, 0-based rounds):")
            player_sequences_chars = [[] for _ in range(n)]
            for p_idx_1based in players: # 1 to n
                for r_idx in rounds: # 0 to n-2
                    is_p_home = pulp.value(is_home[p_idx_1based][r_idx]) == 1
                    if is_p_home:
                        player_sequences_chars[p_idx_1based - 1].append('H')
                    else:
                        player_sequences_chars[p_idx_1based - 1].append('A')
            
            for p_idx_1based in players:
                print(f"  Player {p_idx_1based}: {''.join(player_sequences_chars[p_idx_1based - 1])}")

    else:
         if __name__ == '__main__':
            print(f"Status: {results['status']}")
            print("Solver did not find an optimal solution.")

    return results # Return the dictionary


if __name__ == '__main__':
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    alpha_pen_seq_arg = float(sys.argv[2]) if len(sys.argv) > 2 else config.ALPHA
    beta_arg = float(sys.argv[3]) if len(sys.argv) > 3 else config.BETA
    time_limit_arg = int(sys.argv[4]) if len(sys.argv) > 4 else None # Assuming time limit was 4th arg

    if n_arg % 2 != 0:
        print(f"Error: Number of players (n={n_arg}) must be even.")
        sys.exit(1)

    print(f"Running single exact model instance: n={n_arg}, alpha={alpha_pen_seq_arg:.2f}, beta={beta_arg:.2f}, time_limit={time_limit_arg}")

    results = solve_exact(n_arg, alpha_pen_seq=alpha_pen_seq_arg, beta=beta_arg, time_limit=time_limit_arg)

    # The detailed printout, including H/A sequences, is now handled within solve_exact's __main__ block.
    # We can add a summary here if needed, but it might be redundant.
    if results["status"] != "Optimal":
        print(f"Solver did not find an optimal solution for n={n_arg}, alpha={alpha_pen_seq_arg:.2f}, beta={beta_arg:.2f}.")
        sys.exit(1)
    else:
        # A brief confirmation that it finished, details are printed by solve_exact
        print(f"\nSingle run finished for n={n_arg}, alpha={alpha_pen_seq_arg:.2f}, beta={beta_arg:.2f}. Status: {results['status']}.")
    sys.exit(0) # Exit successfully after single run
