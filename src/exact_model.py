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
import argparse
import csv
import numpy as np # For np.arange
# Import the specific functions needed, including the updated denominator calculator
from .metrics import calculate_max_home_strength_denominator, normalize_home_strength, normalize_total_pen_seq, normalize_max_dev
from . import config # Import the configuration

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

    # Normalization denominators (constants for a given n)
    # Use the correct theoretical maximum S_max calculation for HomeStrength
    denom_hs = calculate_max_home_strength_denominator(n)
    denom_ps = n * (n - 2.0) if n > 2 else 1.0
    denom_md = (n - 1.0) / 2.0 if n > 1 else 1.0 # Kept original for MaxDev

    # Ensure denominators are at least 1.0 to avoid division by zero or inflation
    denom_hs = max(denom_hs, 1.0)
    denom_ps = max(denom_ps, 1.0)
    denom_md = max(denom_md, 1.0)

    # Normalized objective function: Minimize HS_norm + alpha*PS_norm + beta*MD_norm
    # Since the new home_strength_term is always non-negative, we minimize it directly.
    prob += (1/denom_hs) * home_strength_term + \
            (alpha_pen_seq/denom_ps) * total_penalites_sequence_term + \
            (beta/denom_md) * max_deviation_term
            
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
    # Check if specific grid arguments are provided. If not, run a default single instance.
    if any(arg.startswith('--grid') for arg in sys.argv):
        # Grid search mode
        parser = argparse.ArgumentParser(description="Run MILP for fair round-robin scheduling over a grid of alpha and beta values.")
        parser.add_argument("--n", type=int, required=True, help="Number of players (must be even).")
        parser.add_argument("--grid_alpha_start", type=float, required=True, help="Start value for alpha grid.")
        parser.add_argument("--grid_alpha_end", type=float, required=True, help="End value for alpha grid.")
        parser.add_argument("--grid_alpha_step", type=float, required=True, help="Step value for alpha grid.")
        parser.add_argument("--grid_beta_start", type=float, required=True, help="Start value for beta grid.")
        parser.add_argument("--grid_beta_end", type=float, required=True, help="End value for beta grid.")
        parser.add_argument("--grid_beta_step", type=float, required=True, help="Step value for beta grid.")
        parser.add_argument("--output_csv", type=str, required=True, help="Path to save the CSV results.")
        parser.add_argument("--time_limit", type=int, default=None, help="Time limit for the solver in seconds for each run.")
        args = parser.parse_args()

        if args.n % 2 != 0:
            print(f"Error: Number of players (n={args.n}) must be even.")
            sys.exit(1)
        
        if args.grid_alpha_step <= 0 or args.grid_beta_step <= 0:
            print("Error: Alpha and Beta steps must be positive.")
            sys.exit(1)

        alpha_values = np.arange(args.grid_alpha_start, args.grid_alpha_end + args.grid_alpha_step, args.grid_alpha_step)
        if not np.isclose(alpha_values[-1], args.grid_alpha_end) and alpha_values[-1] < args.grid_alpha_end :
             alpha_values = np.append(alpha_values, args.grid_alpha_end)
        if alpha_values[-1] > args.grid_alpha_end and not np.isclose(alpha_values[-1], args.grid_alpha_end):
            alpha_values = alpha_values[:-1]

        beta_values = np.arange(args.grid_beta_start, args.grid_beta_end + args.grid_beta_step, args.grid_beta_step)
        if not np.isclose(beta_values[-1], args.grid_beta_end) and beta_values[-1] < args.grid_beta_end:
            beta_values = np.append(beta_values, args.grid_beta_end)
        if beta_values[-1] > args.grid_beta_end and not np.isclose(beta_values[-1], args.grid_beta_end):
            beta_values = beta_values[:-1]

        print(f"Starting calibration for n={args.n}")
        print(f"Alpha range: {args.grid_alpha_start} to {args.grid_alpha_end} step {args.grid_alpha_step} ({len(alpha_values)} values)")
        print(f"Beta range: {args.grid_beta_start} to {args.grid_beta_end} step {args.grid_beta_step} ({len(beta_values)} values)")
        print(f"Output CSV: {args.output_csv}")
        print(f"Time limit per run: {args.time_limit if args.time_limit is not None else 'None'} seconds.")

        fieldnames = [
            "alpha", "beta", "home_strength_norm", "ps_norm", "md_norm",
            "z_norm_calculated", "z_norm_solver",
            "raw_home_strength", "raw_ps", "raw_md", "status"
        ]

        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            total_runs = len(alpha_values) * len(beta_values)
            current_run = 0

            for alpha_val in alpha_values:
                for beta_val in beta_values:
                    current_run += 1
                    print(f"\nRunning ({current_run}/{total_runs}): n={args.n}, alpha={alpha_val:.2f}, beta={beta_val:.2f}")
                    
                    try:
                        results = solve_exact(args.n, alpha_pen_seq=alpha_val, beta=beta_val, time_limit=args.time_limit)
                        
                        row_data = {
                            "alpha": f"{alpha_val:.2f}", 
                            "beta": f"{beta_val:.2f}",
                            "status": results["status"]
                        }

                        if results["status"] == "Optimal" and results["metrics"]:
                            raw_home_strength = results["metrics"]["home_strength"]
                            raw_ps = results["metrics"]["penalites_sequence"]
                            raw_md = results["metrics"]["max_deviation"]

                            home_strength_norm = normalize_home_strength(raw_home_strength, args.n)
                            ps_norm = normalize_total_pen_seq(raw_ps, args.n)
                            md_norm = normalize_max_dev(raw_md, args.n)
                            
                            # Calculate z_norm using the new objective structure (no abs)
                            z_norm_calculated = home_strength_norm + alpha_val * ps_norm + beta_val * md_norm

                            row_data.update({
                                "home_strength_norm": f"{home_strength_norm:.4f}", # This is HS_norm_new
                                "ps_norm": f"{ps_norm:.4f}",
                                "md_norm": f"{md_norm:.4f}",
                                "z_norm_calculated": f"{z_norm_calculated:.4f}",
                                "z_norm_solver": f"{results['objective_value']:.4f}" if results['objective_value'] is not None else "N/A",
                                "raw_home_strength": f"{raw_home_strength:.2f}",
                                "raw_ps": f"{raw_ps:.0f}",
                                "raw_md": f"{raw_md:.2f}"
                            })
                        else:
                            for key in ["home_strength_norm", "ps_norm", "md_norm", "z_norm_calculated", "z_norm_solver", "raw_home_strength", "raw_ps", "raw_md"]:
                                row_data[key] = "N/A"
                        
                        writer.writerow(row_data)
                        csvfile.flush() 

                    except ValueError as e:
                        print(f"Error during solve_exact for alpha={alpha_val}, beta={beta_val}: {e}")
                        writer.writerow({
                            "alpha": f"{alpha_val:.2f}", "beta": f"{beta_val:.2f}", "status": "ValueError",
                            "home_strength_norm": "N/A", "ps_norm": "N/A", "md_norm": "N/A",
                            "z_norm_calculated": "N/A", "z_norm_solver": "N/A",
                            "raw_home_strength": "N/A", "raw_ps": "N/A", "raw_md": "N/A"
                        })
                    except pulp.PulpSolverError as e:
                        print(f"Solver Error for alpha={alpha_val}, beta={beta_val}: {e}")
                        writer.writerow({
                            "alpha": f"{alpha_val:.2f}", "beta": f"{beta_val:.2f}", "status": "PulpSolverError",
                            "home_strength_norm": "N/A", "ps_norm": "N/A", "md_norm": "N/A",
                            "z_norm_calculated": "N/A", "z_norm_solver": "N/A",
                            "raw_home_strength": "N/A", "raw_ps": "N/A", "raw_md": "N/A"
                        })
                    except Exception as e:
                        print(f"An unexpected error occurred for alpha={alpha_val}, beta={beta_val}: {e}")
                        writer.writerow({
                            "alpha": f"{alpha_val:.2f}", "beta": f"{beta_val:.2f}", "status": f"UnexpectedError: {type(e).__name__}",
                            "home_strength_norm": "N/A", "ps_norm": "N/A", "md_norm": "N/A",
                            "z_norm_calculated": "N/A", "z_norm_solver": "N/A",
                            "raw_home_strength": "N/A", "raw_ps": "N/A", "raw_md": "N/A"
                        })
        print(f"\nCalibration finished. Results saved to {args.output_csv}")
    else:
        # Single run mode (not grid search)
        parser_single = argparse.ArgumentParser(description="Run MILP for a single instance.")
        parser_single.add_argument("--n", type=int, default=6, help="Number of players (must be even). Default is 6.")
        parser_single.add_argument("--alpha", type=float, default=config.ALPHA, help=f"Alpha weight for penalty sequence. Default is {config.ALPHA} from config.")
        parser_single.add_argument("--beta", type=float, default=config.BETA, help=f"Beta weight for max deviation. Default is {config.BETA} from config.")
        parser_single.add_argument("--time_limit", type=int, default=None, help="Time limit for the solver in seconds. Default is no limit.")
        
        args_single = parser_single.parse_args()

        if args_single.n % 2 != 0:
            print(f"Error: Number of players (n={args_single.n}) must be even.")
            sys.exit(1)

        print(f"Running single exact model instance: n={args_single.n}, alpha={args_single.alpha:.2f}, beta={args_single.beta:.2f}, time_limit={args_single.time_limit}")
        
        results = solve_exact(args_single.n, alpha_pen_seq=args_single.alpha, beta=args_single.beta, time_limit=args_single.time_limit)
        
        # The detailed printout, including H/A sequences, is now handled within solve_exact's __main__ block.
        # We can add a summary here if needed, but it might be redundant.
        if results["status"] != "Optimal":
            print(f"Solver did not find an optimal solution for n={args_single.n}, alpha={args_single.alpha:.2f}, beta={args_single.beta:.2f}.")
            sys.exit(1)
        else:
            # A brief confirmation that it finished, details are printed by solve_exact
            print(f"\nSingle run finished for n={args_single.n}, alpha={args_single.alpha:.2f}, beta={args_single.beta:.2f}. Status: {results['status']}.")
        sys.exit(0) # Exit successfully after single run

    # This part below is for grid search, ensure it's not reached by single run due to sys.exit(0) above.
    # The following check should ideally be inside the grid block or handled by argparse structure.
    # For now, this structure assumes if --grid args are present, args object is from that parser.
    # If we reached here, it means grid arguments were parsed by the first parser.
    
    # Defensive check, though single run mode should exit before this.
    if not hasattr(args, 'grid_alpha_step') or args.grid_alpha_step <= 0 or args.grid_beta_step <= 0 :
        print("Error: Grid step arguments are invalid or missing for grid mode.")
        sys.exit(1)

    alpha_values = np.arange(args.grid_alpha_start, args.grid_alpha_end + args.grid_alpha_step, args.grid_alpha_step)
    # Ensure the end value is included if the step doesn't perfectly align, by slightly adjusting the end for arange
    if not np.isclose(alpha_values[-1], args.grid_alpha_end) and alpha_values[-1] < args.grid_alpha_end :
         alpha_values = np.append(alpha_values, args.grid_alpha_end)
    if alpha_values[-1] > args.grid_alpha_end and not np.isclose(alpha_values[-1], args.grid_alpha_end): # If overshot, remove last
        alpha_values = alpha_values[:-1]


    beta_values = np.arange(args.grid_beta_start, args.grid_beta_end + args.grid_beta_step, args.grid_beta_step)
    if not np.isclose(beta_values[-1], args.grid_beta_end) and beta_values[-1] < args.grid_beta_end:
        beta_values = np.append(beta_values, args.grid_beta_end)
    if beta_values[-1] > args.grid_beta_end and not np.isclose(beta_values[-1], args.grid_beta_end):
        beta_values = beta_values[:-1]


    print(f"Starting calibration for n={args.n}")
    print(f"Alpha range: {args.grid_alpha_start} to {args.grid_alpha_end} step {args.grid_alpha_step} ({len(alpha_values)} values)")
    print(f"Beta range: {args.grid_beta_start} to {args.grid_beta_end} step {args.grid_beta_step} ({len(beta_values)} values)")
    print(f"Output CSV: {args.output_csv}")
    print(f"Time limit per run: {args.time_limit if args.time_limit is not None else 'None'} seconds.")

    fieldnames = [
        "alpha", "beta", "home_strength_norm", "ps_norm", "md_norm",
        "z_norm_calculated", "z_norm_solver",
        "raw_home_strength", "raw_ps", "raw_md", "status"
    ]

    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total_runs = len(alpha_values) * len(beta_values)
        current_run = 0

        for alpha_val in alpha_values:
            for beta_val in beta_values:
                current_run += 1
                print(f"\nRunning ({current_run}/{total_runs}): n={args.n}, alpha={alpha_val:.2f}, beta={beta_val:.2f}")
                
                try:
                    results = solve_exact(args.n, alpha_pen_seq=alpha_val, beta=beta_val, time_limit=args.time_limit)
                    
                    row_data = {
                        "alpha": f"{alpha_val:.2f}", # Store with consistent formatting
                        "beta": f"{beta_val:.2f}",
                        "status": results["status"]
                    }

                    if results["status"] == "Optimal" and results["metrics"]:
                        raw_home_strength = results["metrics"]["home_strength"]
                        raw_ps = results["metrics"]["penalites_sequence"]
                        raw_md = results["metrics"]["max_deviation"]

                        home_strength_norm = normalize_home_strength(raw_home_strength, args.n)
                        ps_norm = normalize_total_pen_seq(raw_ps, args.n)
                        md_norm = normalize_max_dev(raw_md, args.n)
                        
                        # Calculate z_norm using the new objective structure (no abs)
                        z_norm_calculated = home_strength_norm + alpha_val * ps_norm + beta_val * md_norm

                        row_data.update({
                            "home_strength_norm": f"{home_strength_norm:.4f}", # This is HS_norm_new
                            "ps_norm": f"{ps_norm:.4f}",
                            "md_norm": f"{md_norm:.4f}",
                            "z_norm_calculated": f"{z_norm_calculated:.4f}",
                            "z_norm_solver": f"{results['objective_value']:.4f}" if results['objective_value'] is not None else "N/A",
                            "raw_home_strength": f"{raw_home_strength:.2f}",
                            "raw_ps": f"{raw_ps:.0f}",
                            "raw_md": f"{raw_md:.2f}"
                        })
                    else:
                        # Fill with N/A if not optimal or metrics missing
                        for key in ["home_strength_norm", "ps_norm", "md_norm", "z_norm_calculated", "z_norm_solver", "raw_home_strength", "raw_ps", "raw_md"]:
                            row_data[key] = "N/A"
                    
                    writer.writerow(row_data)
                    csvfile.flush() # Ensure data is written progressively

                except ValueError as e:
                    print(f"Error during solve_exact for alpha={alpha_val}, beta={beta_val}: {e}")
                    writer.writerow({
                        "alpha": f"{alpha_val:.2f}", "beta": f"{beta_val:.2f}", "status": "ValueError",
                        "home_strength_norm": "N/A", "ps_norm": "N/A", "md_norm": "N/A",
                        "z_norm_calculated": "N/A", "z_norm_solver": "N/A",
                        "raw_home_strength": "N/A", "raw_ps": "N/A", "raw_md": "N/A"
                    })
                except pulp.PulpSolverError as e:
                    print(f"Solver Error for alpha={alpha_val}, beta={beta_val}: {e}")
                    writer.writerow({
                        "alpha": f"{alpha_val:.2f}", "beta": f"{beta_val:.2f}", "status": "PulpSolverError",
                        "home_strength_norm": "N/A", "ps_norm": "N/A", "md_norm": "N/A",
                        "z_norm_calculated": "N/A", "z_norm_solver": "N/A",
                        "raw_home_strength": "N/A", "raw_ps": "N/A", "raw_md": "N/A"
                    })
                except Exception as e:
                    print(f"An unexpected error occurred for alpha={alpha_val}, beta={beta_val}: {e}")
                    writer.writerow({
                        "alpha": f"{alpha_val:.2f}", "beta": f"{beta_val:.2f}", "status": f"UnexpectedError: {type(e).__name__}",
                        "home_strength_norm": "N/A", "ps_norm": "N/A", "md_norm": "N/A",
                        "z_norm_calculated": "N/A", "z_norm_solver": "N/A",
                        "raw_home_strength": "N/A", "raw_ps": "N/A", "raw_md": "N/A"
                    })
    print(f"\nCalibration finished. Results saved to {args.output_csv}")
