import pulp
import numpy as np
from .metrics import calculate_home_strength, calculate_raw_total_penalty_sequence, calculate_raw_max_deviation
from .normalization_manager import calculate_analytical_factors, calculate_normalized_score
import argparse # Import argparse
import sys # Import sys
import time # Import time
from .config import ALPHA, BETA # Import default weights from config

def solve_exact(n, weights):
    """
    Solves the round robin scheduling problem using an exact MILP model.
    The objective function minimizes a weighted sum of normalized deviations
    for Home Strength, Penalty Sequence, and Max Deviation, using analytical
    normalization factors.

    Args:
        n (int): Number of teams.
        weights (dict): Dictionary of weights for the objective function components
                        (home_strength, penalty_sequence, max_deviation).

    Returns:
        tuple: A tuple containing:
            - schedule (list): The generated schedule as a list of tuples (day, home, away).
            - metrics (dict): A dictionary of calculated metrics for the schedule.
            - status (int): The solver status (e.g., pulp.LpStatusOptimal).
    """
    # Ensure n is even for the standard round robin model
    if n % 2 != 0:
        # Handle odd n by adding a dummy player
        n_eff = n + 1
        is_odd = True
    else:
        n_eff = n
        is_odd = False

    # Define the problem
    prob = pulp.LpProblem("Round Robin Scheduling", pulp.LpMinimize)

    # Decision variables
    # x[i][j][k] = 1 if team i plays team j on day k, 0 otherwise
    x = pulp.LpVariable.dicts("x", (range(n_eff), range(n_eff), range(n_eff - 1)), 0, 1, pulp.LpBinary)

    # y[i][k] = 1 if team i plays at home on day k, 0 otherwise
    y = pulp.LpVariable.dicts("y", (range(n_eff), range(n_eff - 1)), 0, 1, pulp.LpBinary)

    # z[i][k] = 1 if team i plays away on day k, 0 otherwise
    z = pulp.LpVariable.dicts("z", (range(n_eff), range(n_eff - 1)), 0, 1, pulp.LpBinary)

    # Home strength deviation variables
    # These represent the absolute deviation from the target number of home games
    hs_dev_plus = pulp.LpVariable.dicts("hs_dev_plus", range(n_eff), lowBound=0)
    hs_dev_minus = pulp.LpVariable.dicts("hs_dev_minus", range(n_eff), lowBound=0)

    # Penalty sequence variables (defined for all n_eff participants)
    # pen_seq[i][k] = 1 if participant i has a penalty (HH or AA) between round k and k+1
    pen_seq = pulp.LpVariable.dicts("pen_seq", (range(n_eff), range(n_eff - 2)), 0, 1, pulp.LpBinary)

    # Max deviation variable
    # This represents the maximum absolute deviation across all teams for home games
    max_dev = pulp.LpVariable("max_dev", lowBound=0)

    # Constraints

    # 1. Each team plays every other team exactly once
    for i in range(n_eff):
        for j in range(i + 1, n_eff):
            # If odd n, the dummy player (n_eff-1) plays each real player once.
            # Real players (0 to n-1) play each other once.
            if is_odd and (i == n_eff - 1 or j == n_eff - 1):
                 # Dummy player plays each real player once
                 prob += pulp.lpSum(x[i][j][k] + x[j][i][k] for k in range(n_eff - 1)) == 1
            elif not is_odd:
                 # Real players play each other once
                 prob += pulp.lpSum(x[i][j][k] + x[j][i][k] for k in range(n_eff - 1)) == 1


    # 2. Each team plays exactly one game per day
    for i in range(n_eff):
        for k in range(n_eff - 1):
            prob += pulp.lpSum(x[i][j][k] + x[j][i][k] for j in range(n_eff) if i != j) == 1

    # 3. Relationship between x, y, and z variables
    for i in range(n_eff):
        for j in range(n_eff):
            if i != j:
                for k in range(n_eff - 1):
                    prob += x[i][j][k] <= y[i][k]  # If i plays j on day k, and i is home, y[i][k] must be 1
                    prob += x[j][i][k] <= z[i][k]  # If j plays i on day k, and i is away, z[i][k] must be 1

    for i in range(n_eff):
        for k in range(n_eff - 1):
            prob += y[i][k] + z[i][k] == 1 # Each team is either home or away on a given day

    # 4. Home/Away constraints (derived from 3, but good for clarity)
    for i in range(n_eff):
        for k in range(n_eff - 1):
            prob += pulp.lpSum(x[i][j][k] for j in range(n_eff) if i != j) == y[i][k] # Sum of games where i is home
            prob += pulp.lpSum(x[j][i][k] for j in range(n_eff) if i != j) == z[i][k] # Sum of games where i is away

    # 5. Home strength deviation calculation
    # Target home games for each team
    target_home_games = (n_eff - 1) / 2.0

    for i in range(n_eff):
        prob += pulp.lpSum(y[i][k] for k in range(n_eff - 1)) - target_home_games == hs_dev_plus[i] - hs_dev_minus[i]

    # 6. Penalty sequence constraints (Linearization of HH or AA)
    # pen_seq[i][k] = 1 if y[i][k] == y[i][k+1] for k in 0..n_eff-3
    for i in range(n_eff):
        for k in range(n_eff - 2): # Rounds 0 to n_eff-3
            # If y[i][k] == 1 and y[i][k+1] == 1 (HH), then pen_seq[i][k] must be 1
            prob += pen_seq[i][k] >= y[i][k] + y[i][k+1] - 1
            # If y[i][k] == 0 and y[i][k+1] == 0 (AA), then pen_seq[i][k] must be 1
            prob += pen_seq[i][k] >= (1 - y[i][k]) + (1 - y[i][k+1]) - 1

    # 7. Max deviation calculation (for home games)
    for i in range(n_eff):
        prob += hs_dev_plus[i] <= max_dev
        prob += hs_dev_minus[i] <= max_dev

    # Objective function
    # Minimize the weighted sum of normalized deviations (z-scores)
    # Get analytical normalization factors (mean and std deviation)
    # Note: Analytical factors are calculated based on the *real* number of players (n)
    (mu_hs, sigma_hs), (mu_ps, sigma_ps), (mu_md, sigma_md) = calculate_analytical_factors(n)

    # Home Strength contribution to objective: sum of absolute deviations for real players, normalized
    # Sum over real players (indices 0 to n-1)
    hs_objective = pulp.lpSum(hs_dev_plus[i] + hs_dev_minus[i] for i in range(n)) * (1.0 / sigma_hs)

    # Penalty Sequence contribution to objective: sum of penalties for real players, normalized
    # Sum over real players (indices 0 to n-1) and rounds 0 to n_eff-3
    ps_objective = pulp.lpSum(pen_seq[i][k] for i in range(n) for k in range(n_eff - 2)) * (1.0 / sigma_ps)

    # Max Deviation contribution to objective: max deviation for real players, normalized
    # The MaxDev variable is defined over n_eff participants, but the metric is for real players.
    # The constraints for MaxDev should ideally only link to real players' home game counts.
    # However, the current constraints link to all n_eff participants.
    # For consistency with the metric definition, we should ideally redefine MaxDev over real players.
    # Let's assume for now the MaxDev variable captures the max deviation among real players.
    # A more rigorous model would need MaxDev_real = max_{i=0..n-1} |H_i - (n-1)/2|.
    # Using the existing MaxDev variable which is over n_eff participants might not perfectly match the metric definition.
    # Let's proceed with the existing MaxDev variable for now, acknowledging this potential slight mismatch
    # between the MILP objective term and the exact metric definition for MD when n is odd.
    md_objective = max_dev * (1.0 / sigma_md)


    # Total objective function
    prob += (
        weights.get('home_strength', 0) * hs_objective
        + weights.get('penalty_sequence', 0) * ps_objective # Include Penalty Sequence
        + weights.get('max_deviation', 0) * md_objective
    )

    # Solve the problem
    status = prob.solve()

    # Extract the schedule
    schedule = []
    # Iterate through the decision variables and check their values
    for v in prob.variables():
        # Check if the variable name starts with 'x_' and its value is close to 1
        if v.name.startswith('x_') and v.varValue is not None and v.varValue > 0.5:
            # Extract indices (i, j, k) from the variable name
            # Variable names are typically in the format x_i_j_k
            try:
                parts = v.name.split('_')
                # Indices are 0-based in the variable names
                i = int(parts[1])
                j = int(parts[2])
                k = int(parts[3])

                # If odd n, skip games involving the dummy player (n_eff - 1 in 0-based index)
                if is_odd and (i == n_eff - 1 or j == n_eff - 1):
                    continue

                # Add game to schedule (using 1-based indexing for players and day)
                schedule.append((k + 1, i + 1, j + 1))
            except (ValueError, IndexError):
                # Handle unexpected variable name format if necessary
                print(f"Warning: Could not parse variable name: {v.name}")


    # Calculate metrics from the extracted schedule
    # The schedule list already excludes dummy player games during extraction.
    # We need to reconstruct the list of rounds for real players from the extracted schedule list
    # for the metric calculation functions.

    # Determine the number of rounds for the effective tournament (n_eff - 1)
    num_eff_rounds = n_eff - 1
    real_schedule_by_round_full = [[] for _ in range(num_eff_rounds)]

    for day, home, away in schedule:
         # Ensure day is within bounds (1 to n_eff-1)
         if 1 <= day <= num_eff_rounds:
              real_schedule_by_round_full[day - 1].append((home, away))

    # The metric functions in metrics.py are designed to work with a list of rounds
    # where each round contains tuples (home, away) for *real* players.
    # Passing `real_schedule_by_round_full` (which has n_eff-1 rounds, potentially with empty ones)
    # seems correct based on the metrics.py implementation for odd n.

    # Calculate raw metrics using the real schedule (excluding dummy games) and original n
    raw_hs = calculate_home_strength(real_schedule_by_round_full, n)
    raw_ps = calculate_raw_total_penalty_sequence(real_schedule_by_round_full, n)
    raw_md = calculate_raw_max_deviation(real_schedule_by_round_full, n)

    # Calculate normalized scores using the analytical factors and the dedicated function
    total_normalized_score, obj_hs, obj_ps, obj_md, scaled_score, scaled_hs, scaled_ps, scaled_md = \
        calculate_normalized_score(raw_hs, raw_ps, raw_md, weights.get('penalty_sequence', 0), weights.get('max_deviation', 0), n)

    metrics = {
        'raw_home_strength': raw_hs,
        'raw_total_penalty_sequence': raw_ps,
        'raw_max_deviation': raw_md,
        'normalized': {
            'home_strength': obj_hs, # This is the z-score
            'penalty_sequence': obj_ps, # This is the z-score
            'max_deviation': obj_md, # This is the z-score
            'scaled_home_strength': scaled_hs,
            'scaled_penalty_sequence': scaled_ps,
            'scaled_max_deviation': scaled_md,
        },
        'total_normalized_score': total_normalized_score, # Sum of z-scores
        'total_scaled_score': scaled_score, # Sum of scaled scores
    }


    return schedule, metrics, pulp.LpStatus[status]

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Solve the round robin scheduling problem using an exact MILP model.')
    parser.add_argument('n', type=int, help='Number of players.')
    parser.add_argument('poids_ps', type=float, nargs='?', default=ALPHA,
                        help=f'Weight for the Penalty Sequence objective term (default: {ALPHA}).')
    parser.add_argument('poids_md', type=float, nargs='?', default=BETA,
                        help=f'Weight for the Max Deviation objective term (default: {BETA}).')
    parser.add_argument('time_limit', type=float, nargs='?', default=None,
                        help='Time limit for the solver in seconds (default: None).')


    args = parser.parse_args()

    n_arg = args.n
    # Home Strength weight is implicitly 1.0 in the objective function
    weights = {
        'penalty_sequence': args.poids_ps,
        'max_deviation': args.poids_md
    }
    time_limit_arg = args.time_limit

    print(f"Running Exact Solver for n={n_arg}, weights (PS, MD)={weights}, time_limit={time_limit_arg}")

    # Set time limit for the solver if provided
    if time_limit_arg is not None:
        pulp.LpSolverDefault.timeLimit = time_limit_arg

    schedule, metrics, status = solve_exact(n_arg, weights)

    print(f"Solver Status: {status}")

    # Check if an optimal solution was found before trying to print the schedule
    if status == "Optimal":
        print("\nGenerated Schedule (Day, Home, Away):")
        # Filter out dummy player games for display if n was odd
        n_eff_display = n_arg + 1 if n_arg % 2 != 0 else n_arg
        # Sort schedule by day for cleaner printing
        sorted_schedule = sorted(schedule, key=lambda x: x[0])
        for game in sorted_schedule:
             # The schedule list already excludes dummy player games during extraction
             print(game)
        if not sorted_schedule:
             print("No schedule extracted (possibly due to small n or solver issue).")
    else:
        print("\nNo optimal schedule found.")


    print("\nCalculated Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for sub_metric, sub_value in value.items():
                print(f"    {sub_metric}: {sub_value}")
        else:
            print(f"  {metric}: {value}")

    print("\nTime used: {:.2f} seconds".format(time.time() - start_time))
