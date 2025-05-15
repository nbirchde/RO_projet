import pulp
import numpy as np
from .metrics import calculate_home_strength, calculate_raw_total_penalty_sequence, calculate_raw_max_deviation
from .normalization_manager import calculate_analytical_factors, calculate_normalized_score

def solve_exact(n, weights):
    """
    Solves the round robin scheduling problem using an exact MILP model.
    The objective function minimizes a weighted sum of normalized deviations
    for Home Strength and Max Deviation, using analytical normalization factors.
    Penalty Sequence is not directly modeled in this MILP formulation.

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

    # 6. Max deviation calculation (for home games)
    for i in range(n_eff):
        prob += hs_dev_plus[i] <= max_dev
        prob += hs_dev_minus[i] <= max_dev

    # Objective function
    # Minimize the weighted sum of normalized deviations (z-scores)
    # Get analytical normalization factors (mean and std deviation)
    (mu_hs, sigma_hs), (mu_ps, sigma_ps), (mu_md, sigma_md) = calculate_analytical_factors(n)

    # The objective function minimizes the sum of z-scores for the metrics
    # that can be directly modeled in the MILP (Home Strength and Max Deviation).
    # Penalty Sequence is calculated post-solution.

    # Home Strength contribution to objective: sum of absolute deviations, normalized
    # Use multiplication by reciprocal for normalization in MILP
    hs_objective = pulp.lpSum(hs_dev_plus[i] + hs_dev_minus[i] for i in range(n_eff)) * (1.0 / sigma_hs)

    # Max Deviation contribution to objective: max deviation, normalized
    # Use multiplication by reciprocal for normalization in MILP
    md_objective = max_dev * (1.0 / sigma_md)

    # Total objective function
    prob += (
        weights.get('home_strength', 0) * hs_objective
        # + weights.get('penalty_sequence', 0) * ps_objective # Penalty sequence not directly modeled
        + weights.get('max_deviation', 0) * md_objective
    )

    # Solve the problem
    status = prob.solve()

    # Extract the schedule
    schedule = []
    for k in range(n_eff - 1):
        for i in range(n_eff):
            for j in range(n_eff):
                # Check if the variable is in the problem and its value is close to 1
                if (i, j, k) in x and pulp.value(x[i][j][k]) is not None and pulp.value(x[i][j][k]) > 0.5:
                    # If odd n, skip games involving the dummy player (n_eff - 1)
                    if is_odd and (i == n_eff - 1 or j == n_eff - 1):
                        continue
                    schedule.append((k + 1, i + 1, j + 1)) # Day, Home, Away (1-based indexing)

    # Calculate metrics from the extracted schedule
    # Filter out games with the dummy player for metric calculation if n is odd
    if is_odd:
        real_schedule = [(d, h, a) for (d, h, a) in schedule if h != n_eff and a != n_eff]
    else:
        real_schedule = schedule

    # Calculate raw metrics using the real schedule and original n
    raw_hs = calculate_home_strength(real_schedule, n)
    raw_ps = calculate_raw_total_penalty_sequence(real_schedule, n)
    raw_md = calculate_raw_max_deviation(real_schedule, n)

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
    # Example usage
    n_teams = 6 # Change to 5 for odd number of teams
    weights = {'home_strength': 1, 'penalty_sequence': 1, 'max_deviation': 1}

    schedule, metrics, status = solve_exact(n_teams, weights)

    print(f"Solver Status: {status}")
    print("\nGenerated Schedule (Day, Home, Away):")
    # Filter out dummy player games for display if n was odd
    n_eff_display = n_teams + 1 if n_teams % 2 != 0 else n_teams
    for game in schedule:
         if game[1] != n_eff_display and game[2] != n_eff_display:
             print(game)


    print("\nCalculated Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for sub_metric, sub_value in value.items():
                print(f"    {sub_metric}: {sub_value}") # Print raw values as is, formatted for floats
        else:
            print(f"  {metric}: {value}") # Print raw values as is, formatted for floats

    # Example with odd number of teams
    n_teams_odd = 5
    schedule_odd, metrics_odd, status_odd = solve_exact(n_teams_odd, weights)

    print(f"\n--- Solving for n={n_teams_odd} (Odd) ---")
    print(f"Solver Status: {status_odd}")
    print("\nGenerated Schedule (Day, Home, Away) - excluding dummy player games:")
    # Filter out dummy player games for display
    n_eff_display_odd = n_teams_odd + 1 if n_teams_odd % 2 != 0 else n_teams_odd
    for game in schedule_odd:
        if game[1] != n_eff_display_odd and game[2] != n_eff_display_odd:
            print(game)

    print("\nCalculated Metrics (for real teams):")
    for metric, value in metrics_odd.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for sub_metric, sub_value in value.items():
                print(f"    {sub_metric}: {sub_value}") # Print raw values as is, formatted for floats
        else:
            print(f"  {metric}: {value}") # Print raw values as is, formatted for floats
