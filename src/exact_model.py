import pulp
import numpy as np
from .metrics import calculate_home_strength, calculate_raw_total_penalty_sequence, calculate_raw_max_deviation
from .normalization_manager import calculate_analytical_factors, calculate_normalized_score
import argparse
import sys
import time
from .config import ALPHA, BETA

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
    if n % 2 != 0:
        n_eff = n + 1
        is_odd = True
    else:
        n_eff = n
        is_odd = False

    prob = pulp.LpProblem("Round Robin Scheduling", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (range(n_eff), range(n_eff), range(n_eff - 1)), 0, 1, pulp.LpBinary)

    y = pulp.LpVariable.dicts("y", (range(n_eff), range(n_eff - 1)), 0, 1, pulp.LpBinary)

    z = pulp.LpVariable.dicts("z", (range(n_eff), range(n_eff - 1)), 0, 1, pulp.LpBinary)

    hs_dev_plus = pulp.LpVariable.dicts("hs_dev_plus", range(n_eff), lowBound=0)
    hs_dev_minus = pulp.LpVariable.dicts("hs_dev_minus", range(n_eff), lowBound=0)

    pen_seq = pulp.LpVariable.dicts("pen_seq", (range(n_eff), range(n_eff - 2)), 0, 1, pulp.LpBinary)

    max_dev = pulp.LpVariable("max_dev", lowBound=0)

    # Constraints

    for i in range(n_eff):
        for j in range(i + 1, n_eff):
            if is_odd and (i == n_eff - 1 or j == n_eff - 1):
                 prob += pulp.lpSum(x[i][j][k] + x[j][i][k] for k in range(n_eff - 1)) == 1
            elif not is_odd:
                 prob += pulp.lpSum(x[i][j][k] + x[j][i][k] for k in range(n_eff - 1)) == 1


    for i in range(n_eff):
        for k in range(n_eff - 1):
            prob += pulp.lpSum(x[i][j][k] + x[j][i][k] for j in range(n_eff) if i != j) == 1

    for i in range(n_eff):
        for j in range(n_eff):
            if i != j:
                for k in range(n_eff - 1):
                    prob += x[i][j][k] <= y[i][k]
                    prob += x[j][i][k] <= z[i][k]

    for i in range(n_eff):
        for k in range(n_eff - 1):
            prob += y[i][k] + z[i][k] == 1

    for i in range(n_eff):
        for k in range(n_eff - 1):
            prob += pulp.lpSum(x[i][j][k] for j in range(n_eff) if i != j) == y[i][k]
            prob += pulp.lpSum(x[j][i][k] for j in range(n_eff) if i != j) == z[i][k]

    target_home_games = (n_eff - 1) / 2.0

    for i in range(n_eff):
        prob += pulp.lpSum(y[i][k] for k in range(n_eff - 1)) - target_home_games == hs_dev_plus[i] - hs_dev_minus[i]

    for i in range(n_eff):
        for k in range(n_eff - 2):
            prob += pen_seq[i][k] >= y[i][k] + y[i][k+1] - 1
            prob += pen_seq[i][k] >= (1 - y[i][k]) + (1 - y[i][k+1]) - 1

    for i in range(n_eff):
        prob += hs_dev_plus[i] <= max_dev
        prob += hs_dev_minus[i] <= max_dev

    (mu_hs, sigma_hs), (mu_ps, sigma_ps), (mu_md, sigma_md) = calculate_analytical_factors(n)

    hs_objective = pulp.lpSum(hs_dev_plus[i] + hs_dev_minus[i] for i in range(n)) * (1.0 / sigma_hs)

    ps_objective = pulp.lpSum(pen_seq[i][k] for i in range(n) for k in range(n_eff - 2)) * (1.0 / sigma_ps)

    md_objective = max_dev * (1.0 / sigma_md)


    prob += (
        weights.get('home_strength', 0) * hs_objective
        + weights.get('penalty_sequence', 0) * ps_objective
        + weights.get('max_deviation', 0) * md_objective
    )

    status = prob.solve()

    schedule = []
    for v in prob.variables():
        if v.name.startswith('x_') and v.varValue is not None and v.varValue > 0.5:
            try:
                parts = v.name.split('_')
                i = int(parts[1])
                j = int(parts[2])
                k = int(parts[3])

                if is_odd and (i == n_eff - 1 or j == n_eff - 1):
                    continue

                schedule.append((k + 1, i + 1, j + 1))
            except (ValueError, IndexError):
                print(f"Warning: Could not parse variable name: {v.name}")


    num_eff_rounds = n_eff - 1
    real_schedule_by_round_full = [[] for _ in range(num_eff_rounds)]

    for day, home, away in schedule:
         if 1 <= day <= num_eff_rounds:
              real_schedule_by_round_full[day - 1].append((home, away))

    raw_hs = calculate_home_strength(real_schedule_by_round_full, n)
    raw_ps = calculate_raw_total_penalty_sequence(real_schedule_by_round_full, n)
    raw_md = calculate_raw_max_deviation(real_schedule_by_round_full, n)

    total_normalized_score, obj_hs, obj_ps, obj_md, scaled_score, scaled_hs, scaled_ps, scaled_md = \
        calculate_normalized_score(raw_hs, raw_ps, raw_md, weights.get('penalty_sequence', 0), weights.get('max_deviation', 0), n)

    metrics = {
        'raw_home_strength': raw_hs,
        'raw_total_penalty_sequence': raw_ps,
        'raw_max_deviation': raw_md,
        'normalized': {
            'home_strength': obj_hs,
            'penalty_sequence': obj_ps,
            'max_deviation': obj_md,
            'scaled_home_strength': scaled_hs,
            'scaled_penalty_sequence': scaled_ps,
            'scaled_max_deviation': scaled_md,
        },
        'total_normalized_score': total_normalized_score,
        'total_scaled_score': scaled_score,
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
    weights = {
        'penalty_sequence': args.poids_ps,
        'max_deviation': args.poids_md
    }
    time_limit_arg = args.time_limit

    print(f"Running Exact Solver for n={n_arg}, weights (PS, MD)={weights}, time_limit={time_limit_arg}")

    if time_limit_arg is not None:
        pulp.LpSolverDefault.timeLimit = time_limit_arg

    schedule, metrics, status = solve_exact(n_arg, weights)

    print(f"Solver Status: {status}")

    if status == "Optimal":
        print("\nGenerated Schedule (Day, Home, Away):")
        n_eff_display = n_arg + 1 if n_arg % 2 != 0 else n_arg
        sorted_schedule = sorted(schedule, key=lambda x: x[0])
        for game in sorted_schedule:
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
