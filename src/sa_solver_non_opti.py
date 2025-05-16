import sys
import os
import random
import math
import copy
import numpy as np
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

from src.metrics import calculate_home_strength, calculate_raw_max_deviation, calculate_raw_total_penalty_sequence
from src.normalization_manager import calculate_normalized_score
from src.schedule_utils import initial_schedule
import src.config as config


def compute_metrics(schedule, n):
    """
    Computes the raw fairness metrics for a given schedule.

    Metrics:
        - Raw HomeStrength: Sum of max(0, away_rank - home_rank) for all matches.
        - Raw Total Penalty Sequence: Total count of consecutive home or away games for any player.
        - Raw Max Deviation: Maximum absolute deviation of the number of home games
          per player from the ideal average ((n-1)/2).

    Args:
        schedule (list): The schedule, a list of rounds, each a list of (home, away) tuples (1-based indices).
        n (int): The number of players.

    Returns:
        tuple: (raw_hs, raw_ps, raw_md) raw metrics
    """

    raw_home_strength = calculate_home_strength(schedule, n)
    raw_penalites_sequence = calculate_raw_total_penalty_sequence(schedule, n)
    raw_max_dev = calculate_raw_max_deviation(schedule, n)

    return raw_home_strength, raw_penalites_sequence, raw_max_dev


def neighbor(schedule, n, prob_flip=0.95, prob_swap_rounds=0, prob_swap_players=0.05):
    """
    Generates a neighboring schedule by either:
    - flipping the home/away assignment of a randomly selected match.
    - swapping two entire rounds (days) in the schedule.
    - swapping two players throughout the entire schedule.

    The choice of move is made based on the provided probabilities.

    Args:
        schedule (list): The current schedule (with 1-based indices).
        n (int): The number of players (real players, excluding dummy).

    Returns:
        list: A new schedule representing a neighbor of the input schedule.

    """
    if n == 0 or not schedule: return schedule
    new_sched = copy.deepcopy(schedule)
    num_rounds = len(schedule)
    round_len = len(schedule[0])

    total_prob = prob_flip + prob_swap_rounds + prob_swap_players
    if total_prob != 1:
        prob_flip /= total_prob
        prob_swap_rounds /= total_prob
        prob_swap_players /= total_prob

    move_type = None
    rand_val = random.random()

    if rand_val < prob_flip:
        move_type = "flip_home_away"
    elif rand_val < prob_flip + prob_swap_rounds:
        move_type = "swap_rounds"
    else:
        move_type = "swap_players"

    if move_type == "flip_home_away":
        while True:
            round_idx = random.randrange(num_rounds)
            match_idx = random.randrange(round_len)
            home, away = new_sched[round_idx][match_idx]
            if home is not None and away is not None:
                break

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


from src.normalization_manager import calculate_normalized_score, calculate_analytical_factors
from src.metrics import calculate_home_strength, calculate_raw_max_deviation, calculate_raw_total_penalty_sequence, get_all_fairness_metrics

def solve_sa(n, iterations=10000, initial_temp=1.0, cooling_rate=0.95, alpha_pen_seq=None, beta_obj=None, seed=51, time_budget=None):
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
        time_budget (float, optional): Time budget in seconds. If set, overrides iterations.

    Returns:
        tuple: (best_schedule, best_norm_score_calculated, best_raw_metrics, best_analytical_metrics, best_scaled_score_calculated, best_scaled_metrics)
    """
    random.seed(seed)

    # Support for time budget
    import time

    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA

    # Calculate analytical normalization factors once
    (mu_hs, sigma_hs), (mu_ps, sigma_ps), (mu_md, sigma_md) = calculate_analytical_factors(n)

    current_sched = initial_schedule(n)

    # Calculate initial raw metrics
    raw_c_home_strength, raw_c_pen_seq, raw_c_max_dev = compute_metrics(current_sched, n)

    # Calculate initial normalized score using analytical factors
    initial_norm_score, obj_c_home_strength, obj_c_pen_seq, obj_c_max_dev, _, _, _, _ = calculate_normalized_score(
        raw_c_home_strength, raw_c_pen_seq, raw_c_max_dev,
        alpha_pen_seq, beta_obj, n
    )
    current_norm_score = initial_norm_score # Use the weighted sum of z-scores for SA

    best_sched = copy.deepcopy(current_sched)
    best_norm_score = current_norm_score
    # Store best raw metrics found so far
    best_raw_metrics_found = (raw_c_home_strength, raw_c_pen_seq, raw_c_max_dev)


    T = initial_temp

    print(f"Initial Score (Analytical Normalized): {current_norm_score:.4f} (HS: {obj_c_home_strength:.4f}, PS: {obj_c_pen_seq:.4f}, MD: {obj_c_max_dev:.4f})")

    it = 0
    start_time = time.time()
    last_print = 0
    while True:
        # Check time budget
        if time_budget is not None:
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                break
        else:
            if it >= iterations:
                break

        candidate_sched = neighbor(current_sched, n)

        # Calculate raw metrics for candidate schedule
        raw_cand_home_strength, raw_cand_pen_seq, raw_cand_max_dev = compute_metrics(candidate_sched, n)

        # Calculate normalized score for candidate using analytical factors
        candidate_norm_score, obj_cand_home_strength, obj_cand_pen_seq, obj_cand_max_dev, _, _, _, _ = calculate_normalized_score(
            raw_cand_home_strength, raw_cand_pen_seq, raw_cand_max_dev,
            alpha_pen_seq, beta_obj, n
        )

        delta_norm_score = candidate_norm_score - current_norm_score

        if delta_norm_score < 0:
            accept = True
        else:
            probability = math.exp(-delta_norm_score / T)
            accept = random.random() < probability

        if accept:
            current_sched = candidate_sched
            current_norm_score = candidate_norm_score

            if current_norm_score < best_norm_score:
                best_sched = copy.deepcopy(current_sched)
                best_norm_score = current_norm_score
                # Update best raw metrics found
                best_raw_metrics_found = (raw_cand_home_strength, raw_cand_pen_seq, raw_cand_max_dev)


        T *= cooling_rate
        if T < 1e-5:
            T = 1e-5

        # Print progress every 10% of iterations or every 10% of time budget
        if time_budget is not None:
            elapsed = time.time() - start_time
            if time_budget > 0 and elapsed - last_print >= time_budget / 10:
                print(f"Time {elapsed:.1f}/{time_budget}s, Temp: {T:.4f}, Current: {current_norm_score:.4f}, Best: {best_norm_score:.4f}")
                last_print = elapsed
        else:
            if it % max(1, (iterations // 10)) == 0:
                print(f"Iter {it}/{iterations}, Temp: {T:.4f}, Current: {current_norm_score:.4f}, Best: {best_norm_score:.4f}")
        it += 1

    # Calculate final metrics for the best schedule found using calculate_normalized_score
    best_norm_score_calculated, best_anal_hs, best_anal_ps, best_anal_md, \
    best_scaled_score_calculated, best_scaled_hs, best_scaled_ps, best_scaled_md = calculate_normalized_score(
        best_raw_metrics_found[0], best_raw_metrics_found[1], best_raw_metrics_found[2],
        alpha_pen_seq, beta_obj, n
    )

    best_analytical_metrics = (best_anal_hs, best_anal_ps, best_anal_md)
    best_scaled_metrics = (best_scaled_hs, best_scaled_ps, best_scaled_md)

    print(f"Final Best Score (Analytical Normalized): {best_norm_score_calculated:.4f} (HS: {best_anal_hs:.4f}, PS: {best_anal_ps:.4f}, MD: {best_anal_md:.4f})")

    # Return the best schedule, the analytical normalized score, raw metrics, analytical metrics, scaled score, and scaled metrics
    return best_sched, best_norm_score_calculated, best_raw_metrics_found, best_analytical_metrics, best_scaled_score_calculated, best_scaled_metrics


def main():
    start_time = time.time()

    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    # If argument 2 is a float, treat as time budget, else as iterations
    iters_arg = None
    time_budget_arg = None
    if len(sys.argv) > 2:
        try:
            # Try to parse as float (time budget)
            val = float(sys.argv[2])
            if '.' in sys.argv[2] or val < 100:  # Heuristic: treat as seconds if float or small int
                time_budget_arg = val
            else:
                iters_arg = int(val)
        except ValueError:
            iters_arg = int(sys.argv[2])
    else:
        iters_arg = 10000

    alpha_pen_seq_arg = float(sys.argv[3]) if len(sys.argv) > 3 else config.ALPHA
    beta_arg = float(sys.argv[4]) if len(sys.argv) > 4 else config.BETA

    if time_budget_arg is not None:
        print(f"Running SA for n={n_arg}, time_budget={time_budget_arg}s, alpha_pen_seq={alpha_pen_seq_arg}, beta={beta_arg}")
        best_schedule, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics = solve_sa(
            n_arg,
            iterations=100000000,  # Large number, will be overridden by time_budget
            alpha_pen_seq=alpha_pen_seq_arg,
            beta_obj=beta_arg,
            time_budget=time_budget_arg
        )
    else:
        print(f"Running SA for n={n_arg}, iterations={iters_arg}, alpha_pen_seq={alpha_pen_seq_arg}, beta={beta_arg}")
        best_schedule, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics = solve_sa(
            n_arg,
            iterations=iters_arg,
            alpha_pen_seq=alpha_pen_seq_arg,
            beta_obj=beta_arg
        )

    # Unpack raw, analytical, and scaled metrics
    final_raw_hs, final_raw_ps, final_raw_md = raw_metrics
    final_anal_hs, final_anal_ps, final_anal_md = analytical_metrics
    final_scaled_hs, final_scaled_ps, final_scaled_md = scaled_metrics

    print(f"\n--- Best SA Schedule (Overall Analytical Normalized Score: {best_norm_score:.4f}) ---")
    print(f"Final Best Score (Analytical Normalized): {best_norm_score:.6f}")
    print(f"best_home_strength: {final_raw_hs}, best_pen_seq: {final_raw_ps}, best_max_dev: {final_raw_md}")

    # Get all metrics using the new function from the final best_schedule
    all_metrics = get_all_fairness_metrics(best_schedule, n_arg)

    print("--- Detailed Fairness Metrics ---")
    print(f"  Number of Players (n): {all_metrics.get('num_players', 'N/A')}")
    print(f"  Number of Rounds: {all_metrics.get('num_rounds', 'N/A')}")

    print("\n  Home Strength:")
    raw_hs_val = all_metrics.get('raw_home_strength', 'N/A')
    print(f"    Raw: {raw_hs_val:.4f}" if isinstance(raw_hs_val, float) else f"    Raw: {raw_hs_val}")
    print(f"    Analytical Norm (Z-Score): {final_anal_hs:.4f}")
    print(f"    Scaled Metric ([0,1]): {final_scaled_hs:.4f}")

    print("\n  Total Penalty Sequence (Breaks):")
    raw_tps_val = all_metrics.get('raw_total_penalty_sequence', 'N/A')
    print(f"    Raw: {raw_tps_val}" if isinstance(raw_tps_val, int) else f"    Raw: {raw_tps_val}")
    print(f"    Analytical Norm (Z-Score): {final_anal_ps:.4f}")
    print(f"    Scaled Metric ([0,1]): {final_scaled_ps:.4f}")

    print("\n  Max Deviation (from ideal home games):")
    raw_md_val = all_metrics.get('raw_max_deviation', 'N/A')
    print(f"    Raw: {raw_md_val:.4f}" if isinstance(raw_md_val, float) else f"    Raw: {raw_md_val}")
    print(f"    Analytical Norm (Z-Score): {final_anal_md:.4f}")
    print(f"    Scaled Metric ([0,1]): {final_scaled_md:.4f}")

    print("\n  Home Games Per Player (Player ID: Count):")
    home_games = all_metrics.get('home_games_per_player', [])
    if home_games:
        for i, count in enumerate(home_games):
            print(f"    Player {i+1}: {count}")
    elif all_metrics.get('num_players', 0) > 0 :
        for i in range(all_metrics['num_players']):
             print(f"    Player {i+1}: 0")
    else:
        print("    N/A")

    print("\n  Player H/A Sequences (Player ID: Sequence):")
    player_sequences = all_metrics.get('player_ha_sequences', [])
    if player_sequences:
         for i, seq_str in enumerate(player_sequences):
             print(f"    Player {i+1}: {seq_str}")
    elif all_metrics.get('num_players', 0) > 0:
         for i in range(all_metrics['num_players']):
             print(f"    Player {i+1}: ")
    else:
        print("    N/A")
    print("---------------------------------")

    print("\nFinal Schedule:")
    if best_schedule:
        for r, rnd in enumerate(best_schedule):
            match_strs = [f"{h}-{a}" for h, a in rnd]
            print(f"Round {r+1}: {', '.join(match_strs)}")
    else:
        print("No schedule generated.")

    print("\nTime used: {:.2f} seconds".format(time.time() - start_time))

if __name__ == '__main__':
    main()
