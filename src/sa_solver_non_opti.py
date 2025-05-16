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
import src.config as config

def initial_schedule(n):
    """
    Generates an initial round-robin schedule using the circle method.

    Assigns home/away arbitrarily in the first instance. Handles odd 'n'
    by adding a dummy player (None) for bye rounds.

    Args:
        n (int): Number of players.

    Returns:
        list: A list of rounds, where each round is a list of
              (home_player, away_player) tuples (using 1-based indexing).
    """
    players = list(range(1, n + 1))
    original_n = n
    if n % 2:
        players.append(None)
        n_effective = n + 1
    else:
        n_effective = n

    half = n_effective // 2
    schedule = []
    for r in range(n_effective - 1):
        round_pairs = []
        for i in range(half):
            p1 = players[i]
            p2 = players[n_effective - 1 - i]
            if p1 is not None and p2 is not None:
                round_pairs.append((p1, p2))
            elif p1 is None or p2 is None:
                 pass

        if round_pairs:
             schedule.append(round_pairs)

        if n_effective > 1:
            fixed_player = players[0]
            rotated_part = [players[n_effective - 1]] + players[1:n_effective - 1]
            players = [fixed_player] + rotated_part

    return schedule


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

    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA

    current_sched = initial_schedule(n)

    obj_c_home_strength, obj_c_pen_seq, obj_c_max_dev = compute_metrics(current_sched, n)

    current_norm_score = calculate_norm_score(obj_c_home_strength, obj_c_pen_seq, obj_c_max_dev, alpha_pen_seq, beta_obj)

    best_sched = copy.deepcopy(current_sched)
    best_norm_score = current_norm_score
    best_metrics = (obj_c_home_strength, obj_c_pen_seq, obj_c_max_dev)

    T = initial_temp

    print(f"Initial Score (normalized): {current_norm_score:.2f} (HS: {obj_c_home_strength:.2f}, PS: {obj_c_pen_seq:.2f}, MD: {obj_c_max_dev:.2f})")

    for it in range(iterations):
        candidate_sched = neighbor(current_sched, n)

        obj_cand_home_strength, obj_cand_pen_seq, obj_cand_max_dev = compute_metrics(candidate_sched, n)
        candidate_norm_score = calculate_norm_score(obj_cand_home_strength, obj_cand_pen_seq, obj_cand_max_dev, alpha_pen_seq, beta_obj)

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
                best_metrics = (obj_cand_home_strength, obj_cand_pen_seq, obj_cand_max_dev)

        T *= cooling_rate
        if T < 1e-5:
             T = 1e-5

        if it % (iterations // 10) == 0:
             print(f"Iter {it}/{iterations}, Temp: {T:.4f}, Current: {current_norm_score:.2f}, Best: {best_norm_score:.2f}")

    final_home_strength, final_penalites_sequence, final_max_dev = best_metrics
    print(f"Final Best Score (normalized): {best_norm_score:.2f} (HS: {final_home_strength:.2f}, PS: {final_penalites_sequence:.2f}, MD: {final_max_dev:.2f})")

    return best_sched, best_norm_score, best_metrics


def main():
    start_time = time.time()

    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    iters_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    alpha_pen_seq_arg = float(sys.argv[3]) if len(sys.argv) > 3 else config.ALPHA
    beta_arg = float(sys.argv[4]) if len(sys.argv) > 4 else config.BETA


    print(f"Running SA for n={n_arg}, iterations={iters_arg}, alpha_pen_seq={alpha_pen_seq_arg}, beta={beta_arg}")

    best_schedule, best_score, (final_home_strength, final_penalites_sequence, final_max_dev) = solve_sa(
        n_arg,
        iterations=iters_arg,
        alpha_pen_seq=alpha_pen_seq_arg,
        beta_obj=beta_arg
    )

    print(f"\n--- Best SA Schedule (Score: {best_score:.2f}) ---")
    print(f"Metrics: HomeStrength={final_home_strength:.4f}, Total Pénalités Séquence={final_penalites_sequence:.4f}, Max Deviation={final_max_dev:.4f}")

    print("\nFinal Schedule:")
    for r, rnd in enumerate(best_schedule):
        match_strs = [f"{h}-{a}" for h, a in rnd]
        print(f"Round {r+1}: {', '.join(match_strs)}")

    print("\nTime used: {:.2f} seconds".format(time.time() - start_time))
if __name__ == '__main__':
    main()
