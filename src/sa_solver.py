import sys
import random
import math
import copy
import numpy as np
import concurrent.futures as cf
import multiprocessing as mp
import time
import logging
import numba
import os
import argparse # Import argparse

current_dir_sa = os.path.dirname(os.path.abspath(__file__))
project_root_sa = os.path.abspath(os.path.join(current_dir_sa, os.pardir))
if project_root_sa not in sys.path:
    sys.path.insert(0, project_root_sa)

from src import config
from src.metrics import calculate_home_strength, get_all_fairness_metrics, calculate_raw_total_penalty_sequence, calculate_raw_max_deviation
from src.normalization_manager import calculate_normalized_score, calculate_analytical_factors
from src.schedule_utils import initial_schedule
from src.packed_array_utils import get_status_packed, set_status_packed, PLAYERS_PER_BYTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
log = logging.getLogger(__name__)

PENALTY_LUT = np.array([1, 0, 0, 1], dtype=np.int8)


@numba.njit(fastmath=True, cache=True)
def calculate_unnormalized_score_numba(home_strength, penalites_sequence, max_dev, alpha_pen_seq, beta_obj):
    return home_strength + alpha_pen_seq * penalites_sequence + beta_obj * max_dev

@numba.njit(fastmath=True, cache=True)
def _update_pen_numba_packed(player, round_idx, old_player_status_at_round, new_player_status_at_round,
                             packed_seq_arr, current_pen_seq_val, rounds):
    # If player is the dummy player (0), no penalty sequence applies
    if player == 0:
        return current_pen_seq_val

    pen_change = np.int64(0)
    if round_idx > 0:
        status_prev_round = get_status_packed(packed_seq_arr, player, round_idx - 1)
        idx_old_link_prev = (status_prev_round << 1) | old_player_status_at_round
        idx_new_link_prev = (status_prev_round << 1) | new_player_status_at_round
        pen_change += np.int64(PENALTY_LUT[idx_new_link_prev]) - np.int64(PENALTY_LUT[idx_old_link_prev])
    if round_idx < rounds - 1:
        status_next_round = get_status_packed(packed_seq_arr, player, round_idx + 1)
        idx_old_link_next = (old_player_status_at_round << 1) | status_next_round
        idx_new_link_next = (new_player_status_at_round << 1) | status_next_round
        pen_change += np.int64(PENALTY_LUT[idx_new_link_next]) - np.int64(PENALTY_LUT[idx_old_link_next])
    set_status_packed(packed_seq_arr, player, round_idx, new_player_status_at_round)
    return current_pen_seq_val + pen_change

import time

@numba.njit(fastmath=True, cache=True)
def sa_loop(schedule_h_input, schedule_a_input, home_cnt_input, packed_seq_input,
            iterations, T0, cooling, alpha_pen_seq, beta_obj, ideal_home_games,
            n,
            mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md,
            seed, log_interval):
    np.random.seed(seed)

    rounds = schedule_h_input.shape[0]
    matches_per_round = schedule_h_input.shape[1]
    # n is now passed as a parameter
    schedule_h = schedule_h_input.copy()
    schedule_a = schedule_a_input.copy()
    home_cnt = home_cnt_input.copy()
    packed_seq = packed_seq_input.copy()
    initial_schedule_h_snapshot = schedule_h_input.copy()
    initial_schedule_a_snapshot = schedule_a_input.copy()
    initial_home_cnt_snapshot = home_cnt_input.copy()
    initial_packed_seq_snapshot = packed_seq_input.copy()
    initial_T0_for_replay = T0

    # Initial HomeStrength calculation: sum max(0, away_rank - home_rank)
    current_home_strength = np.float64(0)
    for r_loop_init in range(rounds):
        for m_loop_init in range(matches_per_round):
            home_player = schedule_h[r_loop_init, m_loop_init]
            away_player = schedule_a[r_loop_init, m_loop_init]
            current_home_strength += max(0, away_player - home_player)

    current_pen_seq = np.int64(0)
    if rounds > 1 and n > 0:
        for player_idx_loop in range(1, n + 1):
            for r_loop in range(rounds - 1):
                status_curr = get_status_packed(packed_seq, player_idx_loop, r_loop)
                status_next = get_status_packed(packed_seq, player_idx_loop, r_loop + 1)
                lut_idx = (status_curr << 1) | status_next
                current_pen_seq += np.int64(PENALTY_LUT[lut_idx])

    current_max_dev = 0.0
    if ideal_home_games >= 0 and n > 0:
        for i in range(1, n + 1):
            deviation = abs(home_cnt[i] - ideal_home_games)
            if deviation > current_max_dev: current_max_dev = deviation

    # Calculate initial normalized score using passed analytical factors
    obj_hs_init = (current_home_strength - mu_hs) / sigma_hs
    obj_ps_init = (current_pen_seq - mu_ps) / sigma_ps
    obj_md_init = (current_max_dev - mu_md) / sigma_md
    current_norm_score = obj_hs_init + alpha_pen_seq * obj_ps_init + beta_obj * obj_md_init

    current_unnorm_score = calculate_unnormalized_score_numba(current_home_strength, current_pen_seq, current_max_dev,
                                                      alpha_pen_seq, beta_obj)

    best_norm_score_found = current_norm_score
    actual_best_unnorm_score = current_unnorm_score
    actual_best_home_strength = current_home_strength
    actual_best_pen_seq = current_pen_seq
    actual_best_max_dev = current_max_dev
    best_found_iteration = -1
    T = T0

    # random indices will be generated on-the-fly

    # keep a copy of best schedule
    best_schedule_h_saved = schedule_h.copy()
    best_schedule_a_saved = schedule_a.copy()
    best_packed_seq_saved = packed_seq.copy()

    for it in range(iterations):
        rnd_idx = np.random.randint(0, rounds)
        match_idx = np.random.randint(0, matches_per_round)
        h = schedule_h[rnd_idx, match_idx] # Original home player
        a = schedule_a[rnd_idx, match_idx] # Original away player

        # Skip move if either player is the dummy player (represented as 0)
        if h == 0 or a == 0:
            T *= cooling
            if log_interval > 0 and it % log_interval == 0:
                # Use passed analytical factors for logging z-scores
                z_hs = (current_home_strength - mu_hs) / sigma_hs
                z_ps = (current_pen_seq - mu_ps) / sigma_ps
                z_md = (current_max_dev - mu_md) / sigma_md

                print("SA_LOOP_PROGRESS: Iter:", it,
                      "Temp:", T,
                      "CurrNormScore:", current_norm_score,
                      "BestNormScore:", best_norm_score_found,
                      "UnnormScore:", current_unnorm_score,
                      "(HS:", current_home_strength,
                      ",PS:", current_pen_seq,
                      ",MD:", current_max_dev,
                      "), z_HS:", z_hs,
                      ", z_PS:", z_ps,
                      ", z_MD:", z_md)
            continue # Skip the rest of the loop for this iteration

        # Incremental HomeStrength update for new definition: max(0, h-a) - max(0, a-h)
        # This is the change when flipping h vs a
        hs_change = max(0, h - a) - max(0, a - h)
        current_home_strength += hs_change

        max_dev_before_move = current_max_dev
        old_dev_h = abs(home_cnt[h] - ideal_home_games)
        home_cnt[h] -= 1
        new_dev_h = abs(home_cnt[h] - ideal_home_games)
        old_dev_a = abs(home_cnt[a] - ideal_home_games)
        home_cnt[a] += 1
        new_dev_a = abs(home_cnt[a] - ideal_home_games)
        possible_new_max_dev = max_dev_before_move
        if new_dev_h > possible_new_max_dev: possible_new_max_dev = new_dev_h
        if new_dev_a > possible_new_max_dev: possible_new_max_dev = new_dev_a
        if (old_dev_h == current_max_dev and new_dev_h < old_dev_h) or \
           (old_dev_a == current_max_dev and new_dev_a < old_dev_a):
             if new_dev_h < possible_new_max_dev and new_dev_a < possible_new_max_dev:
                  if n > 0:
                      current_max_dev = 0.0
                      for i in range(1, n + 1): # Iterate only through real players 1..n
                          deviation = abs(home_cnt[i] - ideal_home_games)
                          if deviation > current_max_dev: current_max_dev = deviation
                  else: current_max_dev = 0.0
             else: current_max_dev = possible_new_max_dev
        else: current_max_dev = possible_new_max_dev

        # Update penalty sequence only for real players
        if 1 <= h <= n:
            current_pen_seq = _update_pen_numba_packed(h, rnd_idx, 1, 0, packed_seq, current_pen_seq, rounds)
        if 1 <= a <= n:
            current_pen_seq = _update_pen_numba_packed(a, rnd_idx, 0, 1, packed_seq, current_pen_seq, rounds)

        schedule_h[rnd_idx, match_idx] = a # New home player
        schedule_a[rnd_idx, match_idx] = h # New away player

        # Calculate candidate normalized score using passed analytical factors
        obj_hs_cand = (current_home_strength - mu_hs) / sigma_hs
        obj_ps_cand = (current_pen_seq - mu_ps) / sigma_ps
        obj_md_cand = (current_max_dev - mu_md) / sigma_md
        candidate_norm_score = obj_hs_cand + alpha_pen_seq * obj_ps_cand + beta_obj * obj_md_cand

        delta_norm_score = candidate_norm_score - current_norm_score
        accept = delta_norm_score < 0 or np.random.random() < math.exp(-delta_norm_score / T)

        if accept:
            current_norm_score = candidate_norm_score
            current_unnorm_score = calculate_unnormalized_score_numba(current_home_strength, current_pen_seq, current_max_dev, alpha_pen_seq, beta_obj)
            if current_norm_score < best_norm_score_found:
                best_norm_score_found = current_norm_score
                actual_best_unnorm_score = current_unnorm_score
                actual_best_home_strength = current_home_strength
                actual_best_pen_seq = current_pen_seq
                actual_best_max_dev = current_max_dev
                best_found_iteration = it
                best_schedule_h_saved = schedule_h.copy()
                best_schedule_a_saved = schedule_a.copy()
                best_packed_seq_saved = packed_seq.copy()
        else: # Revert changes
            schedule_h[rnd_idx, match_idx] = h
            schedule_a[rnd_idx, match_idx] = a
            # Revert home_cnt only for real players
            if 1 <= h <= n:
                home_cnt[h] += 1
            if 1 <= a <= n:
                home_cnt[a] -= 1

            # Revert penalty sequence only for real players
            if 1 <= h <= n:
                current_pen_seq = _update_pen_numba_packed(h, rnd_idx, 0, 1, packed_seq, current_pen_seq, rounds)
            if 1 <= a <= n:
                current_pen_seq = _update_pen_numba_packed(a, rnd_idx, 1, 0, packed_seq, current_pen_seq, rounds)

            current_home_strength -= hs_change
            current_max_dev = max_dev_before_move

        T *= cooling
        if log_interval > 0 and it % log_interval == 0:
            # Use passed analytical factors for logging z-scores
            z_hs = (current_home_strength - mu_hs) / sigma_hs
            z_ps = (current_pen_seq - mu_ps) / sigma_ps
            z_md = (current_max_dev - mu_md) / sigma_md

            print("SA_LOOP_PROGRESS: Iter:", it,
                  "Temp:", T,
                  "CurrNormScore:", current_norm_score,
                  "BestNormScore:", best_norm_score_found,
                  "UnnormScore:", current_unnorm_score,
                  "(HS:", current_home_strength,
                  ",PS:", current_pen_seq,
                  ",MD:", current_max_dev,
                  "), z_HS:", z_hs,
                  ", z_PS:", z_ps,
                  ", z_MD:", z_md)

    return best_schedule_h_saved, best_schedule_a_saved, best_packed_seq_saved, \
           actual_best_unnorm_score, actual_best_home_strength, actual_best_pen_seq, actual_best_max_dev

def sa_loop_with_time_budget(schedule_h_input, schedule_a_input, home_cnt_input, packed_seq_input,
                             iterations, T0, cooling, alpha_pen_seq, beta_obj, ideal_home_games,
                             n, mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md,
                             seed, log_interval, time_budget_sec, chunk_size=10000):  # smaller chunks for better budget control
    """
    Python wrapper that repeatedly calls the Numba sa_loop in chunks,
    enforcing a wall-clock time budget while keeping high performance.
    """
    start_time = time.time()
    # Start with the full iteration allowance that was calculated by the
    # caller.  The outer loop will still stop as soon as the wall‑clock
    # budget is exhausted, but keeping a finite *remaining* counter makes
    # sure the geometric cooling schedule matches the intended number of
    # iterations so we don’t run indefinitely at a too‑high temperature.
    remaining = iterations if iterations > 0 else float('inf')
    best_u = float('inf')
    best_res = None
    # initialize temperature for continuous cooling across chunks
    T_current = T0
    # Try multiple chunks until time or iterations exhausted
    while remaining > 0 and (time_budget_sec <= 0 or time.time() - start_time < time_budget_sec):
        this_chunk = remaining if remaining < chunk_size else chunk_size
        # Call compiled SA loop with current temperature
        res = sa_loop(
            schedule_h_input, schedule_a_input, home_cnt_input, packed_seq_input,
            this_chunk, T_current, cooling, alpha_pen_seq, beta_obj, ideal_home_games,
            n, mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md,
            seed, log_interval
        )
        # res = (h_arr, a_arr, p_arr, unnorm_score, hs, ps, md)
        u_score = res[3]
        if u_score < best_u:
            best_u = u_score
            best_res = res
        # continue from best found state for next chunk
        schedule_h_input, schedule_a_input, packed_seq_input = res[0], res[1], res[2]
        # recompute home_cnt_input from updated home schedule
        home_cnt_input = np.zeros_like(home_cnt_input)
        rounds, matches = schedule_h_input.shape
        for r in range(rounds):
            for m in range(matches):
                p = schedule_h_input[r, m]
                if p > 0:
                    home_cnt_input[p] += 1
        remaining -= this_chunk
        seed += 1  # change seed for next chunk to diversify search
        # update temperature by geometric cooling over this_chunk iterations
        T_current *= math.pow(cooling, this_chunk)
    # If no budget or no best found, do a single full run
    if (time_budget_sec <= 0 and best_res is None) or best_res is None:
        return sa_loop(
            schedule_h_input, schedule_a_input, home_cnt_input, packed_seq_input,
            iterations, T0, cooling, alpha_pen_seq, beta_obj, ideal_home_games,
            n, mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md,
            seed, log_interval
        )
    return best_res

def _sa_worker(args):
    n_arg, iterations_arg, seed_arg, num_threads_for_this_worker_arg, kwargs_arg = args
    numba.set_num_threads(num_threads_for_this_worker_arg)
    # Extract expected arguments from kwargs_arg and pass them to solve_sa
    # Extract expected arguments from kwargs_arg and pass them to solve_sa
    return solve_sa(
        n_arg,
        iterations=iterations_arg,
        seed=seed_arg,
        alpha_pen_seq=kwargs_arg.get('alpha_pen_seq', config.ALPHA),
        beta_obj=kwargs_arg.get('beta_obj', config.BETA),
        log_interval_sa_loop=kwargs_arg.get('log_interval_sa_loop', 0),
        time_budget_sec=kwargs_arg.get('time_budget_sec', 0.0)
    )

def _convert_schedule_to_np_format(schedule_list, n):
    rounds = len(schedule_list)
    matches_per_round = len(schedule_list[0]) if schedule_list else 0 # Get actual matches per round
    schedule_h = np.zeros((rounds, matches_per_round), dtype=np.int64)
    schedule_a = np.zeros((rounds, matches_per_round), dtype=np.int64)
    home_cnt = np.zeros(n + 1, dtype=np.int64) # Size n+1 for 1-based indexing up to n
    packed_rows = 0
    if n > 0: packed_rows = (n - 1) // PLAYERS_PER_BYTE + 1 # Packed seq only for real players 1..n
    packed_seq = np.zeros((packed_rows, rounds), dtype=np.uint8)
    for r_idx, rnd_matches in enumerate(schedule_list):
        for m_idx, (h, a) in enumerate(rnd_matches):
            # h and a can be None (dummy player)
            schedule_h[r_idx, m_idx] = h if h is not None else 0 # Represent dummy as 0
            schedule_a[r_idx, m_idx] = a if a is not None else 0 # Represent dummy as 0

            # Only update home_cnt and packed_seq for real players
            if h is not None:
                home_cnt[h] += 1
                if 1 <= h <= n: # Ensure player ID is within expected range for packed_seq
                    set_status_packed(packed_seq, h, r_idx, 1)
            if a is not None:
                 if 1 <= a <= n: # Ensure player ID is within expected range for packed_seq
                    set_status_packed(packed_seq, a, r_idx, 0)

    return schedule_h, schedule_a, home_cnt, packed_seq

def compute_metrics(schedule, n):
    players = list(range(1, n + 1))
    if not schedule: return 0, 0, 0
    rounds = len(schedule)
    if rounds == 0: return 0, 0, 0

    # Convert schedule list to NumPy format to use packed_seq for penalty calc
    _s_h, _s_a, _hc, current_packed_seq = _convert_schedule_to_np_format(schedule, n)

    ideal_home_games = (n - 1) / 2.0

    raw_home_strength = calculate_home_strength(schedule, n)

    home_games_count = {i: 0 for i in players}
    for r_idx, rnd_matches in enumerate(schedule):
        for h_player, a_player in rnd_matches:
            home_games_count[h_player] += 1

    penalites_sequence = np.int64(0)
    if rounds > 1 and n > 0:
        for player_idx_loop in range(1, n + 1):
            for r_loop in range(rounds - 1):
                status_curr = get_status_packed(current_packed_seq, player_idx_loop, r_loop)
                status_next = get_status_packed(current_packed_seq, player_idx_loop, r_loop + 1)
                lut_idx = (status_curr << 1) | status_next
                penalites_sequence += np.int64(PENALTY_LUT[lut_idx])

    max_dev = 0.0
    if ideal_home_games >= 0 and n > 0:
        for i in players:
            deviation = abs(home_games_count[i] - ideal_home_games)
            if deviation > max_dev:
                max_dev = deviation

    return raw_home_strength, penalites_sequence, max_dev

def solve_sa(n, iterations=10000, initial_temp=1.5, cooling_rate=0.97,
             alpha_pen_seq=None, beta_obj=None, seed=42,
             log_interval_sa_loop=0,
             start_time_sec=0.0, time_budget_sec=0.0): # Add time parameters
    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA
    # Auto‑tune the geometric cooling factor so that the temperature
    # drops to roughly 0.1 % of its initial value after the allotted
    # number of iterations.  This prevents the schedule from “freezing”
    # too early when we run millions of iterations for a time budget.
    target_fraction = 1e-3            # T_final / T_initial
    if iterations > 0:
        effective_cooling_rate = math.exp(math.log(target_fraction) / iterations)
    else:                              # Fallback for the degenerate case
        effective_cooling_rate = cooling_rate

    log.info(f"Autotuned cooling_rate to {effective_cooling_rate:.10f} "
             f"so that T reaches {target_fraction}·T0 after {iterations} iterations.")

    random.seed(seed)
    np.random.seed(seed)

    log.info(f"solve_sa received iterations: {iterations}, time_budget_sec: {time_budget_sec}") # Add this log

    # Calculate analytical normalization factors once per SA run
    (mu_hs, sigma_hs), (mu_ps, sigma_ps), (mu_md, sigma_md) = calculate_analytical_factors(n)

    # Always generate a new random initial schedule
    log.info(f"Generating initial random schedule for n={n}.")
    current_sched_list = initial_schedule(n) # Uses random, so seed set above matters
    if not current_sched_list:
        log.warning("Initial schedule is empty.")
        # Return raw metrics and analytical metrics as 0 or inf for failure case, and scaled metrics as 0.5 (sigmoid center)
        return [], float('inf'), (float('inf'), float('inf'), float('inf')), (float('inf'), float('inf'), float('inf')), \
               float('inf'), (0.5, 0.5, 0.5) # Return scaled score and individual scaled metrics

    # Calculate initial metrics and scores for the random schedule
    c_home_strength, c_penalites_sequence, c_max_dev = compute_metrics(current_sched_list, n)

    # Calculate initial normalized score using analytical factors and get scaled scores
    initial_norm_score, initial_anal_hs, initial_anal_ps, initial_anal_md, \
    initial_scaled_score, initial_scaled_hs, initial_scaled_ps, initial_scaled_md = calculate_normalized_score(
        c_home_strength, c_penalites_sequence, c_max_dev,
        alpha_pen_seq, beta_obj, n # Pass n
    )
    current_unnorm_score = c_home_strength + alpha_pen_seq * c_penalites_sequence + beta_obj * c_max_dev # Calculate unnormalized for reporting

    # Calculate ideal home games
    ideal_home_games = (n - 1) / 2.0

    # Convert initial schedule list to NumPy format
    initial_schedule_h, initial_schedule_a, initial_home_cnt, initial_packed_seq_arr = _convert_schedule_to_np_format(current_sched_list, n)

    rounds = len(current_sched_list)
    matches_per_round = n // 2

    if rounds > 0 and matches_per_round > 0:
        # Ensure iterations is not excessively large if not using time budget
        if time_budget_sec <= 0:
             max_iterations_cap = 10**9 # Cap iterations to prevent memory issues
             if iterations > max_iterations_cap:
                  log.warning(f"Requested iterations ({iterations}) exceeds cap ({max_iterations_cap}). Capping at {max_iterations_cap}.")
                  iterations = max_iterations_cap
        # If using time budget, iterations can be very large, controlled by time check

        # random indices now generated inside sa_loop – no pre-allocation needed
        log.info(f"Random indices will be generated inside sa_loop.")
    else:
        log.warning("No rounds or matches to process for random index generation.")
        # Return raw metrics and analytical metrics as 0 or inf for failure case, and scaled metrics as 0.5 (sigmoid center)
        return [], float('inf'), (float('inf'), float('inf'), float('inf')), (float('inf'), float('inf'), float('inf')), \
               float('inf'), (0.5, 0.5, 0.5) # Return scaled score and individual scaled metrics

    # Trigger Numba compilation ahead of time with a tiny 1-iteration call
    try:
        _ = sa_loop(
            initial_schedule_h, initial_schedule_a, initial_home_cnt, initial_packed_seq_arr,
            1, initial_temp, effective_cooling_rate, alpha_pen_seq, beta_obj, ideal_home_games,
            n, mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md,
            seed, 0
        )
    except Exception:
        pass  # ignore compile errors here

    # ------------------------------------------------------------------
    #  🔄  Warm‑up speed test
    # ------------------------------------------------------------------
    # If we have a wall‑clock budget, measure the *true* iteration speed
    # now that all Numba kernels are hot.  Then retune both the iteration
    # target and the geometric cooling factor so that the temperature
    # reaches the desired fraction exactly at the end of the budget.
    if time_budget_sec and time_budget_sec > 0.0:
        warm_iters = min(50_000, max(2_000, iterations // 10))
        t0_speed = time.time()
        _ = sa_loop(
            initial_schedule_h.copy(),   # work on fresh copies so we don’t disturb the real run
            initial_schedule_a.copy(),
            initial_home_cnt.copy(),
            initial_packed_seq_arr.copy(),
            warm_iters,
            initial_temp,
            effective_cooling_rate,
            alpha_pen_seq,
            beta_obj,
            ideal_home_games,
            n, mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md,
            seed,
            0    # no logging
        )
        elapsed_speed = max(1e-6, time.time() - t0_speed)
        measured_rate = warm_iters / elapsed_speed
        iterations = int(measured_rate * time_budget_sec * 0.98)  # leave ~2 % buffer
        log.info(f"Warm‑up measured {measured_rate:.1f} iter/s; "
                 f"adjusting main run to {iterations} iterations.")

        # Retune the cooling schedule so that T falls to 0.1 % of T0 after
        # the *new* iteration budget.
        effective_cooling_rate = math.exp(math.log(target_fraction) / max(1, iterations))
        log.info(f"Retuned cooling_rate to {effective_cooling_rate:.10f}")

    # ------------------------------------------------------------------
    #  🚀  Main simulated‑annealing run (single compiled pass)
    # ------------------------------------------------------------------
    start_time_loop = time.time()

    best_schedule_h_arr, best_schedule_a_arr, final_best_packed_seq, \
    best_unnorm_score_found, best_home_strength, best_pen_seq, best_max_dev = sa_loop(
        initial_schedule_h, initial_schedule_a, initial_home_cnt, initial_packed_seq_arr,
        iterations, initial_temp, effective_cooling_rate, alpha_pen_seq, beta_obj, ideal_home_games,
        n, mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md,
        seed, log_interval_sa_loop
    )

    elapsed = time.time() - start_time_loop
    log.info(
        f"SA loop finished in {elapsed:.4f} seconds. "
        f"({iterations:,d} iterations → {iterations/elapsed:,.1f} it/s)"
    )

    # Convert best schedule NumPy arrays back to list format
    best_sched_list = []
    rounds = best_schedule_h_arr.shape[0] # Use shape from returned array
    matches_per_round = best_schedule_h_arr.shape[1] # Use shape from returned array
    for r in range(rounds):
        round_list = []
        for m_idx in range(matches_per_round):
            # Use the best found schedule arrays from sa_loop
            round_list.append((best_schedule_h_arr[r, m_idx], best_schedule_a_arr[r, m_idx]))
        best_sched_list.append(round_list)

    # Calculate the analytical normalized and scaled metrics of the best found schedule
    best_norm_score_calculated, best_anal_hs, best_anal_ps, best_anal_md, \
    best_scaled_score_calculated, best_scaled_hs, best_scaled_ps, best_scaled_md = calculate_normalized_score(
        best_home_strength, best_pen_seq, best_max_dev,
        alpha_pen_seq, beta_obj, n # Pass n
    )

    best_raw_metrics = (best_home_strength, best_pen_seq, best_max_dev)
    best_analytical_metrics = (best_anal_hs, best_anal_ps, best_anal_md)
    best_scaled_metrics = (best_scaled_hs, best_scaled_ps, best_scaled_md)


    log.info(f"Final Best Score (Analytical Normalized): {best_norm_score_calculated:.4f}")
    log.info(f"Final Best Analytical Norms (Z-Scores): (HS: {best_anal_hs:.4f}, PS: {best_anal_ps:.4f}, MD: {best_anal_md:.4f})")
    log.info(f"Final Best Scaled Score: {best_scaled_score_calculated:.4f}")
    log.info(f"Final Best Scaled Metrics ([0,1]): (HS: {best_scaled_hs:.4f}, PS: {best_scaled_ps:.4f}, MD: {best_scaled_md:.4f})")


    # Return the best schedule, the analytical normalized score, raw metrics, analytical metrics, scaled score, and scaled metrics
    return best_sched_list, best_norm_score_calculated, best_raw_metrics, best_analytical_metrics, best_scaled_score_calculated, best_scaled_metrics

def solve_sa_parallel(n, iterations, runs=6, seed=42, executor=None,
                      start_time_sec=0.0, time_budget_sec=0.0, **kwargs): # Add time parameters
    num_threads_per_worker = max(1, (os.cpu_count() or 1) // runs)
    log.info(f"Parallel SA: Target {runs} chains, {os.cpu_count() or 'N/A'} CPU cores detected. Assigning {num_threads_per_worker} Numba threads per worker.")


    seeds = [seed + i for i in range(runs)]
    # Pass n, time budget, and other kwargs to each worker
    worker_kwargs = {
        **kwargs,
        'n_players': n,
        'time_budget_sec': time_budget_sec
    }
    args_list = [(n, iterations, s, num_threads_per_worker, worker_kwargs) for s in seeds]
    results_list = []

    if executor:
        # The worker now returns (best_sched_list, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics)
        future_to_seed = {executor.submit(_sa_worker, arg): arg[2] for arg in args_list}
        for future in cf.as_completed(future_to_seed):
            s = future_to_seed[future]
            try:
                result = future.result()
                results_list.append(result)
            except Exception as exc:
                log.exception(f"Seed {s} in parallel SA run generated an exception.")
    else:
        log.info(f"No shared executor provided to solve_sa_parallel, creating a new one with max_workers={runs}.")
        with cf.ProcessPoolExecutor(max_workers=runs, mp_context=mp.get_context("spawn")) as exe:
            # Use list() to force execution and get results
            # The worker now returns (best_sched_list, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics)
            results_list = list(exe.map(_sa_worker, args_list))

    if not results_list:
        log.error("All parallel SA runs failed to produce results.")
        # If no runs succeeded, return default empty
        # Return best_schedule, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics
        return [], float('inf'), (float('inf'), float('inf'), float('inf')), (float('inf'), float('inf'), float('inf')), \
               float('inf'), (0.5, 0.5, 0.5) # Return scaled score and individual scaled metrics

    # Find the best result among all parallel runs based on the analytical normalized score
    best_result_from_runs = None
    best_scaled_score_from_runs = float('inf')

    # results_list contains tuples: (best_sched_list, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics)
    for result in results_list:
        # The fifth element is the scaled score
        scaled_score = result[4]
        if scaled_score < best_scaled_score_from_runs:
            best_scaled_score_from_runs = scaled_score
            best_result_from_runs = result # Store the entire result tuple

    # overall_best_schedule_list, overall_best_norm_score, overall_best_raw_metrics, overall_best_analytical_metrics, overall_best_scaled_score, overall_best_scaled_metrics = best_result_from_runs
    # Return the components of the best result tuple
    return best_result_from_runs[0], best_result_from_runs[1], best_result_from_runs[2], best_result_from_runs[3], best_result_from_runs[4], best_result_from_runs[5]

# import argparse # Moved import to the top


def main():
    parser = argparse.ArgumentParser(description='Run Simulated Annealing solver for schedule optimization.')
    parser.add_argument('n', type=int, help='Number of players.')
    parser.add_argument('-i', '--iterations', type=int, default=1000000,
                        help='Number of iterations for the SA loop (default: 100000). Ignored if --time_budget is provided.')
    parser.add_argument('-t', '--time_budget', type=float,
                        help='Time budget in seconds for the SA loop. Overrides --iterations if provided.')
    parser.add_argument('alpha', type=float, nargs='?', default=config.ALPHA,
                        help=f'Weight for penalty sequence objective (default: {config.ALPHA}).')
    parser.add_argument('beta', type=float, nargs='?', default=config.BETA,
                        help=f'Weight for max deviation objective (default: {config.BETA}).')
    parser.add_argument('runs', type=int, nargs='?', default=1,
                        help='Number of parallel runs (default: 1).')
    parser.add_argument('--test_iterations', type=int, default=1000,
                        help='Number of iterations for the test run to estimate iteration rate when using time budget (default: 1000).')
    parser.add_argument('--log_interval', type=int, default=0,
                        help='Interval for logging progress inside the SA loop (default: 0, disabled).')


    args = parser.parse_args()

    n_arg = args.n
    alpha_pen_seq_arg = args.alpha
    beta_arg = args.beta
    runs_arg = args.runs
    test_iterations_arg = args.test_iterations
    iters_arg = args.iterations # Default value or value from -i
    time_budget_arg = args.time_budget
    log_interval_arg = args.log_interval # Capture log interval argument

    if time_budget_arg is not None:
        log.info(f"Time budget specified: {time_budget_arg} seconds.")
        # Estimate how many iterations fit in the budget using a short test run
        log.info(f"Estimating iteration rate...")
        test_iters = test_iterations_arg
        while True:
            t0_est = time.time()
            _ = solve_sa(n_arg, iterations=test_iters,
                         alpha_pen_seq=alpha_pen_seq_arg, beta_obj=beta_arg,
                         log_interval_sa_loop=0)
            elapsed_est = max(1e-6, time.time() - t0_est)
            if elapsed_est >= 0.25 or test_iters >= 1_000_000:
                break
            test_iters *= 4
        iter_rate = test_iters / elapsed_est
        iters_arg = int(iter_rate * time_budget_arg * 0.95)
        hard_cap = 200_000_000_000
        if iters_arg > hard_cap:
            log.info(f"Capping iterations at {hard_cap} for memory safety.")
            iters_arg = hard_cap
        if iters_arg < 10:
            iters_arg = 10
        log.info(f"Estimated iteration rate: {iter_rate:.2f} it/s. Using {iters_arg} iterations for the main run.")
    else:
        iters_arg = args.iterations

    log.info(f"Running SA for n={n_arg}, iterations={iters_arg}, alpha_pen_seq={alpha_pen_seq_arg}, beta={beta_arg}, parallel_runs={runs_arg}")

    sa_kwargs = {
        'alpha_pen_seq': alpha_pen_seq_arg,
        'beta_obj': beta_arg,
        'log_interval_sa_loop': log_interval_arg
    }

    # Use parallel runs if requested; each chain will respect the time budget if provided
    if runs_arg > 1:
        # Parallel chains, each respecting time budget
        best_schedule, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics = solve_sa_parallel(
            n_arg,
            iterations=iters_arg,
            runs=runs_arg,
            time_budget_sec=time_budget_arg,
            **sa_kwargs
        )
    else:
        # Single chain with time budget enforcement
        best_schedule, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics = solve_sa(
            n_arg,
            iterations=iters_arg,
            time_budget_sec=time_budget_arg,
            **sa_kwargs
        )

    # Optional final top-up run if time budget was used and time remains
    if time_budget_arg is not None:
        # Note: This requires tracking elapsed time *outside* the solve_sa call,
        # which is not currently done in main().
        # For simplicity based on the provided diff, we'll skip the top-up for now
        # as the diff doesn't include the necessary time tracking in main().
        pass # Placeholder for potential future top-up logic

    # Unpack raw, analytical, and scaled metrics
    final_raw_hs, final_raw_ps, final_raw_md = raw_metrics
    final_anal_hs, final_anal_ps, final_anal_md = analytical_metrics
    final_scaled_hs, final_scaled_ps, final_scaled_md = scaled_metrics

    log.info(f"\n--- Best SA Schedule (Overall Analytical Normalized Score: {best_norm_score:.4f}) ---")

    # Get all metrics using the new function from the final best_schedule
    all_metrics = get_all_fairness_metrics(best_schedule, n_arg)

    log.info("--- Detailed Fairness Metrics ---")
    log.info(f"  Number of Players (n): {all_metrics.get('num_players', 'N/A')}")
    log.info(f"  Number of Rounds: {all_metrics.get('num_rounds', 'N/A')}")

    log.info("\n  Home Strength:")
    raw_hs_val = all_metrics.get('raw_home_strength', 'N/A')
    log.info(f"    Raw: {raw_hs_val:.4f}" if isinstance(raw_hs_val, float) else f"    Raw: {raw_hs_val}")
    log.info(f"    Analytical Norm (Z-Score): {final_anal_hs:.4f}")
    log.info(f"    Scaled Metric ([0,1]): {final_scaled_hs:.4f}")

    log.info("\n  Total Penalty Sequence (Breaks):")
    raw_tps_val = all_metrics.get('raw_total_penalty_sequence', 'N/A')
    log.info(f"    Raw: {raw_tps_val}" if isinstance(raw_tps_val, int) else f"    Raw: {raw_tps_val}")
    log.info(f"    Analytical Norm (Z-Score): {final_anal_ps:.4f}")
    log.info(f"    Scaled Metric ([0,1]): {final_scaled_ps:.4f}")

    log.info("\n  Max Deviation (from ideal home games):")
    raw_md_val = all_metrics.get('raw_max_deviation', 'N/A')
    log.info(f"    Raw: {raw_md_val:.4f}" if isinstance(raw_md_val, float) else f"    Raw: {raw_md_val}")
    log.info(f"    Analytical Norm (Z-Score): {final_anal_md:.4f}")
    log.info(f"    Scaled Metric ([0,1]): {final_scaled_md:.4f}")

    #log.info("\n  Home Games Per Player (Player ID: Count):")
    #home_games = all_metrics.get('home_games_per_player', [])
    #if home_games:
    #    for i, count in enumerate(home_games):
    #        log.info(f"    Player {i+1}: {count}")
    #elif all_metrics.get('num_players', 0) > 0:
    #    for i in range(all_metrics['num_players']):
    #         log.info(f"    Player {i+1}: 0")
    #else:
    #    log.info("    N/A")

    #log.info("\n  Player H/A Sequences (Player ID: Sequence):")
    #player_sequences = all_metrics.get('player_ha_sequences', [])
    #if player_sequences:
    #    for i, seq_str in enumerate(player_sequences):
    #        log.info(f"    Player {i+1}: {seq_str}")
    #elif all_metrics.get('num_players', 0) > 0:
    #     for i in range(all_metrics['num_players']):
    #         log.info(f"    Player {i+1}: ")
    #else:
    #    log.info("    N/A")
    #log.info("---------------------------------")

    # --- Print Schedule and Metrics to CSV ---
    csv_filename = f"sa_schedule_n{n_arg}.csv"
    log.info(f"\n--- Writing schedule and metrics to {csv_filename} ---")
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)

            # Write Schedule
            if best_schedule:
                rounds = len(best_schedule)
                matches_per_round = len(best_schedule[0]) if best_schedule else 0 # Use actual matches per round
                schedule_header = ["Round"] + [f"Match {i+1}" for i in range(matches_per_round)]
                writer.writerow(schedule_header)

                for r_idx, rnd in enumerate(best_schedule):
                    round_row = [f"Round {r_idx+1}"] + [f"{h}v{a}(H)" for h, a in rnd]
                    writer.writerow(round_row)
                writer.writerow([]) # Empty row for separation

            # Write Metrics
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Overall Analytical Normalized Score (Z-sum)", best_norm_score])
            writer.writerow(["Overall Scaled Score ([0,1] sum)", scaled_score])
            writer.writerow(["Raw Home Strength", final_raw_hs])
            writer.writerow(["Analytical Norm HS (Z-Score)", final_anal_hs])
            writer.writerow(["Scaled HS ([0,1])", final_scaled_hs])
            writer.writerow(["Raw Penalty Sequence", final_raw_ps])
            writer.writerow(["Analytical Norm PS (Z-Score)", final_anal_ps])
            writer.writerow(["Scaled PS ([0,1])", final_scaled_ps])
            writer.writerow(["Raw Max Deviation", final_raw_md])
            writer.writerow(["Analytical Norm MD (Z-Score)", final_anal_md])
            writer.writerow(["Scaled MD ([0,1])", final_scaled_md])

            home_games = all_metrics.get('home_games_per_player', [])
            if home_games:
                for i, count in enumerate(home_games):
                    writer.writerow([f"Home Games Player {i+1}", count])

            player_sequences = all_metrics.get('player_ha_sequences', [])
            if player_sequences:
                 for i, seq_str in enumerate(player_sequences):
                     writer.writerow([f"Sequence Player {i+1}", seq_str])

        log.info(f"Successfully wrote schedule and metrics to {csv_filename}")

    except Exception as e:
        log.error(f"Error writing CSV file {csv_filename}: {e}")

if __name__ == '__main__':
    main()
