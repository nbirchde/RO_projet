#!/usr/bin/env python3
"""
Simulated Annealing (SA) heuristic for the fair round-robin scheduling problem.
"""

import sys
import random
import math
import copy
import numpy as np
import concurrent.futures as cf
import multiprocessing as mp
import time
import logging # Ensure logging is imported
import numba # Ensure numba is imported
import os # Ensure os is imported

from . import config # Explicit relative import
# Corrected import: Use existing functions from metrics.py
from .metrics import calculate_home_strength, get_all_fairness_metrics, calculate_max_home_strength_denominator # Explicit relative import
# from .schedule_generator import generate_random_schedule_from_matches, generate_random_schedule_n_players # Explicit relative import -> REMOVED as file does not exist

from .packed_array_utils import get_status_packed, set_status_packed, PLAYERS_PER_BYTE # Changed to relative import

# Ensure logging setup is present and correct
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s') # Adjusted level for calibration runs
log = logging.getLogger(__name__)

PENALTY_LUT = np.array([1, 0, 0, 1], dtype=np.int8)

# Function to compute empirical normalization factors
def get_empirical_normalization_factors(n, num_samples=200, seed=42):
    """
    Generates a sample of schedules and computes median and std dev for HS, PS, MD.
    """
    random.seed(seed) # For initial_schedule generation
    np.random.seed(seed) # For any np randomness if used by helpers

    schedules_hs = []
    schedules_ps = []
    schedules_md = []

    log.info(f"Generating {num_samples} samples for n={n} to calculate empirical normalization factors...")
    # Ensure n is appropriate for initial_schedule if it has constraints (e.g. even n)
    # initial_schedule internally handles odd n by using n+1 for generation.
    
    for i in range(num_samples):
        sched_list = initial_schedule(n) # initial_schedule is defined later in this file
        if not sched_list and n > 1 : # initial_schedule might return empty for n=0 or n=1
             # For n > 1, if it's empty, it might be an issue or expected for very small n.
             # initial_schedule for n=2 (example) should produce a schedule.
             # If n is odd, initial_schedule adds a dummy player, so original_n is used.
             # Let's assume compute_metrics handles empty sched_list gracefully.
             pass

        # compute_metrics is defined later in this file.
        # It returns: raw_home_strength, penalites_sequence, max_dev
        hs, ps, md = compute_metrics(sched_list, n) # compute_metrics is defined later
        schedules_hs.append(hs)
        schedules_ps.append(ps)
        schedules_md.append(md)
    
    if not schedules_hs: # If all samples failed or num_samples was 0
        log.warning(f"No samples collected for n={n}. Using default sigmas=1.0, medians=0.0.")
        # Return placeholder values that won't cause division by zero
        return 0.0, 1.0, 0.0, 1.0, 0.0, 1.0

    med_hs = np.median(schedules_hs)
    sigma_hs = np.std(schedules_hs, ddof=1) if len(schedules_hs) > 1 else 1.0
    med_ps = np.median(schedules_ps)
    sigma_ps = np.std(schedules_ps, ddof=1) if len(schedules_ps) > 1 else 1.0
    med_md = np.median(schedules_md)
    sigma_md = np.std(schedules_md, ddof=1) if len(schedules_md) > 1 else 1.0

    # Handle cases where sigma might be zero or very small
    sigma_hs = max(sigma_hs, 1.0) # If std is 0 or too small, use 1.0 to avoid issues.
    sigma_ps = max(sigma_ps, 1.0)
    sigma_md = max(sigma_md, 1.0)
    
    log.info(f"Empirical factors for n={n}: med_hs={med_hs:.2f}, sigma_hs={sigma_hs:.2f}, med_ps={med_ps:.2f}, sigma_ps={sigma_ps:.2f}, med_md={med_md:.2f}, sigma_md={sigma_md:.2f}")
    return med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md

@numba.njit(fastmath=True, cache=True)
def calculate_normalized_score(home_strength, penalites_sequence, max_dev, alpha_pen_seq, beta_obj,
                               # max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx): # OLD PARAMS
                               med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md): # NEW PARAMS + medians
    # norm_home_strength = home_strength / max_home_strength_approx
    # norm_penalites_sequence = penalites_sequence / max_penalites_sequence_approx
    # norm_maxdev = max_dev / max_maxdev_approx
    # return norm_home_strength + alpha_pen_seq * norm_penalites_sequence + beta_obj * norm_maxdev

    # Ensure sigmas are not zero to prevent division by zero.
    # get_empirical_normalization_factors ensures sigmas are >= 1.0
    
    obj_hs = (home_strength - med_hs) / sigma_hs
    obj_ps = (penalites_sequence - med_ps) / sigma_ps
    obj_md = (max_dev - med_md) / sigma_md
    return obj_hs + alpha_pen_seq * obj_ps + beta_obj * obj_md

@numba.njit(fastmath=True, cache=True)
def calculate_unnormalized_score(home_strength, penalites_sequence, max_dev, alpha_pen_seq, beta_obj):
     return home_strength + alpha_pen_seq * penalites_sequence + beta_obj * max_dev

@numba.njit(fastmath=True, cache=True)
def _update_pen_numba_packed(player, round_idx, old_player_status_at_round, new_player_status_at_round, 
                             packed_seq_arr, current_pen_seq_val, rounds):
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

@numba.njit(fastmath=True, cache=True) # Removed parallel=True
def sa_loop(schedule_h_input, schedule_a_input, home_cnt_input, packed_seq_input,
            rnd_round_idx_arr, rnd_match_idx_arr,
            iterations, T0, cooling, alpha_pen_seq, beta_obj, ideal_home_games,
            # max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx, # OLD PARAMS (max_delta_approx was for hs)
            med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md, # NEW empirical factors
            seed, log_interval): # Added log_interval
    np.random.seed(seed)
    rounds = schedule_h_input.shape[0]
    matches_per_round = schedule_h_input.shape[1]
    n = home_cnt_input.shape[0] - 1
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
            current_home_strength += max(0, away_player - home_player) # Use new definition
    
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
    # max_delta_approx is the parameter name for max_home_strength_approx in this function
    current_norm_score = calculate_normalized_score(current_home_strength, current_pen_seq, current_max_dev,
                                                    alpha_pen_seq, beta_obj,
                                                    # max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx) # OLD PARAMS
                                                    med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md) # NEW PARAMS + medians
    current_unnorm_score = calculate_unnormalized_score(current_home_strength, current_pen_seq, current_max_dev,
                                                      alpha_pen_seq, beta_obj)
    best_norm_score_found = current_norm_score 
    actual_best_unnorm_score = current_unnorm_score
    actual_best_home_strength = current_home_strength
    actual_best_pen_seq = current_pen_seq
    actual_best_max_dev = current_max_dev
    best_found_iteration = -1
    T = T0
    for it in range(iterations):
        rnd_idx = rnd_round_idx_arr[it]
        match_idx = rnd_match_idx_arr[it]
        h = schedule_h[rnd_idx, match_idx] # Original home player
        a = schedule_a[rnd_idx, match_idx] # Original away player
        
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
                      for i in range(1, n + 1):
                          deviation = abs(home_cnt[i] - ideal_home_games)
                          if deviation > current_max_dev: current_max_dev = deviation
                  else: current_max_dev = 0.0
             else: current_max_dev = possible_new_max_dev
        else: current_max_dev = possible_new_max_dev
        current_pen_seq = _update_pen_numba_packed(h, rnd_idx, 1, 0, packed_seq, current_pen_seq, rounds)
        current_pen_seq = _update_pen_numba_packed(a, rnd_idx, 0, 1, packed_seq, current_pen_seq, rounds)
        schedule_h[rnd_idx, match_idx] = a # New home player
        schedule_a[rnd_idx, match_idx] = h # New away player
        
        candidate_norm_score = calculate_normalized_score(current_home_strength, current_pen_seq, current_max_dev,
                                                          alpha_pen_seq, beta_obj,
                                                          # max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx) # OLD PARAMS
                                                          med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md) # NEW PARAMS + medians
        delta_norm_score = candidate_norm_score - current_norm_score # current_norm_score is score before this move
        accept = delta_norm_score < 0 or np.random.random() < math.exp(-delta_norm_score / T)
        
        if accept:
            current_norm_score = candidate_norm_score # current_norm_score is now updated to the accepted state
            current_unnorm_score = calculate_unnormalized_score(current_home_strength, current_pen_seq, current_max_dev, alpha_pen_seq, beta_obj)
            if current_norm_score < best_norm_score_found:
                best_norm_score_found = current_norm_score
                actual_best_unnorm_score = current_unnorm_score 
                actual_best_home_strength = current_home_strength
                actual_best_pen_seq = current_pen_seq
                actual_best_max_dev = current_max_dev
                best_found_iteration = it
        else: # Revert changes
            schedule_h[rnd_idx, match_idx] = h # Revert to original home
            schedule_a[rnd_idx, match_idx] = a # Revert to original away
            home_cnt[h] += 1 # Revert home count for h
            home_cnt[a] -= 1 # Revert home count for a
            current_pen_seq = _update_pen_numba_packed(h, rnd_idx, 0, 1, packed_seq, current_pen_seq, rounds) # Revert h to home
            current_pen_seq = _update_pen_numba_packed(a, rnd_idx, 1, 0, packed_seq, current_pen_seq, rounds) # Revert a to away
            # Revert HomeStrength change using the same logic
            current_home_strength -= hs_change
            current_max_dev = max_dev_before_move
            # current_norm_score remains as it was before this candidate move
            
        T *= cooling
        if log_interval > 0 and it % log_interval == 0:
            # Calculate z-scores for logging
            # sigma_X are guaranteed to be >= 1.0 by get_empirical_normalization_factors
            z_hs = (current_home_strength - med_hs) / sigma_hs
            z_ps = (current_pen_seq - med_ps) / sigma_ps
            z_md = (current_max_dev - med_md) / sigma_md
            
            # Original f-string causing Numba error:
            # print(f"SA_LOOP_PROGRESS: Iter: {it}, Temp: {T:.2e}, CurrNormScore: {current_norm_score:.3f}, BestNormScore: {best_norm_score_found:.3f}, UnnormScore: {current_unnorm_score:.2f} (HS:{current_home_strength:.1f},PS:{current_pen_seq},MD:{current_max_dev:.1f}), z_HS:{z_hs:.3f}, z_PS:{z_ps:.3f}, z_MD:{z_md:.3f}")
            
            # Numba-compatible print statement:
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
            
    if best_found_iteration == -1: # No better solution found than initial
        return initial_schedule_h_snapshot, initial_schedule_a_snapshot, initial_packed_seq_snapshot, \
               actual_best_unnorm_score, actual_best_home_strength, actual_best_pen_seq, actual_best_max_dev
    if best_found_iteration == iterations - 1: # Best was the last one
         return schedule_h, schedule_a, packed_seq, \
                actual_best_unnorm_score, actual_best_home_strength, actual_best_pen_seq, actual_best_max_dev

    # Replay to get the best found state if it wasn't the last one
    np.random.seed(seed) 
    replayed_schedule_h = initial_schedule_h_snapshot.copy()
    replayed_schedule_a = initial_schedule_a_snapshot.copy()
    replayed_home_cnt = initial_home_cnt_snapshot.copy()
    replayed_packed_seq = initial_packed_seq_snapshot.copy()
    
    # Replay initial HomeStrength calculation: sum max(0, away_rank - home_rank)
    replayed_current_home_strength = np.float64(0)
    for r_loop_rep in range(rounds):
        for m_loop_rep in range(matches_per_round):
            home_player_rep = replayed_schedule_h[r_loop_rep, m_loop_rep]
            away_player_rep = replayed_schedule_a[r_loop_rep, m_loop_rep]
            replayed_current_home_strength += max(0, away_player_rep - home_player_rep) # Use new definition

    replayed_current_pen_seq = np.int64(0)
    if rounds > 1 and n > 0:
        for player_idx_loop in range(1, n + 1):
            for r_loop in range(rounds - 1):
                status_curr = get_status_packed(replayed_packed_seq, player_idx_loop, r_loop)
                status_next = get_status_packed(replayed_packed_seq, player_idx_loop, r_loop + 1)
                lut_idx = (status_curr << 1) | status_next
                replayed_current_pen_seq += np.int64(PENALTY_LUT[lut_idx])
    replayed_current_max_dev = 0.0
    if ideal_home_games >= 0 and n > 0:
        for i in range(1, n + 1):
            deviation = abs(replayed_home_cnt[i] - ideal_home_games)
            if deviation > replayed_current_max_dev: replayed_current_max_dev = deviation
    replayed_current_norm_score = calculate_normalized_score(replayed_current_home_strength, replayed_current_pen_seq, replayed_current_max_dev,
                                                             alpha_pen_seq, beta_obj,
                                                             # max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx) # OLD PARAMS
                                                             med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md) # NEW PARAMS + medians
    replayed_T = initial_T0_for_replay
    for k_it in range(best_found_iteration + 1):
        r_idx = rnd_round_idx_arr[k_it] 
        m_idx = rnd_match_idx_arr[k_it] 
        h_replay = replayed_schedule_h[r_idx, m_idx]
        a_replay = replayed_schedule_a[r_idx, m_idx]
        
        home_strength_before_move_replay = replayed_current_home_strength # Store before modification
        pen_seq_before_move_replay = replayed_current_pen_seq
        max_dev_before_move_replay = replayed_current_max_dev
        home_cnt_h_before_replay = replayed_home_cnt[h_replay]
        home_cnt_a_before_replay = replayed_home_cnt[a_replay]
        seq_h_char_before_replay = get_status_packed(replayed_packed_seq, h_replay, r_idx)
        seq_a_char_before_replay = get_status_packed(replayed_packed_seq, a_replay, r_idx)
        
        # Replay incremental HomeStrength update for new definition
        hs_change_replay = max(0, h_replay - a_replay) - max(0, a_replay - h_replay)
        replayed_current_home_strength += hs_change_replay
        
        replayed_home_cnt[h_replay] -= 1
        replayed_home_cnt[a_replay] += 1
        old_dev_h_replay = abs(home_cnt_h_before_replay - ideal_home_games)
        new_dev_h_replay = abs(replayed_home_cnt[h_replay] - ideal_home_games)
        old_dev_a_replay = abs(home_cnt_a_before_replay - ideal_home_games)
        new_dev_a_replay = abs(replayed_home_cnt[a_replay] - ideal_home_games)
        possible_new_max_dev_replay = max_dev_before_move_replay
        if new_dev_h_replay > possible_new_max_dev_replay: possible_new_max_dev_replay = new_dev_h_replay
        if new_dev_a_replay > possible_new_max_dev_replay: possible_new_max_dev_replay = new_dev_a_replay
        if (old_dev_h_replay == max_dev_before_move_replay and new_dev_h_replay < old_dev_h_replay) or \
           (old_dev_a_replay == max_dev_before_move_replay and new_dev_a_replay < old_dev_a_replay):
            if new_dev_h_replay < possible_new_max_dev_replay and new_dev_a_replay < possible_new_max_dev_replay :
                if n > 0:
                    replayed_current_max_dev = 0.0
                    for i_player in range(1, n + 1):
                        dev = abs(replayed_home_cnt[i_player] - ideal_home_games)
                        if dev > replayed_current_max_dev: replayed_current_max_dev = dev
                else: replayed_current_max_dev = 0.0
            else: replayed_current_max_dev = possible_new_max_dev_replay
        else: replayed_current_max_dev = possible_new_max_dev_replay
        replayed_current_pen_seq = _update_pen_numba_packed(h_replay, r_idx, 1, 0, replayed_packed_seq, replayed_current_pen_seq, rounds)
        replayed_current_pen_seq = _update_pen_numba_packed(a_replay, r_idx, 0, 1, replayed_packed_seq, replayed_current_pen_seq, rounds)
        replayed_schedule_h[r_idx, m_idx] = a_replay
        replayed_schedule_a[r_idx, m_idx] = h_replay
        
        candidate_norm_score_replay = calculate_normalized_score(replayed_current_home_strength, replayed_current_pen_seq, replayed_current_max_dev,
                                                                 alpha_pen_seq, beta_obj,
                                                                 # max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx) # OLD PARAMS
                                                                 med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md) # NEW PARAMS + medians
        delta_norm_score_replay = candidate_norm_score_replay - replayed_current_norm_score # replayed_current_norm_score is score before this move
        accept_replay = delta_norm_score_replay < 0 or np.random.random() < math.exp(-delta_norm_score_replay / replayed_T)
        
        if accept_replay:
            replayed_current_norm_score = candidate_norm_score_replay # Update score for next iteration's comparison
        else: # Revert move
            replayed_schedule_h[r_idx, m_idx] = h_replay
            replayed_schedule_a[r_idx, m_idx] = a_replay
            replayed_home_cnt[h_replay] = home_cnt_h_before_replay
            replayed_home_cnt[a_replay] = home_cnt_a_before_replay
            set_status_packed(replayed_packed_seq, h_replay, r_idx, seq_h_char_before_replay)
            set_status_packed(replayed_packed_seq, a_replay, r_idx, seq_a_char_before_replay)
            replayed_current_pen_seq = pen_seq_before_move_replay
            # Revert HomeStrength change using the same logic
            replayed_current_home_strength -= hs_change_replay
            replayed_current_max_dev = max_dev_before_move_replay
            # replayed_current_norm_score remains as it was
            
        replayed_T *= cooling
        if replayed_T < 1e-5: replayed_T = 1e-5 # Prevent T from becoming too small
        
    return replayed_schedule_h, replayed_schedule_a, replayed_packed_seq, \
           actual_best_unnorm_score, actual_best_home_strength, actual_best_pen_seq, actual_best_max_dev

def _sa_worker(args):
    n_arg, iterations_arg, seed_arg, num_threads_for_this_worker_arg, kwargs_arg = args
    numba.set_num_threads(num_threads_for_this_worker_arg)
    return solve_sa(n_arg, iterations=iterations_arg, seed=seed_arg, **kwargs_arg)

def initial_schedule(n):
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
                # Randomly assign home/away
                if random.random() < 0.5:
                    round_pairs.append((p1, p2))
                else:
                    round_pairs.append((p2, p1))
        if round_pairs:
             schedule.append(round_pairs)
        # Player 0 (or the first player in the list) is fixed, others rotate.
        # This is standard for the circle method.
        fixed_player = players[0]
        rotated_part = [players[-1]] + players[1:-1]
        players = [fixed_player] + rotated_part
    return schedule[:original_n - 1]

def _convert_schedule_to_np_format(schedule_list, n):
    rounds = len(schedule_list)
    matches_per_round = n // 2
    schedule_h = np.zeros((rounds, matches_per_round), dtype=np.int64)
    schedule_a = np.zeros((rounds, matches_per_round), dtype=np.int64)
    home_cnt = np.zeros(n + 1, dtype=np.int64)
    packed_rows = 0
    if n > 0: packed_rows = (n - 1) // PLAYERS_PER_BYTE + 1
    packed_seq = np.zeros((packed_rows, rounds), dtype=np.uint8)
    for r_idx, rnd_matches in enumerate(schedule_list):
        for m_idx, (h, a) in enumerate(rnd_matches):
            schedule_h[r_idx, m_idx] = h
            schedule_a[r_idx, m_idx] = a
            home_cnt[h] += 1
            if n > 0:
                set_status_packed(packed_seq, h, r_idx, 1)
                set_status_packed(packed_seq, a, r_idx, 0)
    return schedule_h, schedule_a, home_cnt, packed_seq

def compute_metrics(schedule, n):
    players = list(range(1, n + 1))
    if not schedule: return 0, 0, 0 # Adjusted tuple size
    rounds = len(schedule)
    if rounds == 0: return 0, 0, 0 # Adjusted tuple size
    
    _s_h, _s_a, _hc, current_packed_seq = _convert_schedule_to_np_format(schedule, n) # Used for packed_seq for penalty calc

    ideal_home_games = (n - 1) / 2.0
    
    # Use the new central function to calculate raw home strength
    # schedule is already in list format: [[(h,a), ...], ...]
    raw_home_strength = calculate_home_strength(schedule, n)
    
    home_games_count = {i: 0 for i in players} # Still needed for max_dev
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
    
    # total_breaks calculation removed
    return raw_home_strength, penalites_sequence, max_dev

def neighbor(schedule, n):
    new_sched = copy.deepcopy(schedule)
    if n % 2 != 0:
        raise ValueError("Neighbor generation requires even n for this schedule structure.")
    num_rounds = n - 1
    if num_rounds <= 0: return new_sched
    try:
        valid_round_indices = [r for r, rnd in enumerate(new_sched) if rnd]
        if not valid_round_indices: return new_sched
        rnd_idx = random.choice(valid_round_indices)
        round_len = len(new_sched[rnd_idx])
        if round_len == 0: return new_sched
        match_idx = random.randrange(round_len)
        home, away = new_sched[rnd_idx][match_idx]
        new_sched[rnd_idx][match_idx] = (away, home)
    except IndexError as e:
        log.warning(f"IndexError during neighbor generation (schedule might be malformed?). Error: {e}")
        return schedule
    except Exception as e:
         log.warning(f"Unexpected error during neighbor generation: {e}")
         return schedule
    return new_sched

def solve_sa(n, iterations=10000, initial_temp=1.5, cooling_rate=0.97, 
             alpha_pen_seq=None, beta_obj=None, seed=42, 
             log_interval_sa_loop=0, num_empirical_samples=200,
             med_hs=None, sigma_hs=None, med_ps=None, sigma_ps=None, med_md=None, sigma_md=None): # Added empirical factor params
    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA
    
    random.seed(seed)
    np.random.seed(seed)

    if n % 2 != 0:
        log.info(f"Odd n={n} detected. Initial schedule generation will use n+1 internally via initial_schedule function.")
        # initial_schedule handles odd n by adding a dummy player.

    # Get empirical normalization factors
    if med_hs is None or sigma_hs is None or med_ps is None or sigma_ps is None or med_md is None or sigma_md is None:
        log.info(f"Empirical factors not provided, calculating for n={n} with {num_empirical_samples} samples.")
        # Pass a seed for this calculation to make it deterministic.
        # Using seed + 1 to differentiate from the main SA randomness if that's ever an issue.
        med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md = get_empirical_normalization_factors(
            n, num_samples=num_empirical_samples, seed=seed + 1 if seed is not None else 43
        )
    else:
        log.info(f"Using provided empirical factors for n={n}.")

    initial_sched_list = initial_schedule(n) # Uses random, so seed set above matters
    if not initial_sched_list:
        log.warning("Initial schedule is empty.")
        return [], 0, (0, 0, 0) # Adjusted tuple size
    rounds = len(initial_sched_list)
    matches_per_round = n // 2
    schedule_h = np.zeros((rounds, matches_per_round), dtype=np.int64)
    schedule_a = np.zeros((rounds, matches_per_round), dtype=np.int64)
    home_cnt = np.zeros(n + 1, dtype=np.int64)
    packed_rows = 0
    if n > 0: packed_rows = (n - 1) // PLAYERS_PER_BYTE + 1
    packed_seq_arr = np.zeros((packed_rows, rounds), dtype=np.uint8)
    for r_init in range(rounds):
        for m_idx_init, (h_init, a_init) in enumerate(initial_sched_list[r_init]):
            schedule_h[r_init, m_idx_init] = h_init
            schedule_a[r_init, m_idx_init] = a_init
            home_cnt[h_init] += 1
            if n > 0:
                set_status_packed(packed_seq_arr, h_init, r_init, 1)
                set_status_packed(packed_seq_arr, a_init, r_init, 0)
    
    # Use the correct theoretical maximum S_max calculation for HomeStrength normalization denominator
    # max_home_strength_approx = calculate_max_home_strength_denominator(n) # NO LONGER PASSED TO SA_LOOP for objective
    # max_penalites_sequence_approx = n * (n - 2.0) if n > 2 else 1.0 # Ensure float for consistency # NO LONGER PASSED
    # max_maxdev_approx = (n - 1.0) / 2.0 if n > 1 else 1.0 # Ensure float # NO LONGER PASSED

    # Ensure denominators are at least 1.0
    # max_home_strength_approx = max(max_home_strength_approx, 1.0) # Not used by sa_loop objective
    # max_penalites_sequence_approx = max(max_penalites_sequence_approx, 1.0) # Not used
    # max_maxdev_approx = max(max_maxdev_approx, 1.0) # Not used
    
    ideal_home_games = (n - 1) / 2.0
    if rounds > 0 and matches_per_round > 0:
        rnd_round_idx_arr = np.random.randint(0, rounds, size=iterations, dtype=np.int64)
        rnd_match_idx_arr = np.random.randint(0, matches_per_round, size=iterations, dtype=np.int64)
    else:
        log.warning("No rounds or matches to process for random index generation.")
        return [], 0, (0, 0, 0) # Adjusted tuple size
    start_time = time.time()
    T_min = 1e-6
    effective_cooling_rate = cooling_rate
    if iterations > 0 and initial_temp > T_min and initial_temp > 0:
        effective_cooling_rate = (T_min / initial_temp)**(1.0 / iterations)
    elif iterations == 0 and initial_temp > 0 :
        effective_cooling_rate = 1.0
    log.info(f"Starting SA loop with {iterations} iterations. Initial Temp: {initial_temp:.2e}, Cooling Rate (effective): {effective_cooling_rate:.6f}. Log interval: {log_interval_sa_loop if log_interval_sa_loop > 0 else 'disabled'}")
    log.info(f"Using empirical factors for objective: med_hs={med_hs:.2f}, sigma_hs={sigma_hs:.2f}, med_ps={med_ps:.2f}, sigma_ps={sigma_ps:.2f}, med_md={med_md:.2f}, sigma_md={sigma_md:.2f}")
    # Consider T0 and cooling rate adjustments based on user prompt (e.g. T0*1.5, cooling_rate to 0.97)
    # These would be passed into solve_sa by the caller (e.g. run_calibration.py)
    
    best_schedule_h_arr, best_schedule_a_arr, final_best_packed_seq, \
    best_unnorm_score, best_home_strength, best_pen_seq, best_max_dev = sa_loop(
        schedule_h, schedule_a, home_cnt, packed_seq_arr, rnd_round_idx_arr, rnd_match_idx_arr,
        iterations, initial_temp, effective_cooling_rate, alpha_pen_seq, beta_obj, ideal_home_games,
        # max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx, # OLD PARAMS REMOVED
        med_hs, sigma_hs, med_ps, sigma_ps, med_md, sigma_md, # NEW EMPIRICAL FACTORS
        seed, 
        log_interval_sa_loop # Pass the new log_interval parameter
    )
    elapsed = time.time() - start_time
    log.info(f"SA loop finished in {elapsed:.4f} seconds.")
    best_sched_list = []
    for r in range(rounds):
        round_list = []
        for m_idx in range(matches_per_round):
            round_list.append((best_schedule_h_arr[r, m_idx], best_schedule_a_arr[r, m_idx]))
        best_sched_list.append(round_list)
    
    best_metrics = (best_home_strength, best_pen_seq, best_max_dev)
    log.info(f"Final Best Score (Unnormalized): {best_unnorm_score:.2f} (HomeStrength: {best_home_strength}, PénSeq: {best_pen_seq}, MaxDev: {best_max_dev:.2f})")
    return best_sched_list, best_unnorm_score, best_metrics

def solve_sa_parallel(n, iterations, runs=4, seed=42, executor=None, **kwargs):
    num_threads_per_worker = max(1, (os.cpu_count() or 1) // runs)
    log.info(f"Parallel SA: Target {runs} chains, {os.cpu_count() or 'N/A'} CPU cores detected. Assigning {num_threads_per_worker} Numba threads per worker.")
    seeds = [seed + i for i in range(runs)]
    args_list = [(n, iterations, s, num_threads_per_worker, kwargs) for s in seeds]
    results_list = []
    if executor:
        future_to_seed = {executor.submit(_sa_worker, arg): arg[2] for arg in args_list}
        for future in cf.as_completed(future_to_seed):
            s = future_to_seed[future]
            try:
                # result = future.result() # Ensure this is called to propagate exceptions
                results_list.append(future.result()) # Store the actual result
            except Exception as exc:
                # log.error(f"Seed {s} generated an exception: {exc}")
                # Optionally, re-raise or handle more gracefully
                # For now, just logging and continuing, which might lead to empty results_list
                # if all fail.
                log.exception(f"Seed {s} in parallel SA run generated an exception.") # More detailed log
    else:
        log.info(f"No shared executor provided to solve_sa_parallel, creating a new one with max_workers={runs}.")
        with cf.ProcessPoolExecutor(max_workers=runs, mp_context=mp.get_context("spawn")) as exe:
            results_list = list(exe.map(_sa_worker, args_list))
    if not results_list:
        log.error("All parallel SA runs failed to produce results.")
        return [], float('inf'), (float('inf'), float('inf'), float('inf')) # Adjusted tuple size
    best_schedule, best_score, best_metrics = min(results_list, key=lambda x: x[1])
    return best_schedule, best_score, best_metrics

# TODO: Tabu Search parts (_calculate_metrics_for_tabu, generate_top_k_flips_tabu, tabu_search_solver)
# currently use theoretical max approximations for their normalization if they perform any.
# If Tabu search is to be used with the new empirical normalization philosophy for its objective/evaluation,
# these functions will need similar updates to accept and use empirical sigmas (and potentially medians).
# For now, these functions remain unchanged and will use their original normalization logic if called.
# --- REMOVED TABU SEARCH CODE BELOW ---

def main():
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    iters_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    alpha_pen_seq_arg = float(sys.argv[3]) if len(sys.argv) > 3 else config.ALPHA
    beta_arg = float(sys.argv[4]) if len(sys.argv) > 4 else config.BETA
    runs_arg = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    log.info(f"Running SA for n={n_arg}, iterations={iters_arg}, alpha_pen_seq={alpha_pen_seq_arg}, beta={beta_arg}, parallel_runs={runs_arg}")
    
    # Pass explicitly, solve_sa and solve_sa_parallel will use them.
    # If they were to default to config values, we wouldn't need to pass them here if CLI args are absent.
    # However, the current structure of solve_sa_parallel's kwargs means we should pass them.
    sa_kwargs = {'alpha_pen_seq': alpha_pen_seq_arg, 'beta_obj': beta_arg}

    if runs_arg > 1:
        best_schedule, best_score, metrics_tuple = solve_sa_parallel(
            n_arg, runs=runs_arg, iterations=iters_arg, **sa_kwargs
        )
    else:
        best_schedule, best_score, metrics_tuple = solve_sa(
            n_arg, iterations=iters_arg, **sa_kwargs
        )
    final_home_strength, final_penalites_sequence, final_max_dev = metrics_tuple # These are from the solver's internal best state
    log.info(f"\n--- Best SA Schedule (Overall Score from Solver: {best_score:.2f}) ---")
    
    # Get all metrics using the new function from the final best_schedule
    # This ensures all calculations are consistent and includes home games per player
    all_metrics = get_all_fairness_metrics(best_schedule, n_arg)

    log.info("--- Detailed Fairness Metrics ---")
    log.info(f"  Number of Players (n): {all_metrics.get('num_players', 'N/A')}")
    log.info(f"  Number of Rounds: {all_metrics.get('num_rounds', 'N/A')}")
    
    log.info("\n  Home Strength:")
    raw_hs_val = all_metrics.get('raw_home_strength', 'N/A')
    norm_hs_val = all_metrics.get('normalized_home_strength', 'N/A')
    log.info(f"    Raw: {raw_hs_val:.4f}" if isinstance(raw_hs_val, float) else f"    Raw: {raw_hs_val}")
    log.info(f"    Normalized: {norm_hs_val:.4f}" if isinstance(norm_hs_val, float) else f"    Normalized: {norm_hs_val}")

    log.info("\n  Total Penalty Sequence (Breaks):")
    raw_tps_val = all_metrics.get('raw_total_penalty_sequence', 'N/A')
    norm_tps_val = all_metrics.get('normalized_total_penalty_sequence', 'N/A')
    log.info(f"    Raw: {raw_tps_val}" if isinstance(raw_tps_val, int) else f"    Raw: {raw_tps_val}")
    log.info(f"    Normalized: {norm_tps_val:.4f}" if isinstance(norm_tps_val, float) else f"    Normalized: {norm_tps_val}")

    log.info("\n  Max Deviation (from ideal home games):")
    raw_md_val = all_metrics.get('raw_max_deviation', 'N/A')
    norm_md_val = all_metrics.get('normalized_max_deviation', 'N/A')
    log.info(f"    Raw: {raw_md_val:.4f}" if isinstance(raw_md_val, float) else f"    Raw: {raw_md_val}")
    log.info(f"    Normalized: {norm_md_val:.4f}" if isinstance(norm_md_val, float) else f"    Normalized: {norm_md_val}")

    log.info("\n  Home Games Per Player (Player ID: Count):")
    home_games = all_metrics.get('home_games_per_player', [])
    if home_games:
        for i, count in enumerate(home_games):
            log.info(f"    Player {i+1}: {count}")
    elif all_metrics.get('num_players', 0) > 0:
        for i in range(all_metrics['num_players']):
             log.info(f"    Player {i+1}: 0")
    else:
        log.info("    N/A")

    log.info("\n  Player H/A Sequences (Player ID: Sequence):")
    player_sequences = all_metrics.get('player_ha_sequences', [])
    if player_sequences:
        for i, seq_str in enumerate(player_sequences):
            log.info(f"    Player {i+1}: {seq_str}")
    elif all_metrics.get('num_players', 0) > 0:
         for i in range(all_metrics['num_players']):
             log.info(f"    Player {i+1}: ") # Empty sequence
    else:
        log.info("    N/A")
    log.info("---------------------------------")

    # Verification step using original normalization (from get_all_fairness_metrics)
    log.info("\\n--- Verification Against Original Normalization Thresholds ---")
    hs_norm_orig = all_metrics.get('normalized_home_strength', float('inf'))
    ps_norm_orig = all_metrics.get('normalized_total_penalty_sequence', float('inf'))
    md_norm_orig = all_metrics.get('normalized_max_deviation', float('inf'))

    hs_ok = hs_norm_orig <= 0.10
    ps_ok = ps_norm_orig <= 0.20
    md_ok = md_norm_orig <= 0.20

    log.info(f"  Normalized HS (Original): {hs_norm_orig:.4f} (Target: <= 0.10) - Met: {hs_ok}")
    log.info(f"  Normalized PS (Original): {ps_norm_orig:.4f} (Target: <= 0.20) - Met: {ps_ok}")
    log.info(f"  Normalized MD (Original): {md_norm_orig:.4f} (Target: <= 0.20) - Met: {md_ok}")

    if hs_ok and ps_ok and md_ok:
        log.info("  VERIFICATION PASSED: All criteria met.")
    else:
        log.info("  VERIFICATION FAILED: One or more criteria not met.")
    log.info("---------------------------------")

    # log.info("\\n--- Schedule Details ---") # Removed schedule printing
    # for r, rnd in enumerate(best_schedule):
    #     match_strs = [f"{h}v{a}(H)" for h, a in rnd] # Assuming (home, away)
    #     log.info(f"Round {r+1}: {', '.join(match_strs)}")

if __name__ == '__main__':
    main()
