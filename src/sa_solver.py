#!/usr/bin/env python3
"""
Simulated Annealing (SA) heuristic for the fair round-robin scheduling problem.
"""
import sys
import random
import math
import copy
import numpy as np
from collections import defaultdict, deque
import concurrent.futures as cf
import multiprocessing as mp
import time
import os
import numba
import logging
import config # Import the configuration
# analysis_utils import for count_breaks and compute_total_breaks removed
from packed_array_utils import get_status_packed, set_status_packed, PLAYERS_PER_BYTE
# Import necessary functions from metrics, including the updated denominator calculator
from metrics import calculate_home_strength, get_all_fairness_metrics, calculate_max_home_strength_denominator

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

PENALTY_LUT = np.array([1, 0, 0, 1], dtype=np.int8)

@numba.njit(fastmath=True, cache=True)
def calculate_normalized_score(home_strength, penalites_sequence, max_dev, alpha_pen_seq, beta_obj,
                               max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx):
    norm_home_strength = home_strength / max_home_strength_approx
    norm_penalites_sequence = penalites_sequence / max_penalites_sequence_approx
    norm_maxdev = max_dev / max_maxdev_approx
    return norm_home_strength + alpha_pen_seq * norm_penalites_sequence + beta_obj * norm_maxdev

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
            max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx, seed, log_interval): # Added log_interval
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
                                                    max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx)
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
                                                          max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx)
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
            print("SA_LOOP_PROGRESS:", "Iter:", it, "Temp:", T, "CurrUnnormScore:", current_unnorm_score, "BestUnnormScore:", actual_best_unnorm_score)
            
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
                                                             max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx)
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
                                                                 max_delta_approx, max_penalites_sequence_approx, max_maxdev_approx)
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

def solve_sa(n, iterations=10000, initial_temp=1.0, cooling_rate=0.95, alpha_pen_seq=None, beta_obj=None, seed=42, log_interval_sa_loop=0): # Added log_interval_sa_loop
    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA
    random.seed(seed)
    np.random.seed(seed)
    if n % 2 != 0:
        raise ValueError("SA solver currently requires an even number of players (n).")
    initial_sched_list = initial_schedule(n)
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
    max_home_strength_approx = calculate_max_home_strength_denominator(n)
    max_penalites_sequence_approx = n * (n - 2.0) if n > 2 else 1.0 # Ensure float for consistency
    max_maxdev_approx = (n - 1.0) / 2.0 if n > 1 else 1.0 # Ensure float

    # Ensure denominators are at least 1.0
    max_home_strength_approx = max(max_home_strength_approx, 1.0)
    max_penalites_sequence_approx = max(max_penalites_sequence_approx, 1.0)
    max_maxdev_approx = max(max_maxdev_approx, 1.0)
    
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
    
    best_schedule_h_arr, best_schedule_a_arr, final_best_packed_seq, \
    best_unnorm_score, best_home_strength, best_pen_seq, best_max_dev = sa_loop(
        schedule_h, schedule_a, home_cnt, packed_seq_arr, rnd_round_idx_arr, rnd_match_idx_arr,
        iterations, initial_temp, effective_cooling_rate, alpha_pen_seq, beta_obj, ideal_home_games,
        max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx, seed, # Pass new denom name
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
                results_list.append(future.result())
            except Exception as exc:
                log.error(f'Seed {s} generated an exception: {exc}')
                results_list.append(([], float('inf'), (float('inf'), float('inf'), float('inf')))) # Adjusted tuple size
    else:
        log.info(f"No shared executor provided to solve_sa_parallel, creating a new one with max_workers={runs}.")
        with cf.ProcessPoolExecutor(max_workers=runs, mp_context=mp.get_context("spawn")) as exe:
            results_list = list(exe.map(_sa_worker, args_list))
    if not results_list:
        log.error("All parallel SA runs failed to produce results.")
        return [], float('inf'), (float('inf'), float('inf'), float('inf')) # Adjusted tuple size
    best_schedule, best_score, best_metrics = min(results_list, key=lambda x: x[1])
    return best_schedule, best_score, best_metrics

# Updated to accept normalization denominators and return normalized score for decision-making
def _calculate_metrics_for_tabu(schedule_h, schedule_a, home_cnt, packed_seq, n, alpha_pen_seq, beta_obj,
                               max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx):
    rounds = schedule_h.shape[0]
    matches_per_round = schedule_h.shape[1] # Assuming schedule_h is not empty
    # Calculate HomeStrength using the new definition: sum max(0, away-home)
    home_strength = np.float64(0)
    for r_loop in range(rounds):
        for m_loop in range(matches_per_round):
            home_player = schedule_h[r_loop, m_loop]
            away_player = schedule_a[r_loop, m_loop]
            home_strength += max(0, away_player - home_player) # Use new definition
            
    pen_seq = np.int64(0) 
    if rounds > 1 and n > 0:
        for player_idx_loop in range(1, n + 1):
            for r_loop in range(rounds - 1):
                status_curr = get_status_packed(packed_seq, player_idx_loop, r_loop)
                status_next = get_status_packed(packed_seq, player_idx_loop, r_loop + 1)
                lut_idx = (status_curr << 1) | status_next
                pen_seq += np.int64(PENALTY_LUT[lut_idx])
    max_dev = 0.0
    ideal_home_games = (n - 1) / 2.0
    if ideal_home_games >= 0 and n > 0:
        for i in range(1, n + 1):
            deviation = abs(home_cnt[i] - ideal_home_games)
            if deviation > max_dev: max_dev = deviation
    
    # Calculate both normalized and unnormalized scores
    norm_score = calculate_normalized_score(home_strength, pen_seq, max_dev, alpha_pen_seq, beta_obj,
                                            max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx)
    unnorm_score = calculate_unnormalized_score(home_strength, pen_seq, max_dev, alpha_pen_seq, beta_obj)
    
    return norm_score, unnorm_score, (home_strength, pen_seq, max_dev)

Move = tuple[int, int, int, int]

def _apply_move_to_schedule_arrays(move: Move, schedule_h, schedule_a, home_cnt, packed_seq, n):
    r_idx, m_idx, h_orig, a_orig = move
    home_cnt[h_orig] -= 1
    home_cnt[a_orig] += 1
    set_status_packed(packed_seq, h_orig, r_idx, 0)
    set_status_packed(packed_seq, a_orig, r_idx, 1)
    schedule_h[r_idx, m_idx] = a_orig
    schedule_a[r_idx, m_idx] = h_orig
    return schedule_h, schedule_a, home_cnt, packed_seq

def generate_top_k_flips_tabu(current_schedule_arrays, n, k, alpha_pen_seq, beta_obj,
                               max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx):
    current_h_arr, current_a_arr, current_home_cnt_arr, current_packed_seq_arr = current_schedule_arrays
    rounds = current_h_arr.shape[0]
    matches_per_round = current_h_arr.shape[1]
    potential_moves = []
    num_attempts = k * 5
    for _ in range(num_attempts):
        if rounds == 0 or matches_per_round == 0: break
        r_idx = random.randrange(rounds)
        m_idx = random.randrange(matches_per_round)
        h_player = current_h_arr[r_idx, m_idx]
        a_player = current_a_arr[r_idx, m_idx]
        move = (r_idx, m_idx, h_player, a_player)
        temp_h = current_h_arr.copy()
        temp_a = current_a_arr.copy()
        temp_home_cnt = current_home_cnt_arr.copy()
        temp_packed_seq = current_packed_seq_arr.copy()
        _apply_move_to_schedule_arrays(move, temp_h, temp_a, temp_home_cnt, temp_packed_seq, n)
        # Pass normalization denominators
        norm_s, unnorm_s, new_metrics_tuple = _calculate_metrics_for_tabu(
            temp_h, temp_a, temp_home_cnt, temp_packed_seq, n, alpha_pen_seq, beta_obj,
            max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx
        )
        potential_moves.append({'move': move, 'norm_score': norm_s, 'unnorm_score': unnorm_s, 'metrics': new_metrics_tuple, 
                                'arrays': (temp_h, temp_a, temp_home_cnt, temp_packed_seq)})
    potential_moves.sort(key=lambda x: x['norm_score']) # Sort by normalized score
    return potential_moves[:k]

def tabu_search_solver(n_players, initial_schedule_list, iterations=5000, tenure=20, k_neighbors=50,
                       alpha_pen_seq=None, beta_obj=None, seed=42):
    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA
    random.seed(seed)
    np.random.seed(seed)

    # Use the correct theoretical maximum S_max calculation for HomeStrength normalization denominator
    max_home_strength_approx = calculate_max_home_strength_denominator(n_players)
    max_penalites_sequence_approx = n_players * (n_players - 2.0) if n_players > 2 else 1.0
    max_maxdev_approx = (n_players - 1.0) / 2.0 if n_players > 1 else 1.0
    
    # Ensure denominators are at least 1.0
    max_home_strength_approx = max(max_home_strength_approx, 1.0)
    max_penalites_sequence_approx = max(max_penalites_sequence_approx, 1.0)
    max_maxdev_approx = max(max_maxdev_approx, 1.0)

    s_h, s_a, hc, ps = _convert_schedule_to_np_format(initial_schedule_list, n_players)
    
    current_norm_score, current_unnorm_score, current_metrics_tuple = _calculate_metrics_for_tabu(
        s_h, s_a, hc, ps, n_players, alpha_pen_seq, beta_obj,
        max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx
    )
    current_arrays = (s_h, s_a, hc, ps)
    
    best_arrays = tuple(arr.copy() for arr in current_arrays)
    best_norm_score = current_norm_score
    best_unnorm_score = current_unnorm_score # Keep track of best unnormalized score
    best_metrics_tuple = current_metrics_tuple
    
    effective_tenure = max(10, min(int(0.1 * n_players), 75))
    if n_players <= 10: effective_tenure = max(2, n_players -1) if n_players > 1 else 1 # Ensure n_players-1 is not negative
    tabu_list = deque(maxlen=effective_tenure)
    log.info(f"Starting Tabu Search: n={n_players}, iters={iterations}, tenure={effective_tenure}, k_neighbors={k_neighbors}")
    log.info(f"Initial TS Score (Norm): {best_norm_score:.2f}, (Unnorm): {best_unnorm_score:.2f}, Metrics (HS,P,M): {best_metrics_tuple}")

    for i in range(iterations):
        neighbors = generate_top_k_flips_tabu(current_arrays, n_players, k_neighbors, alpha_pen_seq, beta_obj,
                                             max_home_strength_approx, max_penalites_sequence_approx, max_maxdev_approx)
        best_candidate_move_info = None
        for move_info in neighbors:
            tabu_attr = (move_info['move'][0], move_info['move'][1])
            is_tabu = tabu_attr in tabu_list
            # Decisions based on normalized score
            aspiration_met = move_info['norm_score'] < best_norm_score 
            if not is_tabu or aspiration_met:
                if best_candidate_move_info is None or move_info['norm_score'] < best_candidate_move_info['norm_score']:
                    best_candidate_move_info = move_info
                    if is_tabu and aspiration_met:
                        log.debug(f"TS iter {i}: Aspiration criterion met for tabu move {tabu_attr} with norm_score {move_info['norm_score']:.2f} (global best norm: {best_norm_score:.2f})")
        if best_candidate_move_info is None:
            if neighbors:
                log.warning(f"TS iter {i}: All {len(neighbors)} neighbors are tabu and none met aspiration. Picking best tabu move by norm_score.")
                best_candidate_move_info = min(neighbors, key=lambda x: x['norm_score'])
            else:
                log.error(f"TS iter {i}: No neighbors generated. Stopping.")
                break
        current_arrays = best_candidate_move_info['arrays']
        current_norm_score = best_candidate_move_info['norm_score'] # Now using norm_score
        current_unnorm_score = best_candidate_move_info['unnorm_score']
        current_metrics_tuple = best_candidate_move_info['metrics']
        
        chosen_move_details = best_candidate_move_info['move']
        chosen_tabu_attribute = (chosen_move_details[0], chosen_move_details[1])
        tabu_list.append(chosen_tabu_attribute) 
        
        if current_norm_score < best_norm_score: # Compare with best_norm_score
            best_norm_score = current_norm_score
            best_unnorm_score = current_unnorm_score # Store corresponding unnormalized
            best_metrics_tuple = current_metrics_tuple
            best_arrays = tuple(arr.copy() for arr in current_arrays) 
            log.info(f"TS iter {i}: New best norm_score: {best_norm_score:.2f} (unnorm: {best_unnorm_score:.2f}, Metrics HS,P,M: {best_metrics_tuple})")
        if i % 100 == 0 or i == iterations -1 :
             log.info(f"TS iter {i}: Current norm_score: {current_norm_score:.2f} (unnorm: {current_unnorm_score:.2f}), Best norm_score: {best_norm_score:.2f} (Metrics HS,P,M: {current_metrics_tuple})")
    
    best_h_arr, best_a_arr, _best_hc_arr, best_packed_seq_final = best_arrays
    best_sched_list = []
    rounds = best_h_arr.shape[0]
    matches_per_round = best_h_arr.shape[1]
    for r_idx in range(rounds):
        round_list_item = []
        for m_idx_item in range(matches_per_round):
            round_list_item.append((best_h_arr[r_idx, m_idx_item], best_a_arr[r_idx, m_idx_item]))
        best_sched_list.append(round_list_item)
    
    # Return the best unnormalized score and its metrics for consistency with SA solver's reporting
    log.info(f"Tabu Search finished. Final Best Unnormalized Score: {best_unnorm_score:.2f} (Norm: {best_norm_score:.2f}, Metrics HS,P,M: {best_metrics_tuple})")
    return best_sched_list, best_unnorm_score, best_metrics_tuple

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

    # log.info("\n--- Schedule Details ---") # Removed schedule printing
    # for r, rnd in enumerate(best_schedule):
    #     match_strs = [f"{h}v{a}(H)" for h, a in rnd] # Assuming (home, away)
    #     log.info(f"Round {r+1}: {', '.join(match_strs)}")

if __name__ == '__main__':
    main()
