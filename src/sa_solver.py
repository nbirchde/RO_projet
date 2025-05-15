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

# Add the project root directory to sys.path
current_dir_sa = os.path.dirname(os.path.abspath(__file__))
project_root_sa = os.path.abspath(os.path.join(current_dir_sa, os.pardir))
if project_root_sa not in sys.path:
    sys.path.insert(0, project_root_sa)

from src import config # Explicit relative import
# Import necessary functions from metrics.py for raw metric calculation
from src.metrics import calculate_home_strength, get_all_fairness_metrics, calculate_raw_total_penalty_sequence, calculate_raw_max_deviation
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

# Add the project root directory to sys.path
current_dir_sa = os.path.dirname(os.path.abspath(__file__))
project_root_sa = os.path.abspath(os.path.join(current_dir_sa, os.pardir))
if project_root_sa not in sys.path:
    sys.path.insert(0, project_root_sa)

from src import config # Explicit relative import
# Import necessary functions from metrics.py for raw metric calculation
from src.metrics import calculate_home_strength, get_all_fairness_metrics, calculate_raw_total_penalty_sequence, calculate_raw_max_deviation
# Import necessary functions from normalization_manager
from src.normalization_manager import calculate_normalized_score, calculate_analytical_factors # Updated import
# Import schedule generator utility
from src.schedule_utils import initial_schedule

from src.packed_array_utils import get_status_packed, set_status_packed, PLAYERS_PER_BYTE

# Ensure logging setup is present and correct
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s') # Adjusted level for calibration runs
log = logging.getLogger(__name__)

PENALTY_LUT = np.array([1, 0, 0, 1], dtype=np.int8)


@numba.njit(fastmath=True, cache=True)
def calculate_unnormalized_score_numba(home_strength, penalites_sequence, max_dev, alpha_pen_seq, beta_obj):
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

@numba.njit(fastmath=True, cache=True) 
def sa_loop(schedule_h_input, schedule_a_input, home_cnt_input, packed_seq_input,
            rnd_round_idx_arr, rnd_match_idx_arr,
            iterations, T0, cooling, alpha_pen_seq, beta_obj, ideal_home_games,
            n, # Pass n
            mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md, # Pass analytical factors
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
        else: # Revert changes
            schedule_h[rnd_idx, match_idx] = h
            schedule_a[rnd_idx, match_idx] = a
            home_cnt[h] += 1
            home_cnt[a] -= 1
            current_pen_seq = _update_pen_numba_packed(h, rnd_idx, 0, 1, packed_seq, current_pen_seq, rounds)
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
            replayed_current_home_strength += max(0, away_player_rep - home_player_rep)

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

    # Calculate initial normalized score for replay using passed analytical factors
    obj_hs_init_rep = (replayed_current_home_strength - mu_hs) / sigma_hs
    obj_ps_init_rep = (replayed_current_pen_seq - mu_ps) / sigma_ps
    obj_md_init_rep = (replayed_current_max_dev - mu_md) / sigma_md
    replayed_current_norm_score = obj_hs_init_rep + alpha_pen_seq * obj_ps_init_rep + beta_obj * obj_md_init_rep

    replayed_T = initial_T0_for_replay

    for k_it in range(best_found_iteration + 1):
        r_idx = rnd_round_idx_arr[k_it] 
        m_idx = rnd_match_idx_arr[k_it] 
        h_replay = replayed_schedule_h[r_idx, m_idx]
        a_replay = replayed_schedule_a[r_idx, m_idx]
        
        home_strength_before_move_replay = replayed_current_home_strength
        pen_seq_before_move_replay = replayed_current_pen_seq
        max_dev_before_move_replay = replayed_current_max_dev
        home_cnt_h_before_replay = replayed_home_cnt[h_replay]
        home_cnt_a_before_replay = replayed_home_cnt[a_replay]
        seq_h_char_before_replay = get_status_packed(replayed_packed_seq, h_replay, r_idx)
        seq_a_char_before_replay = get_status_packed(replayed_packed_seq, a_replay, r_idx)
        
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
        
        # Calculate candidate normalized score for replay using passed analytical factors
        obj_hs_cand_rep = (replayed_current_home_strength - mu_hs) / sigma_hs
        obj_ps_cand_rep = (replayed_current_pen_seq - mu_ps) / sigma_ps
        obj_md_cand_rep = (replayed_current_max_dev - mu_md) / sigma_md
        candidate_norm_score_replay = obj_hs_cand_rep + alpha_pen_seq * obj_ps_cand_rep + beta_obj * obj_md_cand_rep

        delta_norm_score_replay = candidate_norm_score_replay - replayed_current_norm_score
        accept_replay = delta_norm_score_replay < 0 or np.random.random() < math.exp(-delta_norm_score_replay / replayed_T)
        
        if accept_replay:
            replayed_current_norm_score = candidate_norm_score_replay
        else: # Revert move
            replayed_schedule_h[r_idx, m_idx] = h_replay
            replayed_schedule_a[r_idx, m_idx] = a_replay
            replayed_home_cnt[h_replay] = home_cnt_h_before_replay
            replayed_home_cnt[a_replay] = home_cnt_a_before_replay
            set_status_packed(replayed_packed_seq, h_replay, r_idx, seq_h_char_before_replay)
            set_status_packed(replayed_packed_seq, a_replay, r_idx, seq_a_char_before_replay)
            replayed_current_pen_seq = pen_seq_before_move_replay
            replayed_current_home_strength -= hs_change_replay
            replayed_current_max_dev = max_dev_before_move_replay
            
        replayed_T *= cooling
        if replayed_T < 1e-5: replayed_T = 1e-5
        
    return replayed_schedule_h, replayed_schedule_a, replayed_packed_seq, \
           actual_best_unnorm_score, actual_best_home_strength, actual_best_pen_seq, actual_best_max_dev

def _sa_worker(args):
    n_arg, iterations_arg, seed_arg, num_threads_for_this_worker_arg, kwargs_arg = args
    numba.set_num_threads(num_threads_for_this_worker_arg)
    # Extract expected arguments from kwargs_arg and pass them to solve_sa
    # solve_sa will calculate analytical factors internally
    return solve_sa(
        n_arg,
        iterations=iterations_arg,
        seed=seed_arg,
        alpha_pen_seq=kwargs_arg.get('alpha_pen_seq', config.ALPHA),
        beta_obj=kwargs_arg.get('beta_obj', config.BETA)
    )

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
    if not schedule: return 0, 0, 0
    rounds = len(schedule)
    if rounds == 0: return 0, 0, 0
    
    # Need to convert schedule list to NumPy format to use packed_seq for penalty calc
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
             log_interval_sa_loop=0): # Removed num_empirical_samples and load_best_schedule
    if alpha_pen_seq is None:
        alpha_pen_seq = config.ALPHA
    if beta_obj is None:
        beta_obj = config.BETA

    random.seed(seed)
    np.random.seed(seed)

    if n % 2 != 0:
        log.info(f"Odd n={n} detected. Initial schedule generation will use n+1 internally via initial_schedule function.")

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

    # Calculate ideal home games before converting to numpy format
    ideal_home_games = (n - 1) / 2.0

    # Convert initial schedule list to NumPy format for Numba
    initial_schedule_h, initial_schedule_a, initial_home_cnt, initial_packed_seq_arr = _convert_schedule_to_np_format(current_sched_list, n)

    rounds = len(current_sched_list)
    matches_per_round = n // 2

    if rounds > 0 and matches_per_round > 0:
        rnd_round_idx_arr = np.random.randint(0, rounds, size=iterations, dtype=np.int64)
        rnd_match_idx_arr = np.random.randint(0, matches_per_round, size=iterations, dtype=np.int64)
    else:
        log.warning("No rounds or matches to process for random index generation.")
        # Return raw metrics and analytical metrics as 0 or inf for failure case, and scaled metrics as 0.5 (sigmoid center)
        return [], float('inf'), (float('inf'), float('inf'), float('inf')), (float('inf'), float('inf'), float('inf')), \
               float('inf'), (0.5, 0.5, 0.5) # Return scaled score and individual scaled metrics

    start_time = time.time()
    T_min = 1e-6
    effective_cooling_rate = cooling_rate
    if iterations > 0 and initial_temp > T_min and initial_temp > 0:
        effective_cooling_rate = (T_min / initial_temp)**(1.0 / iterations)
    elif iterations == 0 and initial_temp > 0 :
        effective_cooling_rate = 1.0
    log.info(f"Starting SA loop with {iterations} iterations. Initial Temp: {initial_temp:.2e}, Cooling Rate (effective): {effective_cooling_rate:.6f}. Log interval: {log_interval_sa_loop if log_interval_sa_loop > 0 else 'disabled'}")
    log.info(f"Using analytical factors for objective: mu_hs={mu_hs:.4f}, sigma_hs={sigma_hs:.4f}, mu_ps={mu_ps:.4f}, sigma_ps={sigma_ps:.4f}, mu_md={mu_md:.4f}, sigma_md={sigma_md:.4f}")


    best_schedule_h_arr, best_schedule_a_arr, final_best_packed_seq, \
    best_unnorm_score_found, best_home_strength, best_pen_seq, best_max_dev = sa_loop(
        initial_schedule_h, initial_schedule_a, initial_home_cnt, initial_packed_seq_arr,
        rnd_round_idx_arr, rnd_match_idx_arr,
        iterations, initial_temp, effective_cooling_rate, alpha_pen_seq, beta_obj, ideal_home_games,
        n, # Pass n
        mu_hs, sigma_hs, mu_ps, sigma_ps, mu_md, sigma_md, # Pass analytical factors
        seed,
        log_interval_sa_loop
    )
    elapsed = time.time() - start_time
    log.info(f"SA loop finished in {elapsed:.4f} seconds.")

    # Convert best schedule NumPy arrays back to list format
    best_sched_list = []
    for r in range(rounds):
        round_list = []
        for m_idx in range(matches_per_round):
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

def solve_sa_parallel(n, iterations, runs=4, seed=42, executor=None, **kwargs):
    num_threads_per_worker = max(1, (os.cpu_count() or 1) // runs)
    log.info(f"Parallel SA: Target {runs} chains, {os.cpu_count() or 'N/A'} CPU cores detected. Assigning {num_threads_per_worker} Numba threads per worker.")

    # Removed logic to get empirical normalization factors

    seeds = [seed + i for i in range(runs)]
    # Pass n and other kwargs to the worker
    worker_kwargs = {**kwargs, 'n_players': n} # Pass n
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
    # We should find the best based on the *scaled* score now, as that's the new objective representation
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

# TODO: Tabu Search parts (_calculate_metrics_for_tabu, generate_top_k_flips_tabu, tabu_search_solver)
# currently use theoretical max approximations for their normalization if they perform any.
# If Tabu search is to be used with the new analytical normalization philosophy for its objective/evaluation,
# these functions will need similar updates to accept and use analytical mu and sigma.
# For now, these functions remain unchanged and will use their original normalization logic if called.
import argparse

# --- REMOVED TABU SEARCH CODE BELOW ---

def main():
    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    iters_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    alpha_pen_seq_arg = float(sys.argv[3]) if len(sys.argv) > 3 else config.ALPHA
    beta_arg = float(sys.argv[4]) if len(sys.argv) > 4 else config.BETA
    runs_arg = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    # seed_arg = int(sys.argv[6]) if len(sys.argv) > 6 else 42 # Assuming seed was 6th arg
    # initial_temp_arg = float(sys.argv[7]) if len(sys.argv) > 7 else 1.5 # Assuming initial_temp was 7th arg
    # cooling_rate_arg = float(sys.argv[8]) if len(sys.argv) > 8 else 0.97 # Assuming cooling_rate was 8th arg
    # log_interval_arg = int(sys.argv[9]) if len(sys.argv) > 9 else 0 # Assuming log_interval was 9th arg
    # num_empirical_samples_arg = int(sys.argv[10]) if len(sys.argv) > 10 else 200 # Assuming num_empirical_samples was 10th arg

    # Note: Reverting to sys.argv makes handling optional arguments and their order more rigid.
    # I will only include the most essential arguments (n, iterations, alpha, beta, runs)
    # and let the solve_sa function use its defaults for the SA parameters.

    if n_arg % 2 != 0:
        log.error(f"Error: Number of players (n={n_arg}) must be even.")
        sys.exit(1)

    log.info(f"Running SA for n={n_arg}, iterations={iters_arg}, alpha_pen_seq={alpha_pen_seq_arg}, beta={beta_arg}, parallel_runs={runs_arg}")

    sa_kwargs = {
        'alpha_pen_seq': alpha_pen_seq_arg,
        'beta_obj': beta_arg,
        # Use defaults from solve_sa for seed, initial_temp, cooling_rate, log_interval
    }

    if runs_arg > 1:
        # solve_sa_parallel now returns best_schedule, best_norm_score (analytical), raw_metrics, analytical_metrics (z-scores), scaled_score, scaled_metrics ([0,1])
        best_schedule, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics = solve_sa_parallel(
            n_arg, runs=runs_arg, iterations=iters_arg, **sa_kwargs
        )
    else:
        # solve_sa now returns best_schedule, best_norm_score (analytical), raw_metrics, analytical_metrics (z-scores), scaled_score, scaled_metrics ([0,1])
        best_schedule, best_norm_score, raw_metrics, analytical_metrics, scaled_score, scaled_metrics = solve_sa(
            n_arg, iterations=iters_arg, **sa_kwargs
        )
    
    # Unpack raw, analytical, and scaled metrics
    final_raw_hs, final_raw_ps, final_raw_md = raw_metrics
    final_anal_hs, final_anal_ps, final_anal_md = analytical_metrics
    final_scaled_hs, final_scaled_ps, final_scaled_md = scaled_metrics


    log.info(f"\n--- Best SA Schedule (Overall Analytical Normalized Score: {best_norm_score:.4f}) ---")

    # Get all metrics using the new function from the final best_schedule
    # This ensures all calculations are consistent and includes home games per player
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
                matches_per_round = n_arg // 2
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
