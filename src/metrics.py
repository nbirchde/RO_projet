#!/usr/bin/env python3
"""
Functions for calculating and normalizing fairness metrics for tournament schedules.
"""
import numpy as np # Retaining numpy in case it's useful for other metrics later

def calculate_home_strength(schedule_list, n):
    """
    Calculates the raw HomeStrength metric based on player ranks.
    HS = sum (rank_away_player - rank_home_player) for all matches.
    Assumes player IDs are their ranks (1 to n).

    Args:
        schedule_list (list): A list of rounds, where each round is a list of
                              (home_player_id, away_player_id) tuples.
        n (int): Number of players.

    Returns:
        float: The raw HomeStrength value.
    """
    if not schedule_list or n == 0:
        return 0.0

    raw_hs = 0.0
    for round_matches in schedule_list:
        for home_player, away_player in round_matches:
            # Assuming player IDs are 1-based ranks
            raw_hs += (away_player - home_player)
    return raw_hs

def normalize_home_strength(raw_home_strength, n):
    """
    Normalizes the raw HomeStrength metric.
    The denominator is S_max = n*(n-1)*(n+1)/6, which is the theoretical
    maximum absolute value if player ranks are 1 to n. This aims for a range of [-1, 1].
    A value of 0 implies a neutral schedule in terms of strength advantage,
    while +/-1 implies an extreme schedule.
    """
    if n < 2: # S_max would be 0 or ill-defined for n=0,1. Max HS is 0.
        return 0.0
    
    # S_max = n * (n-1) * (n+1) / 6
    # Ensure integer division is not used if n*(n-1)*(n+1) is not divisible by 6,
    # though for n>=2, n*(n-1)*(n+1) is always divisible by 6.
    denominator = n * (n - 1) * (n + 1) / 6.0
    
    if denominator == 0: # Should only happen if n < 2, already handled.
        return 0.0 # Or handle as an error, but 0 is consistent.
    return raw_home_strength / denominator

def normalize_total_pen_seq(total_pen_seq, n):
    """
    Normalizes the Total Penalty Sequence metric.
    The denominator is a theoretical maximum or a characteristic value.
    """
    if n <= 2: # No sequences possible for n=0,1,2
        return 0.0
    # Denominator: Max possible penalties if each player has n-2 breaks (worst case)
    denominator = n * (n - 2.0) 
    if denominator == 0:
        return 0.0
    return total_pen_seq / denominator

def normalize_max_dev(max_dev, n):
    """
    Normalizes the Max Deviation metric.
    The denominator is a theoretical maximum or a characteristic value.
    """
    if n <= 1:
        return 0.0
    # Denominator: Max possible deviation from the mean number of home games
    # For n even, ideal is (n-1)/2. Max actual can be n/2 or (n-2)/2. Deviation is 0.5.
    # For n odd, ideal is (n-1)/2. Max actual can be (n-1)/2. Deviation is 0.
    # The old denominator was (n-1.0)/2.0.
    # If n is even, (n-1)/2 is x.5. Max dev is 0.5. Denom = (n-1)/2. Result is max_dev / ((n-1)/2)
    # If n=4, (4-1)/2 = 1.5. Max dev is 0.5.  0.5 / 1.5 = 1/3.
    # If n=6, (6-1)/2 = 2.5. Max dev is 0.5.  0.5 / 2.5 = 1/5.
    # This seems to normalize to a range smaller than [0,1] if max_dev is always 0.5 for even n.
    # Let's keep the original normalization for MaxDev as its interpretation is subtle.
    denominator = (n - 1.0) / 2.0
    if denominator == 0: # handles n=1
        return 0.0
    return max_dev / denominator

def calculate_home_games_per_player(schedule_list, n):
    """
    Calculates the number of home games played by each player.
    Player IDs are assumed to be 1 to n.
    Returns a list where index i corresponds to player i+1.
    """
    if n == 0:
        return []
    home_games_counts = [0] * n
    if not schedule_list:
        return home_games_counts
    for round_matches in schedule_list:
        for home_player, _ in round_matches:
            if 1 <= home_player <= n: # Ensure player ID is valid
                home_games_counts[home_player - 1] += 1
    return home_games_counts

def calculate_raw_max_deviation(schedule_list, n):
    """
    Calculates the raw Max Deviation metric: MaxDev = max_i | H_i - (n-1)/2 |.
    H_i is the number of home games for player i.
    """
    if n <= 1:
        return 0.0
    
    home_games_counts = calculate_home_games_per_player(schedule_list, n)
    if not home_games_counts and n > 0 : # schedule_list might be empty
         # All players have 0 home games if schedule is empty
        home_games_counts = [0] * n


    ideal_home_games = (n - 1) / 2.0
    max_dev = 0.0
    # If home_games_counts is empty (e.g. n=0), this loop won't run, max_dev remains 0.0
    for count in home_games_counts:
        deviation = abs(count - ideal_home_games)
        if deviation > max_dev:
            max_dev = deviation
    return max_dev

def calculate_raw_total_penalty_sequence(schedule_list, n):
    """
    Calculates the total number of breaks in the schedule.
    A break is defined as a player playing more than two consecutive
    home games or more than two consecutive away games.
    A sequence of k > 2 identical game types (H or A) contributes k-2 breaks.
    """
    if n == 0 or not schedule_list:
        return 0

    player_schedules = [[] for _ in range(n)] 

    for _round_idx, round_matches in enumerate(schedule_list):
        for home_player, away_player in round_matches:
            if 1 <= home_player <= n:
                player_schedules[home_player - 1].append('H')
            if 1 <= away_player <= n:
                player_schedules[away_player - 1].append('A')
            
    total_breaks = 0
    for i in range(n):
        player_seq = player_schedules[i]
        if len(player_seq) < 3:
            continue

        current_streak_type = player_seq[0]
        current_streak_length = 1
        for j in range(1, len(player_seq)):
            if player_seq[j] == current_streak_type:
                current_streak_length += 1
            else:
                if current_streak_length > 2:
                    total_breaks += (current_streak_length - 2)
                current_streak_type = player_seq[j]
                current_streak_length = 1
        
        if current_streak_length > 2: # Check the last streak
            total_breaks += (current_streak_length - 2)
            
    return total_breaks

def get_all_fairness_metrics(schedule_list, n):
    """
    Calculates and returns all fairness metrics, both raw and normalized,
    and home games per player.
    """
    if n == 0: # Special case for n=0
        return {
            "num_players": 0,
            "num_rounds": len(schedule_list) if schedule_list else 0,
            "raw_home_strength": 0.0,
            "normalized_home_strength": 0.0,
            "home_games_per_player": [],
            "raw_total_penalty_sequence": 0,
            "normalized_total_penalty_sequence": 0.0,
            "raw_max_deviation": 0.0,
            "normalized_max_deviation": 0.0
        }
    
    # Handle empty schedule_list for n > 0
    if not schedule_list:
        raw_hs = 0.0
        home_games = [0] * n
        raw_total_pen_seq = 0
        raw_max_dev = calculate_raw_max_deviation(schedule_list, n) # handles empty schedule
    else:
        raw_hs = calculate_home_strength(schedule_list, n)
        home_games = calculate_home_games_per_player(schedule_list, n)
        raw_total_pen_seq = calculate_raw_total_penalty_sequence(schedule_list, n)
        raw_max_dev = calculate_raw_max_deviation(schedule_list, n)

    norm_hs = normalize_home_strength(raw_hs, n)
    norm_total_pen_seq = normalize_total_pen_seq(raw_total_pen_seq, n)
    norm_max_dev = normalize_max_dev(raw_max_dev, n)

    # Calculate player H/A sequences
    player_ha_sequences_chars = [[] for _ in range(n)]
    if schedule_list: # Only if schedule is not empty
        for r_idx in range(len(schedule_list)):
            # Initialize round for all players if not seen yet (for byes or incomplete schedules)
            # This loop structure assumes schedule_list is dense for players involved in matches.
            # For a proper RR schedule, each player plays each round.
            # Let's build based on matches seen.
            temp_round_assignments = {p: '' for p in range(n)} # 0-indexed player

            for home_player, away_player in schedule_list[r_idx]:
                if 1 <= home_player <= n:
                    temp_round_assignments[home_player - 1] = 'H'
                if 1 <= away_player <= n:
                    temp_round_assignments[away_player - 1] = 'A'
            
            for p_idx in range(n): # p_idx is 0-indexed
                # If player p_idx had a match this round, append H/A.
                # If it was a bye (e.g. odd n, not handled here) or error, it might be empty.
                # For SRR with even n, everyone plays.
                if temp_round_assignments[p_idx]:
                     player_ha_sequences_chars[p_idx].append(temp_round_assignments[p_idx])
                # else: # Should not happen for valid SRR with even n
                #    player_ha_sequences_chars[p_idx].append('-') # Placeholder for bye/missing

    player_ha_sequences_str = ["".join(seq) for seq in player_ha_sequences_chars]

    return {
        "num_players": n,
        "num_rounds": len(schedule_list) if schedule_list else 0,
        "raw_home_strength": raw_hs,
        "normalized_home_strength": norm_hs,
        "home_games_per_player": home_games,
        "raw_total_penalty_sequence": raw_total_pen_seq,
        "normalized_total_penalty_sequence": norm_total_pen_seq,
        "raw_max_deviation": raw_max_dev,
        "normalized_max_deviation": norm_max_dev,
        "player_ha_sequences": player_ha_sequences_str # Added
    }

def pprint_fairness_metrics(metrics_dict):
    """Prints the fairness metrics in a readable format."""
    print("--- Fairness Metrics ---")
    if not metrics_dict:
        print("No metrics to display.")
        return

    print(f"  Number of Players (n): {metrics_dict.get('num_players', 'N/A')}")
    print(f"  Number of Rounds: {metrics_dict.get('num_rounds', 'N/A')}")
    
    print("\n  Home Strength:")
    raw_hs_val = metrics_dict.get('raw_home_strength', 'N/A')
    norm_hs_val = metrics_dict.get('normalized_home_strength', 'N/A')
    print(f"    Raw: {raw_hs_val:.4f}" if isinstance(raw_hs_val, float) else f"    Raw: {raw_hs_val}")
    print(f"    Normalized: {norm_hs_val:.4f}" if isinstance(norm_hs_val, float) else f"    Normalized: {norm_hs_val}")

    print("\n  Total Penalty Sequence (Breaks):")
    raw_tps_val = metrics_dict.get('raw_total_penalty_sequence', 'N/A')
    norm_tps_val = metrics_dict.get('normalized_total_penalty_sequence', 'N/A')
    print(f"    Raw: {raw_tps_val}" if isinstance(raw_tps_val, int) else f"    Raw: {raw_tps_val}")
    print(f"    Normalized: {norm_tps_val:.4f}" if isinstance(norm_tps_val, float) else f"    Normalized: {norm_tps_val}")


    print("\n  Max Deviation (from ideal home games):")
    raw_md_val = metrics_dict.get('raw_max_deviation', 'N/A')
    norm_md_val = metrics_dict.get('normalized_max_deviation', 'N/A')
    print(f"    Raw: {raw_md_val:.4f}" if isinstance(raw_md_val, float) else f"    Raw: {raw_md_val}")
    print(f"    Normalized: {norm_md_val:.4f}" if isinstance(norm_md_val, float) else f"    Normalized: {norm_md_val}")


    print("\n  Home Games Per Player (Player ID: Count):")
    home_games = metrics_dict.get('home_games_per_player', [])
    if home_games:
        for i, count in enumerate(home_games):
            print(f"    Player {i+1}: {count}")
    elif metrics_dict.get('num_players', 0) > 0 : # If n > 0 but home_games is empty (e.g. empty schedule)
        for i in range(metrics_dict['num_players']):
             print(f"    Player {i+1}: 0")
    else:
        print("    N/A")

    print("\n  Player H/A Sequences (Player ID: Sequence):")
    player_sequences = metrics_dict.get('player_ha_sequences', [])
    if player_sequences:
        for i, seq_str in enumerate(player_sequences):
            print(f"    Player {i+1}: {seq_str}")
    elif metrics_dict.get('num_players', 0) > 0:
         for i in range(metrics_dict['num_players']):
             print(f"    Player {i+1}: ") # Empty sequence
    else:
        print("    N/A")
    print("------------------------")

if __name__ == '__main__':
    # Test with n=4 example schedule
    n_test_4 = 4
    schedule_n4_example = [
        [(1, 4), (2, 3)],  # R1: P1:H, P2:H, P3:A, P4:A
        [(4, 3), (1, 2)],  # R2: P1:H, P2:A, P3:A, P4:H
        [(2, 4), (3, 1)]   # R3: P1:A, P2:H, P3:H, P4:A
    ]
    # Expected for this n=4 example:
    # HomeStrength: Raw= (4-1)+(3-2) + (3-4)+(2-1) + (4-2)+(1-3) = (3+1) + (-1+1) + (2-2) = 4+0+0 = 4
    # Normalized HS = 4 / (4*3*5/6) = 4 / 10 = 0.4
    # Home Games: P1:[H,H,A]->2, P2:[H,A,H]->2, P3:[A,A,H]->1, P4:[A,H,A]->1. Counts: [2,2,1,1]
    # Max Deviation: Ideal=(4-1)/2=1.5. Devs: |2-1.5|=0.5, |1-1.5|=0.5. Raw MaxDev=0.5
    # Normalized MaxDev = 0.5 / ((4-1)/2) = 0.5 / 1.5 = 1/3 = 0.3333
    # Penalty Sequence:
    # P1: HHA (No breaks)
    # P2: HAH (No breaks)
    # P3: AAH (No breaks)
    # P4: AHA (No breaks)
    # Raw TotalPenSeq = 0. Normalized = 0.

    print(f"--- Testing with n={n_test_4} Example Schedule ---")
    metrics_n4 = get_all_fairness_metrics(schedule_n4_example, n_test_4)
    pprint_fairness_metrics(metrics_n4)
    # Verification:
    # Raw HS: 4.0, Norm HS: 0.4
    # Home Games: P1:2, P2:2, P3:1, P4:1
    # Raw PenSeq: 0, Norm PenSeq: 0.0
    # Raw MaxDev: 0.5, Norm MaxDev: 0.3333

    # Test with a schedule that should have breaks, e.g., n=3
    # P1: HHH, P2: AAA, P3: HAH (for a hypothetical schedule with 3 games each)
    n_test_3_breaks = 3
    # Schedule: P1 plays HHH, P2 plays AAA, P3 plays HAH
    # This requires 3 rounds for P1, P2, P3 to play 3 games each.
    # Example: (not a round robin, just for testing breaks)
    # R1: (1,x) (2,y) (3,z) -> P1:H, P2:A, P3:H
    # R2: (1,x) (2,y) (3,z) -> P1:H, P2:A, P3:A
    # R3: (1,x) (2,y) (3,z) -> P1:H, P2:A, P3:H
    # Player sequences: P1:HHH (1 break), P2:AAA (1 break), P3:HAH (0 breaks) -> Total 2 breaks
    # For simplicity, let's manually create player_schedules for testing breaks calculation
    
    # Test calculate_raw_total_penalty_sequence directly
    print("\n--- Testing Penalty Sequence Calculation Logic ---")
    test_sched_breaks_n3 = [ # Represents player sequences directly for this sub-test
        ['H','H','H'], # Player 1: 1 break
        ['A','A','A'], # Player 2: 1 break
        ['H','A','H']  # Player 3: 0 breaks
    ]
    # We need to feed a schedule_list to get_all_fairness_metrics
    # Let's make a dummy schedule_list that would result in P1:HHH, P2:AAA, P3:HAH
    # This is tricky if we must maintain valid matches.
    # For n=3, a single RR has 2 games per player. A double RR has 4 games.
    # To get 3 games, it's not a standard RR.
    # Let's use a simpler schedule for n=3 that is a valid RR and test all metrics.
    # n=3, Kirkman Triple System (3 rounds, 1 game per player per round)
    # R1: (1,2) (P3 bye) -> P1:H, P2:A
    # R2: (3,1) (P2 bye) -> P1:A, P3:H
    # R3: (2,3) (P1 bye) -> P2:H, P3:A
    # P1: HA, P2: AH, P3: HA. No breaks.
    # HS: (2-1) + (1-3) + (3-2) = 1 - 2 + 1 = 0. Norm HS = 0.
    # Home Games: P1:1, P2:1, P3:1. Ideal=(3-1)/2=1. MaxDev=0. Norm MaxDev=0.
    schedule_n3_rr = [
        [(1,2)], # P3 bye
        [(3,1)], # P2 bye
        [(2,3)]  # P1 bye
    ]
    print(f"\n--- Testing with n=3 Round Robin Schedule ---")
    metrics_n3 = get_all_fairness_metrics(schedule_n3_rr, 3)
    pprint_fairness_metrics(metrics_n3)

    # Test with n=0
    print("\n--- Testing with n=0 (Empty Metrics) ---")
    metrics_n0 = get_all_fairness_metrics([], 0)
    pprint_fairness_metrics(metrics_n0)

    # Test with n=2, empty schedule
    print("\n--- Testing with n=2, Empty Schedule ---")
    metrics_n2_empty = get_all_fairness_metrics([], 2)
    pprint_fairness_metrics(metrics_n2_empty)
    # Expected: All zeros, Home games P1:0, P2:0

    # Test with n=2, one round schedule (1,2)
    print("\n--- Testing with n=2, Schedule [(1,2)] ---")
    metrics_n2_sched = get_all_fairness_metrics([[(1,2)]], 2)
    pprint_fairness_metrics(metrics_n2_sched)
    # Expected for n=2, schedule [[(1,2)]]:
    # HS: raw=(2-1)=1. NormHS = 1 / (2*1*3/6) = 1/1 = 1.0
    # HomeGames: P1:1, P2:0
    # PenSeq: Raw=0 (P1:H, P2:A). Norm=0
    # MaxDev: Ideal=(2-1)/2=0.5. P1:|1-0.5|=0.5, P2:|0-0.5|=0.5. RawMaxDev=0.5
    # NormMaxDev = 0.5 / ((2-1)/2) = 0.5 / 0.5 = 1.0
