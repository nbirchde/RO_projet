import numpy as np

def calculate_home_strength(schedule_list, n):
    """
    Calculates the raw HomeStrength metric based on player ranks.
    HS = sum max(0, rank_away_player - rank_home_player) for all matches.
    This only counts matches where the home player is stronger (lower rank)
    than the away player.
    Assumes player IDs are their ranks (1 to n).

    Args:
        schedule_list (list): A list of rounds, where each round is a list of
                              (home_player_id, away_player_id) tuples.
        n (int): Number of players.

    Returns:
        float: The raw HomeStrength value (always non-negative).
    """
    if not schedule_list or n == 0:
        return 0.0

    raw_hs = 0.0
    for round_matches in schedule_list:
        for home_player, away_player in round_matches:
            # Ignore matches involving the dummy player (None)
            if home_player is not None and away_player is not None:
                # Assuming player IDs are 1-based ranks
                rank_diff = away_player - home_player
                raw_hs += max(0, rank_diff)
    return raw_hs

def calculate_raw_max_deviation(schedule_list, n):
    """
    Calculates the raw Max Deviation metric: MaxDev = max_i | H_i - (n-1)/2 |.
    H_i is the number of home games for player i.
    """
    if n <= 1:
        return 0.0

    home_games_counts = calculate_home_games_per_player(schedule_list, n)
    if not home_games_counts and n > 0 :
        home_games_counts = [0] * n

    ideal_home_games = (n - 1) / 2.0
    max_dev = 0.0
    for count in home_games_counts:
        deviation = abs(count - ideal_home_games)
        if deviation > max_dev:
            max_dev = deviation
    return max_dev

def calculate_raw_total_penalty_sequence(schedule_list, n):
    """
    Calculates the total number of penalties (breaks) in the schedule
    using the LUT definition (penalty for AA and HH).
    This aligns with the definition used within the SA solver.
    """
    if n == 0 or not schedule_list:
        return 0

    rounds = len(schedule_list)
    if rounds <= 1:
        return 0

    matches_per_round = n // 2
    packed_rows = 0
    if n > 0: packed_rows = (n - 1) // 8 + 1
    packed_seq = np.zeros((packed_rows, rounds), dtype=np.uint8)

    for r_idx, rnd_matches in enumerate(schedule_list):
        for home_player, away_player in rnd_matches:
            if 1 <= home_player <= n:
                 p_zero_based_h = home_player - 1
                 byte_row_idx_h = p_zero_based_h // 8
                 bit_offset_h = p_zero_based_h % 8
                 packed_seq[byte_row_idx_h, r_idx] |= (np.uint8(1) << bit_offset_h)

            if 1 <= away_player <= n:
                 p_zero_based_a = away_player - 1
                 byte_row_idx_a = p_zero_based_a // 8
                 bit_offset_a = p_zero_based_a % 8
                 packed_seq[byte_row_idx_a, r_idx] &= (~(np.uint8(1) << bit_offset_a))

    total_penalties = np.int64(0)
    PENALTY_LUT_LOCAL = np.array([1, 0, 0, 1], dtype=np.int8)

    for player_idx in range(1, n + 1):
        p_zero_based = player_idx - 1
        byte_row_idx = p_zero_based // 8
        bit_offset = p_zero_based % 8

        for r_loop in range(rounds - 1):
            byte_val_curr = packed_seq[byte_row_idx, r_loop]
            status_curr = (byte_val_curr >> bit_offset) & np.uint8(1)

            byte_val_next = packed_seq[byte_row_idx, r_loop + 1]
            status_next = (byte_val_next >> bit_offset) & np.uint8(1)

            lut_idx = (status_curr << 1) | status_next
            total_penalties += np.int64(PENALTY_LUT_LOCAL[lut_idx])

    return total_penalties

def calculate_max_home_strength_denominator(n):
    """
    Calculates the theoretical maximum possible value for the new HomeStrength
    metric (sum max(0, j-i) * x_ijr), used for normalization.
    This maximum occurs when every stronger player 'i' plays at home against
    every weaker player 'j'.
    S_max = sum_{i=1}^{n-1} sum_{j=i+1}^{n} (j - i) = n(n-1)(n+1)/6
    """
    if n < 2:
        return 1.0

    denominator = n * (n - 1) * (n + 1) / 6.0

    return denominator

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
            if 1 <= home_player <= n:
                home_games_counts[home_player - 1] += 1
    return home_games_counts

def calculate_ha_sequences(schedule_list, n):
    player_ha_sequences_chars = [[] for _ in range(n)]
    if schedule_list:
        for r_idx in range(len(schedule_list)):
            temp_round_assignments = {p: '' for p in range(n)}

            for home_player, away_player in schedule_list[r_idx]:
                if 1 <= home_player <= n:
                    temp_round_assignments[home_player - 1] = 'H'
                if 1 <= away_player <= n:
                    temp_round_assignments[away_player - 1] = 'A'

            for p_idx in range(n):
                if temp_round_assignments[p_idx]:
                     player_ha_sequences_chars[p_idx].append(temp_round_assignments[p_idx])

    player_ha_sequences_str = ["".join(seq) for seq in player_ha_sequences_chars]
    return player_ha_sequences_str

def get_all_fairness_metrics(schedule_list, n):
    raw_hs = calculate_home_strength(schedule_list, n)
    home_games = calculate_home_games_per_player(schedule_list, n)
    raw_total_pen_seq = calculate_raw_total_penalty_sequence(schedule_list, n)
    raw_max_dev = calculate_raw_max_deviation(schedule_list, n)

    player_ha_sequences_str = calculate_ha_sequences(schedule_list, n)

    return {
        "num_players": n,
        "num_rounds": len(schedule_list) if schedule_list else 0,
        "raw_home_strength": raw_hs,
        "home_games_per_player": home_games,
        "raw_total_penalty_sequence": raw_total_pen_seq,
        "raw_max_deviation": raw_max_dev,
        "player_ha_sequences": player_ha_sequences_str
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
    print(f"    Raw: {raw_hs_val:.4f}" if isinstance(raw_hs_val, float) else f"    Raw: {raw_hs_val}")

    print("\n  Total Penalty Sequence (Breaks):")
    raw_tps_val = metrics_dict.get('raw_total_penalty_sequence', 'N/A')
    print(f"    Raw: {raw_tps_val}" if isinstance(raw_tps_val, int) else f"    Raw: {raw_tps_val}")

    print("\n  Max Deviation (from ideal home games):")
    raw_md_val = metrics_dict.get('raw_max_deviation', 'N/A')
    print(f"    Raw: {raw_md_val:.4f}" if isinstance(raw_md_val, float) else f"    Raw: {raw_md_val}")

    print("\n  Home Games Per Player (Player ID: Count):")
    home_games = metrics_dict.get('home_games_per_player', [])
    if home_games:
        for i, count in enumerate(home_games):
            print(f"    Player {i+1}: {count}")
    elif metrics_dict.get('num_players', 0) > 0 :
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
             print(f"    Player {i+1}: ")
    else:
        print("    N/A")
    print("------------------------")

if __name__ == '__main__':
    # Test with n=4 example schedule
    n_test_4 = 4
    schedule_n4_example = [
        [(1, 4), (2, 3)],
        [(4, 3), (1, 2)],
        [(2, 4), (3, 1)]
    ]
    print(f"--- Testing with n={n_test_4} Example Schedule ---")
    metrics_n4 = get_all_fairness_metrics(schedule_n4_example, n_test_4)
    pprint_fairness_metrics(metrics_n4)

    # Test with n=3 Round Robin Schedule
    schedule_n3_rr = [
        [(1,2)],
        [(3,1)],
        [(2,3)]
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

    # Test with n=2, one round schedule (1,2)
    print("\n--- Testing with n=2, Schedule [(1,2)] ---")
    metrics_n2_sched = get_all_fairness_metrics([[(1,2)]], 2)
    pprint_fairness_metrics(metrics_n2_sched)
