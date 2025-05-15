import random

def initial_schedule(n):
    """
    Generates an initial round-robin schedule using the circle method.

    Assigns home/away randomly. Handles odd 'n' by adding a dummy player (None)
    for bye rounds, although the main solver logic assumes even 'n'.

    Args:
        n (int): Number of players.

    Returns:
        list: A list of rounds, where each round is a list of
              (home_player, away_player) tuples (using 1-based indexing).
    """
    players = list(range(1, n + 1)) # Use 1-based indexing
    original_n = n
    if n % 2:
        # If n is odd, add a dummy player for the circle method logic
        players.append(None)
        n_effective = n + 1 # Use n_effective for circle method calculation
    else:
        n_effective = n

    half = n_effective // 2
    schedule = []
    for r in range(n_effective - 1): # Iterate through rounds needed for n_effective players
        round_pairs = []
        for i in range(half):
            p1 = players[i]
            p2 = players[n_effective - 1 - i]
            # Only add the match if both players are not the dummy player
            if p1 is not None and p2 is not None:
                # Assign home randomly
                if random.random() < 0.5:
                    round_pairs.append((p1, p2))
                else:
                    round_pairs.append((p2, p1))
        if round_pairs: # Only add non-empty rounds (relevant if n was odd)
             schedule.append(round_pairs)
        # Rotate players for the next round (excluding the fixed player 1)
        # The circle method typically fixes one player and rotates the rest.
        # Let's fix the first player in the current list.
        if n_effective > 1:
            fixed_player = players[0]
            # Rotate the rest of the players
            rotated_part = [players[n_effective - 1]] + players[1:n_effective - 1]
            players = [fixed_player] + rotated_part
        # If n_effective is 1 (original n=0), players list is [None], no rotation needed.

    # Ensure the schedule has the correct number of rounds for the original n
    return schedule[:original_n - 1]
