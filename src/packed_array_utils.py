#!/usr/bin/env python3
import numpy as np
import numba

PLAYERS_PER_BYTE = 8 # For bit-packing player statuses

@numba.njit(cache=True)
def get_status_packed(packed_seq_arr, player_idx, round_idx):
    """Gets a player's H/A status from a bit-packed array."""
    # player_idx is 1-based. Convert to 0-based for bit manipulation.
    p_zero_based = player_idx - 1
    byte_row_idx = p_zero_based // PLAYERS_PER_BYTE
    bit_offset = p_zero_based % PLAYERS_PER_BYTE
    
    byte_val = packed_seq_arr[byte_row_idx, round_idx]
    return (byte_val >> bit_offset) & np.uint8(1)

@numba.njit(cache=True)
def set_status_packed(packed_seq_arr, player_idx, round_idx, new_status):
    """Sets a player's H/A status in a bit-packed array."""
    # player_idx is 1-based.
    p_zero_based = player_idx - 1
    byte_row_idx = p_zero_based // PLAYERS_PER_BYTE
    bit_offset = p_zero_based % PLAYERS_PER_BYTE
    
    current_byte_val = packed_seq_arr[byte_row_idx, round_idx]
    if new_status == 1: # Set bit to 1 (Home)
        packed_seq_arr[byte_row_idx, round_idx] = current_byte_val | (np.uint8(1) << bit_offset)
    else: # Set bit to 0 (Away)
        packed_seq_arr[byte_row_idx, round_idx] = current_byte_val & (~(np.uint8(1) << bit_offset))
