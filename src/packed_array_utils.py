#!/usr/bin/env python3
import numpy as np
import numba

PLAYERS_PER_BYTE = 8

@numba.njit(cache=True)
def get_status_packed(packed_seq_arr, player_idx, round_idx):
    """Gets a player's H/A status from a bit-packed array."""
    p_zero_based = player_idx - 1
    byte_row_idx = p_zero_based // PLAYERS_PER_BYTE
    bit_offset = p_zero_based % PLAYERS_PER_BYTE
    
    byte_val = packed_seq_arr[byte_row_idx, round_idx]
    return (byte_val >> bit_offset) & np.uint8(1)

@numba.njit(cache=True)
def set_status_packed(packed_seq_arr, player_idx, round_idx, new_status):
    """Sets a player's H/A status in a bit-packed array."""
    p_zero_based = player_idx - 1
    byte_row_idx = p_zero_based // PLAYERS_PER_BYTE
    bit_offset = p_zero_based % PLAYERS_PER_BYTE
    
    current_byte_val = packed_seq_arr[byte_row_idx, round_idx]
    if new_status == 1:
        packed_seq_arr[byte_row_idx, round_idx] = current_byte_val | (np.uint8(1) << bit_offset)
    else:
        packed_seq_arr[byte_row_idx, round_idx] = current_byte_val & (~(np.uint8(1) << bit_offset))
