"""Preprocessing utilities for NetHack observations."""

import numpy as np


def sample_to_one_hot_observation(
    chars: np.ndarray, colors: np.ndarray, cursor: np.ndarray
) -> np.ndarray:
    """
    Converts chars, colors, and cursor arrays into a one-hot encoded boolean
    observation tensor of shape (257, 24, 80).

    257 channels = 128 for chars + 128 for colors + 1 for cursor position.
    - Channels 0-127: one-hot for chars (ASCII 0-127)
    - Channels 128-255: one-hot for colors (0-127)
    - Channel 256: cursor (1 at (y, x), else 0)

    Args:
        chars (np.ndarray): (24, 80) array of uint8, ASCII codes of chars.
        colors (np.ndarray): (24, 80) array of int8, color codes [0-127].
        cursor (np.ndarray): (2,) array, (y, x) for cursor position.

    Returns:
        np.ndarray: One-hot tensor, shape (257, 24, 80), dtype=bool.
    """
    H, W = chars.shape
    out = np.zeros((257, H, W), dtype=bool)

    # Chars: one-hot encode into channels 0-127
    idx_rows, idx_cols = np.indices((H, W))
    char_mask = (chars < 128)
    out[chars[char_mask], idx_rows[char_mask], idx_cols[char_mask]] = True

    # Colors: one-hot encode into channels 128-255
    color_mask = (colors >= 0) & (colors < 128)
    out[128 + colors[color_mask].astype(np.uint8), idx_rows[color_mask],
        idx_cols[color_mask]] = True

    # Cursor: channel 256 is True at cursor location
    cy, cx = int(cursor[0]), int(cursor[1])
    if 0 <= cy < H and 0 <= cx < W:
        out[256, cy, cx] = True

    return out


def one_hot_observation_to_sample(one_hot: np.ndarray):
    """
    Converts a one-hot encoded boolean observation tensor (257, 24, 80) back to
    chars, colors, and cursor arrays.

    Args:
        one_hot: One-hot tensor, shape (257, 24, 80), dtype=bool.

    Returns:
        chars (np.ndarray): (24, 80) array of uint8 ASCII codes.
        colors (np.ndarray): (24, 80) array of int8 color codes [0-127].
        cursor (np.ndarray): (2,) array (y, x) for cursor position.
    """
    assert one_hot.shape[0] == 257, "Expected channel dimension of size 257"
    H, W = one_hot.shape[1:]

    # Chars: channels 0-127
    chars = np.zeros((H, W), dtype=np.uint8)
    char_layer = one_hot[0:128]
    chars[...] = char_layer.argmax(axis=0).astype(np.uint8)
    # If no one-hot, the default will be 0 (ASCII NUL)

    # Optionally mask locations without a char one-hot (all False in 0:128):
    # could set to 32 (' ')
    nochar_mask = (char_layer.sum(axis=0) == 0)
    chars[nochar_mask] = 32  # space

    # Colors: channels 128-255
    color_layer = one_hot[128:256]
    colors = color_layer.argmax(axis=0).astype(np.int8)
    # If no one-hot, default will be 0
    nocolor_mask = (color_layer.sum(axis=0) == 0)
    colors[nocolor_mask] = 0

    # Cursor: channel 256
    cursor_mask = one_hot[256]
    cursor_idx = np.argwhere(cursor_mask)
    if cursor_idx.shape[0] == 0:
        # If no cursor marked, fallback to (0,0)
        cursor = np.array([0, 0], dtype=np.int64)
    else:
        # If more than one cursor marked, pick the first
        cursor = cursor_idx[0]

    return chars, colors, cursor
