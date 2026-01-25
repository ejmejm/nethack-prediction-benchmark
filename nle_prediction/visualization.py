"""Visualization utilities for NetHack observations."""

from typing import Optional

import numpy as np


def print_ascii_array(
    ascii_array: np.ndarray,
    colors: Optional[np.ndarray] = None
) -> None:
    """Prints a 2D array of ASCII values with optional ANSI colors.

    Args:
        ascii_array: A 2D numpy array of uint8 ASCII values.
        colors: Optional 2D numpy array of int8 color codes (0-255 for 256-color
            mode, or 0-7 for basic ANSI colors). Must match ascii_array shape.
    """
    if ascii_array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {ascii_array.ndim}D")

    if colors is not None:
        if colors.shape != ascii_array.shape:
            raise ValueError(
                f"Color array shape {colors.shape} doesn't match "
                f"ASCII array shape {ascii_array.shape}"
            )

        for row_idx in range(ascii_array.shape[0]):
            line_parts = []
            for col_idx in range(ascii_array.shape[1]):
                char = chr(ascii_array[row_idx, col_idx])
                color = colors[row_idx, col_idx]
                # Use 256-color ANSI escape: \033[38;5;{color}m
                line_parts.append(f"\033[38;5;{color}m{char}\033[0m")
            print("".join(line_parts))
    else:
        for row in ascii_array:
            print("".join(chr(val) for val in row))
