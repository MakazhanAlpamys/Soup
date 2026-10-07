#!/usr/bin/env python3
"""
Compute the shape implied by "an 8B base fits a 4 GB card".
"""

def parse_size(size_str: str) -> int:
    """Convert a size like '8B' or '4 GB' to bytes."""
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    parts = size_str.strip().split()
    if len(parts) == 1:  # e.g., "8B"
        num, unit = float(parts[0][:-1]), parts[0][-1]
    else:                # e.g., "4 GB"
        num, unit = float(parts[0]), parts[1].upper()
    return int(num * units[unit])

def compute_shape(base_str: str, card_str: str) -> tuple:
    """Return the shape (number of base units) that fit the card."""
    base_bytes = parse_size(base_str)
    card_bytes = parse_size(card_str)
    count = card_bytes // base_bytes
    return (count,)

if __name__ == "__main__":
    base = "8B"
    card = "4 GB"
    shape = compute_shape(base, card)
    print(f"The shape for an {base} base on a {card} card is {shape}.")