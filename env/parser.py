"""Reads the model's stated probability from its answer.

The prompt asks for a final line "Probability: 0.XX". The parser takes the LAST line that starts a
"Probability:" statement and the LAST number on it, so "Probability: 8/17 = 0.47" gives 0.47 (see
writeup/hacks.md, entry 1). Percentages are rejected because the prompt asks for a decimal, and
anything outside [0, 1] is rejected. A rejected answer gets the format penalty in training.
"""
from __future__ import annotations

import re

PROB_LINE = re.compile(r"Probability:([^\n]*)", re.IGNORECASE)
NUMBER = re.compile(r"\d*\.?\d+")


def extract_probability(text: str | None) -> float | None:
    """Last stated probability in [0, 1], or None if missing, a percentage, or out of range."""
    if not text:
        return None
    lines = PROB_LINE.findall(text)
    if not lines:
        return None
    line = lines[-1]
    if "%" in line:
        return None
    numbers = NUMBER.findall(line)
    if not numbers:
        return None
    p = float(numbers[-1])
    return p if 0.0 <= p <= 1.0 else None
