"""
BCI bit rate calculator.

Formula (adapted for 64-square solution space):
    B = log2(N) × max(Sc − Si, 0) / t

where N = 64 (8×8 grid), Sc = correct selections, Si = incorrect, t = elapsed seconds.
log2(64) = 6.0 bits per correct selection.

Updated once per second.
"""

from __future__ import annotations

import math
import time

from config import N_SQUARES


# log2(64) = 6.0
LOG2_N: float = math.log2(N_SQUARES)


class BitRateCalculator:
    """Tracks Sc, Si, and computes live bit rate."""

    def __init__(self) -> None:
        self.sc: int = 0
        self.si: int = 0
        self._start_time: float = 0.0
        self._running: bool = False

    def start(self) -> None:
        """Begin the session timer."""
        self._start_time = time.time()
        self.sc = 0
        self.si = 0
        self._running = True

    def reset(self) -> None:
        self.sc = 0
        self.si = 0
        self._start_time = 0.0
        self._running = False

    def record_correct(self) -> None:
        self.sc += 1

    def record_incorrect(self) -> None:
        self.si += 1

    @property
    def elapsed(self) -> float:
        """Seconds since start (0.0 if not running)."""
        if not self._running:
            return 0.0
        return time.time() - self._start_time

    def get_current_bps(self) -> float:
        """Compute current bit rate in bits per second.

        B = log2(64) × max(Sc − Si, 0) / t
        """
        t = self.elapsed
        if t <= 0:
            return 0.0
        return LOG2_N * max(self.sc - self.si, 0) / t

    @property
    def accuracy(self) -> float:
        """Accuracy as a fraction [0, 1]. Returns 0 if no decisions yet."""
        total = self.sc + self.si
        if total == 0:
            return 0.0
        return self.sc / total

    @property
    def total_decisions(self) -> int:
        return self.sc + self.si
