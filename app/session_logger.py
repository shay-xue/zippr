"""
Session event logger and CSV exporter.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SessionEvent:
    """One scored event during a session."""

    timestamp: float
    target_square: str
    target_direction: Optional[str]
    decoded_action: str
    confidence: float
    arm_x_initial: float
    arm_y_initial: float
    arm_z_initial: float
    arm_x_final: float
    arm_y_final: float
    arm_z_final: float
    delta_x: float
    delta_y: float
    delta_z: float
    correct: bool


CSV_COLUMNS = [
    "timestamp",
    "target_square",
    "target_direction",
    "decoded_action",
    "confidence",
    "arm_x_initial",
    "arm_y_initial",
    "arm_z_initial",
    "arm_x_final",
    "arm_y_final",
    "arm_z_final",
    "delta_x",
    "delta_y",
    "delta_z",
    "correct",
]


class SessionLogger:
    """Accumulates events and exports to CSV."""

    def __init__(self) -> None:
        self.events: List[SessionEvent] = []

    def log(self, event: SessionEvent) -> None:
        self.events.append(event)

    def reset(self) -> None:
        self.events.clear()

    def to_csv_string(self) -> str:
        """Return the full event log as a CSV string."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(CSV_COLUMNS)
        for e in self.events:
            writer.writerow([
                e.timestamp,
                e.target_square,
                e.target_direction,
                e.decoded_action,
                e.confidence,
                e.arm_x_initial,
                e.arm_y_initial,
                e.arm_z_initial,
                e.arm_x_final,
                e.arm_y_final,
                e.arm_z_final,
                e.delta_x,
                e.delta_y,
                e.delta_z,
                e.correct,
            ])
        return buf.getvalue()
