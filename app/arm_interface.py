"""
Arm interface: abstract base, mock (grid-based), and real (SO-101) stubs.

MockArm operates in **grid-coordinate space** — positions are integer
column/row indices on the 8×8 chessboard.  One ``execute_action()`` call
moves the piece exactly one square.

Grep for ``# DIMENSION`` to find every hardcoded spatial constant.
"""

from __future__ import annotations

import abc
import logging
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    GRID_COLS,
    GRID_ROWS,
    GRID_STEP,
    RANDOM_SEED,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Z_MAX,
    WORKSPACE_Z_MIN,
)

logger = logging.getLogger(__name__)


@dataclass
class ArmPosition:
    """End-effector position. For MockArm, x/y are grid column/row."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    theta: float = 0.0
    gripper: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "theta": self.theta,
            "gripper": self.gripper,
        }

    def copy(self) -> "ArmPosition":
        return ArmPosition(
            x=self.x, y=self.y, z=self.z, theta=self.theta, gripper=self.gripper
        )


# ── Direction → delta mapping ────────────────────────────────────────────────
# Each action moves exactly GRID_STEP squares.                     # DIMENSION

DIRECTION_DELTAS: dict[str, tuple[float, float]] = {
    "RIGHT": (+GRID_STEP, 0.0),       # col +1                    # DIMENSION
    "LEFT":  (-GRID_STEP, 0.0),       # col -1                    # DIMENSION
    "UP":    (0.0, +GRID_STEP),       # row +1                    # DIMENSION
    "DOWN":  (0.0, -GRID_STEP),       # row -1                    # DIMENSION
}


def displacement_to_direction(dx: float, dy: float) -> Optional[str]:
    """Map a (Δx, Δy) displacement to the dominant cardinal direction."""
    if abs(dx) == 0 and abs(dy) == 0:
        return None
    if abs(dx) >= abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "UP" if dy > 0 else "DOWN"


# ── Abstract base ────────────────────────────────────────────────────────────

class ArmInterface(abc.ABC):
    """Minimal arm interface — only two methods required."""

    @abc.abstractmethod
    def get_position(self) -> ArmPosition:
        ...

    @abc.abstractmethod
    def execute_action(self, action_label: str) -> bool:
        """Move in the given direction. Returns True on success."""
        ...


# ── Mock arm (grid coordinates) ─────────────────────────────────────────────

class MockArm(ArmInterface):
    """Simulated arm that moves on an 8×8 grid.

    Positions are integer grid coordinates.  Gaussian jitter (σ = 0.0)
    is added by default — set ``noise_sigma`` > 0 for realism, but note
    that the piece position is **rounded** to the nearest square after
    each move so the grid stays discrete.
    """

    def __init__(
        self,
        start_col: int = 0,                        # DIMENSION
        start_row: int = 0,                        # DIMENSION
        noise_sigma: float = 0.0,                  # DIMENSION (mm-equivalent)
    ) -> None:
        self._pos = ArmPosition(
            x=float(start_col),
            y=float(start_row),
            z=0.0,
            theta=0.0,
            gripper=0.0,
        )
        self._noise_sigma = noise_sigma
        self._rng = random.Random(RANDOM_SEED)

    def get_position(self) -> ArmPosition:
        return self._pos.copy()

    def execute_action(self, action_label: str) -> bool:
        delta = DIRECTION_DELTAS.get(action_label.upper())
        if delta is None:
            logger.warning("Unknown action: %s", action_label)
            return False

        dx, dy = delta

        # Add optional jitter                                      # DIMENSION
        if self._noise_sigma > 0:
            dx += self._rng.gauss(0, self._noise_sigma)
            dy += self._rng.gauss(0, self._noise_sigma)

        new_x = self._pos.x + dx
        new_y = self._pos.y + dy

        # Clamp to workspace                                       # DIMENSION
        new_x = max(WORKSPACE_X_MIN, min(WORKSPACE_X_MAX, new_x))
        new_y = max(WORKSPACE_Y_MIN, min(WORKSPACE_Y_MAX, new_y))

        # Snap to nearest grid square (integer coords)
        self._pos.x = round(new_x)
        self._pos.y = round(new_y)

        return True


# ── Real arm (SO-101 stub) ───────────────────────────────────────────────────

class RealArm(ArmInterface):
    """SO-101 robotic arm via the FastAPI server.

    Reads the real arm end-effector position from the FastAPI /api/pose
    endpoint and projects it down onto the chess grid. The perpendicular
    projection maps:
        arm y (shoulder pan, left/right) → grid column (A=0 to H=7)
        arm x (radial reach, near/far)   → grid row    (1=0 to 8=7)

    The physical workspace corners are calibrated so the arm's reachable
    area maps onto the 8×8 board.

    Workspace calibration (metres):                              # DIMENSION
        A1 corner: arm y = -0.16  arm x = 0.08
        H8 corner: arm y = +0.16  arm x = 0.40
    """

    # Physical workspace corners (metres)                          # DIMENSION
    ARM_Y_MIN: float = -0.16   # column A (left)
    ARM_Y_MAX: float =  0.16   # column H (right)
    ARM_X_MIN: float =  0.08   # row 1 (close to base)
    ARM_X_MAX: float =  0.40   # row 8 (far from base)

    def __init__(
        self,
        arm_url: str = "http://127.0.0.1:8000",
        start_col: int = 0,
        start_row: int = 0,
    ) -> None:
        import requests
        self._session = requests.Session()
        self.arm_url = arm_url
        # Fallback grid position in case API read fails
        self._fallback_pos = ArmPosition(
            x=float(start_col), y=float(start_row),
        )

    def _arm_pose_to_grid(self, arm_x: float, arm_y: float) -> tuple[float, float]:
        """Map physical arm (x, y) to continuous grid coordinates.

        arm y → column:  left (-0.16) = A(0), right (+0.16) = H(7)
        arm x → row:     close (0.08) = 1(0), far (0.40) = 8(7)

        Returns continuous (col, row) floats clamped to [0, 7].
        Callers that need discrete squares should round() the result.
        """
        # Normalise to [0, 1] then scale to [0, 7]
        col_f = (arm_y - self.ARM_Y_MIN) / (self.ARM_Y_MAX - self.ARM_Y_MIN) * 7.0
        row_f = (arm_x - self.ARM_X_MIN) / (self.ARM_X_MAX - self.ARM_X_MIN) * 7.0
        # Clamp to grid bounds
        col_f = max(0.0, min(7.0, col_f))
        row_f = max(0.0, min(7.0, row_f))
        return col_f, row_f

    def get_position(self) -> ArmPosition:
        """Read real arm pose and map to grid coordinates.

        Returns continuous grid coordinates for smooth position tracking.
        The gripper field is populated from the API if available.
        """
        try:
            resp = self._session.get(f"{self.arm_url}/api/pose", timeout=1)
            resp.raise_for_status()
            pose = resp.json()
            col_f, row_f = self._arm_pose_to_grid(pose["x"], pose["y"])
            gripper = float(pose.get("gripper", 0.0))
            self._fallback_pos.x = col_f
            self._fallback_pos.y = row_f
            self._fallback_pos.gripper = gripper
            return ArmPosition(x=col_f, y=row_f, z=0.0, gripper=gripper)
        except Exception as exc:
            logger.debug("Arm pose read failed: %s", exc)
            return self._fallback_pos.copy()

    def execute_action(self, action_label: str) -> bool:
        """The arm moves via the bridge script (brain_to_arm.py), not here.

        This method is called by the chess game when a decoder direction is
        registered.  We don't send a separate command — the bridge is already
        forwarding decoder outputs to the arm continuously.  We just re-read
        the arm position (which the bridge has already moved) and let
        get_position() map it to the grid.
        """
        # No-op for physical commands — the bridge handles that.
        # Just return True so the game loop continues.
        return True
