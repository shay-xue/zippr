"""Command models for SO-101 control."""

from __future__ import annotations

from dataclasses import dataclass

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class EndEffectorDeltaCommand:
    """Canonical future decoder-facing command model.

    All values are relative deltas in the robot base frame:
    - `dx`, `dy`, `dz`: end-effector position deltas
    - `d_rot`: tool roll delta about the tool axis
    - `d_jaw`: gripper jaw delta
    """

    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    d_rot: float = 0.0
    d_jaw: float = 0.0


@dataclass(frozen=True)
class JointDeltaCommand:
    """Per-joint delta command used for phase-1 keyboard teleop."""

    shoulder_pan: float = 0.0
    shoulder_lift: float = 0.0
    elbow_flex: float = 0.0
    wrist_flex: float = 0.0
    wrist_roll: float = 0.0
    gripper: float = 0.0
    freeze: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            "shoulder_pan": self.shoulder_pan,
            "shoulder_lift": self.shoulder_lift,
            "elbow_flex": self.elbow_flex,
            "wrist_flex": self.wrist_flex,
            "wrist_roll": self.wrist_roll,
            "gripper": self.gripper,
        }
