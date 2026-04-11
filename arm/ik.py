"""Future IK translation surface for end-effector commands."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .commands import EndEffectorDeltaCommand


@dataclass
class SO101IKTranslator:
    """Placeholder translator from EE deltas to joint targets.

    The future implementation is expected to use a local SO-101 URDF/model and may rely on
    optional dependencies such as `placo`. Phase 1 intentionally leaves this unimplemented.
    """

    urdf_path: str | None = None

    def translate(
        self,
        command: EndEffectorDeltaCommand,
        observation: Mapping[str, Any],
    ) -> dict[str, float]:
        raise NotImplementedError(
            "Inverse kinematics is reserved for a later phase. "
            "Phase 1 only supports direct joint-space teleoperation."
        )
