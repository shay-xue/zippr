"""High-level arm API that translates abstract commands into low-level joint control."""

from __future__ import annotations

from dataclasses import dataclass, field

from .commands import EndEffectorDeltaCommand
from .controller import SO101ArmController, extract_joint_positions
from .ik import ArmPose, SO101IKTranslator


@dataclass(frozen=True)
class EndEffectorDeltaResult:
    sent_targets: dict[str, float]
    merged_targets: dict[str, float]


@dataclass
class SO101ArmAPI:
    """Convenience interface above the low-level controller."""

    controller: SO101ArmController
    ik_translator: SO101IKTranslator
    # Maximum allowed divergence (degrees) between commanded and sent before
    # _last_targets is pulled toward hardware feedback to prevent permanent drift.
    target_sync_limit: float = 5.0
    # EMA blend factor: 1.0 = no smoothing, lower = smoother/slower.
    smoothing_alpha: float = 0.7
    _last_targets: dict[str, float] = field(
        default_factory=dict, init=False, repr=False
    )

    def connect(self) -> None:
        self.controller.connect()

    def disconnect(self) -> None:
        self.controller.disconnect()

    def get_joint_positions(self) -> dict[str, float]:
        return extract_joint_positions(self.controller.get_observation())

    def get_end_effector_pose(self) -> ArmPose:
        return self.ik_translator.forward(self.controller.get_observation())

    def send_end_effector_delta(
        self, command: EndEffectorDeltaCommand
    ) -> EndEffectorDeltaResult:
        observation = self.controller.get_observation()
        targets = self.ik_translator.translate(
            command, observation, last_targets=self._last_targets or None
        )

        # Apply EMA smoothing: blend toward the new IK target from the last
        # sent position so the arm ramps smoothly rather than snapping.
        if self._last_targets and self.smoothing_alpha < 1.0:
            alpha = self.smoothing_alpha
            targets = {
                motor: alpha * targets[motor]
                + (1.0 - alpha) * self._last_targets.get(motor, targets[motor])
                for motor in targets
            }

        sent_targets = self.controller.send_joint_targets(targets)

        # Sync _last_targets: if hardware feedback diverges from commanded by more
        # than target_sync_limit, clamp last_targets toward sent so the next command
        # starts from a reachable base and the gap cannot accumulate indefinitely.
        synced: dict[str, float] = {}
        for motor, commanded in targets.items():
            sent = sent_targets.get(motor, commanded)
            diff = commanded - sent
            if abs(diff) > self.target_sync_limit:
                synced[motor] = sent + (
                    self.target_sync_limit if diff > 0 else -self.target_sync_limit
                )
            else:
                synced[motor] = commanded
        self._last_targets = synced

        joint_positions = extract_joint_positions(observation)
        joint_positions.update(targets)
        return EndEffectorDeltaResult(
            sent_targets=sent_targets,
            merged_targets=joint_positions,
        )
