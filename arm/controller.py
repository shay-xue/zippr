"""SO-101 controller wrapper over LeRobot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .commands import MOTOR_NAMES
from .config import ArmSettings


class RobotInterface(Protocol):
    def connect(self, calibrate: bool = True) -> None: ...

    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: dict[str, float]) -> dict[str, float]: ...

    def disconnect(self) -> None: ...


def extract_joint_positions(observation: Mapping[str, Any]) -> dict[str, float]:
    """Extract the normalized joint positions for the SO-101 motors."""

    positions: dict[str, float] = {}
    for motor in MOTOR_NAMES:
        key = f"{motor}.pos"
        if key not in observation:
            raise KeyError(f"Observation is missing required key: {key}")
        positions[motor] = float(observation[key])
    return positions


def _make_default_dry_run_observation() -> dict[str, float]:
    return {f"{motor}.pos": 0.0 for motor in MOTOR_NAMES}


def _build_default_robot(settings: ArmSettings) -> RobotInterface:
    if not settings.port:
        raise ValueError(
            "SO-101 serial port is not configured. Set `port` in arm/local_config.py or enable dry_run."
        )

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot_config = SO101FollowerConfig(
        port=settings.port,
        id=settings.robot_id,
        use_degrees=settings.use_degrees,
        max_relative_target=settings.max_relative_target,
    )
    return SO101Follower(robot_config)


class SO101ArmController:
    """Thin hardware adapter over LeRobot's SO101Follower."""

    def __init__(
        self,
        settings: ArmSettings,
        robot_factory: Callable[[ArmSettings], RobotInterface] | None = None,
    ) -> None:
        self.settings = settings
        self._robot_factory = robot_factory or _build_default_robot
        self._robot: RobotInterface | None = None
        self._connected = False
        self._dry_run_observation = _make_default_dry_run_observation()

    def connect(self) -> None:
        if self._connected:
            return

        if self.settings.dry_run:
            self._connected = True
            return

        self._robot = self._robot_factory(self.settings)
        self._robot.connect()
        self._connected = True

    def get_observation(self) -> dict[str, Any]:
        self._require_connected()
        if self.settings.dry_run:
            return dict(self._dry_run_observation)
        if self._robot is None:
            raise RuntimeError("Robot has not been initialized.")
        return self._robot.get_observation()

    def send_joint_targets(self, targets: Mapping[str, float]) -> dict[str, float]:
        self._require_connected()
        unknown = set(targets) - set(MOTOR_NAMES)
        if unknown:
            raise ValueError(f"Unknown joint targets: {sorted(unknown)}")

        action = {f"{motor}.pos": float(value) for motor, value in targets.items()}

        if self.settings.dry_run:
            self._dry_run_observation.update(action)
            return {motor: float(value) for motor, value in targets.items()}

        if self._robot is None:
            raise RuntimeError("Robot has not been initialized.")

        sent_action = self._robot.send_action(action)
        return {
            key.removesuffix(".pos"): float(value)
            for key, value in sent_action.items()
            if key.endswith(".pos")
        }

    def disconnect(self) -> None:
        if not self._connected:
            return

        try:
            if not self.settings.dry_run and self._robot is not None:
                self._robot.disconnect()
        finally:
            self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Controller is not connected.")
