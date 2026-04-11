"""Configuration for SO-101 arm control."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .commands import MOTOR_NAMES

DEFAULT_ROBOT_ID = "so101"


def _default_step_sizes() -> dict[str, float]:
    return {
        "shoulder_pan": 2.0,
        "shoulder_lift": 2.0,
        "elbow_flex": 2.0,
        "wrist_flex": 2.0,
        "wrist_roll": 2.0,
        "gripper": 4.0,
    }


def _default_max_relative_target() -> dict[str, float]:
    return {
        "shoulder_pan": 5.0,
        "shoulder_lift": 5.0,
        "elbow_flex": 5.0,
        "wrist_flex": 5.0,
        "wrist_roll": 5.0,
        "gripper": 8.0,
    }


@dataclass(frozen=True)
class ArmSettings:
    """Runtime settings for the SO-101 control package."""

    port: str = ""
    robot_id: str = DEFAULT_ROBOT_ID
    use_degrees: bool = True
    loop_hz: float = 20.0
    step_sizes: dict[str, float] = field(default_factory=_default_step_sizes)
    max_relative_target: dict[str, float] = field(default_factory=_default_max_relative_target)
    dry_run: bool = False

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ArmSettings":
        """Return a new settings object with partial overrides applied."""

        merged = asdict(self)
        for key in ("step_sizes", "max_relative_target"):
            if key in overrides:
                nested = dict(merged[key])
                nested.update(dict(overrides[key]))
                merged[key] = nested

        for key, value in overrides.items():
            if key not in {"step_sizes", "max_relative_target"}:
                merged[key] = value

        return ArmSettings(**merged)


DEFAULT_SETTINGS = ArmSettings()


def _load_settings_from_module(module: Any) -> ArmSettings | None:
    direct_settings = getattr(module, "SETTINGS", None)
    if isinstance(direct_settings, ArmSettings):
        return direct_settings

    if isinstance(getattr(module, "settings", None), ArmSettings):
        return module.settings

    overrides = getattr(module, "SETTINGS_OVERRIDES", None)
    if overrides is None:
        overrides = getattr(module, "settings_overrides", None)

    if overrides is not None:
        return DEFAULT_SETTINGS.with_overrides(overrides)

    get_settings = getattr(module, "get_settings", None)
    if callable(get_settings):
        settings = get_settings()
        if isinstance(settings, ArmSettings):
            return settings
        if isinstance(settings, Mapping):
            return DEFAULT_SETTINGS.with_overrides(settings)

    return None


def load_settings(module_name: str = "arm.local_config") -> ArmSettings:
    """Load checked-in defaults and optionally overlay `arm.local_config`."""

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {module_name, module_name.rsplit(".", 1)[-1]}:
            return DEFAULT_SETTINGS
        raise

    settings = _load_settings_from_module(module)
    if settings is None:
        return DEFAULT_SETTINGS
    return settings


def validate_settings(settings: ArmSettings) -> None:
    """Validate settings before attempting to run a control loop."""

    missing_keys = set(MOTOR_NAMES) - set(settings.step_sizes)
    if missing_keys:
        raise ValueError(f"Missing step sizes for joints: {sorted(missing_keys)}")

    missing_limits = set(MOTOR_NAMES) - set(settings.max_relative_target)
    if missing_limits:
        raise ValueError(f"Missing max_relative_target entries for joints: {sorted(missing_limits)}")
