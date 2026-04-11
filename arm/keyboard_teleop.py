"""Keyboard teleoperation entrypoint for the SO-101 arm."""

from __future__ import annotations

import threading
import time
from collections.abc import Collection
from typing import Any

from .commands import JointDeltaCommand
from .config import ArmSettings, load_settings, validate_settings
from .controller import SO101ArmController, extract_joint_positions

KEY_BINDINGS = {
    "shoulder_pan": ("q", "a"),
    "shoulder_lift": ("w", "s"),
    "elbow_flex": ("e", "d"),
    "wrist_flex": ("r", "f"),
    "wrist_roll": ("t", "g"),
    "gripper": ("y", "h"),
}


def _axis_delta(pressed_keys: Collection[str], positive: str, negative: str, step_size: float) -> float:
    return float((positive in pressed_keys) - (negative in pressed_keys)) * step_size


def build_joint_delta_command(pressed_keys: Collection[str], settings: ArmSettings) -> JointDeltaCommand:
    """Convert active keys into a per-joint delta command."""

    if "space" in pressed_keys:
        return JointDeltaCommand(freeze=True)

    return JointDeltaCommand(
        shoulder_pan=_axis_delta(pressed_keys, "q", "a", settings.step_sizes["shoulder_pan"]),
        shoulder_lift=_axis_delta(pressed_keys, "w", "s", settings.step_sizes["shoulder_lift"]),
        elbow_flex=_axis_delta(pressed_keys, "e", "d", settings.step_sizes["elbow_flex"]),
        wrist_flex=_axis_delta(pressed_keys, "r", "f", settings.step_sizes["wrist_flex"]),
        wrist_roll=_axis_delta(pressed_keys, "t", "g", settings.step_sizes["wrist_roll"]),
        gripper=_axis_delta(pressed_keys, "y", "h", settings.step_sizes["gripper"]),
    )


class JointTargetTracker:
    """Maintains a persistent commanded target over successive keyboard ticks."""

    def __init__(self, settings: ArmSettings) -> None:
        self.settings = settings
        self._targets: dict[str, float] | None = None

    def seed(self, observation: dict[str, Any]) -> dict[str, float]:
        self._targets = extract_joint_positions(observation)
        return dict(self._targets)

    def next_targets(
        self,
        observation: dict[str, Any],
        command: JointDeltaCommand,
    ) -> dict[str, float]:
        if self._targets is None:
            self.seed(observation)

        if command.freeze:
            return dict(self._targets or {})

        current_positions = extract_joint_positions(observation)
        next_targets = dict(self._targets or {})
        for motor, delta in command.as_dict().items():
            desired = float(next_targets[motor] + delta)
            relative_limit = self.settings.max_relative_target[motor]
            current = current_positions[motor]
            desired = min(desired, current + relative_limit)
            desired = max(desired, current - relative_limit)
            if motor == "gripper":
                desired = max(0.0, min(100.0, desired))
            next_targets[motor] = desired

        self._targets = next_targets
        return dict(next_targets)


class KeyboardInput:
    """Low-level keyboard listener backed by pynput."""

    def __init__(self) -> None:
        self._pressed: set[str] = set()
        self._stop_requested = False
        self._listener: Any = None
        self._keyboard_module: Any = None
        self._lock = threading.Lock()

    @property
    def should_exit(self) -> bool:
        with self._lock:
            return self._stop_requested

    def connect(self) -> None:
        keyboard = self._import_keyboard_module()
        self._keyboard_module = keyboard
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def snapshot_pressed_keys(self) -> set[str]:
        with self._lock:
            return set(self._pressed)

    def _on_press(self, key: Any) -> None:
        normalized = self._normalize_key(key)
        if normalized is None:
            return
        with self._lock:
            if normalized == "esc":
                self._stop_requested = True
                return
            self._pressed.add(normalized)

    def _on_release(self, key: Any) -> bool | None:
        normalized = self._normalize_key(key)
        if normalized is None:
            return None
        with self._lock:
            if normalized == "esc":
                self._stop_requested = True
                return False
            self._pressed.discard(normalized)
        return None

    def _normalize_key(self, key: Any) -> str | None:
        keyboard = self._keyboard_module
        if keyboard is None:
            return None
        if hasattr(key, "char") and key.char:
            return str(key.char).lower()
        if key == keyboard.Key.space:
            return "space"
        if key == keyboard.Key.esc:
            return "esc"
        return None

    @staticmethod
    def _import_keyboard_module() -> Any:
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "pynput is required for keyboard teleoperation. Install dependencies from requirements.txt."
            ) from exc
        return keyboard


class KeyboardTeleopRunner:
    """Runs the phase-1 joint-space teleop loop."""

    def __init__(
        self,
        controller: SO101ArmController,
        settings: ArmSettings,
        keyboard_input: KeyboardInput | None = None,
        sleeper: Any = time.sleep,
    ) -> None:
        self.controller = controller
        self.settings = settings
        self.keyboard_input = keyboard_input or KeyboardInput()
        self.sleeper = sleeper
        self.tracker = JointTargetTracker(settings)

    def run(self) -> None:
        validate_settings(self.settings)
        self.controller.connect()
        try:
            initial_observation = self.controller.get_observation()
            self.tracker.seed(initial_observation)
            self.keyboard_input.connect()
            self._print_instructions()

            loop_dt = 1.0 / self.settings.loop_hz
            while not self.keyboard_input.should_exit:
                observation = self.controller.get_observation()
                pressed_keys = self.keyboard_input.snapshot_pressed_keys()
                command = build_joint_delta_command(pressed_keys, self.settings)
                targets = self.tracker.next_targets(observation, command)
                self.controller.send_joint_targets(targets)
                self.sleeper(loop_dt)
        finally:
            self.keyboard_input.disconnect()
            self.controller.disconnect()

    @staticmethod
    def _print_instructions() -> None:
        print("SO-101 keyboard teleop")
        print("q/a shoulder_pan | w/s shoulder_lift | e/d elbow_flex")
        print("r/f wrist_flex   | t/g wrist_roll    | y/h gripper")
        print("space freeze target | esc exit")


def main() -> int:
    settings = load_settings()
    controller = SO101ArmController(settings=settings)
    runner = KeyboardTeleopRunner(controller=controller, settings=settings)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
