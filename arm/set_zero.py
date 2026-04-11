"""Move the SO-101 arm to the zero (0°) position on all joints."""

from __future__ import annotations

import argparse
import time

from .commands import MOTOR_NAMES
from .config import ArmSettings, load_settings
from .controller import SO101ArmController, extract_joint_positions

_ARRIVAL_THRESHOLD_DEG = 0.2
_JOINTS_TO_ZERO = [j for j in MOTOR_NAMES if j != "gripper"]


def _step_toward_zero(
    current: dict[str, float],
    max_relative: dict[str, float],
    include_gripper: bool,
) -> dict[str, float]:
    """Return next targets one tick closer to zero, respecting max_relative_target.

    All motors are always included so LeRobot's max_relative_target key-match
    check passes. Motors not being zeroed are held at their current position.
    """
    targets = {}
    for motor in MOTOR_NAMES:
        pos = current[motor]
        if motor == "gripper" and not include_gripper:
            targets[motor] = pos  # hold in place
            continue
        limit = max_relative[motor]
        step = max(-limit, min(limit, -pos))
        targets[motor] = pos + step
    return targets


def _all_at_zero(positions: dict[str, float], include_gripper: bool) -> bool:
    motors = list(MOTOR_NAMES) if include_gripper else _JOINTS_TO_ZERO
    return all(abs(positions[m]) < _ARRIVAL_THRESHOLD_DEG for m in motors)


def run(settings: ArmSettings, include_gripper: bool = False) -> None:
    controller = SO101ArmController(settings=settings)
    controller.connect()
    try:
        observation = controller.get_observation()
        current = extract_joint_positions(observation)

        print("\nCurrent joint positions:")
        for motor in MOTOR_NAMES:
            marker = ""
            if motor == "gripper" and not include_gripper:
                marker = "  (unchanged)"
            print(f"  {motor:>15}: {current[motor]:>8.2f}°{marker}")

        motors_moving = list(MOTOR_NAMES) if include_gripper else _JOINTS_TO_ZERO
        print(f"\nMoving to 0° on: {', '.join(motors_moving)}")

        try:
            input("\nPress Enter to proceed, Ctrl-C to abort... ")
        except KeyboardInterrupt:
            print("\nAborted.")
            return

        loop_dt = 1.0 / settings.loop_hz
        while True:
            observation = controller.get_observation()
            current = extract_joint_positions(observation)
            if _all_at_zero(current, include_gripper):
                break
            targets = _step_toward_zero(current, settings.max_relative_target, include_gripper)
            controller.send_joint_targets(targets)
            time.sleep(loop_dt)

        observation = controller.get_observation()
        final = extract_joint_positions(observation)
        print("\nFinal joint positions:")
        for motor in MOTOR_NAMES:
            print(f"  {motor:>15}: {final[motor]:>8.2f}°")
        print("\nDone.")

    finally:
        controller.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Move the SO-101 arm to the zero position.")
    parser.add_argument("--port", help="Serial port override (e.g. /dev/ttyUSB0)")
    parser.add_argument(
        "--zero-gripper",
        action="store_true",
        help="Also zero the gripper (default: leave gripper unchanged)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate movement without connecting to hardware",
    )
    args = parser.parse_args()

    settings = load_settings()
    if args.port:
        settings = settings.with_overrides({"port": args.port})
    if args.dry_run:
        settings = settings.with_overrides({"dry_run": True})

    run(settings, include_gripper=args.zero_gripper)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
