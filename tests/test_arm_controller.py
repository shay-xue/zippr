"""Tests for the SO-101 controller adapter."""

from __future__ import annotations

import unittest

from arm.config import ArmSettings
from arm.controller import SO101ArmController


class FakeRobot:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.actions: list[dict[str, float]] = []

    def connect(self, calibrate: bool = True) -> None:
        self.events.append("connect")

    def get_observation(self) -> dict[str, float]:
        self.events.append("get_observation")
        return {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": 2.0,
            "elbow_flex.pos": 3.0,
            "wrist_flex.pos": 4.0,
            "wrist_roll.pos": 5.0,
            "gripper.pos": 6.0,
        }

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.events.append("send_action")
        self.actions.append(action)
        return dict(action)

    def disconnect(self) -> None:
        self.events.append("disconnect")


class SO101ArmControllerTests(unittest.TestCase):
    def test_controller_forwards_calls_to_robot_interface(self) -> None:
        fake_robot = FakeRobot()
        controller = SO101ArmController(
            settings=ArmSettings(port="/dev/mock"),
            robot_factory=lambda _: fake_robot,
        )

        controller.connect()
        observation = controller.get_observation()
        result = controller.send_joint_targets({"shoulder_pan": 10.0, "gripper": 20.0})
        controller.disconnect()

        self.assertEqual(observation["shoulder_pan.pos"], 1.0)
        self.assertEqual(
            fake_robot.actions[0],
            {"shoulder_pan.pos": 10.0, "gripper.pos": 20.0},
        )
        self.assertEqual(result, {"shoulder_pan": 10.0, "gripper": 20.0})
        self.assertEqual(
            fake_robot.events,
            ["connect", "get_observation", "send_action", "disconnect"],
        )

    def test_controller_logs_warning_when_send_action_value_differs(self) -> None:
        class ClampingFakeRobot(FakeRobot):
            def send_action(self, action: dict[str, float]) -> dict[str, float]:
                self.events.append("send_action")
                self.actions.append(action)
                return {"wrist_roll.pos": action["wrist_roll.pos"] - 1.0}

        fake_robot = ClampingFakeRobot()
        controller = SO101ArmController(
            settings=ArmSettings(port="/dev/mock"),
            robot_factory=lambda _: fake_robot,
        )
        controller.connect()
        with self.assertLogs("arm.controller", level="WARNING") as logs:
            result = controller.send_joint_targets({"wrist_roll": 5.0})
        controller.disconnect()

        self.assertEqual(result, {"wrist_roll": 4.0})
        self.assertTrue(
            any("send_action returned values different from requested action" in entry for entry in logs.output)
        )

    def test_controller_dry_run_updates_observation_without_robot(self) -> None:
        controller = SO101ArmController(settings=ArmSettings(dry_run=True))

        controller.connect()
        before = controller.get_observation()
        controller.send_joint_targets({"shoulder_pan": 7.0, "gripper": 11.0})
        after = controller.get_observation()
        controller.disconnect()

        self.assertEqual(before["shoulder_pan.pos"], 0.0)
        self.assertEqual(after["shoulder_pan.pos"], 7.0)
        self.assertEqual(after["gripper.pos"], 11.0)


if __name__ == "__main__":
    unittest.main()
