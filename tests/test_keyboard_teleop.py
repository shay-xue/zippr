"""Tests for keyboard teleop mapping and target tracking."""

from __future__ import annotations

import unittest

from arm.commands import JointDeltaCommand
from arm.config import ArmSettings
from arm.keyboard_teleop import JointTargetTracker, build_joint_delta_command


def _observation(
    shoulder_pan: float = 10.0,
    shoulder_lift: float = 20.0,
    elbow_flex: float = 30.0,
    wrist_flex: float = 40.0,
    wrist_roll: float = 50.0,
    gripper: float = 60.0,
) -> dict[str, float]:
    return {
        "shoulder_pan.pos": shoulder_pan,
        "shoulder_lift.pos": shoulder_lift,
        "elbow_flex.pos": elbow_flex,
        "wrist_flex.pos": wrist_flex,
        "wrist_roll.pos": wrist_roll,
        "gripper.pos": gripper,
    }


class KeyboardTeleopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = ArmSettings()

    def test_build_joint_delta_command_maps_keys(self) -> None:
        command = build_joint_delta_command({"q", "w", "y"}, self.settings)

        self.assertEqual(
            command,
            JointDeltaCommand(
                shoulder_pan=2.0,
                shoulder_lift=2.0,
                gripper=4.0,
            ),
        )

    def test_build_joint_delta_command_freezes_on_space(self) -> None:
        command = build_joint_delta_command({"q", "space"}, self.settings)

        self.assertTrue(command.freeze)
        self.assertEqual(command.as_dict()["shoulder_pan"], 0.0)

    def test_build_joint_delta_command_cancels_opposing_keys(self) -> None:
        command = build_joint_delta_command({"q", "a", "t", "g"}, self.settings)

        self.assertEqual(command.shoulder_pan, 0.0)
        self.assertEqual(command.wrist_roll, 0.0)

    def test_target_tracker_seeds_from_first_observation(self) -> None:
        tracker = JointTargetTracker(self.settings)

        targets = tracker.next_targets(_observation(), JointDeltaCommand())

        self.assertEqual(targets["shoulder_pan"], 10.0)
        self.assertEqual(targets["gripper"], 60.0)

    def test_target_tracker_accumulates_deltas_and_clamps_gripper(self) -> None:
        tracker = JointTargetTracker(self.settings)
        tracker.seed(_observation(gripper=98.0))

        targets = tracker.next_targets(
            _observation(gripper=98.0),
            JointDeltaCommand(gripper=4.0),
        )

        self.assertEqual(targets["gripper"], 100.0)

    def test_target_tracker_limits_joint_targets_relative_to_current_observation(self) -> None:
        tracker = JointTargetTracker(self.settings)
        tracker.seed(_observation())

        tracker.next_targets(_observation(), JointDeltaCommand(shoulder_pan=4.0))
        targets = tracker.next_targets(_observation(), JointDeltaCommand(shoulder_pan=4.0))

        self.assertEqual(targets["shoulder_pan"], 15.0)

    def test_target_tracker_freeze_keeps_previous_target(self) -> None:
        tracker = JointTargetTracker(self.settings)
        tracker.seed(_observation())
        tracker.next_targets(_observation(), JointDeltaCommand(shoulder_pan=2.0))

        targets = tracker.next_targets(_observation(), JointDeltaCommand(freeze=True))

        self.assertEqual(targets["shoulder_pan"], 12.0)


if __name__ == "__main__":
    unittest.main()
