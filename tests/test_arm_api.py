"""Tests for the high-level arm API and IK translation."""

from __future__ import annotations

import math
import tempfile
import unittest
import importlib.util

from arm.api import SO101ArmAPI
from arm.commands import EndEffectorDeltaCommand
from arm.config import ArmSettings
from arm.controller import SO101ArmController
from arm.ik import ArmKinematics, SO101IKTranslator


class FakeRobot:
    def __init__(self) -> None:
        self.actions: list[dict[str, float]] = []
        self.observation = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 15.0,
            "elbow_flex.pos": -25.0,
            "wrist_flex.pos": 10.0,
            "wrist_roll.pos": 5.0,
            "gripper.pos": 20.0,
        }

    def connect(self, calibrate: bool = True) -> None:
        return None

    def get_observation(self) -> dict[str, float]:
        return dict(self.observation)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.actions.append(dict(action))
        self.observation.update(action)
        return dict(action)

    def disconnect(self) -> None:
        return None


class SO101IKTranslatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ikpy_available = importlib.util.find_spec("ikpy") is not None
        if not self.ikpy_available:
            self.translator = None
        else:
            self.translator = SO101IKTranslator(use_degrees=True)
        self.observation = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 10.0,
            "elbow_flex.pos": -20.0,
            "wrist_flex.pos": 5.0,
            "wrist_roll.pos": 3.0,
            "gripper.pos": 40.0,
        }

    def test_zero_delta_preserves_pose_and_direct_axes(self) -> None:
        if not self.ikpy_available:
            self.skipTest("ikpy is not installed in this environment")
        pose_before = self.translator.forward(self.observation)

        targets = self.translator.translate(EndEffectorDeltaCommand(), self.observation)
        pose_after = self.translator.forward({f"{name}.pos": value for name, value in targets.items()})

        self.assertAlmostEqual(pose_after.x, pose_before.x, places=6)
        self.assertAlmostEqual(pose_after.y, pose_before.y, places=6)
        self.assertAlmostEqual(pose_after.z, pose_before.z, places=6)
        self.assertAlmostEqual(targets["wrist_roll"], 3.0)
        self.assertAlmostEqual(targets["gripper"], 40.0)

    def test_translation_applies_cartesian_and_tool_deltas(self) -> None:
        if not self.ikpy_available:
            self.skipTest("ikpy is not installed in this environment")
        pose_before = self.translator.forward(self.observation)
        command = EndEffectorDeltaCommand(dx=-0.01, dy=0.015, dz=-0.005, d_rot=7.0, d_jaw=12.0)

        targets = self.translator.translate(command, self.observation)
        pose_after = self.translator.forward({f"{name}.pos": value for name, value in targets.items()})

        self.assertAlmostEqual(math.hypot(pose_after.x, pose_after.y), math.hypot(pose_before.x, pose_before.y) - 0.01, places=5)
        self.assertAlmostEqual(pose_after.z, pose_before.z - 0.005, places=5)
        self.assertAlmostEqual(math.radians(targets["shoulder_pan"]), 0.015, places=5)
        self.assertAlmostEqual(targets["wrist_roll"], 10.0)
        self.assertAlmostEqual(targets["gripper"], 52.0)

    def test_dy_only_changes_shoulder_pan_target(self) -> None:
        if not self.ikpy_available:
            self.skipTest("ikpy is not installed in this environment")
        targets = self.translator.translate(EndEffectorDeltaCommand(dy=0.2), self.observation)

        self.assertAlmostEqual(math.radians(targets["shoulder_pan"]), 0.2, places=5)
        self.assertAlmostEqual(targets["shoulder_lift"], 10.0)
        self.assertAlmostEqual(targets["elbow_flex"], -20.0)
        self.assertAlmostEqual(targets["wrist_flex"], 5.0)
        self.assertAlmostEqual(targets["wrist_roll"], 3.0)
        self.assertAlmostEqual(targets["gripper"], 40.0)

    def test_roll_only_does_not_recompute_ik_joints(self) -> None:
        if not self.ikpy_available:
            self.skipTest("ikpy is not installed in this environment")
        targets = self.translator.translate(EndEffectorDeltaCommand(d_rot=7.0), self.observation)

        self.assertAlmostEqual(targets["shoulder_pan"], 0.0)
        self.assertAlmostEqual(targets["shoulder_lift"], 10.0)
        self.assertAlmostEqual(targets["elbow_flex"], -20.0)
        self.assertAlmostEqual(targets["wrist_flex"], 5.0)
        self.assertAlmostEqual(targets["wrist_roll"], 10.0)
        self.assertAlmostEqual(targets["gripper"], 40.0)

    def test_gripper_target_is_clamped(self) -> None:
        if not self.ikpy_available:
            self.skipTest("ikpy is not installed in this environment")
        targets = self.translator.translate(
            EndEffectorDeltaCommand(d_jaw=90.0),
            self.observation,
        )

        self.assertEqual(targets["gripper"], 100.0)

    def test_ik_requires_ikpy_dependency(self) -> None:
        if self.ikpy_available:
            self.skipTest("ikpy is installed in this environment")
        with self.assertRaisesRegex(ImportError, "ikpy is required for IK"):
            SO101IKTranslator(use_degrees=True)

    def test_kinematics_can_be_loaded_from_so101_urdf(self) -> None:
        urdf = """<?xml version="1.0"?>
<robot name="so100">
  <joint name="1" type="revolute">
    <origin xyz="-0.0306 0 0.0949" rpy="0 0 0"/>
  </joint>
  <joint name="2" type="revolute">
    <origin xyz="-0.0303992 -0.0182778 -0.0542" rpy="-1.5708 -1.5708 0"/>
  </joint>
  <joint name="3" type="revolute">
    <origin xyz="-0.11257 -0.028 0" rpy="0 0 1.5708"/>
  </joint>
  <joint name="4" type="revolute">
    <origin xyz="-0.1349 0.0052 0" rpy="0 0 -1.5708"/>
  </joint>
  <joint name="5" type="revolute">
    <origin xyz="0 -0.0611 0.0181" rpy="1.5708 0.0486795 3.14159"/>
  </joint>
</robot>
"""
        with tempfile.NamedTemporaryFile("w", suffix=".urdf") as handle:
            handle.write(urdf)
            handle.flush()
            kinematics = ArmKinematics.from_urdf(handle.name, ["1", "2", "3", "4", "5"])

        self.assertAlmostEqual(kinematics.base_height, 0.0949, places=4)
        self.assertAlmostEqual(kinematics.upper_arm_length, 0.1160, places=4)
        self.assertAlmostEqual(kinematics.forearm_length, 0.1350, places=4)
        self.assertAlmostEqual(kinematics.tool_length, 0.0637, places=4)


class SO101ArmAPITests(unittest.TestCase):
    def test_api_translates_end_effector_command_to_low_level_targets(self) -> None:
        if importlib.util.find_spec("ikpy") is None:
            self.skipTest("ikpy is not installed in this environment")
        fake_robot = FakeRobot()
        controller = SO101ArmController(
            settings=ArmSettings(port="/dev/mock"),
            robot_factory=lambda _: fake_robot,
        )
        api = SO101ArmAPI(
            controller=controller,
            ik_translator=SO101IKTranslator(use_degrees=True),
        )

        api.connect()
        initial_pose = api.get_end_effector_pose()
        result = api.send_end_effector_delta(
            EndEffectorDeltaCommand(dx=-0.02, dy=0.0, dz=0.01, d_rot=4.0, d_jaw=-5.0)
        )
        final_pose = api.get_end_effector_pose()
        api.disconnect()

        self.assertEqual(len(fake_robot.actions), 1)
        self.assertTrue(all(key.endswith(".pos") for key in fake_robot.actions[0]))
        self.assertAlmostEqual(math.hypot(final_pose.x, final_pose.y), math.hypot(initial_pose.x, initial_pose.y) - 0.02, places=5)
        self.assertAlmostEqual(final_pose.z, initial_pose.z + 0.01, places=5)
        self.assertAlmostEqual(result.merged_targets["wrist_roll"], 9.0)
        self.assertAlmostEqual(result.merged_targets["gripper"], 15.0)
        self.assertTrue(math.isfinite(result.merged_targets["shoulder_lift"]))

    def test_api_roll_only_preserves_other_joint_targets(self) -> None:
        fake_robot = FakeRobot()
        controller = SO101ArmController(
            settings=ArmSettings(port="/dev/mock"),
            robot_factory=lambda _: fake_robot,
        )

        class FakeTranslator:
            def translate(self, command: EndEffectorDeltaCommand, observation: dict[str, float]) -> dict[str, float]:
                return {
                    "shoulder_pan": observation["shoulder_pan.pos"],
                    "shoulder_lift": observation["shoulder_lift.pos"],
                    "elbow_flex": observation["elbow_flex.pos"],
                    "wrist_flex": observation["wrist_flex.pos"],
                    "wrist_roll": observation["wrist_roll.pos"] + command.d_rot,
                    "gripper": observation["gripper.pos"],
                }

        api = SO101ArmAPI(
            controller=controller,
            ik_translator=FakeTranslator(),
        )

        api.connect()
        result = api.send_end_effector_delta(EndEffectorDeltaCommand(d_rot=4.0))
        api.disconnect()

        self.assertEqual(
            fake_robot.actions,
            [
                {
                    "shoulder_pan.pos": 0.0,
                    "shoulder_lift.pos": 15.0,
                    "elbow_flex.pos": -25.0,
                    "wrist_flex.pos": 10.0,
                    "wrist_roll.pos": 9.0,
                    "gripper.pos": 20.0,
                }
            ],
        )
        self.assertEqual(result.sent_targets["shoulder_lift"], 15.0)
        self.assertEqual(result.sent_targets["elbow_flex"], -25.0)
        self.assertEqual(result.sent_targets["wrist_flex"], 10.0)
        self.assertEqual(result.sent_targets["wrist_roll"], 9.0)


if __name__ == "__main__":
    unittest.main()
