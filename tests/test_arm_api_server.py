"""Tests for the FastAPI arm server."""

from __future__ import annotations

import importlib.util
import unittest

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None

if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient

    from arm.api_server import create_app
    from arm.api import EndEffectorDeltaResult
    from arm.config import ArmSettings
    from arm.ik import ArmPose
else:
    TestClient = None
    create_app = None
    EndEffectorDeltaResult = None
    ArmSettings = None
    ArmPose = None


class FakeArmAPI:
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False
        self.commands: list[object] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def get_joint_positions(self) -> dict[str, float]:
        return {
            "shoulder_pan": 1.0,
            "shoulder_lift": 2.0,
            "elbow_flex": 3.0,
            "wrist_flex": 4.0,
            "wrist_roll": 5.0,
            "gripper": 6.0,
        }

    def get_end_effector_pose(self) -> ArmPose:
        return ArmPose(x=0.1, y=0.2, z=0.3, tool_roll=4.0)

    def send_end_effector_delta(self, command: object) -> dict[str, float]:
        self.commands.append(command)
        return EndEffectorDeltaResult(
            sent_targets={"wrist_roll": 5.0, "gripper": 6.0},
            merged_targets=self.get_joint_positions(),
        )


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class ArmAPIServerTests(unittest.TestCase):
    def test_endpoints_return_arm_state_and_forward_commands(self) -> None:
        fake_api = FakeArmAPI()
        with TestClient(create_app(fake_api)) as client:
            self.assertTrue(fake_api.connected)

            index = client.get("/")
            joints = client.get("/api/joints")
            pose = client.get("/api/pose")
            command = client.post(
                "/api/end-effector-delta",
                json={"dx": 0.1, "dy": -0.2, "dz": 0.3, "d_rot": 4.0, "d_jaw": -5.0},
            )

        self.assertTrue(fake_api.disconnected)
        self.assertEqual(index.status_code, 200)
        self.assertIn("SO-101 Keyboard Command UI", index.text)
        self.assertIn("shoulder pan", index.text)
        self.assertIn("forward reach", index.text)
        self.assertIn("Command + Targets", index.text)
        self.assertIn('id="sent-wrist_roll"', index.text)
        self.assertIn('id="merged-wrist_roll"', index.text)
        self.assertIn("/api/end-effector-delta", index.text)
        self.assertEqual(joints.status_code, 200)
        self.assertEqual(joints.json()["shoulder_pan"], 1.0)
        self.assertEqual(pose.status_code, 200)
        self.assertEqual(pose.json()["z"], 0.3)
        self.assertEqual(command.status_code, 200)
        self.assertEqual(command.json()["sent_targets"]["wrist_roll"], 5.0)
        self.assertEqual(command.json()["merged_targets"]["wrist_roll"], 5.0)
        self.assertEqual(len(fake_api.commands), 1)
        sent = fake_api.commands[0]
        self.assertEqual(sent.dx, 0.1)
        self.assertEqual(sent.dy, -0.2)
        self.assertEqual(sent.dz, 0.3)
        self.assertEqual(sent.d_rot, 4.0)
        self.assertEqual(sent.d_jaw, -5.0)

    def test_index_renders_ui_defaults_from_settings(self) -> None:
        fake_api = FakeArmAPI()
        settings = ArmSettings(
            api_ui_step_sizes={"linear": 0.025, "pan": 0.2, "roll": 9.0, "jaw": 3.0},
            api_ui_repeat_ms=45,
        )

        with TestClient(create_app(fake_api, settings=settings)) as client:
            index = client.get("/")

        self.assertEqual(index.status_code, 200)
        self.assertIn('id="linearStep" type="number" step="0.001" value="0.025"', index.text)
        self.assertIn('id="panStep" type="number" step="0.01" value="0.2"', index.text)
        self.assertIn('id="rollStep" type="number" step="0.1" value="9.0"', index.text)
        self.assertIn('id="jawStep" type="number" step="0.1" value="3.0"', index.text)
        self.assertIn('id="repeatMs" type="number" step="10" min="20" value="45"', index.text)


if __name__ == "__main__":
    unittest.main()
