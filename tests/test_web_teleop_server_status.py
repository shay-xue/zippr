"""Tests for web teleop status reporting and startup errors."""

from __future__ import annotations

import unittest

from arm.config import ArmSettings
from arm.controller import SO101ArmController
from arm.web_teleop_server import BrowserKeyState, RunnerStatus, WebTeleopRunner


class ExplodingRobot:
    def connect(self, calibrate: bool = True) -> None:
        raise RuntimeError("connect failed")

    def get_observation(self) -> dict[str, float]:
        return {}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        return action

    def disconnect(self) -> None:
        return None


class WebTeleopRunnerStatusTests(unittest.TestCase):
    def test_runner_reports_connection_error(self) -> None:
        status = RunnerStatus()
        key_state = BrowserKeyState()
        settings = ArmSettings(port="/dev/mock")
        controller = SO101ArmController(
            settings=settings,
            robot_factory=lambda _: ExplodingRobot(),
        )
        runner = WebTeleopRunner(
            controller=controller,
            settings=settings,
            key_state=key_state,
            status=status,
        )

        runner.start()

        with self.assertRaisesRegex(RuntimeError, "connect failed"):
            runner.wait_until_ready(timeout_s=0.5)

        snapshot = status.snapshot(set())
        self.assertFalse(snapshot["connected"])
        self.assertEqual(snapshot["last_error"], "connect failed")
