"""Tests for arm config loading and validation."""

from __future__ import annotations

import types
import unittest
from unittest import mock

from arm.config import ArmSettings, load_settings


class ArmConfigTests(unittest.TestCase):
    def test_load_settings_uses_defaults_when_local_module_missing(self) -> None:
        missing = ModuleNotFoundError("No module named 'arm.local_config'", name="arm.local_config")
        with mock.patch("arm.config.importlib.import_module", side_effect=missing):
            settings = load_settings()

        self.assertEqual(settings, ArmSettings())

    def test_load_settings_applies_mapping_overrides(self) -> None:
        module = types.SimpleNamespace(
            SETTINGS_OVERRIDES={
                "port": "/dev/mock-port",
                "dry_run": True,
                "step_sizes": {"gripper": 10.0},
            }
        )

        with mock.patch("arm.config.importlib.import_module", return_value=module):
            settings = load_settings()

        self.assertEqual(settings.port, "/dev/mock-port")
        self.assertTrue(settings.dry_run)
        self.assertEqual(settings.step_sizes["gripper"], 10.0)
        self.assertEqual(settings.step_sizes["shoulder_pan"], 2.0)

    def test_load_settings_accepts_direct_settings_object(self) -> None:
        direct = ArmSettings(port="/dev/direct", robot_id="custom", dry_run=True)
        module = types.SimpleNamespace(SETTINGS=direct)

        with mock.patch("arm.config.importlib.import_module", return_value=module):
            settings = load_settings()

        self.assertEqual(settings, direct)


if __name__ == "__main__":
    unittest.main()
