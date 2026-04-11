"""Tests for motor ID scan reporting."""

from __future__ import annotations

import unittest
from unittest import mock

from arm.motor_id_check import format_scan_results, main


class MotorIDCheckTests(unittest.TestCase):
    def test_format_scan_results_with_matches(self) -> None:
        with mock.patch(
            "arm.motor_id_check._model_number_to_name_map",
            return_value={3215: "sts3215"},
        ):
            output = format_scan_results(
                "/dev/mock",
                {
                    1000000: {1: 3215, 6: 3215},
                    115200: {3: 3215},
                },
            )

        self.assertIn("Motor scan results for port: /dev/mock", output)
        self.assertIn("Baudrate 1000000:", output)
        self.assertIn("ID 1: model_number=3215 (sts3215)", output)
        self.assertIn("Baudrate 115200:", output)

    def test_format_scan_results_with_no_matches(self) -> None:
        output = format_scan_results("/dev/mock", {})
        self.assertIn("No responding motors found", output)

    def test_main_uses_configured_port_when_cli_port_missing(self) -> None:
        with mock.patch("arm.motor_id_check.load_settings") as load_settings_mock:
            load_settings_mock.return_value.port = "/dev/from-config"
            with mock.patch("arm.motor_id_check.scan_motor_ids", return_value={}):
                with mock.patch("builtins.print") as print_mock:
                    exit_code = main([])

        self.assertEqual(exit_code, 0)
        printed = print_mock.call_args[0][0]
        self.assertIn("/dev/from-config", printed)


if __name__ == "__main__":
    unittest.main()
