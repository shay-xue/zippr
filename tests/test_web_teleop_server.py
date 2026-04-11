"""Tests for browser teleop server state handling."""

from __future__ import annotations

import unittest

from arm.web_teleop_server import BrowserKeyState


class BrowserKeyStateTests(unittest.TestCase):
    def test_snapshot_returns_current_keys(self) -> None:
        state = BrowserKeyState()
        state.set_keys({"q", "w"})

        self.assertEqual(state.snapshot(), {"q", "w"})
        self.assertFalse(state.should_exit)

    def test_escape_requests_shutdown(self) -> None:
        state = BrowserKeyState()
        state.set_keys({"esc"})

        self.assertTrue(state.should_exit)


if __name__ == "__main__":
    unittest.main()
