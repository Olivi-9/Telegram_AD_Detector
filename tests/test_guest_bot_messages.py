"""Tests for guest bot message handling."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from telegram_ad_detector.core.bot._helpers import is_guest_bot_message


class GuestBotMessageTest(unittest.TestCase):
    def test_detects_guest_bot_message(self) -> None:
        message = SimpleNamespace(
            guest_bot_caller_user=SimpleNamespace(id=123),
            guest_bot_caller_chat=None,
        )

        self.assertTrue(is_guest_bot_message(message))

    def test_ignores_regular_message(self) -> None:
        message = SimpleNamespace(
            guest_bot_caller_user=None,
            guest_bot_caller_chat=None,
        )

        self.assertFalse(is_guest_bot_message(message))


if __name__ == "__main__":
    unittest.main()
