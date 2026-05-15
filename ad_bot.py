"""Compatibility wrapper for the bot entrypoint."""

from __future__ import annotations

from telegram_ad_detector.core.bot.app import app
from telegram_ad_detector.core.bot.main import main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
