"""Compatibility wrapper for bot configuration."""

from __future__ import annotations

from telegram_ad_detector.core.config import (
    BotConfig,
    config,
    print_config,
    validate_config,
)

__all__ = ["BotConfig", "config", "print_config", "validate_config"]
