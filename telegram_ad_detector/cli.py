"""Command-line entrypoints for uv and local execution."""

from __future__ import annotations

from telegram_ad_detector.core.bot.main import main as bot_main
from telegram_ad_detector.core.ml.interactive import run_interactive as predict_main
from telegram_ad_detector.core.ml.training import main as train_main


def bot() -> None:
    """Start the Telegram moderation bot."""
    bot_main()


def predict() -> None:
    """Start the interactive prediction utility."""
    predict_main()


def train() -> None:
    """Run the model training script."""
    train_main()
