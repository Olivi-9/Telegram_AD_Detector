"""Compatibility wrapper for model training."""

from __future__ import annotations

from telegram_ad_detector.core.ml.training import main as training_main


def main() -> None:
    """Run the training workflow."""
    training_main()


if __name__ == "__main__":
    main()
