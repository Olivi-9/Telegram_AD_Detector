"""Compatibility wrapper for interactive prediction."""

from __future__ import annotations

from telegram_ad_detector.core.ml.inference import (
    load_model,
    load_texts_from_file,
    predict_batch,
    predict_text,
)
from telegram_ad_detector.core.ml.interactive import run_interactive

__all__ = [
    "load_model",
    "load_texts_from_file",
    "predict_batch",
    "predict_text",
    "run_interactive",
    "main",
]


def main() -> None:
    """Start the interactive prediction CLI."""
    run_interactive()


if __name__ == "__main__":
    main()
