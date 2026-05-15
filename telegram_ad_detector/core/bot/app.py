"""Pyrogram app setup and logging."""

from __future__ import annotations

import logging

from pyrogram import Client

from ..config import BotConfig, config


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, config.LOG_LEVEL),
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


configure_logging()

logger = logging.getLogger(__name__)
logging.getLogger("pyrogram").setLevel(logging.WARNING)

app = Client(
    "ad_detector_bot",
    bot_token=BotConfig.BOT_TOKEN,
    api_id=BotConfig.API_ID,
    api_hash=BotConfig.API_HASH,
    workdir=".",
)
