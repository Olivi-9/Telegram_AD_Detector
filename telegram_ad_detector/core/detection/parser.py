"""Message parsing helpers."""

from __future__ import annotations

import logging
from typing import Any

from .types import MessageInfo, MessageType

logger = logging.getLogger(__name__)


class MessageParser:
    """Parse Telegram message-like objects."""

    @staticmethod
    def parse(update: Any) -> MessageInfo:
        """Parse a message-like object into MessageInfo.

        Args:
            update: Telegram Update-like object with a message attribute.

        Returns:
            Parsed MessageInfo.
        """
        message = update.message

        logger.debug(
            "开始解析消息: has_text=%s, has_caption=%s, has_photo=%s, has_video=%s, "
            "has_document=%s, has_contact=%s, has_sticker=%s, has_voice=%s",
            bool(getattr(message, "text", None)),
            bool(getattr(message, "caption", None)),
            bool(getattr(message, "photo", None)),
            bool(getattr(message, "video", None)),
            bool(getattr(message, "document", None)),
            bool(getattr(message, "contact", None)),
            bool(getattr(message, "sticker", None)),
            bool(getattr(message, "voice", None)),
        )

        msg_type = MessageType.OTHER
        text = None

        if message.text:
            msg_type = MessageType.TEXT
            text = message.text
        elif message.photo:
            msg_type = MessageType.PHOTO
            text = message.caption if message.caption else None
        elif message.video:
            msg_type = MessageType.VIDEO
            text = message.caption if message.caption else None
        elif message.document:
            msg_type = MessageType.DOCUMENT
            text = message.caption if message.caption else None
        elif message.contact:
            msg_type = MessageType.CONTACT
        elif message.sticker:
            msg_type = MessageType.STICKER
        elif message.voice:
            msg_type = MessageType.VOICE

        has_link = False
        has_mention = False
        text_length = 0

        if text:
            text_length = len(text)
            has_link = "http://" in text or "https://" in text or "www." in text
            has_mention = "@" in text

        is_reply = message.reply_to_message is not None
        is_external_reply = False

        if is_reply:
            replied_msg = message.reply_to_message
            if replied_msg and not replied_msg.from_user:
                is_external_reply = True

        parsed = MessageInfo(
            message_type=msg_type,
            text=text,
            has_link=has_link,
            has_mention=has_mention,
            is_reply=is_reply,
            is_external_reply=is_external_reply,
            text_length=text_length,
        )

        logger.info(
            "消息解析完成: type=%s, text_length=%s, has_link=%s, has_mention=%s, "
            "is_reply=%s, is_external_reply=%s",
            parsed.message_type.value,
            parsed.text_length,
            parsed.has_link,
            parsed.has_mention,
            parsed.is_reply,
            parsed.is_external_reply,
        )

        return parsed
