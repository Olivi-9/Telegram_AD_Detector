"""Rule-based detection logic."""

from __future__ import annotations

import logging
from typing import Optional

from ..config import config
from ..user_state import UserState
from .types import MessageInfo, MessageType, RuleResult

logger = logging.getLogger(__name__)


class RuleEngine:
    """Rule evaluation helpers."""

    @staticmethod
    def check(
        message_info: MessageInfo,
        user_state: UserState,
        user_bio: Optional[str] = None,
    ) -> RuleResult:
        """Evaluate rules for a message.

        Args:
            message_info: Parsed message info.
            user_state: User state.
            user_bio: Optional user bio.

        Returns:
            RuleResult.
        """
        result = RuleResult()

        if config.CHECK_EXTERNAL_REPLY:
            if message_info.is_external_reply and user_state.is_new_member:
                result.external_reply_hit = True

        if config.CHECK_CONTACT_MESSAGE:
            if (
                message_info.message_type == MessageType.CONTACT
                and user_state.is_new_member
            ):
                result.contact_message_hit = True

        if config.CHECK_IMAGE_FROM_NEW_MEMBER:
            if (
                message_info.message_type == MessageType.PHOTO
                and user_state.is_new_member
            ):
                result.image_message_hit = True

        if config.CHECK_STICKER_FROM_NEW_MEMBER:
            if (
                message_info.message_type == MessageType.STICKER
                and user_state.is_new_member
            ):
                result.sticker_message_hit = True

        if user_state.is_new_member and user_bio:
            if "t.me" in user_bio.lower():
                result.bio_link_hit = True

        logger.info(
            "规则检测结果: external_reply=%s, contact=%s, image=%s, sticker=%s, "
            "bio_link=%s, any_hit=%s",
            result.external_reply_hit,
            result.contact_message_hit,
            result.image_message_hit,
            result.sticker_message_hit,
            result.bio_link_hit,
            result.any_hit(),
        )

        return result
