"""Shared bot helper functions."""

from __future__ import annotations

import logging
import time
from typing import Optional

from pyrogram import Client
from pyrogram.enums import ChatMemberStatus
from pyrogram.errors import RPCError
from pyrogram.raw.functions.users.get_full_user import GetFullUser
from pyrogram.raw.types.input_peer_self import InputPeerSelf
from pyrogram.raw.types.input_peer_user import InputPeerUser
from pyrogram.raw.types.input_user import InputUser
from pyrogram.raw.types.input_user_self import InputUserSelf
from pyrogram.raw.types.user_full import UserFull
from pyrogram.types import Message

logger = logging.getLogger(__name__)

GROUP_ANONYMOUS_BOT_ID = 1087968824

_RECENT_MESSAGE_TTL_SECONDS = 10
_RECENT_USER_BIO_TTL_SECONDS = 60

_recent_message_keys: dict[tuple[int, int, bool], float] = {}
_recent_user_bio: dict[int, tuple[str, float]] = {}


def is_duplicate_message(message: Message, is_edited: bool = False) -> bool:
    """Check whether a message was recently processed.

    Args:
        message: Incoming Pyrogram message.
        is_edited: Whether the message is an edited message.

    Returns:
        True if the message appears to be a duplicate.
    """
    if not message.chat or message.id is None:
        return False

    now = time.time()
    key = (message.chat.id, message.id, is_edited)

    expired_keys = [
        k for k, ts in _recent_message_keys.items() if now - ts > _RECENT_MESSAGE_TTL_SECONDS
    ]
    for key_to_remove in expired_keys:
        _recent_message_keys.pop(key_to_remove, None)

    if key in _recent_message_keys:
        return True

    _recent_message_keys[key] = now
    return False


def is_guest_bot_message(message: Message) -> bool:
    """Check whether a message was sent by a guest bot."""
    return bool(
        getattr(message, "guest_bot_caller_user", None)
        or getattr(message, "guest_bot_caller_chat", None)
    )


def should_skip_group_message(message: Message, is_edited: bool) -> bool:
    """Determine whether a group message should be skipped.

    Args:
        message: Incoming Pyrogram message.
        is_edited: Whether the message is edited.

    Returns:
        True if the message should be skipped.
    """
    if not message.from_user:
        return True

    if is_duplicate_message(message, is_edited=is_edited):
        logger.debug("重复消息已跳过处理")
        return True

    if getattr(message.from_user, "is_bot", False):
        return True

    if message.from_user.id == GROUP_ANONYMOUS_BOT_ID:
        logger.debug("跳过群组匿名身份发言")
        return True

    if message.text and message.text.lstrip().startswith("/"):
        return True

    return False


def is_admin_status(status: object) -> bool:
    """Check whether a chat member status is admin/owner."""
    if status in (ChatMemberStatus.OWNER, ChatMemberStatus.ADMINISTRATOR):
        return True
    if isinstance(status, str):
        return status.lower() in ("creator", "owner", "administrator")
    return False


async def try_get_member_joined_date(
    client: Client, chat_id: int | str | None, user_id: int
) -> Optional[object]:
    """Try to retrieve the member joined date.

    Args:
        client: Pyrogram client.
        chat_id: Telegram chat ID.
        user_id: Telegram user ID.

    Returns:
        Joined date or None if unavailable.
    """
    if chat_id is None:
        return None

    try:
        chat_member = await client.get_chat_member(chat_id=chat_id, user_id=user_id)
    except RPCError as exc:
        logger.debug("get_chat_member 失败 (chat_id=%s, user_id=%s): %s", chat_id, user_id, exc)
        return None

    return getattr(chat_member, "joined_date", None)


def format_joined_date_str(joined_date: Optional[object]) -> str:
    """Format a joined date into a string.

    Args:
        joined_date: Joined date value.

    Returns:
        A formatted datetime string.
    """
    if not joined_date:
        return "2000-01-01"
    try:
        return joined_date.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(joined_date)


async def try_get_user_bio(client: Client, user_id: int) -> Optional[str]:
    """Try to fetch a user's bio with caching.

    Args:
        client: Pyrogram client.
        user_id: Telegram user ID.

    Returns:
        Bio text or None.
    """
    now = time.time()
    cached = _recent_user_bio.get(user_id)
    if cached and now - cached[1] <= _RECENT_USER_BIO_TTL_SECONDS:
        logger.debug("命中 bio 缓存 | user_id=%s", user_id)
        return cached[0] or None

    bio_text = None

    try:
        input_peer = await client.resolve_peer(user_id)

        if isinstance(input_peer, InputPeerUser):
            input_user = InputUser(
                user_id=input_peer.user_id,
                access_hash=input_peer.access_hash,
            )
        elif isinstance(input_peer, InputPeerSelf):
            input_user = InputUserSelf()
        else:
            logger.info(
                "resolve_peer 结果不是用户类型，无法获取 bio | user_id=%s, peer_type=%s",
                user_id,
                type(input_peer).__name__,
            )
            return None

        full_result = await client.invoke(GetFullUser(id=input_user))
        full_user = getattr(full_result, "full_user", None)
        if isinstance(full_user, UserFull):
            about = getattr(full_user, "about", None)
            if about is not None:
                bio_text = str(about)
    except RPCError as exc:
        logger.debug("GetFullUser 获取用户 bio 失败 (user_id=%s): %s", user_id, exc)

    if bio_text is not None:
        _recent_user_bio[user_id] = (bio_text, now)
        logger.info("获取用户 bio 成功 | user_id=%s", user_id)
    else:
        logger.info("用户 bio 为空或不可见 | user_id=%s", user_id)

    return bio_text
