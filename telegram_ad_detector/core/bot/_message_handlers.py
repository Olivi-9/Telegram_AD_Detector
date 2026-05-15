"""Group message handlers."""

from __future__ import annotations

import logging
import time

from pyrogram import filters
from pyrogram.types import Message

from ..config import config
from ..detection import get_detection_engine
from ..user_state import get_state_manager
from ._actions import execute_action, send_debug_report
from ._formatting import describe_message_payload, format_bio_for_log, format_debug_report, format_message_text
from ._helpers import (
    is_duplicate_message,
    should_skip_group_message,
    try_get_member_joined_date,
    try_get_user_bio,
)
from ._whitelist import is_group_whitelisted
from .app import app

logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_FILTER = (
    filters.text
    | filters.caption
    | filters.photo
    | filters.video
    | filters.document
    | filters.contact
    | filters.sticker
    | filters.voice
)


class PyrogramUpdateAdapter:
    """Adapter to mimic python-telegram-bot Update objects."""

    def __init__(self, message: Message) -> None:
        self.message = message


async def _process_group_message(
    message: Message, is_edited: bool = False
) -> None:
    """Handle group messages for both new and edited messages."""
    if should_skip_group_message(message, is_edited=is_edited):
        return

    group_id = message.chat.id
    if not is_group_whitelisted(group_id):
        logger.debug("群组 %s 不在白名单中，跳过处理", group_id)
        return

    user_id = message.from_user.id
    username = message.from_user.username

    try:
        state_manager = get_state_manager()
        detection_engine = get_detection_engine()

        user_state = state_manager.record_message(user_id, username)

        if user_state.message_count <= config.NEW_MEMBER_MESSAGE_THRESHOLD:
            if time.time() - user_state.join_time < 5 * 60:
                joined_date = await try_get_member_joined_date(
                    app,
                    chat_id=message.chat.id,
                    user_id=user_id,
                )
                if joined_date:
                    state_manager.update_join_time(user_id, joined_date.timestamp())

        if user_state.is_whitelisted:
            logger.debug("跳过白名单用户 %s", user_id)
            return

        if config.ONLY_NEW_MEMBER and not user_state.is_new_member:
            logger.debug("跳过非新成员 %s", user_id)
            return

        edited_tag = "[已编辑] " if is_edited else ""
        message_text = format_message_text(message)

        user_bio = None
        if user_state.is_new_member:
            user_bio = await try_get_user_bio(app, user_id)
            if user_bio:
                logger.debug(
                    "用户简介检测启用 | user_id=%s, has_t_me=%s",
                    user_id,
                    "t.me" in user_bio.lower(),
                )

        logger.info(
            "%s检测用户 %s (新成员=%s) 的消息 | chat_id=%s, message_id=%s, "
            "text=%r, bio=%r",
            edited_tag,
            user_id,
            user_state.is_new_member,
            group_id,
            message.id,
            message_text,
            format_bio_for_log(user_bio),
        )

        logger.info(
            "消息负载详情 | chat_id=%s, message_id=%s, %s",
            group_id,
            message.id,
            describe_message_payload(message, user_bio=user_bio),
        )

        update_adapter = PyrogramUpdateAdapter(message)
        detection_result = detection_engine.analyze(
            update_adapter, user_state, user_bio=user_bio
        )

        action_executed = False
        action_description = ""

        if not config.DRY_RUN:
            action_executed, action_description = await execute_action(
                app, message, detection_result, state_manager
            )

            logger.info(
                "用户 %s: 风险=%s, 分数=%.2f, 操作=%s, 执行=%s",
                user_id,
                detection_result.risk_level.value,
                detection_result.risk_score,
                detection_result.suggested_action,
                action_executed,
            )

        if config.DEBUG_REPLY:
            report = format_debug_report(detection_result, is_edited=is_edited)

            if action_executed and action_description:
                report += f"\n\n🔧 Action Executed:\n{action_description}"

            await send_debug_report(app, message, report)

        state_manager.auto_save_if_needed()

    except Exception as exc:
        logger.error("处理消息时出错: %s", exc, exc_info=True)


@app.on_message(filters.group & SUPPORTED_CONTENT_FILTER, group=1)
async def handle_message(client, message: Message) -> None:
    """Handle new group messages."""
    await _process_group_message(message, is_edited=False)


@app.on_edited_message(filters.group & SUPPORTED_CONTENT_FILTER, group=1)
async def handle_edited_message(client, message: Message) -> None:
    """Handle edited group messages."""
    logger.info(
        "检测到消息编辑 | chat_id=%s, message_id=%s, user_id=%s",
        message.chat.id,
        message.id,
        message.from_user.id if message.from_user else "N/A",
    )
    await _process_group_message(message, is_edited=True)


@app.on_message(filters.group & filters.new_chat_members)
async def handle_new_member(client, message: Message) -> None:
    """Handle new chat member events."""
    if not message.new_chat_members:
        return

    if is_duplicate_message(message):
        logger.debug("重复的新成员消息已跳过处理")
        return

    state_manager = get_state_manager()

    for member in message.new_chat_members:
        user_id = member.id
        username = member.username
        join_time = message.date.timestamp()

        state_manager.get_or_create(user_id, username)
        state_manager.update_join_time(user_id, join_time)

        logger.info("新成员加入: %s (@%s)", user_id, username)
