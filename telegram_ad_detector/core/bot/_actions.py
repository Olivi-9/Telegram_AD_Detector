"""Moderation action helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Tuple

from pyrogram import Client
from pyrogram.errors import RPCError
from pyrogram.types import Message

from ..config import config
from ..detection.types import DetectionResult
from ..user_state import UserStateManager
from ._helpers import is_admin_status

logger = logging.getLogger(__name__)


async def send_and_delete_later(message_obj: Message, delete_after: int = 15) -> None:
    """Schedule deletion for a sent message.

    Args:
        message_obj: Pyrogram Message to delete.
        delete_after: Delay in seconds.
    """

    async def delete_later() -> None:
        try:
            await asyncio.sleep(delete_after)
            await message_obj.delete()
            logger.debug("已自动删除消息 %s", message_obj.id)
        except Exception as exc:
            logger.warning("自动删除消息失败: %s", exc)

    asyncio.create_task(delete_later())


async def reply_with_auto_delete(
    message: Message,
    text: str,
    delete_after: int = 15,
) -> None:
    """Reply to a message and auto-delete after a delay."""
    sent_msg = await message.reply_text(text)
    asyncio.create_task(send_and_delete_later(sent_msg, delete_after))


async def send_with_auto_delete(
    client: Client,
    chat_id: int,
    text: str,
    delete_after: int = 15,
) -> None:
    """Send a message and auto-delete after a delay."""
    sent_msg = await client.send_message(chat_id=chat_id, text=text)
    asyncio.create_task(send_and_delete_later(sent_msg, delete_after))


async def send_debug_report(client: Client, message: Message, report: str) -> None:
    """Send a debug report message."""
    try:
        if config.USE_REPLY:
            sent_msg = await message.reply_text(report)
        else:
            sent_msg = await client.send_message(chat_id=message.chat.id, text=report)
    except RPCError as exc:
        logger.warning("无法发送调试报告: %s", exc)
        return
    asyncio.create_task(send_and_delete_later(sent_msg))


async def is_protected_member(client: Client, chat_id: int, user_id: int) -> bool:
    """Check if a user is an admin/owner and should be protected."""
    try:
        chat_member = await client.get_chat_member(chat_id=chat_id, user_id=user_id)
    except RPCError as exc:
        logger.debug(
            "get_chat_member failed for protection check (chat_id=%s, user_id=%s): %s",
            chat_id,
            user_id,
            exc,
        )
        return False

    status = getattr(chat_member, "status", None)
    return is_admin_status(status)


async def execute_action(
    client: Client,
    message: Message,
    detection_result: DetectionResult,
    state_manager: UserStateManager,
) -> Tuple[bool, str]:
    """Execute the suggested moderation action.

    Args:
        client: Pyrogram client.
        message: Incoming message.
        detection_result: Detection outcome.
        state_manager: User state manager.

    Returns:
        Tuple of (action_executed, description).
    """
    if config.DRY_RUN:
        return False, "Dry run mode - no action taken"

    user_id = message.from_user.id
    chat_id = message.chat.id

    if await is_protected_member(client, chat_id=chat_id, user_id=user_id):
        return False, "Protected member (owner/administrator) - no action taken"

    action = detection_result.suggested_action
    user_state = detection_result.user_state

    actions_taken = []

    try:
        if action.startswith("NO_ACTION"):
            return False, "No violation detected"

        if action.startswith("KICK"):
            try:
                await client.ban_chat_member(chat_id=chat_id, user_id=user_id)
                actions_taken.append("✅ 已踢出用户")

                try:
                    await message.delete()
                    actions_taken.append("✅ 已删除违规消息")
                except RPCError as exc:
                    actions_taken.append(f"⚠️ 删除消息失败: {exc}")

                reason = ""
                if "IMMEDIATE" in action:
                    reason = f"严重违规 (风险分数: {detection_result.risk_score:.2f})"
                elif "WARN" in action:
                    reason = f"累计 {user_state.warn_count + 1} 次 WARN 级别违规"
                elif "MONITOR" in action:
                    reason = f"累计 {user_state.monitor_count + 1} 次 MONITOR 级别违规"

                notify_msg = f"🛡 已踢出用户 {user_id}\n原因: {reason}"
                await send_with_auto_delete(client, chat_id, notify_msg)

                logger.warning("踢出用户 %s: %s", user_id, reason)

            except RPCError as exc:
                actions_taken.append(f"❌ 踢人失败: {exc}")
                logger.error("踢出用户 %s 失败: %s", user_id, exc)

        elif action.startswith("WARN"):
            if config.WARN_DELETE_MESSAGE:
                try:
                    await message.delete()
                    actions_taken.append("✅ 已删除消息")
                except RPCError as exc:
                    actions_taken.append(f"⚠️ 删除消息失败: {exc}")

            user_state.increment_warn()
            state_manager.save()
            actions_taken.append(
                f"⚠️ WARN 计数: {user_state.warn_count}/{config.WARN_KICK_THRESHOLD}"
            )

            username = message.from_user.username or user_id
            warn_msg = (
                f"⚠️ 警告 @{username}\n"
                "检测到疑似广告行为\n"
                f"当前警告: {user_state.warn_count}/{config.WARN_KICK_THRESHOLD}\n"
                f"达到 {config.WARN_KICK_THRESHOLD} 次将被踢出"
            )
            await send_with_auto_delete(client, chat_id, warn_msg)

            logger.warning(
                "用户 %s WARN 级别违规 (%s/%s)",
                user_id,
                user_state.warn_count,
                config.WARN_KICK_THRESHOLD,
            )

        elif action.startswith("MONITOR"):
            if config.MONITOR_DELETE_MESSAGE:
                try:
                    await message.delete()
                    actions_taken.append("✅ 已删除消息")
                except RPCError as exc:
                    actions_taken.append(f"⚠️ 删除消息失败: {exc}")

            user_state.increment_monitor()
            state_manager.save()
            actions_taken.append(
                "📊 MONITOR 计数: {count}/{threshold}".format(
                    count=user_state.monitor_count,
                    threshold=config.MONITOR_KICK_THRESHOLD,
                )
            )

            username = message.from_user.username or user_id
            monitor_msg = (
                f"📊 提醒 @{username}\n"
                "您的消息已被删除（疑似广告）\n"
                f"新成员前 {config.NEW_MEMBER_MESSAGE_THRESHOLD} 条不能发送 Sticker 和 Photo\n"
                "bio 不要携带加群链接\n"
                f"当前记录: {user_state.monitor_count}/{config.MONITOR_KICK_THRESHOLD}\n"
                f"达到 {config.MONITOR_KICK_THRESHOLD} 次将被踢出"
            )
            await send_with_auto_delete(client, chat_id, monitor_msg)

            logger.info(
                "用户 %s MONITOR 级别违规 (%s/%s)",
                user_id,
                user_state.monitor_count,
                config.MONITOR_KICK_THRESHOLD,
            )

        return True, " | ".join(actions_taken)

    except Exception as exc:
        error_msg = f"执行操作时出错: {exc}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg
