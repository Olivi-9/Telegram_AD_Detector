"""Command handlers for the bot."""

from __future__ import annotations

import logging

from pyrogram import filters
from pyrogram.errors import RPCError
from pyrogram.types import Message

from ..config import config
from ..user_state import get_state_manager
from ._actions import reply_with_auto_delete
from ._helpers import format_joined_date_str, is_admin_status, is_duplicate_message, try_get_member_joined_date
from ._whitelist import is_group_whitelisted
from .app import app

logger = logging.getLogger(__name__)


@app.on_message(filters.command("start"))
async def cmd_start(client, message: Message) -> None:
    """Handle /start command."""
    if is_duplicate_message(message):
        return
    await reply_with_auto_delete(
        message,
        "🛡 广告检测 Bot\n\n"
        "本 Bot 目前仍在完善：\n"
        f"- 干运行: {config.DRY_RUN}\n"
        f"- 仅新成员: {config.ONLY_NEW_MEMBER}\n"
        f"- 调试回复: {config.DEBUG_REPLY}\n\n",
    )


@app.on_message(filters.command("stats"))
async def cmd_stats(client, message: Message) -> None:
    """Handle /stats command."""
    if is_duplicate_message(message):
        return
    state_manager = get_state_manager()
    stats = state_manager.get_stats()

    total_monitor = sum(u.monitor_count for u in state_manager.users.values())
    total_warn = sum(u.warn_count for u in state_manager.users.values())
    users_with_violations = sum(
        1
        for u in state_manager.users.values()
        if u.monitor_count > 0 or u.warn_count > 0
    )

    msg = (
        "📊 Bot 统计信息\n\n"
        f"总用户数: {stats['total_users']}\n"
        f"新成员: {stats['new_members']}\n"
        f"老成员: {stats['old_members']}\n\n"
        "违规统计:\n"
        f"  - 有违规记录: {users_with_violations} 人\n"
        f"  - 总 MONITOR: {total_monitor} 次\n"
        f"  - 总 WARN: {total_warn} 次\n\n"
        "新成员判定规则:\n"
        f"  - 消息数 ≤ {stats['message_threshold']}\n"
        f"  - 加入时间 < {stats['time_threshold_days']:.1f}天\n\n"
        "执行模式:\n"
        f"  - DRY_RUN: {config.DRY_RUN}\n"
        f"  - MONITOR 删除: {config.MONITOR_DELETE_MESSAGE}\n"
        f"  - MONITOR 踢人阈值: {config.MONITOR_KICK_THRESHOLD}\n"
        f"  - WARN 踢人阈值: {config.WARN_KICK_THRESHOLD}"
    )

    await reply_with_auto_delete(message, msg)


@app.on_message(filters.command("config"))
async def cmd_config(client, message: Message) -> None:
    """Handle /config command."""
    if is_duplicate_message(message):
        return
    msg = (
        "⚙️ Bot 配置\n\n"
        "基础设置:\n"
        f"  调试回复: {config.DEBUG_REPLY}\n"
        f"  仅新成员: {config.ONLY_NEW_MEMBER}\n"
        f"  干运行模式: {config.DRY_RUN}\n\n"
        "检测开关:\n"
        f"  使用 ML 模型: {config.USE_ML_MODEL}\n"
        f"  检测外部引用: {config.CHECK_EXTERNAL_REPLY}\n"
        f"  检测联系人: {config.CHECK_CONTACT_MESSAGE}\n"
        f"  检测图片: {config.CHECK_IMAGE_FROM_NEW_MEMBER}\n\n"
        "执行策略:\n"
        f"  MONITOR 删除消息: {config.MONITOR_DELETE_MESSAGE}\n"
        f"  MONITOR 踢人阈值: {config.MONITOR_KICK_THRESHOLD}\n"
        f"  WARN 删除消息: {config.WARN_DELETE_MESSAGE}\n"
        f"  WARN 踢人阈值: {config.WARN_KICK_THRESHOLD}\n"
        f"  HIGH 立即踢人: {config.HIGH_IMMEDIATE_KICK}\n"
    )

    await reply_with_auto_delete(message, msg)


@app.on_message(filters.command("save"))
async def cmd_save(client, message: Message) -> None:
    """Handle /save command."""
    if is_duplicate_message(message):
        return
    state_manager = get_state_manager()
    state_manager.save()
    await reply_with_auto_delete(message, "✅ 用户状态已保存")


@app.on_message(filters.command("get_group_id") & filters.group)
async def cmd_get_group_id(client, message: Message) -> None:
    """Handle /get_group_id command."""
    if is_duplicate_message(message):
        return
    group_id = message.chat.id
    group_title = message.chat.title or "未知群组"

    in_whitelist = is_group_whitelisted(group_id)
    status = "✅ 已在白名单" if in_whitelist else "⚠️ 未在白名单"

    msg = (
        "📋 群组信息\n\n"
        f"群组名称: {group_title}\n"
        f"群组 ID: `{group_id}`\n"
        f"状态: {status}\n\n"
        f"💡 如需添加到白名单，请通知 Olivi 编辑 {config.GROUP_WHITELIST_FILE} 文件"
    )
    await reply_with_auto_delete(message, msg, 15)


@app.on_message(filters.command("get_user_id"))
async def cmd_get_user_id(client, message: Message) -> None:
    """Handle /get_user_id command."""
    if is_duplicate_message(message):
        return
    if message.reply_to_message and message.reply_to_message.from_user:
        target_user = message.reply_to_message.from_user
        target_user_id = target_user.id
        target_username = target_user.username or "无"
        target_firstname = target_user.first_name or "无"
        joined_date = await try_get_member_joined_date(
            app,
            chat_id=message.chat.id,
            user_id=target_user_id,
        )
        joined_group_date = format_joined_date_str(joined_date)

        msg = (
            "👤 用户信息\n\n"
            f"用户 ID: `{target_user_id}`\n"
            f"用户名: @{target_username}\n"
            f"昵称: {target_firstname}\n"
            f"入群时间: {joined_group_date}"
        )
    else:
        user = message.from_user
        user_id = user.id
        username = user.username or "无"
        firstname = user.first_name or "无"
        joined_date = await try_get_member_joined_date(
            app,
            chat_id=message.chat.id,
            user_id=user_id,
        )
        joined_group_date = format_joined_date_str(joined_date)

        msg = (
            "👤 您的用户信息\n\n"
            f"用户 ID: `{user_id}`\n"
            f"用户名: @{username}\n"
            f"显示名: {firstname}\n"
            f"入群时间: {joined_group_date}\n\n"
            "💡 回复某人的消息使用此命令可获取对方的 ID"
        )

    await reply_with_auto_delete(message, msg, 15)


@app.on_message(filters.command("ad_whitelist") & filters.group)
async def cmd_whitelist(client, message: Message) -> None:
    """Handle /ad_whitelist command."""
    if is_duplicate_message(message):
        return
    group_id = message.chat.id
    if not is_group_whitelisted(group_id):
        await reply_with_auto_delete(
            message, "❌ 此群组未在白名单中\n请联系 Olivi 将群组添加到白名单"
        )
        return

    try:
        chat_member = await app.get_chat_member(
            chat_id=message.chat.id, user_id=message.from_user.id
        )
    except RPCError as exc:
        logger.error("检查管理员权限失败: %s", exc)
        await reply_with_auto_delete(message, "❌ 无法验证权限")
        return

    if not is_admin_status(getattr(chat_member, "status", None)):
        await reply_with_auto_delete(message, "❌ 只有管理员可以使用此命令")
        return

    state_manager = get_state_manager()

    parts = message.text.split()
    args = parts[1:] if len(parts) > 1 else []

    if not args or args[0] == "help":
        help_text = (
            "🤍 白名单管理\n\n"
            "用法：\n"
            "`/ad_whitelist add <user_id>` - 添加用户\n"
            "`/ad_whitelist remove <user_id>` - 移除用户\n"
            "`/ad_whitelist list` - 查看白名单\n\n"
            "示例：\n"
            "`/ad_whitelist add 123456789`\n"
            "或回复用户消息：`/ad_whitelist add`"
        )
        await reply_with_auto_delete(message, help_text, 20)
        return

    action = args[0].lower()

    if action == "list":
        whitelist_users = [
            (uid, user.username)
            for uid, user in state_manager.users.items()
            if user.is_whitelisted
        ]

        if not whitelist_users:
            msg = "🤍 白名单为空"
        else:
            msg = f"🤍 白名单用户 ({len(whitelist_users)}):\n\n"
            for uid, username in whitelist_users:
                username_str = f"@{username}" if username else "(无用户名)"
                msg += f"• {uid} - {username_str}\n"

        await reply_with_auto_delete(message, msg, 15)
        return

    if action not in ["add", "remove"]:
        await reply_with_auto_delete(message, "❌ 无效操作，使用 /ad_whitelist help 查看帮助")
        return

    target_user_id = None
    target_username = None

    if message.reply_to_message and message.reply_to_message.from_user:
        target_user_id = message.reply_to_message.from_user.id
        target_username = message.reply_to_message.from_user.username
    elif len(args) > 1:
        try:
            target_user_id = int(args[1])
        except ValueError:
            await reply_with_auto_delete(message, "❌ 无效的 user_id")
            return
    else:
        await reply_with_auto_delete(message, "❌ 请提供 user_id 或回复用户消息")
        return

    if action == "add":
        success = state_manager.add_to_whitelist(target_user_id)
        if success:
            username_str = (
                f"@{target_username}" if target_username else str(target_user_id)
            )
            msg = f"✅ 已将 {username_str} 添加到白名单"
            logger.info(
                "管理员 %s 将用户 %s 添加到白名单",
                message.from_user.id,
                target_user_id,
            )
        else:
            msg = "ℹ️ 用户已在白名单中"
    else:
        success = state_manager.remove_from_whitelist(target_user_id)
        if success:
            username_str = (
                f"@{target_username}" if target_username else str(target_user_id)
            )
            msg = f"✅ 已将 {username_str} 从白名单移除"
            logger.info(
                "管理员 %s 将用户 %s 从白名单移除",
                message.from_user.id,
                target_user_id,
            )
        else:
            msg = "ℹ️ 用户不在白名单中"

    await reply_with_auto_delete(message, msg)
