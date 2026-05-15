import logging
import os
import sys
import asyncio
import time
from pyrogram import Client, filters
from pyrogram.enums import ChatMemberStatus
from pyrogram.types import Message
from pyrogram.raw.functions.users.get_full_user import GetFullUser
from pyrogram.raw.types.input_peer_self import InputPeerSelf
from pyrogram.raw.types.input_peer_user import InputPeerUser
from pyrogram.raw.types.input_user import InputUser
from pyrogram.raw.types.input_user_self import InputUserSelf
from pyrogram.raw.types.user_full import UserFull
from pyrogram.errors import RPCError
import json

from bot_config import BotConfig, config, validate_config, print_config
from user_state import get_state_manager
from detection_engine import get_detection_engine


# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, config.LOG_LEVEL),
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

logging.getLogger("pyrogram").setLevel(logging.WARNING)


GROUP_ANONYMOUS_BOT_ID = 1087968824

# 群组白名单（全局变量）
_group_whitelist: set[int] = set()

# 去重缓存：防止同一条消息被重复处理
# key = (chat_id, message_id, is_edited)
_recent_message_keys: dict[tuple[int, int, bool], float] = {}
_recent_message_ttl_seconds = 10

# 用户简介缓存（仅用于新成员 t.me 检测）
_recent_user_bio: dict[int, tuple[str, float]] = {}
_recent_user_bio_ttl_seconds = 1 * 60

# 全局 Client 实例（使用配置中的 BOT_TOKEN）
app = Client(
    "ad_detector_bot",
    bot_token=BotConfig.BOT_TOKEN,
    api_id=BotConfig.API_ID,
    api_hash=BotConfig.API_HASH,
    workdir=".",
)


def load_group_whitelist() -> None:
    """加载群组白名单"""
    global _group_whitelist
    if not os.path.exists(config.GROUP_WHITELIST_FILE):
        _group_whitelist = set()
        save_group_whitelist_file()
        logger.info("已创建默认群组白名单文件")
        return

    try:
        with open(config.GROUP_WHITELIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"加载群组白名单失败: {e}")
        _group_whitelist = set()
        return

    _group_whitelist = set(data.get("groups", []))
    logger.info(f"已加载 {len(_group_whitelist)} 个白名单群组")


def save_group_whitelist_file() -> None:
    """保存群组白名单到文件"""
    payload = {
        "groups": list(_group_whitelist),
        "description": "群组白名单配置文件。只有在此列表中的群组才能使用 bot 功能。",
        "usage": "手动编辑此文件，在 groups 数组中添加群组 ID（负数）",
    }
    try:
        with open(config.GROUP_WHITELIST_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(f"保存群组白名单失败: {e}")


def is_group_whitelisted(group_id: int) -> bool:
    """检查群组是否在白名单中"""
    # 如果白名单为空，则允许所有群组（首次使用时的默认行为）
    if not _group_whitelist:
        logger.debug("群组白名单为空，允许所有群组使用（请配置 group_whitelist.json）")
        return True
    return group_id in _group_whitelist


def is_duplicate_message(message: Message, is_edited: bool = False) -> bool:
    """检测是否为重复消息（同一 chat_id + message_id + 编辑状态）。"""
    if not message.chat or message.id is None:
        return False

    now = time.time()
    key = (message.chat.id, message.id, is_edited)

    # 清理过期记录
    expired_keys = [
        k
        for k, ts in _recent_message_keys.items()
        if now - ts > _recent_message_ttl_seconds
    ]
    for k in expired_keys:
        _recent_message_keys.pop(k, None)

    if key in _recent_message_keys:
        return True

    _recent_message_keys[key] = now
    return False


def format_message_text(message: Message, max_len: int = 200) -> str:
    """格式化消息文本用于日志（截断并压缩空白）"""
    text = message.text or message.caption or ""

    if not text and message.contact:
        first_name = getattr(message.contact, "first_name", "") or ""
        last_name = getattr(message.contact, "last_name", "") or ""
        phone_number = getattr(message.contact, "phone_number", "") or ""
        text = f"[contact] {first_name} {last_name} {phone_number}".strip()

    if not text and message.sticker:
        emoji = getattr(message.sticker, "emoji", "") or ""
        set_name = getattr(message.sticker, "set_name", "") or ""
        text = f"[sticker] emoji={emoji} set={set_name}".strip()

    if not text and message.photo:
        text = "[photo]"

    text = " ".join(text.split())
    if len(text) > max_len:
        return f"{text[:max_len]}...(truncated)"
    return text


def format_bio_for_log(user_bio: str | None, max_len: int = 180) -> str:
    """格式化用户 bio 用于日志（压缩空白并截断）。"""
    if not user_bio:
        return "(none)"

    compact = " ".join(str(user_bio).split())
    if len(compact) > max_len:
        return f"{compact[:max_len]}...(truncated)"
    return compact


def describe_message_payload(message: Message, user_bio: str | None = None) -> str:
    """输出消息负载细节，便于排查类型识别问题。"""
    details = {
        "has_text": bool(message.text),
        "has_caption": bool(message.caption),
        "has_photo": bool(message.photo),
        "has_video": bool(message.video),
        "has_document": bool(message.document),
        "has_contact": bool(message.contact),
        "has_sticker": bool(message.sticker),
        "has_voice": bool(message.voice),
        "media_group_id": getattr(message, "media_group_id", None),
        "bio_link_hit": bool(user_bio and "t.me" in user_bio.lower()),
    }

    if message.contact:
        details["contact_phone"] = getattr(message.contact, "phone_number", None)
        details["contact_user_id"] = getattr(message.contact, "user_id", None)

    if message.sticker:
        details["sticker_emoji"] = getattr(message.sticker, "emoji", None)
        details["sticker_set_name"] = getattr(message.sticker, "set_name", None)

    if message.photo:
        details["photo_file_id"] = getattr(message.photo, "file_id", None)

    return ", ".join([f"{k}={v}" for k, v in details.items()])


async def send_and_delete_later(message_obj, delete_after: int = 15) -> None:
    """
    创建异步任务，在指定时间后删除消息

    参数:
        message_obj: 已发送的 Message 对象
        delete_after: 删除延迟（秒），默认15秒
    """

    async def delete_later():
        try:
            await asyncio.sleep(delete_after)
            await message_obj.delete()
            logger.debug(f"已自动删除消息 {message_obj.id}")
        except Exception as e:
            logger.warning(f"自动删除消息失败: {e}")

    # 创建异步任务，不阻塞
    asyncio.create_task(delete_later())


async def try_get_member_joined_date(
    client: Client, chat_id: int | str | None, user_id: int
):
    """尽力获取用户入群时间（joined_date）。

    注意：joined_date 并非所有场景都有（权限/群类型/接口限制）。
    返回 datetime 或 None。
    """
    if chat_id is None:
        return None

    try:
        chat_member = await client.get_chat_member(chat_id=chat_id, user_id=user_id)
    except RPCError as e:
        logger.debug(
            f"get_chat_member 失败 (chat_id={chat_id}, user_id={user_id}): {e}"
        )
        return None

    joined_date = getattr(chat_member, "joined_date", None)
    return joined_date


async def format_joined_date_str(
    client: Client, chat_id: int | str | None, user_id: int
) -> str:
    joined_date = await try_get_member_joined_date(
        client, chat_id=chat_id, user_id=user_id
    )
    if not joined_date:
        return "2000-01-01"
    try:
        return joined_date.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(joined_date)


async def try_get_user_bio(client: Client, user_id: int) -> str | None:
    """尽力获取用户简介（bio），失败时返回 None。"""
    now = time.time()
    cached = _recent_user_bio.get(user_id)
    if cached and now - cached[1] <= _recent_user_bio_ttl_seconds:
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
    except RPCError as e:
        logger.debug(f"GetFullUser 获取用户 bio 失败 (user_id={user_id}): {e}")

    if bio_text is not None:
        _recent_user_bio[user_id] = (bio_text, now)
        logger.info(
            "获取用户 bio 成功 | user_id=%s, source=GetFullUser.about, bio=%s",
            user_id,
            format_bio_for_log(bio_text),
        )
    else:
        logger.info("用户 bio 为空或不可见 | user_id=%s", user_id)

    return bio_text


def is_admin_status(status) -> bool:
    if status in (ChatMemberStatus.OWNER, ChatMemberStatus.ADMINISTRATOR):
        return True
    if isinstance(status, str):
        return status.lower() in ("creator", "owner", "administrator")
    return False


async def is_protected_member(client: Client, chat_id: int, user_id: int) -> bool:
    """Return True for owners/admins to avoid moderation actions."""
    try:
        chat_member = await client.get_chat_member(chat_id=chat_id, user_id=user_id)
    except RPCError as e:
        logger.debug(
            "get_chat_member failed for protection check (chat_id=%s, user_id=%s): %s",
            chat_id,
            user_id,
            e,
        )
        return False

    status = getattr(chat_member, "status", None)
    return is_admin_status(status)


async def reply_with_auto_delete(
    message: Message,
    text: str,
    delete_after: int = 15,
) -> None:
    sent_msg = await message.reply_text(text)
    asyncio.create_task(send_and_delete_later(sent_msg, delete_after))


async def send_with_auto_delete(
    client: Client,
    chat_id: int,
    text: str,
    delete_after: int = 15,
) -> None:
    sent_msg = await client.send_message(chat_id=chat_id, text=text)
    asyncio.create_task(send_and_delete_later(sent_msg, delete_after))


async def send_debug_report(
    client: Client,
    message: Message,
    report: str,
) -> None:
    try:
        if config.USE_REPLY:
            sent_msg = await message.reply_text(report)
        else:
            sent_msg = await client.send_message(chat_id=message.chat.id, text=report)
    except RPCError as e:
        logger.warning(f"无法发送调试报告: {e}")
        return
    asyncio.create_task(send_and_delete_later(sent_msg))


def should_skip_group_message(message: Message, is_edited: bool) -> bool:
    # 跳过没有 from_user 的消息（系统消息等）
    if not message.from_user:
        return True

    # 去重：避免相同消息被处理两次
    if is_duplicate_message(message, is_edited=is_edited):
        logger.debug("重复消息已跳过处理")
        return True

    # 忽略所有 bot 账号消息（包括本 bot 自己）
    if getattr(message.from_user, "is_bot", False):
        return True

    # 跳过群组匿名身份发言（GroupAnonymousBot）
    if message.from_user.id == GROUP_ANONYMOUS_BOT_ID:
        logger.debug("跳过群组匿名身份发言")
        return True

    # 群内指令消息不参与广告识别
    if message.text and message.text.lstrip().startswith("/"):
        return True

    return False


async def execute_action(
    client: Client, message: Message, detection_result, state_manager
) -> tuple[bool, str]:
    """
    执行检测结果建议的操作

    参数:
        client: Pyrogram Client 对象
        message: Pyrogram Message 对象
        detection_result: 检测结果
        state_manager: 用户状态管理器

    返回:
        (是否执行了操作, 操作描述)
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
        # 根据建议操作执行
        if action.startswith("NO_ACTION"):
            # 正常消息，不执行任何操作
            return False, "No violation detected"

        elif action.startswith("KICK"):
            # 踢人操作
            try:
                await client.ban_chat_member(chat_id=chat_id, user_id=user_id)
                actions_taken.append("✅ 已踢出用户")

                # 删除违规消息
                try:
                    await message.delete()
                    actions_taken.append("✅ 已删除违规消息")
                except RPCError as e:
                    actions_taken.append(f"⚠️ 删除消息失败: {e}")

                # 发送通知
                reason = ""
                if "IMMEDIATE" in action:
                    reason = f"严重违规 (风险分数: {detection_result.risk_score:.2f})"
                elif "WARN" in action:
                    reason = f"累计 {user_state.warn_count + 1} 次 WARN 级别违规"
                elif "MONITOR" in action:
                    reason = f"累计 {user_state.monitor_count + 1} 次 MONITOR 级别违规"

                notify_msg = f"🛡 已踢出用户 {user_id}\n原因: {reason}"
                await send_with_auto_delete(client, chat_id, notify_msg)

                logger.warning(f"踢出用户 {user_id}: {reason}")

            except RPCError as e:
                actions_taken.append(f"❌ 踢人失败: {e}")
                logger.error(f"踢出用户 {user_id} 失败: {e}")

        elif action.startswith("WARN"):
            # WARN 级别：删除消息 + 增加计数
            if config.WARN_DELETE_MESSAGE:
                try:
                    await message.delete()
                    actions_taken.append("✅ 已删除消息")
                except RPCError as e:
                    actions_taken.append(f"⚠️ 删除消息失败: {e}")

            user_state.increment_warn()
            state_manager.save()
            actions_taken.append(
                f"⚠️ WARN 计数: {user_state.warn_count}/{config.WARN_KICK_THRESHOLD}"
            )

            # 发送警告
            username = message.from_user.username or user_id
            warn_msg = (
                f"⚠️ 警告 @{username}\n"
                f"检测到疑似广告行为\n"
                f"当前警告: {user_state.warn_count}/{config.WARN_KICK_THRESHOLD}\n"
                f"达到 {config.WARN_KICK_THRESHOLD} 次将被踢出"
            )
            await send_with_auto_delete(client, chat_id, warn_msg)

            logger.warning(
                f"用户 {user_id} WARN 级别违规 ({user_state.warn_count}/{config.WARN_KICK_THRESHOLD})"
            )

        elif action.startswith("MONITOR"):
            # MONITOR 级别：删除消息 + 增加计数
            if config.MONITOR_DELETE_MESSAGE:
                try:
                    await message.delete()
                    actions_taken.append("✅ 已删除消息")
                except RPCError as e:
                    actions_taken.append(f"⚠️ 删除消息失败: {e}")

            user_state.increment_monitor()
            state_manager.save()  # 立即保存
            actions_taken.append(
                f"📊 MONITOR 计数: {user_state.monitor_count}/{config.MONITOR_KICK_THRESHOLD}"
            )

            # 发送提醒
            username = message.from_user.username or user_id
            monitor_msg = (
                f"📊 提醒 @{username}\n"
                f"您的消息已被删除（疑似广告）\n"
                f"新成员前 {config.NEW_MEMBER_MESSAGE_THRESHOLD} 条不能发送 Sticker 和 Photo\n"
                f"bio 不要携带加群链接\n"
                f"当前记录: {user_state.monitor_count}/{config.MONITOR_KICK_THRESHOLD}\n"
                f"达到 {config.MONITOR_KICK_THRESHOLD} 次将被踢出"
            )
            await send_with_auto_delete(client, chat_id, monitor_msg)

            logger.info(
                f"用户 {user_id} MONITOR 级别违规 ({user_state.monitor_count}/{config.MONITOR_KICK_THRESHOLD})"
            )

        return True, " | ".join(actions_taken)

    except Exception as e:
        error_msg = f"执行操作时出错: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


def format_debug_report(detection_result, is_edited: bool = False) -> str:
    """
    格式化调试报告
    严格按照 MVP.md 的标准格式

    参数:
        detection_result: DetectionResult 对象
        is_edited: 是否为编辑后的消息

    返回:
        格式化的调试报告字符串
    """
    user = detection_result.user_state
    msg = detection_result.message_info
    rule = detection_result.rule_result
    ml = detection_result.ml_result

    # 构建报告
    edited_tag = " [Edited]" if is_edited else ""
    lines = [f"Anti-Ad Debug Report{edited_tag}\n"]

    # 用户信息
    lines.append("User:")
    lines.append(f"- ID: {user.user_id}")
    if user.username:
        lines.append(f"- Username: @{user.username}")
    lines.append(f"- New member: {'YES' if user.is_new_member else 'NO'}")
    lines.append(f"- Message count: {user.message_count}")
    lines.append(f"- Joined: {user.get_join_time_str()}")
    lines.append("")

    # 消息信息
    lines.append("Message:")
    lines.append(f"- Type: {msg.message_type.value}")
    if msg.text_length > 0:
        lines.append(f"- Length: {msg.text_length}")
    lines.append(f"- Has link: {'YES' if msg.has_link else 'NO'}")
    lines.append(f"- Has mention: {'YES' if msg.has_mention else 'NO'}")
    if msg.is_reply:
        lines.append(f"- Is reply: YES")
        lines.append(f"- External reply: {'YES' if msg.is_external_reply else 'NO'}")
    lines.append("")

    # 规则引擎
    lines.append("Rule Engine:")
    lines.append(f"- External reply: {'YES' if rule.external_reply_hit else 'NO'}")
    lines.append(f"- Contact message: {'YES' if rule.contact_message_hit else 'NO'}")
    lines.append(f"- Image message: {'YES' if rule.image_message_hit else 'NO'}")
    lines.append(f"- Sticker message: {'YES' if rule.sticker_message_hit else 'NO'}")
    lines.append(f"- Bio contains t.me: {'YES' if rule.bio_link_hit else 'NO'}")
    lines.append("")

    # ML 检测
    lines.append("ML Detection:")
    lines.append(f"- Invoked: {'YES' if ml.invoked else 'NO'}")
    if ml.invoked:
        if ml.error:
            lines.append(f"- Error: {ml.error}")
        else:
            lines.append(f"- Result: {ml.result}")
            if ml.confidence is not None:
                lines.append(f"- Confidence: {ml.confidence:.3f}")
            if ml.latency_ms is not None:
                lines.append(f"- Latency: {ml.latency_ms:.0f}ms")
    lines.append("")

    # 最终决策
    lines.append("Final Decision:")
    lines.append(f"- Risk level: {detection_result.risk_level.value}")
    lines.append(f"- Risk score: {detection_result.risk_score:.3f}")
    lines.append(f"- Suggested action: {detection_result.suggested_action}")

    # 显示累计次数（如果有）
    if user.monitor_count > 0 or user.warn_count > 0:
        lines.append("")
        lines.append("Violation History:")
        if user.monitor_count > 0:
            lines.append(
                f"- MONITOR count: {user.monitor_count}/{config.MONITOR_KICK_THRESHOLD}"
            )
        if user.warn_count > 0:
            lines.append(
                f"- WARN count: {user.warn_count}/{config.WARN_KICK_THRESHOLD}"
            )

    report = "\n".join(lines)
    return report


# 创建一个兼容的 Update 对象，用于 detection_engine
class PyrogramUpdateAdapter:
    """适配器：将 Pyrogram Message 转换为类似 python-telegram-bot 的 Update 对象"""

    def __init__(self, message: Message):
        self.message = message


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


async def _process_group_message(
    client: Client, message: Message, is_edited: bool = False
):
    """
    处理群消息的核心逻辑（新消息和编辑消息共用）

    参数:
        client: Pyrogram Client 对象
        message: Pyrogram Message 对象
        is_edited: 是否为编辑后的消息
    """
    if should_skip_group_message(message, is_edited=is_edited):
        return

    # 检查群组白名单
    group_id = message.chat.id
    if not is_group_whitelisted(group_id):
        logger.debug(f"群组 {group_id} 不在白名单中，跳过处理")
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
                    client,
                    chat_id=message.chat.id,
                    user_id=user_id,
                )
                if joined_date:
                    state_manager.update_join_time(user_id, joined_date.timestamp())

        # 检查白名单
        if user_state.is_whitelisted:
            logger.debug(f"跳过白名单用户 {user_id}")
            return

        # 判断是否需要检测
        if config.ONLY_NEW_MEMBER and not user_state.is_new_member:
            # 不是新成员，跳过（MVP 可以选择对所有人回显）
            logger.debug(f"跳过非新成员 {user_id}")
            return

        # 执行完整检测流程
        edited_tag = "[已编辑] " if is_edited else ""
        message_text = format_message_text(message)

        user_bio = None
        if user_state.is_new_member:
            user_bio = await try_get_user_bio(client, user_id)
            if user_bio:
                logger.debug(
                    "用户简介检测启用 | user_id=%s, has_t_me=%s",
                    user_id,
                    "t.me" in user_bio.lower(),
                )

        logger.info(
            "%s检测用户 %s (新成员=%s) 的消息 | chat_id=%s, message_id=%s, text=%r, bio=%r",
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

        # 创建兼容的 Update 对象
        update_adapter = PyrogramUpdateAdapter(message)
        detection_result = detection_engine.analyze(
            update_adapter, user_state, user_bio=user_bio
        )

        # 执行操作（如果不是 DRY_RUN 模式）
        action_executed = False
        action_description = ""

        if not config.DRY_RUN:
            # 执行实际操作
            action_executed, action_description = await execute_action(
                client, message, detection_result, state_manager
            )

            logger.info(
                f"用户 {user_id}: 风险={detection_result.risk_level.value}, "
                f"分数={detection_result.risk_score:.2f}, "
                f"操作={detection_result.suggested_action}, "
                f"执行={action_executed}"
            )

        # 是否回复调试信息
        if config.DEBUG_REPLY:
            # 生成报告
            report = format_debug_report(detection_result, is_edited=is_edited)

            # 添加执行信息
            if action_executed and action_description:
                report += f"\n\n🔧 Action Executed:\n{action_description}"

            await send_debug_report(client, message, report)

        # 自动保存用户状态
        state_manager.auto_save_if_needed()

    except Exception as e:
        logger.error(f"处理消息时出错: {e}", exc_info=True)


@app.on_message(filters.group & SUPPORTED_CONTENT_FILTER, group=1)
async def handle_message(client: Client, message: Message):
    """
    处理所有新群消息
    """
    await _process_group_message(client, message, is_edited=False)


@app.on_edited_message(filters.group & SUPPORTED_CONTENT_FILTER, group=1)
async def handle_edited_message(client: Client, message: Message):
    """
    处理已编辑的群消息
    """
    logger.info(
        f"检测到消息编辑 | chat_id={message.chat.id}, message_id={message.id}, "
        f"user_id={message.from_user.id if message.from_user else 'N/A'}"
    )
    await _process_group_message(client, message, is_edited=True)


@app.on_message(filters.group & filters.new_chat_members)
async def handle_new_member(client: Client, message: Message):
    """
    处理新成员加入事件

    更新用户的真实加入时间
    """
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

        # 创建或更新用户状态
        user_state = state_manager.get_or_create(user_id, username)
        state_manager.update_join_time(user_id, join_time)

        logger.info(f"新成员加入: {user_id} (@{username})")


@app.on_message(filters.command("start"))
async def cmd_start(client: Client, message: Message):
    """处理 /start 命令"""
    if is_duplicate_message(message):
        return
    await reply_with_auto_delete(
        message,
        "🛡 广告检测 Bot\n\n"
        "本 Bot 目前仍在完善：\n"
        f"- 干运行: {config.DRY_RUN}\n"
        f"- 仅新成员: {config.ONLY_NEW_MEMBER}\n"
        f"- 调试回复: {config.DEBUG_REPLY}\n\n"
    )


@app.on_message(filters.command("stats"))
async def cmd_stats(client: Client, message: Message):
    """处理 /stats 命令 - 显示统计信息"""
    if is_duplicate_message(message):
        return
    state_manager = get_state_manager()
    stats = state_manager.get_stats()

    # 计算违规统计
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
        f"违规统计:\n"
        f"  - 有违规记录: {users_with_violations} 人\n"
        f"  - 总 MONITOR: {total_monitor} 次\n"
        f"  - 总 WARN: {total_warn} 次\n\n"
        f"新成员判定规则:\n"
        f"  - 消息数 ≤ {stats['message_threshold']}\n"
        f"  - 加入时间 < {stats['time_threshold_days']:.1f}天\n\n"
        f"执行模式:\n"
        f"  - DRY_RUN: {config.DRY_RUN}\n"
        f"  - MONITOR 删除: {config.MONITOR_DELETE_MESSAGE}\n"
        f"  - MONITOR 踢人阈值: {config.MONITOR_KICK_THRESHOLD}\n"
        f"  - WARN 踢人阈值: {config.WARN_KICK_THRESHOLD}"
    )

    await reply_with_auto_delete(message, msg)


@app.on_message(filters.command("config"))
async def cmd_config(client: Client, message: Message):
    """处理 /config 命令 - 显示配置"""
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
async def cmd_save(client: Client, message: Message):
    """处理 /save 命令 - 手动保存状态"""
    if is_duplicate_message(message):
        return
    state_manager = get_state_manager()
    state_manager.save()
    await reply_with_auto_delete(message, "✅ 用户状态已保存")


@app.on_message(filters.command("get_group_id") & filters.group)
async def cmd_get_group_id(client: Client, message: Message):
    """处理 /get_group_id 命令 - 获取当前群组 ID"""
    if is_duplicate_message(message):
        return
    group_id = message.chat.id
    group_title = message.chat.title or "未知群组"

    # 检查是否在白名单中
    in_whitelist = is_group_whitelisted(group_id)
    status = "✅ 已在白名单" if in_whitelist else "⚠️ 未在白名单"

    msg = (
        f"📋 群组信息\n\n"
        f"群组名称: {group_title}\n"
        f"群组 ID: `{group_id}`\n"
        f"状态: {status}\n\n"
        f"💡 如需添加到白名单，请通知 Olivi 编辑 {config.GROUP_WHITELIST_FILE} 文件"
    )
    await reply_with_auto_delete(message, msg, 15)


@app.on_message(filters.command("get_user_id"))
async def cmd_get_user_id(client: Client, message: Message):
    """处理 /get_user_id 命令 - 获取用户 ID"""
    if is_duplicate_message(message):
        return
    # 如果是回复消息，获取被回复用户的 ID
    if message.reply_to_message and message.reply_to_message.from_user:
        target_user = message.reply_to_message.from_user
        target_user_id = target_user.id
        target_username = target_user.username or "无"
        target_firstname = target_user.first_name or "无"
        joined_group_date = await format_joined_date_str(
            client,
            chat_id=message.chat.id,
            user_id=target_user_id,
        )

        msg = (
            f"👤 用户信息\n\n"
            f"用户 ID: `{target_user_id}`\n"
            f"用户名: @{target_username}\n"
            f"昵称: {target_firstname}\n"
            f"入群时间: {joined_group_date}"
        )
    else:
        # 获取命令发送者的 ID
        user = message.from_user
        user_id = user.id
        username = user.username or "无"
        firstname = user.first_name or "无"
        joined_group_date = await format_joined_date_str(
            client,
            chat_id=message.chat.id,
            user_id=user_id,
        )

        msg = (
            f"👤 您的用户信息\n\n"
            f"用户 ID: `{user_id}`\n"
            f"用户名: @{username}\n"
            f"显示名: {firstname}\n"
            f"入群时间: {joined_group_date}\n\n"
            f"💡 回复某人的消息使用此命令可获取对方的 ID"
        )

    await reply_with_auto_delete(message, msg, 15)


@app.on_message(filters.command("ad_whitelist") & filters.group)
async def cmd_whitelist(client: Client, message: Message):
    """处理 /ad_whitelist 命令 - 管理白名单（仅管理员）

    用法：
        /ad_whitelist add <user_id> - 添加用户到白名单
        /ad_whitelist remove <user_id> - 从白名单移除用户
        /ad_whitelist list - 列出所有白名单用户
    """
    if is_duplicate_message(message):
        return
    # 检查群组是否在白名单中
    group_id = message.chat.id
    if not is_group_whitelisted(group_id):
        await reply_with_auto_delete(
            message, "❌ 此群组未在白名单中\n请联系 Olivi 将群组添加到白名单"
        )
        return

    # 检查管理员权限
    try:
        chat_member = await client.get_chat_member(
            chat_id=message.chat.id, user_id=message.from_user.id
        )
    except RPCError as e:
        logger.error(f"检查管理员权限失败: {e}")
        await reply_with_auto_delete(message, "❌ 无法验证权限")
        return

    if not is_admin_status(getattr(chat_member, "status", None)):
        await reply_with_auto_delete(message, "❌ 只有管理员可以使用此命令")
        return

    state_manager = get_state_manager()

    # 解析命令参数
    parts = message.text.split()
    args = parts[1:] if len(parts) > 1 else []

    # 无参数或帮助
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

    # 列出白名单
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

    # 添加或移除操作
    if action not in ["add", "remove"]:
        await reply_with_auto_delete(
            message, "❌ 无效操作，使用 /ad_whitelist help 查看帮助"
        )
        return

    # 确定目标用户
    target_user_id = None
    target_username = None

    # 如果是回复消息
    if message.reply_to_message and message.reply_to_message.from_user:
        target_user_id = message.reply_to_message.from_user.id
        target_username = message.reply_to_message.from_user.username
    # 如果提供了 user_id
    elif len(args) > 1:
        try:
            target_user_id = int(args[1])
        except ValueError:
            await reply_with_auto_delete(message, "❌ 无效的 user_id")
            return
    else:
        await reply_with_auto_delete(message, "❌ 请提供 user_id 或回复用户消息")
        return

    # 执行操作
    if action == "add":
        success = state_manager.add_to_whitelist(target_user_id)
        if success:
            username_str = (
                f"@{target_username}" if target_username else str(target_user_id)
            )
            msg = f"✅ 已将 {username_str} 添加到白名单"
            logger.info(
                f"管理员 {message.from_user.id} 将用户 {target_user_id} 添加到白名单"
            )
        else:
            msg = f"ℹ️ 用户已在白名单中"
    else:  # remove
        success = state_manager.remove_from_whitelist(target_user_id)
        if success:
            username_str = (
                f"@{target_username}" if target_username else str(target_user_id)
            )
            msg = f"✅ 已将 {username_str} 从白名单移除"
            logger.info(
                f"管理员 {message.from_user.id} 将用户 {target_user_id} 从白名单移除"
            )
        else:
            msg = f"ℹ️ 用户不在白名单中"

    await reply_with_auto_delete(message, msg)


def main():
    """Bot 主函数"""
    print("=" * 60)
    print("广告检测 Bot (MVP - Kurigram/Pyrogram)")
    print("=" * 60)

    load_group_whitelist()

    print_config()

    valid, msg = validate_config()
    print(msg)

    if not valid:
        print("\n请先配置 Telegram 凭据（Pyrogram 必需）:")
        print("  export TELEGRAM_BOT_TOKEN='your_bot_token_here'")
        print("  export TELEGRAM_API_ID='123456'")
        print("  export TELEGRAM_API_HASH='your_api_hash_here'")
        sys.exit(1)

    print("\n正在启动 Bot...")
    print("✓ Bot 启动成功！")
    print("\n等待消息...")
    print("按 Ctrl+C 停止\n")
    print("=" * 60)

    app.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Bot 已停止")
        # 保存状态
        state_manager = get_state_manager()
        state_manager.save()
        print("✓ 用户状态已保存")
