"""Formatting helpers for bot output."""

from __future__ import annotations

from typing import Optional

from pyrogram.types import Message

from ..config import config
from ..detection.types import DetectionResult
from ...utils.text import compact_text, truncate_text


def format_message_text(message: Message, max_len: int = 200) -> str:
    """Format message content for logging.

    Args:
        message: Pyrogram message object.
        max_len: Maximum length of the formatted text.

    Returns:
        Formatted message text.
    """
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

    compacted = compact_text(text)
    return truncate_text(compacted, max_len)


def format_bio_for_log(user_bio: Optional[str], max_len: int = 180) -> str:
    """Format user bio for logging.

    Args:
        user_bio: User bio text.
        max_len: Maximum length of the formatted text.

    Returns:
        Formatted bio string.
    """
    if not user_bio:
        return "(none)"

    compacted = compact_text(str(user_bio))
    return truncate_text(compacted, max_len)


def describe_message_payload(message: Message, user_bio: Optional[str] = None) -> str:
    """Describe a message payload for debugging.

    Args:
        message: Pyrogram message.
        user_bio: Optional user bio.

    Returns:
        A debug string describing the payload.
    """
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

    return ", ".join([f"{key}={value}" for key, value in details.items()])


def format_debug_report(
    detection_result: DetectionResult, is_edited: bool = False
) -> str:
    """Format a debug report for the detection result.

    Args:
        detection_result: Full detection result.
        is_edited: Whether the message was edited.

    Returns:
        Debug report string.
    """
    user = detection_result.user_state
    msg = detection_result.message_info
    rule = detection_result.rule_result
    ml = detection_result.ml_result

    edited_tag = " [Edited]" if is_edited else ""
    lines = [f"Anti-Ad Debug Report{edited_tag}\n"]

    lines.append("User:")
    lines.append(f"- ID: {user.user_id}")
    if user.username:
        lines.append(f"- Username: @{user.username}")
    lines.append(f"- New member: {'YES' if user.is_new_member else 'NO'}")
    lines.append(f"- Message count: {user.message_count}")
    lines.append(f"- Joined: {user.get_join_time_str()}")
    lines.append("")

    lines.append("Message:")
    lines.append(f"- Type: {msg.message_type.value}")
    if msg.text_length > 0:
        lines.append(f"- Length: {msg.text_length}")
    lines.append(f"- Has link: {'YES' if msg.has_link else 'NO'}")
    lines.append(f"- Has mention: {'YES' if msg.has_mention else 'NO'}")
    if msg.is_reply:
        lines.append("- Is reply: YES")
        lines.append(f"- External reply: {'YES' if msg.is_external_reply else 'NO'}")
    lines.append("")

    lines.append("Rule Engine:")
    lines.append(f"- External reply: {'YES' if rule.external_reply_hit else 'NO'}")
    lines.append(f"- Contact message: {'YES' if rule.contact_message_hit else 'NO'}")
    lines.append(f"- Image message: {'YES' if rule.image_message_hit else 'NO'}")
    lines.append(f"- Sticker message: {'YES' if rule.sticker_message_hit else 'NO'}")
    lines.append(f"- Bio contains t.me: {'YES' if rule.bio_link_hit else 'NO'}")
    lines.append("")

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

    lines.append("Final Decision:")
    lines.append(f"- Risk level: {detection_result.risk_level.value}")
    lines.append(f"- Risk score: {detection_result.risk_score:.3f}")
    lines.append(f"- Suggested action: {detection_result.suggested_action}")

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

    return "\n".join(lines)
