"""Group whitelist persistence helpers."""

from __future__ import annotations

import json
import logging
import os

from ..config import config

logger = logging.getLogger(__name__)

_group_whitelist: set[int] = set()


def load_group_whitelist() -> None:
    """Load the group whitelist from disk."""
    global _group_whitelist
    if not os.path.exists(config.GROUP_WHITELIST_FILE):
        _group_whitelist = set()
        save_group_whitelist_file()
        logger.info("已创建默认群组白名单文件")
        return

    try:
        with open(config.GROUP_WHITELIST_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("加载群组白名单失败: %s", exc)
        _group_whitelist = set()
        return

    _group_whitelist = set(data.get("groups", []))
    logger.info("已加载 %s 个白名单群组", len(_group_whitelist))


def save_group_whitelist_file() -> None:
    """Persist the group whitelist to disk."""
    payload = {
        "groups": list(_group_whitelist),
        "description": "群组白名单配置文件。只有在此列表中的群组才能使用 bot 功能。",
        "usage": "手动编辑此文件，在 groups 数组中添加群组 ID（负数）",
    }
    try:
        with open(config.GROUP_WHITELIST_FILE, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.error("保存群组白名单失败: %s", exc)


def is_group_whitelisted(group_id: int) -> bool:
    """Check whether a group is whitelisted.

    Args:
        group_id: Telegram group ID.

    Returns:
        True if the group is allowed.
    """
    if not _group_whitelist:
        logger.debug("群组白名单为空，允许所有群组使用（请配置 group_whitelist.json）")
        return True
    return group_id in _group_whitelist
