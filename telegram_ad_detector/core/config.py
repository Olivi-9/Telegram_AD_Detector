"""Bot configuration and environment loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


from dotenv import find_dotenv, load_dotenv

_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)


def _get_first_env(var_names: Iterable[str]) -> str:
    """Return the first non-empty environment variable value.

    Args:
        var_names: Candidate environment variable names.

    Returns:
        The first non-empty value or an empty string.
    """
    for name in var_names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return ""


def _get_env_int(var_names: Iterable[str], default: int = 0) -> int:
    """Read an integer from environment variables.

    Args:
        var_names: Candidate environment variable names.
        default: Fallback value.

    Returns:
        Parsed integer or the default.
    """
    raw = _get_first_env(var_names)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class BotConfig:
    """Bot configuration values."""

    # ============ Telegram Bot configuration ============
    BOT_TOKEN: str = _get_first_env(["TELEGRAM_BOT_TOKEN", "BOT_TOKEN"])
    API_ID: int = _get_env_int(["TELEGRAM_API_ID", "API_ID"], default=0)
    API_HASH: str = _get_first_env(["TELEGRAM_API_HASH", "API_HASH"])

    # ============ MVP debug switches ============
    DEBUG_REPLY: bool = False
    ONLY_NEW_MEMBER: bool = True
    DRY_RUN: bool = False

    # ============ New member rules ============
    NEW_MEMBER_MESSAGE_THRESHOLD: int = 5
    NEW_MEMBER_TIME_THRESHOLD: int = 5 * 24 * 60 * 60  # 5 days

    # ============ Behavior rule toggles ============
    CHECK_EXTERNAL_REPLY: bool = True
    CHECK_CONTACT_MESSAGE: bool = True
    CHECK_IMAGE_FROM_NEW_MEMBER: bool = True
    CHECK_STICKER_FROM_NEW_MEMBER: bool = True
    USE_ML_MODEL: bool = True

    # ============ ML model configuration ============
    MODEL_PATH: str = "ad_model.pkl"
    TFIDF_PATH: str = "tfidf_vectorizer.pkl"
    EMBEDDING_PATH: str = "embedding_model.pkl"
    ML_CONFIDENCE_THRESHOLD: float = 0.7

    # ============ Risk thresholds ============
    RISK_SCORE_HIGH: float = 0.7
    RISK_SCORE_MEDIUM: float = 0.5

    # ============ Execution strategy ============
    MONITOR_DELETE_MESSAGE: bool = True
    MONITOR_KICK_THRESHOLD: int = 3
    WARN_DELETE_MESSAGE: bool = True
    WARN_KICK_THRESHOLD: int = 2
    HIGH_IMMEDIATE_KICK: bool = True

    # ============ Persistence ============
    USER_STATE_FILE: str = "bot_user_states.json"
    GROUP_WHITELIST_FILE: str = "group_whitelist.json"
    AUTO_SAVE_USER_STATE: bool = True
    AUTO_SAVE_INTERVAL: int = 300

    # ============ Logging ============
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "bot_debug.log"

    # ============ Other ============
    MAX_REPLY_LENGTH: int = 4000
    USE_REPLY: bool = True


config = BotConfig()


def validate_config() -> tuple[bool, str]:
    """Validate configuration values.

    Returns:
        Tuple of (is_valid, message).
    """
    if not config.BOT_TOKEN:
        return False, "❌ 错误：未设置 TELEGRAM_BOT_TOKEN 环境变量"

    if not config.API_ID or config.API_ID <= 0:
        return (
            False,
            "❌ 错误：未设置 TELEGRAM_API_ID（必须是正整数）。\n"
            "请在 https://my.telegram.org 获取 api_id/api_hash 后导出：\n"
            "  export TELEGRAM_API_ID='123456'\n"
            "  export TELEGRAM_API_HASH='abcdef123456...'",
        )
    if not config.API_HASH:
        return (
            False,
            "❌ 错误：未设置 TELEGRAM_API_HASH。\n"
            "请在 https://my.telegram.org 获取 api_id/api_hash 后导出：\n"
            "  export TELEGRAM_API_HASH='abcdef123456...'",
        )

    if config.USE_ML_MODEL:
        if not os.path.exists(config.MODEL_PATH):
            return False, f"❌ 错误：找不到模型文件 {config.MODEL_PATH}"
        if not os.path.exists(config.TFIDF_PATH):
            return False, f"❌ 错误：找不到 TF-IDF 文件 {config.TFIDF_PATH}"
        if not os.path.exists(config.EMBEDDING_PATH):
            return False, f"❌ 错误：找不到 Embedding 文件 {config.EMBEDDING_PATH}"

    return True, "✓ 配置验证通过"


def print_config() -> None:
    """Print configuration values for debugging."""
    print("=" * 60)
    print("Bot 配置信息")
    print("=" * 60)
    print(f"API_ID: {config.API_ID if config.API_ID else '(未设置)'}")
    if config.API_HASH:
        print("API_HASH: (已设置)")
    else:
        print("API_HASH: (未设置)")
    print(f"BOT_TOKEN: {'(已设置)' if config.BOT_TOKEN else '(未设置)'}")
    print(f"调试回复: {config.DEBUG_REPLY}")
    print(f"仅新成员: {config.ONLY_NEW_MEMBER}")
    print(f"干运行模式: {config.DRY_RUN}")
    print(f"新成员消息阈值: {config.NEW_MEMBER_MESSAGE_THRESHOLD}")
    print(f"新成员时间阈值: {config.NEW_MEMBER_TIME_THRESHOLD}秒")
    print(f"使用 ML 模型: {config.USE_ML_MODEL}")
    print(f"ML 置信度阈值: {config.ML_CONFIDENCE_THRESHOLD}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    valid, msg = validate_config()
    print(msg)
