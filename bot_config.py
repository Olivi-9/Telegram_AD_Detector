"""
Bot 配置文件
包含所有可调整的参数和开关
"""

import os
from dataclasses import dataclass


def _load_env_file(env_path: str | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ (no overrides)."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(__file__), ".env")

    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for line in env_file:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if raw.startswith("export "):
                    raw = raw[len("export ") :].strip()
                if "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip()
                os.environ.setdefault(key, value)
    except (OSError, UnicodeDecodeError):
        # Ignore .env loading errors to avoid blocking startup.
        return


_load_env_file()


def _get_first_env(var_names: list[str]) -> str:
    """Return the first non-empty environment variable value."""
    for name in var_names:
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return ""


def _get_env_int(var_names: list[str], default: int = 0) -> int:
    raw = _get_first_env(var_names)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class BotConfig:
    """Bot 配置类"""

    # ============ Telegram Bot 配置 ============
    # 从环境变量读取 Bot Token
    BOT_TOKEN: str = _get_first_env(["TELEGRAM_BOT_TOKEN", "BOT_TOKEN"])
    # Pyrogram 需要 api_id/api_hash（即使是 bot_token 模式也需要）
    API_ID: int = _get_env_int(["TELEGRAM_API_ID", "API_ID"], default=0)
    API_HASH: str = _get_first_env(["TELEGRAM_API_HASH", "API_HASH"])

    # ============ MVP 调试开关 ============
    # 是否回复调试信息（MVP 阶段建议开启）
    DEBUG_REPLY: bool = False

    # 是否只对新成员生效（False = 对所有人都回显）
    ONLY_NEW_MEMBER: bool = True

    # 干运行模式（True = 只分析不执行，False = 实际执行操作）
    DRY_RUN: bool = False

    # ============ 新成员判定规则 ============
    # 消息数量阈值（<= 该值视为新成员）
    NEW_MEMBER_MESSAGE_THRESHOLD: int = 5

    # 加入时间阈值（秒，< 该值视为新成员）
    NEW_MEMBER_TIME_THRESHOLD: int = 5 * 24 * 60 * 60  # 5天

    # ============ 行为规则开关 ============
    # 是否检测外部引用消息
    CHECK_EXTERNAL_REPLY: bool = True

    # 是否检测联系人消息
    CHECK_CONTACT_MESSAGE: bool = True

    # 是否检测图片消息（新成员）
    CHECK_IMAGE_FROM_NEW_MEMBER: bool = True

    # 是否检测贴纸消息（新成员）
    CHECK_STICKER_FROM_NEW_MEMBER: bool = True

    # 是否使用 ML 模型检测文本
    USE_ML_MODEL: bool = True

    # ============ ML 模型配置 ============
    # 模型文件路径
    MODEL_PATH: str = "ad_model.pkl"
    TFIDF_PATH: str = "tfidf_vectorizer.pkl"
    EMBEDDING_PATH: str = "embedding_model.pkl"

    # 模型置信度阈值（>= 该值判定为广告）
    ML_CONFIDENCE_THRESHOLD: float = 0.7

    # ============ 风险等级阈值 ============
    # 风险评分阈值（根据新的分数分布调整）
    RISK_SCORE_HIGH: float = 0.7  # >= 高风险（立即踢人）
    RISK_SCORE_MEDIUM: float = 0.5  # >= 中风险（WARN，高置信度广告）
    # < MEDIUM = 低风险（MONITOR，低置信度广告）
    # = 0 = 无风险（NO_ACTION，正常消息）

    # ============ 执行模式配置 ============
    # MONITOR 级别：删除消息 + 警告，达到阈值踢人
    MONITOR_DELETE_MESSAGE: bool = True
    MONITOR_KICK_THRESHOLD: int = 3

    # WARN 级别：达到阈值踢人
    WARN_DELETE_MESSAGE: bool = True
    WARN_KICK_THRESHOLD: int = 2

    # HIGH 级别：立即踢人
    HIGH_IMMEDIATE_KICK: bool = True

    # ============ 数据持久化 ============
    # 用户状态数据文件
    USER_STATE_FILE: str = "bot_user_states.json"

    # 群组白名单配置文件（只有白名单群组可以使用 bot）
    GROUP_WHITELIST_FILE: str = "group_whitelist.json"

    # 是否自动保存用户状态
    AUTO_SAVE_USER_STATE: bool = True

    # 自动保存间隔（秒）
    AUTO_SAVE_INTERVAL: int = 300  # 5分钟

    # ============ 日志配置 ============
    # 日志级别（DEBUG, INFO, WARNING, ERROR）
    LOG_LEVEL: str = "INFO"

    # 日志文件路径
    LOG_FILE: str = "bot_debug.log"

    # ============ 其他 ============
    # 回复消息最大长度（Telegram 限制 4096）
    MAX_REPLY_LENGTH: int = 4000

    # 是否使用 Reply 方式回复
    USE_REPLY: bool = True


# 全局配置实例
config = BotConfig()


def validate_config() -> tuple[bool, str]:
    """
    验证配置是否合法

    返回:
        (是否合法, 错误信息)
    """
    if not config.BOT_TOKEN:
        return False, "❌ 错误：未设置 TELEGRAM_BOT_TOKEN 环境变量"

    # Pyrogram / Kurigram 必需：API_ID + API_HASH
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


def print_config():
    """打印当前配置（调试用）"""
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
