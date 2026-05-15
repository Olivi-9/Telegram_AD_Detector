"""Bot main entrypoint."""

from __future__ import annotations

import sys

from ..config import print_config, validate_config
from ._whitelist import load_group_whitelist
from .app import app


def _register_handlers() -> None:
    """Register Pyrogram handlers."""
    from . import _command_handlers  # noqa: F401
    from . import _message_handlers  # noqa: F401


def main() -> None:
    """Start the Telegram moderation bot."""
    print("=" * 60)
    print("广告检测 Bot (MVP - Kurigram/Pyrogram)")
    print("=" * 60)

    load_group_whitelist()
    _register_handlers()

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
        from ..user_state import get_state_manager

        state_manager = get_state_manager()
        state_manager.save()
        print("✓ 用户状态已保存")
