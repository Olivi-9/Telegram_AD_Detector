"""User state tracking for moderation decisions."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

from .config import config


@dataclass
class UserState:
    """State for a single user.

    Attributes:
        user_id: Telegram user ID.
        username: Telegram username.
        join_time: Unix timestamp when the user was first seen or joined.
        message_count: Total number of messages observed.
        join_time_verified: Whether join_time is a verified join timestamp.
        first_message_time: Timestamp of the first observed message.
        last_message_time: Timestamp of the last observed message.
        monitor_count: MONITOR violation count.
        warn_count: WARN violation count.
        is_whitelisted: Whether the user is whitelisted.
    """

    user_id: int
    username: Optional[str]
    join_time: float
    message_count: int
    join_time_verified: bool = False
    first_message_time: Optional[float] = None
    last_message_time: Optional[float] = None
    monitor_count: int = 0
    warn_count: int = 0
    is_whitelisted: bool = False

    @property
    def is_new_member(self) -> bool:
        """Check whether the user is considered a new member.

        Returns:
            True if the user is considered new.
        """
        if self.join_time_verified:
            time_since_join = time.time() - self.join_time
            return time_since_join < config.NEW_MEMBER_TIME_THRESHOLD

        if self.message_count <= config.NEW_MEMBER_MESSAGE_THRESHOLD:
            return True

        time_since_join = time.time() - self.join_time
        if time_since_join < config.NEW_MEMBER_TIME_THRESHOLD:
            return True

        return False

    def get_join_time_str(self) -> str:
        """Get a human-friendly join-time string.

        Returns:
            A compact relative time string.
        """
        elapsed = time.time() - self.join_time

        if elapsed < 60:
            return f"{int(elapsed)}s ago"
        if elapsed < 3600:
            return f"{int(elapsed / 60)}m ago"
        if elapsed < 86400:
            return f"{int(elapsed / 3600)}h ago"
        return f"{int(elapsed / 86400)}d ago"

    def increment_message(self) -> None:
        """Increment message counters and timestamps."""
        self.message_count += 1
        current_time = time.time()

        if self.first_message_time is None:
            self.first_message_time = current_time

        self.last_message_time = current_time

    def increment_monitor(self) -> None:
        """Increment the MONITOR violation counter."""
        self.monitor_count += 1

    def increment_warn(self) -> None:
        """Increment the WARN violation counter."""
        self.warn_count += 1

    def should_kick(self) -> tuple[bool, str]:
        """Determine whether the user should be kicked.

        Returns:
            Tuple of (should_kick, reason).
        """
        if self.warn_count >= config.WARN_KICK_THRESHOLD:
            return True, f"WARN violations: {self.warn_count}"

        if self.monitor_count >= config.MONITOR_KICK_THRESHOLD:
            return True, f"MONITOR violations: {self.monitor_count}"

        return False, ""


class UserStateManager:
    """Manage user states with persistence."""

    def __init__(self, storage_file: str = "bot_user_states.json") -> None:
        """Initialize the state manager.

        Args:
            storage_file: Path to the persistence file.
        """
        self.storage_file = storage_file
        self.users: dict[int, UserState] = {}
        self.last_save_time = time.time()
        self.load()

    def get_or_create(self, user_id: int, username: Optional[str] = None) -> UserState:
        """Get an existing user state or create a new one.

        Args:
            user_id: Telegram user ID.
            username: Optional username.

        Returns:
            UserState instance.
        """
        if user_id not in self.users:
            self.users[user_id] = UserState(
                user_id=user_id,
                username=username,
                join_time=time.time(),
                message_count=0,
            )
        else:
            if username and self.users[user_id].username != username:
                self.users[user_id].username = username

        return self.users[user_id]

    def update_join_time(self, user_id: int, join_timestamp: float) -> None:
        """Update verified join time for a user.

        Args:
            user_id: Telegram user ID.
            join_timestamp: Join timestamp.
        """
        if user_id not in self.users:
            return

        user_state = self.users[user_id]

        if not user_state.join_time_verified:
            user_state.join_time = join_timestamp
            user_state.join_time_verified = True
            return

        if join_timestamp < user_state.join_time:
            user_state.join_time = join_timestamp

    def record_message(self, user_id: int, username: Optional[str] = None) -> UserState:
        """Record a message for the user.

        Args:
            user_id: Telegram user ID.
            username: Optional username.

        Returns:
            Updated user state.
        """
        user_state = self.get_or_create(user_id, username)
        user_state.increment_message()
        return user_state

    def get_state(self, user_id: int) -> Optional[UserState]:
        """Get a user state without creating a new one.

        Args:
            user_id: Telegram user ID.

        Returns:
            UserState or None.
        """
        return self.users.get(user_id)

    def save(self) -> None:
        """Persist user states to disk."""
        try:
            data = {
                str(user_id): asdict(state) for user_id, state in self.users.items()
            }

            with open(self.storage_file, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)

            self.last_save_time = time.time()

        except Exception as exc:
            print(f"⚠️  保存用户状态失败: {exc}")

    def load(self) -> None:
        """Load user states from disk."""
        if not os.path.exists(self.storage_file):
            print(f"ℹ️  状态文件不存在，将创建新文件: {self.storage_file}")
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            for user_id_str, state_dict in data.items():
                user_id = int(user_id_str)
                if "join_time_verified" not in state_dict:
                    join_time = state_dict.get("join_time")
                    first_message_time = state_dict.get("first_message_time")
                    if (
                        isinstance(join_time, (int, float))
                        and isinstance(first_message_time, (int, float))
                        and join_time < first_message_time - 60
                    ):
                        state_dict["join_time_verified"] = True
                    else:
                        state_dict["join_time_verified"] = False

                self.users[user_id] = UserState(**state_dict)

            print(f"✓ 加载了 {len(self.users)} 个用户状态")

        except Exception as exc:
            print(f"⚠️  加载用户状态失败: {exc}")

    def auto_save_if_needed(self) -> None:
        """Auto-save based on the configured interval."""
        if not config.AUTO_SAVE_USER_STATE:
            return

        elapsed = time.time() - self.last_save_time
        if elapsed >= config.AUTO_SAVE_INTERVAL:
            self.save()

    def add_to_whitelist(self, user_id: int) -> bool:
        """Add a user to the whitelist.

        Args:
            user_id: Telegram user ID.

        Returns:
            True if added, False if already whitelisted.
        """
        user_state = self.get_or_create(user_id)
        if user_state.is_whitelisted:
            return False
        user_state.is_whitelisted = True
        self.save()
        return True

    def remove_from_whitelist(self, user_id: int) -> bool:
        """Remove a user from the whitelist.

        Args:
            user_id: Telegram user ID.

        Returns:
            True if removed, False if not whitelisted.
        """
        user_state = self.get_state(user_id)
        if not user_state or not user_state.is_whitelisted:
            return False
        user_state.is_whitelisted = False
        self.save()
        return True

    def is_user_whitelisted(self, user_id: int) -> bool:
        """Check whether the user is whitelisted.

        Args:
            user_id: Telegram user ID.

        Returns:
            True if whitelisted.
        """
        user_state = self.get_state(user_id)
        return user_state.is_whitelisted if user_state else False

    def get_stats(self) -> dict[str, int | float]:
        """Return aggregate statistics for debugging.

        Returns:
            Dictionary of stats.
        """
        new_members = sum(1 for user in self.users.values() if user.is_new_member)
        whitelisted = sum(1 for user in self.users.values() if user.is_whitelisted)

        return {
            "total_users": len(self.users),
            "new_members": new_members,
            "old_members": len(self.users) - new_members,
            "whitelisted_users": whitelisted,
            "message_threshold": config.NEW_MEMBER_MESSAGE_THRESHOLD,
            "time_threshold_days": config.NEW_MEMBER_TIME_THRESHOLD / 86400,
        }


_state_manager: Optional[UserStateManager] = None


def get_state_manager() -> UserStateManager:
    """Get the global user state manager.

    Returns:
        UserStateManager singleton.
    """
    global _state_manager
    if _state_manager is None:
        _state_manager = UserStateManager(config.USER_STATE_FILE)
    return _state_manager


if __name__ == "__main__":
    manager = UserStateManager("test_states.json")
    user1 = manager.record_message(123456, "test_user")
    print(f"用户1 是否新成员: {user1.is_new_member}")
    print(f"用户1 消息数: {user1.message_count}")

    for _ in range(20):
        manager.record_message(123456, "test_user")

    user1 = manager.get_state(123456)
    print(f"用户1 消息数（更新后）: {user1.message_count}")
    print(f"用户1 是否新成员: {user1.is_new_member}")

    manager.save()
    print("✓ 测试完成")
