"""
用户状态管理模块
跟踪群成员的状态信息，用于新成员判定
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from datetime import datetime, timezone
import os


@dataclass
class UserState:
    """单个用户的状态信息"""

    user_id: int
    username: Optional[str]
    join_time: float  # Unix 时间戳
    message_count: int
    join_time_verified: bool = (
        False  # join_time 是否来自真实入群信息（join 事件 / get_chat_member）
    )
    first_message_time: Optional[float] = None
    last_message_time: Optional[float] = None
    # 违规计数
    monitor_count: int = 0  # MONITOR 级别违规次数
    warn_count: int = 0  # WARN 级别违规次数
    is_whitelisted: bool = False  # 是否在白名单中

    @property
    def is_new_member(self) -> bool:
        """
        判断是否为新成员
        规则：消息数 <= 15 或 加入时间 < 7天
        """
        from bot_config import config

        # 如果 join_time 已经被真实入群时间校准，则以“入群时间”为准，
        # 避免 bot 重启/首次见到导致老成员被误判为新成员。
        if self.join_time_verified:
            time_since_join = time.time() - self.join_time
            return time_since_join < config.NEW_MEMBER_TIME_THRESHOLD

        # 规则1: 消息数判定
        if self.message_count <= config.NEW_MEMBER_MESSAGE_THRESHOLD:
            return True

        # 规则2: 时间判定
        time_since_join = time.time() - self.join_time
        if time_since_join < config.NEW_MEMBER_TIME_THRESHOLD:
            return True

        return False

    def get_join_time_str(self) -> str:
        """获取友好的加入时间字符串"""
        elapsed = time.time() - self.join_time

        if elapsed < 60:
            return f"{int(elapsed)}s ago"
        elif elapsed < 3600:
            return f"{int(elapsed/60)}m ago"
        elif elapsed < 86400:
            return f"{int(elapsed/3600)}h ago"
        else:
            return f"{int(elapsed/86400)}d ago"

    def increment_message(self):
        """增加消息计数"""
        self.message_count += 1
        current_time = time.time()

        if self.first_message_time is None:
            self.first_message_time = current_time

        self.last_message_time = current_time

    def increment_monitor(self):
        """增加 MONITOR 违规计数"""
        self.monitor_count += 1

    def increment_warn(self):
        """增加 WARN 违规计数"""
        self.warn_count += 1

    def should_kick(self) -> tuple[bool, str]:
        """判断是否应该踢出用户

        返回:
            (是否踢出, 原因)
        """
        from bot_config import config

        if self.warn_count >= config.WARN_KICK_THRESHOLD:
            return True, f"WARN violations: {self.warn_count}"

        if self.monitor_count >= config.MONITOR_KICK_THRESHOLD:
            return True, f"MONITOR violations: {self.monitor_count}"

        return False, ""


class UserStateManager:
    """用户状态管理器"""

    def __init__(self, storage_file: str = "bot_user_states.json"):
        """
        初始化用户状态管理器

        参数:
            storage_file: 状态持久化文件路径
        """
        self.storage_file = storage_file
        self.users: Dict[int, UserState] = {}
        self.last_save_time = time.time()

        # 尝试加载已有数据
        self.load()

    def get_or_create(self, user_id: int, username: Optional[str] = None) -> UserState:
        """
        获取或创建用户状态

        参数:
            user_id: Telegram 用户 ID
            username: 用户名（可选）

        返回:
            UserState 对象
        """
        if user_id not in self.users:
            # 创建新用户状态
            self.users[user_id] = UserState(
                user_id=user_id,
                username=username,
                join_time=time.time(),  # 首次见到即记录为加入时间
                message_count=0,
            )
        else:
            # 更新用户名（如果变化）
            if username and self.users[user_id].username != username:
                self.users[user_id].username = username

        return self.users[user_id]

    def update_join_time(self, user_id: int, join_timestamp: float):
        """
        更新用户加入时间（当真实获取到 join 事件时）

        参数:
            user_id: 用户 ID
            join_timestamp: 加入时间戳
        """
        if user_id not in self.users:
            return

        user_state = self.users[user_id]

        # 第一次获得“真实入群时间”时：直接写入并标记为已校准。
        if not user_state.join_time_verified:
            user_state.join_time = join_timestamp
            user_state.join_time_verified = True
            return

        # 已校准过：只在拿到更早的时间戳时更新（更可信的最早入群时间）。
        if join_timestamp < user_state.join_time:
            user_state.join_time = join_timestamp

    def record_message(self, user_id: int, username: Optional[str] = None) -> UserState:
        """
        记录用户发送消息

        参数:
            user_id: 用户 ID
            username: 用户名（可选）

        返回:
            更新后的 UserState
        """
        user_state = self.get_or_create(user_id, username)
        user_state.increment_message()
        return user_state

    def get_state(self, user_id: int) -> Optional[UserState]:
        """
        获取用户状态（不创建新状态）

        参数:
            user_id: 用户 ID

        返回:
            UserState 或 None
        """
        return self.users.get(user_id)

    def save(self):
        """将用户状态持久化到文件"""
        try:
            data = {
                str(user_id): asdict(state) for user_id, state in self.users.items()
            }

            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.last_save_time = time.time()

        except Exception as e:
            print(f"⚠️  保存用户状态失败: {e}")

    def load(self):
        """从文件加载用户状态"""
        if not os.path.exists(self.storage_file):
            print(f"ℹ️  状态文件不存在，将创建新文件: {self.storage_file}")
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for user_id_str, state_dict in data.items():
                user_id = int(user_id_str)
                # 兼容旧状态文件：早期版本没有 join_time_verified 字段
                if "join_time_verified" not in state_dict:
                    join_time = state_dict.get("join_time")
                    first_message_time = state_dict.get("first_message_time")
                    # 如果 join_time 明显早于首次发言时间，通常意味着 join_time 来自 join 事件/接口而非“首次见到”
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

        except Exception as e:
            print(f"⚠️  加载用户状态失败: {e}")

    def auto_save_if_needed(self):
        """根据配置自动保存（避免频繁 IO）"""
        from bot_config import config

        if not config.AUTO_SAVE_USER_STATE:
            return

        elapsed = time.time() - self.last_save_time
        if elapsed >= config.AUTO_SAVE_INTERVAL:
            self.save()

    def add_to_whitelist(self, user_id: int) -> bool:
        """将用户添加到白名单

        参数:
            user_id: 用户 ID

        返回:
            是否成功（如果用户已在白名单中返回 False）
        """
        user_state = self.get_or_create(user_id)
        if user_state.is_whitelisted:
            return False
        user_state.is_whitelisted = True
        self.save()
        return True

    def remove_from_whitelist(self, user_id: int) -> bool:
        """从白名单中移除用户

        参数:
            user_id: 用户 ID

        返回:
            是否成功（如果用户不在白名单中返回 False）
        """
        user_state = self.get_state(user_id)
        if not user_state or not user_state.is_whitelisted:
            return False
        user_state.is_whitelisted = False
        self.save()
        return True

    def is_user_whitelisted(self, user_id: int) -> bool:
        """检查用户是否在白名单中

        参数:
            user_id: 用户 ID

        返回:
            是否在白名单中
        """
        user_state = self.get_state(user_id)
        return user_state.is_whitelisted if user_state else False

    def get_stats(self) -> dict:
        """获取统计信息（调试用）"""
        from bot_config import config

        new_members = sum(1 for u in self.users.values() if u.is_new_member)
        whitelisted = sum(1 for u in self.users.values() if u.is_whitelisted)

        return {
            "total_users": len(self.users),
            "new_members": new_members,
            "old_members": len(self.users) - new_members,
            "whitelisted_users": whitelisted,
            "message_threshold": config.NEW_MEMBER_MESSAGE_THRESHOLD,
            "time_threshold_days": config.NEW_MEMBER_TIME_THRESHOLD / 86400,
        }


# 全局单例
_state_manager: Optional[UserStateManager] = None


def get_state_manager() -> UserStateManager:
    """获取全局用户状态管理器单例"""
    global _state_manager
    if _state_manager is None:
        from bot_config import config

        _state_manager = UserStateManager(config.USER_STATE_FILE)
    return _state_manager


if __name__ == "__main__":
    # 测试代码
    manager = UserStateManager("test_states.json")

    # 模拟用户
    user1 = manager.record_message(123456, "test_user")
    print(f"用户1 是否新成员: {user1.is_new_member}")
    print(f"用户1 消息数: {user1.message_count}")

    # 增加消息
    for _ in range(20):
        manager.record_message(123456, "test_user")

    user1 = manager.get_state(123456)
    print(f"用户1 消息数（更新后）: {user1.message_count}")
    print(f"用户1 是否新成员: {user1.is_new_member}")

    # 保存
    manager.save()
    print("✓ 测试完成")
