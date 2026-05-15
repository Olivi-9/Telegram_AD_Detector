"""Detection domain types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..user_state import UserState


class MessageType(Enum):
    """Message type enumeration."""

    TEXT = "TEXT"
    PHOTO = "PHOTO"
    VIDEO = "VIDEO"
    DOCUMENT = "DOCUMENT"
    CONTACT = "CONTACT"
    STICKER = "STICKER"
    VOICE = "VOICE"
    OTHER = "OTHER"


class RiskLevel(Enum):
    """Risk level enumeration."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class MessageInfo:
    """Parsed message information."""

    message_type: MessageType
    text: Optional[str] = None
    has_link: bool = False
    has_mention: bool = False
    is_reply: bool = False
    is_external_reply: bool = False
    text_length: int = 0

    def __str__(self) -> str:
        return (
            f"Type: {self.message_type.value}, "
            f"Length: {self.text_length}, "
            f"Link: {self.has_link}, "
            f"Mention: {self.has_mention}"
        )


@dataclass
class RuleResult:
    """Rule-based detection result."""

    external_reply_hit: bool = False
    contact_message_hit: bool = False
    image_message_hit: bool = False
    sticker_message_hit: bool = False
    bio_link_hit: bool = False

    def any_hit(self) -> bool:
        """Check whether any rule was triggered."""
        return (
            self.external_reply_hit
            or self.contact_message_hit
            or self.image_message_hit
            or self.sticker_message_hit
            or self.bio_link_hit
        )

    def hit_count(self) -> int:
        """Return the number of triggered rules."""
        return sum(
            [
                self.external_reply_hit,
                self.contact_message_hit,
                self.image_message_hit,
                self.sticker_message_hit,
                self.bio_link_hit,
            ]
        )


@dataclass
class MLResult:
    """ML-based detection result."""

    invoked: bool
    result: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DetectionResult:
    """Full detection result."""

    user_state: UserState
    message_info: MessageInfo
    rule_result: RuleResult
    ml_result: MLResult
    risk_level: RiskLevel
    suggested_action: str
    risk_score: float
