"""Detection engine orchestration."""

from __future__ import annotations

import logging
from typing import Optional

from ..config import config
from ..user_state import UserState
from .ml import MLDetector
from .parser import MessageParser
from .rules import RuleEngine
from .types import DetectionResult, MLResult, MessageType, RiskLevel, RuleResult

logger = logging.getLogger(__name__)


class DetectionEngine:
    """Orchestrate rule-based and ML-based detection."""

    def __init__(self) -> None:
        self.ml_detector = MLDetector()
        if config.USE_ML_MODEL:
            self.ml_detector.load_model()

    def analyze(
        self, update: object, user_state: UserState, user_bio: Optional[str] = None
    ) -> DetectionResult:
        """Analyze a message and return a detection result.

        Args:
            update: Telegram Update-like object.
            user_state: User state.
            user_bio: Optional user bio.

        Returns:
            DetectionResult.
        """
        message_info = MessageParser.parse(update)
        rule_result = RuleEngine.check(message_info, user_state, user_bio=user_bio)

        ml_result = MLResult(invoked=False)
        if message_info.message_type == MessageType.TEXT and message_info.text:
            ml_result = self.ml_detector.detect(message_info.text)
        else:
            logger.debug(
                "跳过 ML 检测: message_type=%s, has_text=%s",
                message_info.message_type.value,
                bool(message_info.text),
            )

        if ml_result.invoked and ml_result.error:
            logger.warning("ML 检测失败: %s", ml_result.error)
        elif ml_result.invoked and not self._is_ml_ad_result(ml_result.result):
            logger.debug(
                "ML 未判定广告: result=%r, confidence=%r",
                ml_result.result,
                ml_result.confidence,
            )

        risk_score = self._calculate_risk_score(rule_result, ml_result)
        risk_level = self._determine_risk_level(risk_score)
        suggested_action = self._suggest_action(
            risk_level, user_state, rule_result, ml_result, risk_score
        )

        logger.info(
            "检测决策: risk_level=%s, risk_score=%.3f, suggested_action=%s, "
            "ml_invoked=%s, ml_result=%s",
            risk_level.value,
            risk_score,
            suggested_action,
            ml_result.invoked,
            ml_result.result,
        )

        return DetectionResult(
            user_state=user_state,
            message_info=message_info,
            rule_result=rule_result,
            ml_result=ml_result,
            risk_level=risk_level,
            suggested_action=suggested_action,
            risk_score=risk_score,
        )

    def _calculate_risk_score(self, rule_result: RuleResult, ml_result: MLResult) -> float:
        """Compute a risk score in the range [0, 1]."""
        score = 0.0

        if rule_result.external_reply_hit:
            score += 0.85
        if rule_result.contact_message_hit:
            score += 0.85
        if rule_result.image_message_hit:
            score += 0.25
        if rule_result.bio_link_hit:
            score += 0.25

        if ml_result.invoked and self._is_ml_ad_result(ml_result.result):
            if ml_result.confidence:
                score += min(ml_result.confidence, 1.0)
            else:
                score += 0.5

        return min(score, 1.0)

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine the risk level based on the score."""
        if risk_score >= config.RISK_SCORE_HIGH:
            return RiskLevel.HIGH
        if risk_score >= config.RISK_SCORE_MEDIUM:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _suggest_action(
        self,
        risk_level: RiskLevel,
        user_state: UserState,
        rule_result: RuleResult,
        ml_result: MLResult,
        risk_score: float,
    ) -> str:
        """Suggest a moderation action for the detection result."""
        if config.DRY_RUN:
            suffix = " (debug only)"
        else:
            suffix = ""

        has_violation = (
            ml_result.invoked and self._is_ml_ad_result(ml_result.result)
        ) or rule_result.any_hit()

        if risk_level == RiskLevel.HIGH:
            if config.HIGH_IMMEDIATE_KICK:
                return f"KICK_IMMEDIATE{suffix}"
            return f"KICK{suffix}"

        if risk_level == RiskLevel.MEDIUM:
            if user_state.warn_count + 1 >= config.WARN_KICK_THRESHOLD:
                return f"KICK_ON_WARN{suffix}"
            return f"WARN{suffix}"

        if not has_violation:
            return f"NO_ACTION{suffix}"

        if user_state.monitor_count + 1 >= config.MONITOR_KICK_THRESHOLD:
            return f"KICK_ON_MONITOR{suffix}"
        return f"MONITOR{suffix}"

    @staticmethod
    def _is_ml_ad_result(result: Optional[str]) -> bool:
        """Check whether an ML result indicates an ad."""
        if result is None:
            return False

        normalized = str(result).strip().lower()
        if not normalized:
            return False

        return normalized in {"广告", "ad", "spam"} or "广告" in normalized


_detection_engine: Optional[DetectionEngine] = None


def get_detection_engine() -> DetectionEngine:
    """Return the global detection engine singleton."""
    global _detection_engine
    if _detection_engine is None:
        _detection_engine = DetectionEngine()
    return _detection_engine


if __name__ == "__main__":
    print("检测引擎模块测试")
    engine = DetectionEngine()
    print("✓ 检测引擎初始化成功")
