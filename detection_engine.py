"""Compatibility wrapper for the detection engine."""

from __future__ import annotations

from telegram_ad_detector.core.detection import (
    DetectionEngine,
    DetectionResult,
    MessageInfo,
    MessageType,
    MLResult,
    RiskLevel,
    RuleResult,
    get_detection_engine,
)

__all__ = [
    "DetectionEngine",
    "DetectionResult",
    "MessageInfo",
    "MessageType",
    "MLResult",
    "RiskLevel",
    "RuleResult",
    "get_detection_engine",
]
