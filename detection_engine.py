"""
消息检测引擎
实现四类行为的检测逻辑 + ML 模型调用
"""

import logging
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from bot_config import config
from user_state import UserState


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型枚举"""

    TEXT = "TEXT"
    PHOTO = "PHOTO"
    VIDEO = "VIDEO"
    DOCUMENT = "DOCUMENT"
    CONTACT = "CONTACT"
    STICKER = "STICKER"
    VOICE = "VOICE"
    OTHER = "OTHER"


class RiskLevel(Enum):
    """风险等级"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class MessageInfo:
    """消息信息（解析后）"""

    message_type: MessageType
    text: Optional[str] = None
    has_link: bool = False
    has_mention: bool = False
    is_reply: bool = False
    is_external_reply: bool = False
    text_length: int = 0

    def __str__(self):
        return (
            f"Type: {self.message_type.value}, "
            f"Length: {self.text_length}, "
            f"Link: {self.has_link}, "
            f"Mention: {self.has_mention}"
        )

@dataclass
class RuleResult:
    """规则检测结果"""

    external_reply_hit: bool = False
    contact_message_hit: bool = False
    image_message_hit: bool = False
    sticker_message_hit: bool = False
    bio_link_hit: bool = False  # 简介链接检测

    def any_hit(self) -> bool:
        """是否有任何规则命中"""
        return (
            self.external_reply_hit
            or self.contact_message_hit
            or self.image_message_hit
            or self.sticker_message_hit
            or self.bio_link_hit
        )

    def hit_count(self) -> int:
        """命中规则数量"""
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
    """ML 模型检测结果"""

    invoked: bool
    result: Optional[str] = None  # "广告" 或 "正常"
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class DetectionResult:
    """完整的检测结果"""

    user_state: UserState
    message_info: MessageInfo
    rule_result: RuleResult
    ml_result: MLResult
    risk_level: RiskLevel
    suggested_action: str
    risk_score: float


class MessageParser:
    """消息解析器"""

    @staticmethod
    def parse(update) -> MessageInfo:
        """
        解析 Telegram Message 对象

        参数:
            update: Telegram Update 对象

        返回:
            MessageInfo
        """
        message = update.message

        logger.debug(
            "开始解析消息: has_text=%s, has_caption=%s, has_photo=%s, has_video=%s, has_document=%s, has_contact=%s, has_sticker=%s, has_voice=%s",
            bool(getattr(message, "text", None)),
            bool(getattr(message, "caption", None)),
            bool(getattr(message, "photo", None)),
            bool(getattr(message, "video", None)),
            bool(getattr(message, "document", None)),
            bool(getattr(message, "contact", None)),
            bool(getattr(message, "sticker", None)),
            bool(getattr(message, "voice", None)),
        )

        # 识别消息类型
        msg_type = MessageType.OTHER
        text = None

        if message.text:
            msg_type = MessageType.TEXT
            text = message.text
        elif message.photo:
            msg_type = MessageType.PHOTO
            # 图片可能带 caption
            text = message.caption if message.caption else None
        elif message.video:
            msg_type = MessageType.VIDEO
            text = message.caption if message.caption else None
        elif message.document:
            msg_type = MessageType.DOCUMENT
            text = message.caption if message.caption else None
        elif message.contact:
            msg_type = MessageType.CONTACT
        elif message.sticker:
            msg_type = MessageType.STICKER
        elif message.voice:
            msg_type = MessageType.VOICE

        # 文本分析
        has_link = False
        has_mention = False
        text_length = 0

        if text:
            text_length = len(text)
            has_link = "http://" in text or "https://" in text or "www." in text
            has_mention = "@" in text

        # Reply 检测
        is_reply = message.reply_to_message is not None
        is_external_reply = False

        if is_reply:
            # 判断是否外部引用（简化版：检查 message_id 是否在当前会话中）
            # 注意：真实场景需要维护消息历史
            replied_msg = message.reply_to_message
            # 如果 replied_msg 的 chat_id 与当前不同 → 外部引用
            # Telegram 的 reply_to_message 通常是同一个 chat 内的
            # 真正的"外部引用"需要更复杂的检测
            # 这里简化为：如果 reply_to_message 为空用户 → 潜在外部
            if replied_msg and not replied_msg.from_user:
                is_external_reply = True

        parsed = MessageInfo(
            message_type=msg_type,
            text=text,
            has_link=has_link,
            has_mention=has_mention,
            is_reply=is_reply,
            is_external_reply=is_external_reply,
            text_length=text_length,
        )

        logger.info(
            "消息解析完成: type=%s, text_length=%s, has_link=%s, has_mention=%s, is_reply=%s, is_external_reply=%s",
            parsed.message_type.value,
            parsed.text_length,
            parsed.has_link,
            parsed.has_mention,
            parsed.is_reply,
            parsed.is_external_reply,
        )

        return parsed


class RuleEngine:
    """规则引擎"""

    @staticmethod
    def check(
        message_info: MessageInfo,
        user_state: UserState,
        user_bio: Optional[str] = None,
    ) -> RuleResult:
        """
        检查行为规则

        参数:
            message_info: 消息信息
            user_state: 用户状态

        返回:
            RuleResult
        """
        result = RuleResult()

        # 外部引用消息检测
        if config.CHECK_EXTERNAL_REPLY:
            if message_info.is_external_reply and user_state.is_new_member:
                result.external_reply_hit = True

        # 联系人消息检测
        if config.CHECK_CONTACT_MESSAGE:
            if (
                message_info.message_type == MessageType.CONTACT
                and user_state.is_new_member
            ):
                result.contact_message_hit = True

        # 图片消息检测
        if config.CHECK_IMAGE_FROM_NEW_MEMBER:
            if (
                message_info.message_type == MessageType.PHOTO
                and user_state.is_new_member
            ):
                result.image_message_hit = True

        # 贴纸消息检测
        if config.CHECK_STICKER_FROM_NEW_MEMBER:
            if (
                message_info.message_type == MessageType.STICKER
                and user_state.is_new_member
            ):
                result.sticker_message_hit = True

        # 用户简介链接检测（仅新成员）
        if user_state.is_new_member and user_bio:
            if "t.me" in user_bio.lower():
                result.bio_link_hit = True

        logger.info(
            "规则检测结果: external_reply=%s, contact=%s, image=%s, sticker=%s, bio_link=%s, any_hit=%s",
            result.external_reply_hit,
            result.contact_message_hit,
            result.image_message_hit,
            result.sticker_message_hit,
            result.bio_link_hit,
            result.any_hit(),
        )

        return result


class MLDetector:
    """ML 模型检测器"""

    def __init__(self):
        """初始化检测器（延迟加载模型）"""
        self.model = None
        self.tfidf_vectorizer = None
        self.embedding_model = None
        self.emb_weight = 1.0
        self.model_loaded = False

    def load_model(self):
        """加载 ML 模型"""
        if self.model_loaded:
            return True

        if not config.USE_ML_MODEL:
            return False

        try:
            from predict import load_model

            logger.info("正在加载 ML 模型...")
            self.model, self.tfidf_vectorizer, self.embedding_model, emb_w = (
                load_model()
            )
            self.emb_weight = emb_w if emb_w is not None else 1.0

            if all([self.model, self.tfidf_vectorizer, self.embedding_model]):
                self.model_loaded = True
                logger.info("✓ ML 模型加载成功")
                return True
            else:
                logger.warning("⚠️  ML 模型加载失败")
                return False

        except Exception as e:
            logger.exception("⚠️  加载 ML 模型出错")
            return False

    def detect(self, text: str) -> MLResult:
        """
        使用 ML 模型检测文本

        参数:
            text: 待检测文本

        返回:
            MLResult
        """
        if not config.USE_ML_MODEL:
            return MLResult(invoked=False)

        safe_text = self._sanitize_text(text)
        if not safe_text:
            return MLResult(invoked=False)

        # 确保模型已加载
        if not self.model_loaded:
            if not self.load_model():
                return MLResult(invoked=True, error="Model not loaded")

        try:
            from predict import predict_text

            start_time = time.time()

            result, confidence = predict_text(
                safe_text,
                self.model,
                self.tfidf_vectorizer,
                self.embedding_model,
                show_confidence=True,
                emb_weight=self.emb_weight,
            )

            latency = (time.time() - start_time) * 1000  # 转为毫秒

            return MLResult(
                invoked=True, result=result, confidence=confidence, latency_ms=latency
            )

        except Exception as e:
            error_msg = str(e)

            if self._is_encoding_error(error_msg):
                repaired_text = self._sanitize_text(text, aggressive=True)
                if repaired_text:
                    try:
                        from predict import predict_text

                        start_time = time.time()
                        result, confidence = predict_text(
                            repaired_text,
                            self.model,
                            self.tfidf_vectorizer,
                            self.embedding_model,
                            show_confidence=True,
                            emb_weight=self.emb_weight,
                        )
                        latency = (time.time() - start_time) * 1000
                        logger.warning("ML 文本包含异常字符，已使用容错清洗重试")
                        return MLResult(
                            invoked=True,
                            result=result,
                            confidence=confidence,
                            latency_ms=latency,
                        )
                    except Exception as retry_error:
                        return MLResult(invoked=True, error=str(retry_error))

            return MLResult(invoked=True, error=error_msg)

    @staticmethod
    def _sanitize_text(text: str, aggressive: bool = False) -> str:
        if text is None:
            return ""

        normalized = unicodedata.normalize("NFKC", str(text))

        cleaned_chars = []
        for char in normalized:
            codepoint = ord(char)

            if 0xD800 <= codepoint <= 0xDFFF:
                continue

            category = unicodedata.category(char)
            if category.startswith("C") and char not in {"\n", "\r", "\t"}:
                continue

            cleaned_chars.append(char)

        cleaned = "".join(cleaned_chars)

        if aggressive:
            try:
                cleaned = cleaned.encode("utf-16", "surrogatepass").decode(
                    "utf-16", "ignore"
                )
            except Exception:
                pass

            cleaned = " ".join(cleaned.split())

        return cleaned.strip()

    @staticmethod
    def _is_encoding_error(error_msg: str) -> bool:
        if not error_msg:
            return False
        lowered = error_msg.lower()
        return (
            "codec can't decode" in lowered
            or "utf-16" in lowered
            or "unicode" in lowered
            or "token" in lowered
            and "decode" in lowered
        )


class DetectionEngine:
    """检测引擎（整合所有检测逻辑）"""

    def __init__(self):
        """初始化检测引擎"""
        self.ml_detector = MLDetector()

        # 预加载模型（可选）
        if config.USE_ML_MODEL:
            self.ml_detector.load_model()

    def analyze(
        self, update, user_state: UserState, user_bio: Optional[str] = None
    ) -> DetectionResult:
        """
        完整分析消息

        参数:
            update: Telegram Update 对象
            user_state: 用户状态

        返回:
            DetectionResult
        """
        # Step 1: 解析消息
        message_info = MessageParser.parse(update)

        # Step 2: 规则检测
        rule_result = RuleEngine.check(message_info, user_state, user_bio=user_bio)

        # Step 3: ML 检测（仅文本）
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

        # Step 4: 计算风险分数
        risk_score = self._calculate_risk_score(rule_result, ml_result)

        # Step 5: 确定风险等级
        risk_level = self._determine_risk_level(risk_score)

        # Step 6: 建议操作（传入检测结果用于判断）
        suggested_action = self._suggest_action(
            risk_level, user_state, rule_result, ml_result, risk_score
        )

        logger.info(
            "检测决策: risk_level=%s, risk_score=%.3f, suggested_action=%s, ml_invoked=%s, ml_result=%s",
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

    def _calculate_risk_score(
        self, rule_result: RuleResult, ml_result: MLResult
    ) -> float:
        """
        计算风险分数（0.0 - 1.0）

        规则命中 + ML 置信度综合

        分数分布设计：
        - 正常消息：0
        - 低置信度广告(0.3)：0.24 → MONITOR
        - 中置信度广告(0.6)：0.48 → MONITOR
        - 高置信度广告(0.8)：0.64 → WARN
        - 极高置信度(0.95)：0.76 → WARN
        - 规则命中+高置信度：≥0.8 → HIGH
        """
        score = 0.0

        # 规则命中贡献
        if rule_result.external_reply_hit:
            score += 0.85  # 外部引用
        if rule_result.contact_message_hit:
            score += 0.85  # 联系人卡片
        if rule_result.image_message_hit:
            score += 0.25  # 新成员发图
        if rule_result.bio_link_hit:
            score += 0.25  # 新成员简介含 t.me

        # ML 模型贡献
        if ml_result.invoked and self._is_ml_ad_result(ml_result.result):
            if ml_result.confidence:
                score += min(ml_result.confidence, 1.0)
            else:
                score += 0.5

        return min(score, 1.0)

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """根据风险分数确定风险等级"""
        if risk_score >= config.RISK_SCORE_HIGH:
            return RiskLevel.HIGH
        elif risk_score >= config.RISK_SCORE_MEDIUM:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _suggest_action(
        self,
        risk_level: RiskLevel,
        user_state: UserState,
        rule_result: RuleResult,
        ml_result: MLResult,
        risk_score: float,
    ) -> str:
        """
        建议操作

        关键逻辑：
        - 只有明确有违规行为（ML判定为广告 OR 规则命中）才执行操作
        - LOW 级别 + 无违规 = NO_ACTION（正常消息）
        - LOW 级别 + 有违规 = MONITOR（低置信度广告）
        - MEDIUM 级别 = WARN（高置信度广告）
        - HIGH 级别 = KICK（严重违规）

        返回:
            操作建议字符串
        """
        if config.DRY_RUN:
            suffix = " (debug only)"
        else:
            suffix = ""

        # 判断是否有明确违规
        has_violation = (
            ml_result.invoked and self._is_ml_ad_result(ml_result.result)
        ) or rule_result.any_hit()

        if risk_level == RiskLevel.HIGH:
            if config.HIGH_IMMEDIATE_KICK:
                return f"KICK_IMMEDIATE{suffix}"
            else:
                return f"KICK{suffix}"

        elif risk_level == RiskLevel.MEDIUM:
            # MEDIUM = WARN（高置信度广告）
            # 检查是否达到 WARN 踢人阈值
            if user_state.warn_count + 1 >= config.WARN_KICK_THRESHOLD:
                return f"KICK_ON_WARN{suffix}"
            else:
                return f"WARN{suffix}"

        else:  # LOW
            # 关键判断：只有明确违规才执行 MONITOR
            if not has_violation:
                # 正常消息，不执行任何操作
                return f"NO_ACTION{suffix}"

            # 有违规但置信度低 → MONITOR（删除消息+计数）
            if user_state.monitor_count + 1 >= config.MONITOR_KICK_THRESHOLD:
                return f"KICK_ON_MONITOR{suffix}"
            else:
                return f"MONITOR{suffix}"

    @staticmethod
    def _is_ml_ad_result(result: Optional[str]) -> bool:
        """判断 ML 输出是否表示广告（兼容不同格式返回值）"""
        if result is None:
            return False

        normalized = str(result).strip().lower()
        if not normalized:
            return False

        return normalized in {"广告", "ad", "spam"} or "广告" in normalized


# 全局单例
_detection_engine: Optional[DetectionEngine] = None


def get_detection_engine() -> DetectionEngine:
    """获取全局检测引擎单例"""
    global _detection_engine
    if _detection_engine is None:
        _detection_engine = DetectionEngine()
    return _detection_engine


if __name__ == "__main__":
    # 测试代码
    print("检测引擎模块测试")
    engine = DetectionEngine()
    print("✓ 检测引擎初始化成功")
