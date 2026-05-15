"""ML-based detection helper."""

from __future__ import annotations

import logging
import time
import unicodedata
from typing import Any, Optional

from ..config import config
from .types import MLResult

logger = logging.getLogger(__name__)


class MLDetector:
    """ML model wrapper used by the detection engine."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.tfidf_vectorizer: Optional[Any] = None
        self.embedding_model: Optional[Any] = None
        self.emb_weight: float = 1.0
        self.model_loaded: bool = False

    def load_model(self) -> bool:
        """Load the ML model and vectorizers.

        Returns:
            True if the model is loaded successfully.
        """
        if self.model_loaded:
            return True

        if not config.USE_ML_MODEL:
            return False

        try:
            from ..ml.inference import load_model

            logger.info("正在加载 ML 模型...")
            self.model, self.tfidf_vectorizer, self.embedding_model, emb_w = (
                load_model()
            )
            self.emb_weight = emb_w if emb_w is not None else 1.0

            if all([self.model, self.tfidf_vectorizer, self.embedding_model]):
                self.model_loaded = True
                logger.info("✓ ML 模型加载成功")
                return True

            logger.warning("⚠️  ML 模型加载失败")
            return False

        except Exception:
            logger.exception("⚠️  加载 ML 模型出错")
            return False

    def detect(self, text: str) -> MLResult:
        """Run ML detection on a text.

        Args:
            text: Input text.

        Returns:
            MLResult.
        """
        if not config.USE_ML_MODEL:
            return MLResult(invoked=False)

        safe_text = self._sanitize_text(text)
        if not safe_text:
            return MLResult(invoked=False)

        if not self.model_loaded and not self.load_model():
            return MLResult(invoked=True, error="Model not loaded")

        try:
            from ..ml.inference import predict_text

            start_time = time.time()

            result, confidence = predict_text(
                safe_text,
                self.model,
                self.tfidf_vectorizer,
                self.embedding_model,
                show_confidence=True,
                emb_weight=self.emb_weight,
            )

            latency = (time.time() - start_time) * 1000

            return MLResult(
                invoked=True,
                result=result,
                confidence=confidence,
                latency_ms=latency,
            )

        except Exception as exc:
            error_msg = str(exc)

            if self._is_encoding_error(error_msg):
                repaired_text = self._sanitize_text(text, aggressive=True)
                if repaired_text:
                    try:
                        from ..ml.inference import predict_text

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
