"""Model inference helpers."""

from __future__ import annotations

import time
from typing import Any

import joblib
from scipy.sparse import csr_matrix, hstack

MODEL_PATH = "ad_model.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"
EMBEDDING_PATH = "embedding_model.pkl"
EMB_WEIGHT_PATH = "emb_weight.pkl"


def load_model() -> tuple[Any | None, Any | None, Any | None, float | None]:
    """Load trained model artifacts.

    Returns:
        Tuple of (model, tfidf_vectorizer, embedding_model, emb_weight).
    """
    try:
        print("正在加载模型...")
        start_time = time.time()

        model = joblib.load(MODEL_PATH)
        tfidf_vectorizer = joblib.load(TFIDF_PATH)
        embedding_model = joblib.load(EMBEDDING_PATH)
        emb_weight = joblib.load(EMB_WEIGHT_PATH)

        load_time = time.time() - start_time
        print(f"✓ 模型加载成功！（耗时 {load_time:.2f}秒）")
        return model, tfidf_vectorizer, embedding_model, emb_weight

    except FileNotFoundError:
        print("❌ 错误：找不到模型文件！")
        print("请先运行 main.py 训练并保存模型。")
        return None, None, None, None
    except Exception as exc:
        print(f"❌ 加载模型时出错: {exc}")
        return None, None, None, None


def predict_text(
    text: str,
    model: Any,
    tfidf_vectorizer: Any,
    embedding_model: Any,
    show_confidence: bool = True,
    emb_weight: float = 1.0,
) -> tuple[str, float | None]:
    """Predict a single text using fused features.

    Args:
        text: Input text.
        model: Trained classifier.
        tfidf_vectorizer: TF-IDF vectorizer.
        embedding_model: Sentence embedding model.
        show_confidence: Whether to compute confidence scores.
        emb_weight: Embedding feature weight.

    Returns:
        Tuple of (result, confidence).
    """
    tfidf_vec = tfidf_vectorizer.transform([text])
    emb_vec = embedding_model.encode([text], normalize_embeddings=True)
    emb_vec_sparse = csr_matrix(emb_vec)

    final_vec = hstack([tfidf_vec, emb_vec_sparse * emb_weight])

    prediction = model.predict(final_vec)[0]
    result = "广告" if prediction == 1 else "正常"

    confidence = None
    if show_confidence and hasattr(model, "decision_function"):
        score = model.decision_function(final_vec)[0]
        confidence = abs(score)

    return result, confidence


def predict_batch(
    text_list: list[str],
    model: Any,
    tfidf_vectorizer: Any,
    embedding_model: Any,
    emb_weight: float = 1.0,
) -> list[tuple[str, str, float | None]]:
    """Predict a batch of texts.

    Args:
        text_list: Input texts.
        model: Trained classifier.
        tfidf_vectorizer: TF-IDF vectorizer.
        embedding_model: Sentence embedding model.
        emb_weight: Embedding feature weight.

    Returns:
        List of (text, result, confidence) tuples.
    """
    if not text_list:
        return []

    print(f"\n正在批量预测 {len(text_list)} 条文本...")
    start_time = time.time()

    tfidf_vec = tfidf_vectorizer.transform(text_list)
    emb_vec = embedding_model.encode(
        text_list, normalize_embeddings=True, show_progress_bar=True
    )
    emb_vec_sparse = csr_matrix(emb_vec)

    final_vec = hstack([tfidf_vec, emb_vec_sparse * emb_weight])

    predictions = model.predict(final_vec)

    confidences = None
    if hasattr(model, "decision_function"):
        scores = model.decision_function(final_vec)
        confidences = [abs(score) for score in scores]

    results = []
    for i, (text, pred) in enumerate(zip(text_list, predictions)):
        result = "广告" if pred == 1 else "正常"
        conf = confidences[i] if confidences else None
        results.append((text, result, conf))

    elapsed = time.time() - start_time
    print(
        "✓ 批量预测完成！（耗时 {elapsed:.2f}秒，平均 {avg:.3f}秒/条）".format(
            elapsed=elapsed,
            avg=elapsed / len(text_list),
        )
    )

    return results


def load_texts_from_file(file_path: str) -> list[str]:
    """Load texts from a file, one per line.

    Args:
        file_path: Path to the text file.

    Returns:
        List of text lines.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            texts = [line.strip() for line in handle if line.strip()]
        print(f"✓ 从 {file_path} 读取了 {len(texts)} 条文本")
        return texts
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
        return []
    except Exception as exc:
        print(f"❌ 读取文件出错: {exc}")
        return []
