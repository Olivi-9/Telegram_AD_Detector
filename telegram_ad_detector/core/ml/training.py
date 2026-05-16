"""Training workflow for the ad detector model."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

TRAIN_FILE = "train.csv"
MODEL_PATH = "ad_model.pkl"
TFIDF_PATH = "tfidf_vectorizer.pkl"
EMBEDDING_PATH = "embedding_model.pkl"
EMB_WEIGHT_PATH = "emb_weight.pkl"


def load_training_data(csv_path: str = TRAIN_FILE) -> tuple[list[str], list[int]]:
    """Load and sanitize training data.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Tuple of (texts, labels).
    """
    df = pd.read_csv(csv_path, sep=",", header=0, names=["label", "text"])

    df.dropna(inplace=True)
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)

    return df["text"].tolist(), df["label"].tolist()


def split_data(
    texts: list[str],
    labels: list[int],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Split the dataset into train/test sets."""
    return train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )


def build_vectorizers() -> FeatureUnion:
    """Build TF-IDF vectorizers for word and character n-grams."""
    word_vectorizer: TransformerMixin = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        max_features=6000,
        max_df=0.8,
        min_df=2,
    )

    char_vectorizer: TransformerMixin = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 8),
        min_df=2,
        max_features=6000,
        max_df=0.8,
    )

    return FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])


def compute_embeddings(
    embedding_model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
    show_progress_bar: bool = True,
) -> csr_matrix:
    """Compute sentence embeddings and return them as a sparse matrix."""
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )
    return csr_matrix(embeddings)


def find_best_emb_weight(
    x_train_tfidf: csr_matrix,
    x_train_emb: csr_matrix,
    y_train: list[int],
    weight_candidates: np.ndarray | None = None,
) -> float:
    """Search for the best embedding weight on a validation split."""
    if weight_candidates is None:
        weight_candidates = np.arange(0.5, 3.5, 0.2)

    (
        x_sub_tfidf,
        x_val_tfidf,
        x_sub_emb,
        x_val_emb,
        y_sub,
        y_val,
    ) = train_test_split(
        x_train_tfidf,
        x_train_emb,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    best_weight = None
    best_score = -1.0

    print("\n开始自动搜索最佳 EMB_WEIGHT...")
    for weight in weight_candidates:
        x_sub_final = hstack([x_sub_tfidf, x_sub_emb * weight])
        x_val_final = hstack([x_val_tfidf, x_val_emb * weight])

        temp_model = LinearSVC(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            dual=True,
            random_state=40,
        )
        temp_model.fit(x_sub_final, y_sub)
        y_val_pred = temp_model.predict(x_val_final)
        score = f1_score(y_val, y_val_pred, pos_label=1)

        print(f"EMB_WEIGHT={weight:.1f} -> val f1(pos_label=1)={score:.4f}")

        if score > best_score:
            best_score = score
            best_weight = float(weight)

    print(f"最佳 EMB_WEIGHT: {best_weight:.1f} (val f1(pos_label=1)={best_score:.4f})")
    return best_weight


def train_model(x_train_final: csr_matrix, y_train: list[int]) -> GridSearchCV:
    """Train the LinearSVC model using grid search."""
    print("\n开始网格搜索最优参数...")

    param_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0], "max_iter": [2000, 5000]}

    base_model = LinearSVC(
        class_weight="balanced",
        dual=True,
        random_state=40,
    )

    pos_f1_scorer = make_scorer(f1_score, pos_label=1)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring=pos_f1_scorer,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(x_train_final, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_model(
    model: LinearSVC, x_test_final: csr_matrix, y_test: list[int]
) -> None:
    """Evaluate the trained model and print metrics."""
    y_pred = model.predict(x_test_final)

    print("\n" + "=" * 60)
    print("模型表现评估：")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["正常", "广告"]))

    print("\n混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))
    print("          预测:正常  预测:广告")
    print("实际:正常    TN        FP")
    print("实际:广告    FN        TP")


def save_artifacts(
    model: LinearSVC,
    tfidf_vectorizer: FeatureUnion,
    embedding_model: SentenceTransformer,
    emb_weight: float,
) -> None:
    """Persist trained artifacts to disk."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(tfidf_vectorizer, TFIDF_PATH)
    joblib.dump(embedding_model, EMBEDDING_PATH)
    joblib.dump(emb_weight, EMB_WEIGHT_PATH)
    print("\n✓ 模型已保存")


def predict_samples(
    model: LinearSVC,
    tfidf_vectorizer: FeatureUnion,
    embedding_model: SentenceTransformer,
    emb_weight: float,
    samples: list[str],
) -> None:
    """Run predictions for sample texts and print results."""
    tfidf_vec = tfidf_vectorizer.transform(samples)

    emb_vec = embedding_model.encode(samples, normalize_embeddings=True)
    emb_vec_sparse = csr_matrix(emb_vec)

    final_vec = hstack([tfidf_vec, emb_vec_sparse * emb_weight])

    preds = model.predict(final_vec)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(final_vec)
        for text, pred, score in zip(samples, preds, scores):
            confidence = abs(score)
            print(f"内容: {text}")
            print(f"预测结果: {'广告' if pred == 1 else '正常'}")
            print(f"置信度: {confidence:.3f}")
            print("-" * 50)
    else:
        for text, pred in zip(samples, preds):
            print(f"内容: {text}")
            print(f"预测结果: {'广告' if pred == 1 else '正常'}")
            print("-" * 50)


def run_training() -> None:
    """Run the full training workflow."""
    texts, labels = load_training_data(TRAIN_FILE)

    x_train, x_test, y_train, y_test = split_data(texts, labels)

    tfidf_vectorizer = build_vectorizers()

    print("正在提取 TF-IDF 特征...")
    x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)
    print(f"TF-IDF 特征维度: {x_train_tfidf.shape}")

    print("\n正在生成句子嵌入...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    x_train_emb = compute_embeddings(embedding_model, x_train)
    x_test_emb = compute_embeddings(embedding_model, x_test)

    emb_weight = find_best_emb_weight(x_train_tfidf, x_train_emb, y_train)

    x_train_final = hstack([x_train_tfidf, x_train_emb * emb_weight])
    x_test_final = hstack([x_test_tfidf, x_test_emb * emb_weight])

    print(f"最终特征维度: {x_train_final.shape}")

    grid_search = train_model(x_train_final, y_train)
    model = grid_search.best_estimator_

    evaluate_model(model, x_test_final, y_test)

    save_artifacts(model, tfidf_vectorizer, embedding_model, emb_weight)

    print("\n" + "=" * 60)
    print("测试样例预测：")
    print("=" * 60)

    test_samples = [
        "🎁抖音代刷礼物｜有号就能做｜无风险｜日结5000起 @KKK588",
        "可保养读书妹，找零装逼专用 @wanshiruyib",
        "爆款通道｜0.5TRX 免费转U｜@K9trx",
        "给妹妹拍照，一张1000,轻灰铲，想做的看簡介",
        "免费试用，抖音小黑屋解封，账号安全保护，找我就对了 @KKK88",
        "柬埔寨木牌会所｜多地同步营业｜在线优先 @myhs",
        "还在家里躺？又不甘心一个人躺？来跟我干下个月娶老婆看届介",
        "喔这边新搞的路子。懂的都懂，瞄一眼就行👀",
    ]

    predict_samples(model, tfidf_vectorizer, embedding_model, emb_weight, test_samples)


def main() -> None:
    """Entry point for model training."""
    run_training()


if __name__ == "__main__":
    main()
