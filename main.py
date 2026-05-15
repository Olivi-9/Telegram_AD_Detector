import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from sentence_transformers import SentenceTransformer


# ======================
# 数据加载
# ======================
df = pd.read_csv("train.csv", sep=",", header=0, names=["label", "text"])

df.dropna(inplace=True)
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)

texts = df["text"].tolist()
labels = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# ======================
# TF-IDF 分支（优化参数）
# ======================
# 参数说明：
# - max_features: 限制特征数量，避免特征过多导致过拟合
# - max_df: 过滤在超过80%文档中出现的词（可能是无意义高频词）
# - min_df: 至少在2个文档中出现（降低了原来的3，增加特征覆盖）
word_vectorizer: TransformerMixin = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"(?u)\b\w+\b",
    max_features=6000,  # 限制词特征数量
    max_df=0.8,  # 过滤高频词
    min_df=2,  # 降低最小文档频率
)

# 参数说明：
# - ngram_range: 从(2,10)缩小到(2,8)，减少噪声特征
# - max_features: 限制字符n-gram特征数量
# - min_df: 降到2，增加特征覆盖
char_vectorizer: TransformerMixin = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 8),  # 缩小范围，减少噪声
    min_df=2,  # 降低阈值
    max_features=6000,  # 限制特征数
    max_df=0.8,  # 过滤极高频字符组合
)

tfidf_vectorizer = FeatureUnion(
    [
        ("word", word_vectorizer),
        ("char", char_vectorizer),
    ]
)

print("正在提取 TF-IDF 特征...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF 特征维度: {X_train_tfidf.shape}")

pos_f1_scorer = make_scorer(f1_score, pos_label=1)


# ======================
# Sentence Embeddings
# ======================
print("\n正在生成句子嵌入...")
# embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

X_train_emb = embedding_model.encode(
    X_train, batch_size=64, show_progress_bar=True, normalize_embeddings=True
)

X_test_emb = embedding_model.encode(
    X_test, batch_size=64, show_progress_bar=True, normalize_embeddings=True
)

X_train_emb_sparse = csr_matrix(X_train_emb)
X_test_emb_sparse = csr_matrix(X_test_emb)


def find_best_emb_weight(
    X_train_tfidf_data,
    X_train_emb_data,
    y_train_data,
    weight_candidates=None,
):
    """在训练集内部自动搜索最佳 embedding 权重，避免使用测试集调参。"""
    if weight_candidates is None:
        weight_candidates = np.arange(0.5, 3.5, 0.2)

    (
        X_sub_tfidf,
        X_val_tfidf,
        X_sub_emb,
        X_val_emb,
        y_sub,
        y_val,
    ) = train_test_split(
        X_train_tfidf_data,
        X_train_emb_data,
        y_train_data,
        test_size=0.2,
        random_state=42,
        stratify=y_train_data,
    )

    best_weight = None
    best_score = -1.0

    print("\n开始自动搜索最佳 EMB_WEIGHT...")
    for weight in weight_candidates:
        X_sub_final = hstack([X_sub_tfidf, X_sub_emb * weight])
        X_val_final = hstack([X_val_tfidf, X_val_emb * weight])

        temp_model = LinearSVC(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            dual=True,
            random_state=40,
        )
        temp_model.fit(X_sub_final, y_sub)
        y_val_pred = temp_model.predict(X_val_final)
        score = f1_score(y_val, y_val_pred, pos_label=1)

        print(f"EMB_WEIGHT={weight:.1f} -> val f1(pos_label=1)={score:.4f}")

        if score > best_score:
            best_score = score
            best_weight = float(weight)

    print(f"最佳 EMB_WEIGHT: {best_weight:.1f} (val f1(pos_label=1)={best_score:.4f})")
    return best_weight


EMB_WEIGHT = find_best_emb_weight(X_train_tfidf, X_train_emb_sparse, y_train)


# ======================
# 特征融合
# ======================
X_train_final = hstack([X_train_tfidf, X_train_emb_sparse * EMB_WEIGHT])

X_test_final = hstack([X_test_tfidf, X_test_emb_sparse * EMB_WEIGHT])

print(f"最终特征维度: {X_train_final.shape}")


# ======================
# 模型训练（使用网格搜索优化参数）
# ======================
print("\n开始网格搜索最优参数...")

# 参数说明：
# - C: 正则化强度的倒数，越小正则化越强
#   - C=0.1: 强正则化，防止过拟合，适合特征多、数据少
#   - C=1.0: 中等正则化（默认值）
#   - C=5.0: 弱正则化，允许更复杂的决策边界
# - max_iter: 最大迭代次数，增加以确保收敛
# - class_weight: 'balanced' 自动调整权重平衡类别
param_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0], "max_iter": [2000, 5000]}

base_model = LinearSVC(
    class_weight="balanced",
    dual=True,
    random_state=40,  # 当样本数 > 特征数时，dual=True更快
)

# 使用3折交叉验证
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=3,
    scoring=pos_f1_scorer,  # 使用正类F1分数（pos_label=1）
    n_jobs=-1,  # 使用所有CPU核心
    verbose=2,
)

grid_search.fit(X_train_final, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_


# ======================
# 评估
# ======================
y_pred = model.predict(X_test_final)

print("\n" + "=" * 60)
print("模型表现评估：")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["正常", "广告"]))

print("\n混淆矩阵：")
print(confusion_matrix(y_test, y_pred))
print("          预测:正常  预测:广告")
print("实际:正常    TN        FP")
print("实际:广告    FN        TP")


# ======================
# 推理函数
# ======================
def predict_new_text(text_list):
    """预测新文本是否为广告"""
    tfidf_vec = tfidf_vectorizer.transform(text_list)

    emb_vec = embedding_model.encode(text_list, normalize_embeddings=True)
    emb_vec_sparse = csr_matrix(emb_vec)

    final_vec = hstack([tfidf_vec, emb_vec_sparse * EMB_WEIGHT])

    preds = model.predict(final_vec)

    # 如果支持decision_function，可以获取置信度
    if hasattr(model, "decision_function"):
        scores = model.decision_function(final_vec)
        for text, p, score in zip(text_list, preds, scores):
            confidence = abs(score)
            print(f"内容: {text}")
            print(f"预测结果: {'广告' if p == 1 else '正常'}")
            print(f"置信度: {confidence:.3f}")
            print("-" * 50)
    else:
        for text, p in zip(text_list, preds):
            print(f"内容: {text}")
            print(f"预测结果: {'广告' if p == 1 else '正常'}")
            print("-" * 50)


# ======================
# 保存 / 加载
# ======================
def save_my_ai():
    joblib.dump(model, "ad_model.pkl")
    joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    joblib.dump(embedding_model, "embedding_model.pkl")
    joblib.dump(EMB_WEIGHT, "emb_weight.pkl")
    print("\n✓ 模型已保存")


def load_my_ai():
    m = joblib.load("ad_model.pkl")
    v = joblib.load("tfidf_vectorizer.pkl")
    e = joblib.load("embedding_model.pkl")
    return m, v, e


save_my_ai()


# ======================
# 测试
# ======================
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

predict_new_text(test_samples)
