
import warnings

warnings.filterwarnings("ignore")

import os
import sys
import joblib
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Font setup (support Chinese characters on Windows)
# ──────────────────────────────────────────────
import matplotlib.font_manager as fm

_CHINESE_FONTS = ["Microsoft YaHei"]
_available = {f.name for f in fm.fontManager.ttflist}
for _font in _CHINESE_FONTS:
    if _font in _available:
        matplotlib.rcParams["font.family"] = _font
        break
matplotlib.rcParams["axes.unicode_minus"] = False

plt.style.use("seaborn-v0_8-whitegrid")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MAX_VIZ_SAMPLES = 1000  # cap for t-SNE / PCA scatter (speed)
COLORS = ["#2196F3", "#F44336"]  # Blue = Normal, Red = Ad
LABELS = ["Normal", "Ad"]
SEED = 42

# ──────────────────────────────────────────────
# 1. Load saved artifacts
# ──────────────────────────────────────────────
print("=" * 60)
print("Loading saved model artifacts …")
print("=" * 60)

for fname in [
    "ad_model.pkl",
    "tfidf_vectorizer.pkl",
    "embedding_model.pkl",
    "emb_weight.pkl",
]:
    if not os.path.exists(fname):
        sys.exit(f"❌  Missing file: {fname}\n   Please run main.py first.")

model = joblib.load("ad_model.pkl")
tfidf_vec = joblib.load("tfidf_vectorizer.pkl")
embedding_model = joblib.load("embedding_model.pkl")
emb_weight = joblib.load("emb_weight.pkl")
print(f"✓  EMB_WEIGHT = {emb_weight:.2f}")

# ──────────────────────────────────────────────
# 2. Load data & reproduce the same train/test split
# ──────────────────────────────────────────────
print("\nLoading train.csv …")
df = pd.read_csv("train.csv", sep=",", header=0, names=["label", "text"])
df.dropna(inplace=True)
df["label"] = df["label"].astype(int)
df["text"] = df["text"].astype(str)

all_texts = df["text"].tolist()
all_labels = np.array(df["label"].tolist())

X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    all_texts, all_labels, test_size=0.2, random_state=SEED
)
print(f"  Total: {len(all_texts)}  Train: {len(X_train_txt)}  Test: {len(X_test_txt)}")

# Subsample for scatter visualizations
rng = np.random.default_rng(SEED)
if len(all_texts) > MAX_VIZ_SAMPLES:
    idx = rng.choice(len(all_texts), MAX_VIZ_SAMPLES, replace=False)
    viz_texts = [all_texts[i] for i in idx]
    viz_labels = all_labels[idx]
else:
    viz_texts = all_texts
    viz_labels = all_labels

print(f"  Viz sample: {len(viz_texts)}")

# ──────────────────────────────────────────────
# 3. Encode features
# ──────────────────────────────────────────────
print("\nEncoding sentence embeddings for visualization sample …")
viz_emb = embedding_model.encode(
    viz_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
)

print("\nEncoding sentence embeddings for test set …")
test_emb = embedding_model.encode(
    X_test_txt, batch_size=64, show_progress_bar=True, normalize_embeddings=True
)

test_tfidf = tfidf_vec.transform(X_test_txt)
test_emb_sparse = csr_matrix(test_emb)
X_test_final = hstack([test_tfidf, test_emb_sparse * emb_weight])
y_pred = model.predict(X_test_final)

print("\nClassification report on test set:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Ad"]))

# ──────────────────────────────────────────────
# 4. Dimensionality reduction (embeddings → 2D)
# ──────────────────────────────────────────────
print("Running PCA on embeddings …")
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(viz_emb)
pca_var = pca.explained_variance_ratio_

print("Running t-SNE on embeddings …")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000, n_jobs=-1)
X_tsne = tsne.fit_transform(viz_emb)

# ──────────────────────────────────────────────
# 5. Train a 2-D SVM on PCA features (for boundary visualisation)
# ──────────────────────────────────────────────
print("Training 2D SVM on PCA-reduced embeddings …")
svm_2d = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced", random_state=SEED)
svm_2d.fit(X_pca, viz_labels)

# Build a mesh grid for the decision surface
pad = 1.0
x_lo, x_hi = X_pca[:, 0].min() - pad, X_pca[:, 0].max() + pad
y_lo, y_hi = X_pca[:, 1].min() - pad, X_pca[:, 1].max() + pad
step = (x_hi - x_lo) / 300
xx, yy = np.meshgrid(
    np.arange(x_lo, x_hi, step),
    np.arange(y_lo, y_hi, step),
)
Z_mesh = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Decision score heatmap (distance from hyper-plane)
Z_score = svm_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# ──────────────────────────────────────────────
# 6. TF-IDF feature importance from the trained model
# ──────────────────────────────────────────────
tfidf_names = tfidf_vec.get_feature_names_out()  # word + char features
n_tfidf = len(tfidf_names)
coef_full = model.coef_[0]
coef_tfidf = coef_full[:n_tfidf]

TOP_N = 15
top_pos_i = np.argsort(coef_tfidf)[-TOP_N:][::-1]  # most "Ad" features
top_neg_i = np.argsort(coef_tfidf)[:TOP_N]  # most "Normal" features
feat_idx = np.concatenate([top_pos_i, top_neg_i])
feat_coefs = coef_tfidf[feat_idx]
feat_names = [tfidf_names[i] for i in feat_idx]
bar_colors = ["#F44336" if c > 0 else "#2196F3" for c in feat_coefs]

# ──────────────────────────────────────────────
# 7. Build Dashboard Figure
# ──────────────────────────────────────────────
fig = plt.figure(figsize=(22, 17))
fig.suptitle(
    "SVM Telegram Advertisement Detector — Visualization Dashboard",
    fontsize=17,
    fontweight="bold",
    y=0.99,
)

legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLORS, LABELS)]

# ── 7a: t-SNE ──────────────────────────────────
ax1 = fig.add_subplot(2, 2, 1)
for lv, col, name in zip([0, 1], COLORS, LABELS):
    m = viz_labels == lv
    ax1.scatter(
        X_tsne[m, 0],
        X_tsne[m, 1],
        c=col,
        label=name,
        alpha=0.65,
        s=18,
        linewidths=0,
    )
ax1.set_title("t-SNE  ·  Sentence Embeddings", fontsize=13, fontweight="bold")
ax1.set_xlabel("t-SNE Dim 1")
ax1.set_ylabel("t-SNE Dim 2")
ax1.legend(handles=legend_patches, fontsize=10, loc="best")
ax1.set_xticks([])
ax1.set_yticks([])

# ── 7b: PCA + Decision Boundary ────────────────
ax2 = fig.add_subplot(2, 2, 2)

# Soft background: decision score heatmap
ax2.contourf(xx, yy, Z_score, levels=50, cmap="RdBu_r", alpha=0.25)
# Hard decision boundary
ax2.contour(
    xx,
    yy,
    Z_mesh,
    levels=[0.5],
    colors=["#424242"],
    linewidths=[1.8],
    linestyles=["--"],
)

for lv, col, name in zip([0, 1], COLORS, LABELS):
    m = viz_labels == lv
    ax2.scatter(
        X_pca[m, 0],
        X_pca[m, 1],
        c=col,
        label=name,
        alpha=0.65,
        s=18,
        linewidths=0,
    )

ax2.set_xlim(x_lo, x_hi)
ax2.set_ylim(y_lo, y_hi)
var_pct = pca_var.sum() * 100
ax2.set_title(
    f"PCA + SVM Decision Boundary  (variance explained: {var_pct:.1f}%)",
    fontsize=13,
    fontweight="bold",
)
ax2.set_xlabel(f"PC1  ({pca_var[0]*100:.1f}%)")
ax2.set_ylabel(f"PC2  ({pca_var[1]*100:.1f}%)")
ax2.legend(handles=legend_patches, fontsize=10, loc="best")
ax2.set_xticks([])
ax2.set_yticks([])

# ── 7c: Feature Weights ────────────────────────
ax3 = fig.add_subplot(2, 2, 3)
y_pos = np.arange(len(feat_names))
bars = ax3.barh(y_pos, feat_coefs, color=bar_colors, alpha=0.85, height=0.75)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(feat_names, fontsize=8.5)
ax3.axvline(x=0, color="#424242", linewidth=1.0)
ax3.set_title(
    f"Top {TOP_N} TF-IDF Feature Weights  (Red → Ad, Blue → Normal)",
    fontsize=13,
    fontweight="bold",
)
ax3.set_xlabel("SVM Coefficient")
ax3.grid(True, axis="x", alpha=0.4)

# ── 7d: Confusion Matrix ───────────────────────
ax4 = fig.add_subplot(2, 2, 4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Ad"],
    yticklabels=["Normal", "Ad"],
    ax=ax4,
    annot_kws={"size": 16},
    linewidths=0.5,
    linecolor="white",
)
tn, fp, fn, tp = cm.ravel()
prec = tp / (tp + fp) if (tp + fp) else 0.0
rec = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
acc = (tn + tp) / cm.sum()

ax4.set_title(
    f"Confusion Matrix  (test set, n={cm.sum()})\n"
    f"Acc={acc:.3f}   F1={f1:.3f}   Prec={prec:.3f}   Recall={rec:.3f}",
    fontsize=12,
    fontweight="bold",
)
ax4.set_xlabel("Predicted", fontsize=11)
ax4.set_ylabel("Actual", fontsize=11)

# ── Final layout ──────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = "svm_visualization.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n✓  Saved → {os.path.abspath(out_path)}")
plt.show()
