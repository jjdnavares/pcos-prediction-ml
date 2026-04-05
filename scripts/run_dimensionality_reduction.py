"""
Dimensionality reduction — PCA analysis and t-SNE/UMAP visualization.

Outputs:
    visualizations/eda/  — PCA and t-SNE/UMAP plots
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.config import (
    TRAIN_DATA_FILE, TEST_DATA_FILE, MODEL_CONFIG_FILE,
    VISUALIZATIONS_DIR, PLOT_DPI, PLOT_STYLE
)

EDA_DIR = VISUALIZATIONS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use(PLOT_STYLE)

# ── Load data ─────────────────────────────────────────────────────────────────
with open(MODEL_CONFIG_FILE) as f:
    config = json.load(f)
feature_names = config.get("features") or config.get("selected_features")

target = "PCOS (Y/N)"

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

# Train/test CSVs are already scaled (pipeline scales before saving)
y = test_df[target].values
X_scaled = test_df[feature_names].values
print(f"Data shape: {X_scaled.shape}, Classes: {np.bincount(y.astype(int))}")

# ── 1. PCA — explained variance ──────────────────────────────────────────────
pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_90 = np.argmax(cum_var >= 90) + 1
n_95 = np.argmax(cum_var >= 95) + 1

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
            pca_full.explained_variance_ratio_ * 100, color="#5B9BD5", alpha=0.7)
axes[0].set_title("Scree Plot", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")

# Cumulative variance
axes[1].plot(range(1, len(cum_var) + 1), cum_var, "o-", color="#ED7D31", markersize=4)
axes[1].axhline(y=90, color="red", linestyle="--", alpha=0.7, label=f"90% → {n_90} PCs")
axes[1].axhline(y=95, color="green", linestyle="--", alpha=0.7, label=f"95% → {n_95} PCs")
axes[1].set_title("Cumulative Variance Explained", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].legend()

plt.suptitle("PCA — Variance Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "09_pca_variance.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print(f"✓ PCA variance plot saved (90% at {n_90} PCs, 95% at {n_95} PCs)")

# ── 2. PCA — 2D projection ───────────────────────────────────────────────────
pca_2d = PCA(n_components=2, random_state=42)
X_pca = pca_2d.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm",
                     alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA — 2D Projection by PCOS Status", fontsize=13, fontweight="bold")
legend = ax.legend(*scatter.legend_elements(), loc="best", title="Class")
legend.get_texts()[0].set_text("Non-PCOS")
legend.get_texts()[1].set_text("PCOS")
plt.tight_layout()
plt.savefig(EDA_DIR / "10_pca_2d_scatter.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ PCA 2D scatter saved")

# ── 3. PCA — loadings heatmap ────────────────────────────────────────────────
n_show = min(5, len(feature_names))
loadings = pd.DataFrame(
    pca_full.components_[:n_show],
    columns=feature_names,
    index=[f"PC{i+1}" for i in range(n_show)]
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax)
ax.set_title(f"PCA Loadings — Top {n_show} Components", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "11_pca_loadings.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ PCA loadings heatmap saved")

# ── 4. t-SNE — 2D visualization ──────────────────────────────────────────────
print("Computing t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="coolwarm",
                     alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_title("t-SNE — 2D Visualization by PCOS Status", fontsize=13, fontweight="bold")
legend = ax.legend(*scatter.legend_elements(), loc="best", title="Class")
legend.get_texts()[0].set_text("Non-PCOS")
legend.get_texts()[1].set_text("PCOS")
plt.tight_layout()
plt.savefig(EDA_DIR / "12_tsne_2d_scatter.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ t-SNE 2D scatter saved")

# ── 5. PCA vs t-SNE side by side ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, X_proj, title in [(axes[0], X_pca, "PCA"), (axes[1], X_tsne, "t-SNE")]:
    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap="coolwarm",
                         alpha=0.6, edgecolors="k", linewidths=0.3, s=40)
    ax.set_title(title, fontsize=13, fontweight="bold")
    legend = ax.legend(*scatter.legend_elements(), loc="best", title="Class")
    legend.get_texts()[0].set_text("Non-PCOS")
    legend.get_texts()[1].set_text("PCOS")

plt.suptitle("Dimensionality Reduction Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "13_pca_vs_tsne.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ PCA vs t-SNE comparison saved")

print(f"\n✅ All dimensionality reduction plots saved to {EDA_DIR}")
