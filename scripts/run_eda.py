"""
Applied EDA — distributions, relationships, and class-level comparisons.

Outputs:
    visualizations/eda/  — all EDA plots
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import (
    RAW_DATA_FILE, RAW_SHEET_NAME, VISUALIZATIONS_DIR,
    PLOT_DPI, PLOT_STYLE, CLEANED_DATA_FILE
)

EDA_DIR = VISUALIZATIONS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use(PLOT_STYLE)
sns.set_palette("husl")

# ── Load cleaned data ────────────────────────────────────────────────────────
df = pd.read_csv(CLEANED_DATA_FILE)
target = "PCOS (Y/N)"
print(f"Loaded cleaned data: {df.shape}")

# ── 1. Class distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df[target].value_counts()
labels = ["Non-PCOS", "PCOS"]

axes[0].bar(labels, counts.values, color=["#5B9BD5", "#ED7D31"])
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")
axes[0].set_title("Class Distribution (Count)")
axes[0].set_ylabel("Count")

axes[1].pie(counts.values, labels=labels, autopct="%1.1f%%",
            colors=["#5B9BD5", "#ED7D31"], startangle=90)
axes[1].set_title("Class Distribution (%)")

plt.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "01_class_distribution.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ Class distribution plot saved")

# ── 2. Numeric feature distributions ─────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")
# Pick key clinical features
key_features = [
    " Age (yrs)", "BMI", "LH(mIU/mL)", "FSH(mIU/mL)",
    "Vit D3 (ng/mL)", "RBS(mg/dl)", "Endometrium (mm)",
    "Follicle No. (L)", "Follicle No. (R)", "Cycle length(days)",
    "TSH (mIU/L)", "AMH(ng/mL)"
]
key_features = [f for f in key_features if f in df.columns]

n = len(key_features)
ncols = 3
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for i, col in enumerate(key_features):
    sns.histplot(data=df, x=col, hue=target, kde=True, ax=axes[i],
                 palette=["#5B9BD5", "#ED7D31"], alpha=0.6)
    axes[i].set_title(col, fontsize=10)
    axes[i].legend(["Non-PCOS", "PCOS"], fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Distributions by PCOS Status", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(EDA_DIR / "02_feature_distributions.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ Feature distributions plot saved")

# ── 3. Boxplots: key features by class ───────────────────────────────────────
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for i, col in enumerate(key_features):
    sns.boxplot(data=df, x=target, y=col, ax=axes[i],
                palette=["#5B9BD5", "#ED7D31"])
    axes[i].set_xticklabels(["Non-PCOS", "PCOS"])
    axes[i].set_title(col, fontsize=10)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Feature Boxplots by PCOS Status", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(EDA_DIR / "03_boxplots_by_class.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ Boxplots saved")

# ── 4. Correlation heatmap ───────────────────────────────────────────────────
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EDA_DIR / "04_correlation_heatmap.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ Correlation heatmap saved")

# ── 5. Top correlations with target ──────────────────────────────────────────
target_corr = df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 7))
colors = sns.color_palette("RdYlGn_r", len(target_corr))
target_corr.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Top 15 Features — Correlation with PCOS", fontsize=13, fontweight="bold")
ax.set_xlabel("|Pearson Correlation|")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(EDA_DIR / "05_target_correlation.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ Target correlation plot saved")

# ── 6. Symptom prevalence by PCOS status ─────────────────────────────────────
symptoms = ["Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
            "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)"]
symptoms = [s for s in symptoms if s in df.columns]

symptom_prev = df.groupby(target)[symptoms].mean()
symptom_prev.index = ["Non-PCOS", "PCOS"]

fig, ax = plt.subplots(figsize=(10, 6))
symptom_prev.T.plot(kind="barh", ax=ax, color=["#5B9BD5", "#ED7D31"])
ax.set_title("Symptom Prevalence by PCOS Status", fontsize=13, fontweight="bold")
ax.set_xlabel("Proportion")
ax.legend(title="Class")
plt.tight_layout()
plt.savefig(EDA_DIR / "06_symptom_prevalence.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ Symptom prevalence plot saved")

# ── 7. Pairplot of top features ──────────────────────────────────────────────
top4 = target_corr.index[:4].tolist()
pair_df = df[top4 + [target]].copy()
pair_df[target] = pair_df[target].map({0: "Non-PCOS", 1: "PCOS"})

g = sns.pairplot(pair_df, hue=target, palette=["#5B9BD5", "#ED7D31"],
                 diag_kind="kde", plot_kws={"alpha": 0.5})
g.figure.suptitle("Pairplot — Top 4 Correlated Features", y=1.02, fontsize=14, fontweight="bold")
plt.savefig(EDA_DIR / "07_pairplot_top_features.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ Pairplot saved")

# ── 8. Missing values summary ────────────────────────────────────────────────
raw_df = pd.read_excel(RAW_DATA_FILE, sheet_name=RAW_SHEET_NAME)
missing = raw_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.5)))
    missing.plot(kind="barh", ax=ax, color="#ED7D31")
    ax.set_title("Missing Values in Raw Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "08_missing_values.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    print("✓ Missing values plot saved")
else:
    print("✓ No missing values to plot")

print(f"\n✅ All EDA plots saved to {EDA_DIR}")
