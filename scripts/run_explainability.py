"""
Feature importance & explainability — SHAP, LIME, and model-based importances.

Outputs:
    visualizations/explainability/  — all explainability plots
"""

import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from src.config import (
    BEST_MODEL_FILE, SCALER_FILE, MODEL_CONFIG_FILE,
    TRAIN_DATA_FILE, TEST_DATA_FILE, VISUALIZATIONS_DIR,
    PLOT_DPI, PLOT_STYLE, SELECTED_FEATURES_FILE
)

EXPL_DIR = VISUALIZATIONS_DIR / "explainability"
EXPL_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use(PLOT_STYLE)

# ── Load artifacts ────────────────────────────────────────────────────────────
model = joblib.load(BEST_MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

with open(MODEL_CONFIG_FILE) as f:
    config = json.load(f)
feature_names = config.get("features") or config.get("selected_features")

train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)

target = "PCOS (Y/N)"
y_train = train_df[target].values
y_test = test_df[target].values

# Train/test CSVs are already scaled (pipeline scales before saving)
X_train_scaled = train_df[feature_names].values
X_test_scaled = test_df[feature_names].values

print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Model: {type(model).__name__}")
print(f"Features ({len(feature_names)}): {feature_names}")

# ── 1. Model-based feature importance (Logistic Regression coefficients) ─────
fig, ax = plt.subplots(figsize=(10, 7))
coefs = model.coef_[0]
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefs
}).sort_values("Coefficient", key=abs, ascending=True)

colors = ["#ED7D31" if c > 0 else "#5B9BD5" for c in importance_df["Coefficient"]]
importance_df.plot(x="Feature", y="Coefficient", kind="barh", ax=ax, color=colors, legend=False)
ax.set_title("Logistic Regression Coefficients (Scaled Features)", fontsize=13, fontweight="bold")
ax.set_xlabel("Coefficient Value")
ax.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(EXPL_DIR / "01_model_coefficients.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ Model coefficients plot saved")

# ── 2. SHAP values (global) ──────────────────────────────────────────────────
print("Computing SHAP values...")
explainer = shap.LinearExplainer(model, X_train_scaled, feature_names=feature_names)
shap_values = explainer(X_test_scaled)

# Summary plot (bee swarm)
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot (Bee Swarm)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(EXPL_DIR / "02_shap_summary_beeswarm.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ SHAP bee swarm plot saved")

# Bar plot (mean |SHAP|)
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names,
                  plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(EXPL_DIR / "03_shap_importance_bar.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ SHAP bar importance plot saved")

# ── 3. SHAP dependence plots for top 3 features ─────────────────────────────
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
top3_idx = np.argsort(mean_abs_shap)[-3:][::-1]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, idx in enumerate(top3_idx):
    shap.dependence_plot(idx, shap_values.values, X_test_scaled,
                         feature_names=feature_names, ax=axes[i], show=False)
plt.suptitle("SHAP Dependence Plots — Top 3 Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EXPL_DIR / "04_shap_dependence_top3.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ SHAP dependence plots saved")

# ── 4. SHAP waterfall for individual predictions ─────────────────────────────
# Pick one PCOS-positive and one PCOS-negative from test set
pos_idx = np.where(y_test == 1)[0][0]
neg_idx = np.where(y_test == 0)[0][0]

for label, idx in [("positive", pos_idx), ("negative", neg_idx)]:
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(shap_values[idx], show=False)
    plt.title(f"SHAP Waterfall — PCOS {label.capitalize()} Case (Test #{idx})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EXPL_DIR / f"05_shap_waterfall_{label}.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
print("✓ SHAP waterfall plots saved")

# ── 5. LIME explanations ─────────────────────────────────────────────────────
print("Computing LIME explanations...")
lime_explainer = LimeTabularExplainer(
    X_train_scaled,
    feature_names=feature_names,
    class_names=["Non-PCOS", "PCOS"],
    mode="classification",
    random_state=42
)

for label, idx in [("positive", pos_idx), ("negative", neg_idx)]:
    exp = lime_explainer.explain_instance(
        X_test_scaled[idx],
        model.predict_proba,
        num_features=len(feature_names)
    )
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(10, 7)
    plt.title(f"LIME Explanation — PCOS {label.capitalize()} Case (Test #{idx})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(EXPL_DIR / f"06_lime_{label}.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
print("✓ LIME explanation plots saved")

# ── 6. SHAP vs Coefficient comparison ────────────────────────────────────────
comparison_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient (abs)": np.abs(coefs),
    "SHAP (mean abs)": mean_abs_shap
})
comparison_df["Coef Rank"] = comparison_df["Coefficient (abs)"].rank(ascending=False).astype(int)
comparison_df["SHAP Rank"] = comparison_df["SHAP (mean abs)"].rank(ascending=False).astype(int)
comparison_df = comparison_df.sort_values("SHAP Rank")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

comparison_df.sort_values("Coefficient (abs)", ascending=True).plot(
    x="Feature", y="Coefficient (abs)", kind="barh", ax=axes[0], color="#5B9BD5", legend=False)
axes[0].set_title("By |Coefficient|", fontsize=12, fontweight="bold")

comparison_df.sort_values("SHAP (mean abs)", ascending=True).plot(
    x="Feature", y="SHAP (mean abs)", kind="barh", ax=axes[1], color="#ED7D31", legend=False)
axes[1].set_title("By Mean |SHAP Value|", fontsize=12, fontweight="bold")

plt.suptitle("Feature Importance: Coefficients vs SHAP", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(EXPL_DIR / "07_shap_vs_coefficients.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close()
print("✓ SHAP vs coefficients comparison saved")

print(f"\n✅ All explainability plots saved to {EXPL_DIR}")
