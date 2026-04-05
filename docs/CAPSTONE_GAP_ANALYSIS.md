# Capstone Gap Analysis

> Comparison of project deliverables against Capstone Project Requirements.
> Last updated: 2026-04-05

---

## Step 1: Problem Understanding & Framing — DONE

- [x] Business and data science problem framed
- [x] Classification task defined
- [x] Success metrics specified (Accuracy, AUC, Recall, F1)
- [x] Business KPIs identified (screening cost reduction, missed-case minimization)

---

## Step 2: Data Collection & Understanding — DONE

- [x] Public dataset sourced (Kaggle PCOS dataset, 541 patients, 41 features)
- [x] Feature types and missing values summarized
- [x] **Data dictionary** (variables, types, units, allowed values) — see `docs/DATA_DICTIONARY.md`

---

## Step 3: Data Preprocessing, Applied EDA & Feature Engineering — DONE

- [x] Data cleaning (nulls, outliers, duplicates)
- [x] Feature engineering (scaling, binning, domain-derived features)
- [x] Feature selection (consensus method: RFE, MI, correlation)
- [x] **Applied EDA report** — see `visualizations/eda/` (8 plots: class distribution, feature distributions, boxplots, correlation heatmap, target correlation, symptom prevalence, pairplot, missing values)
- [x] **Feature importance & explainability** — see `visualizations/explainability/` (SHAP bee swarm, SHAP bar, SHAP dependence, SHAP waterfall, LIME, model coefficients, SHAP vs coefficients comparison)
- [x] **Dimensionality reduction** — see `visualizations/eda/` (PCA variance/scree, PCA 2D scatter, PCA loadings heatmap, t-SNE 2D scatter, PCA vs t-SNE comparison)

---

## Step 4: Model Implementation — DONE

- [x] Logistic Regression
- [x] Decision Tree
- [x] Random Forest
- [x] XGBoost
- [x] Gradient Boosting
- [x] LightGBM
- [x] **SVM** — added to `src/model_training.py` and `src/config.py`
- [x] Model comparison with relevant metrics
- [x] Reproducibility (saved configs and artifacts in models/)

---

## Step 5: Critical Thinking — Ethical AI & Bias Auditing — DONE

- [x] **Model explainability** (SHAP, LIME, PDP, ICE) — see `visualizations/explainability/`
- [x] **Limitations addressed** (imbalance, leakage, overfitting) — see `docs/BIAS_FAIRNESS_ANALYSIS.md` Section 2
- [x] **Bias detection & fairness audits** across sensitive groups (age, BMI) — see report Section 3
- [x] **Fairness metrics** (demographic parity, equalized odds, disparate impact) — see report Section 3.3–3.4
- [x] **Mitigation proposals** (reweighting, thresholds, augmentation, post-processing) — see report Section 4
- [x] **Deliverable: "Bias & Fairness Analysis"** — `docs/BIAS_FAIRNESS_ANALYSIS.md`

---

## Step 6: Final Presentation & Communication — DONE

- [x] **Technical presentation** (12 slides) — see `docs/presentations/TECHNICAL_PRESENTATION_DECK.md`
  - Covers: problem, EDA, features, methodology, results, explainability, bias, limitations, reproducibility
- [x] **Business presentation** (9 slides) — see `docs/presentations/BUSINESS_PRESENTATION_DECK.md`
  - Covers: business problem, approach, findings, business value, recommendations, risks, ROI projection
- [x] **Presentation guide** — see `docs/presentations/PRESENTATION_GUIDE.md`

---

## Step 7: GitHub Profile & Upload — DONE

- [x] Public GitHub repo with open-source structure
- [x] src/, notebooks/, data/, models/ directories present
- [x] Reproducible code
- [ ] Final report (pending Steps 5 & 6 completion)

---

## Step 8: Deployment & MLOps (Optional) — MOSTLY DONE

- [x] Local deployment via FastAPI
- [x] Reproducible environment (requirements.txt, Docker)
- [x] CI checks (unit tests for API and data processing)
- [x] **Experiment tracking** (MLflow) — integrated into `src/model_training.py` and `scripts/train_pipeline.py`
- [ ] **Versioning & rollback plan**
- [ ] **Demo** (GIF/screencast of running app)
- [ ] Deployment guide

---

## Step 9: Use of Generative AI (Optional) — DONE

- [x] Document how Generative AI was used — see `README.md` "Use of Generative AI" section
- [x] Code + examples in repo (AI-assisted scripts throughout `src/`, `scripts/`, `app/`)
- [ ] Demo video

---

## Priority Summary

| Priority | Item | Step |
|----------|------|------|
| ~~CRITICAL~~ | ~~Bias & Fairness Analysis (SHAP/LIME, audits, fairness metrics, mitigations)~~ | ~~5~~ |
| ~~CRITICAL~~ | ~~Two presentation decks (technical + business)~~ | ~~6~~ |
| ~~HIGH~~ | ~~SHAP/LIME feature explainability~~ | ~~3 & 5~~ |
| ~~HIGH~~ | ~~PCA / dimensionality reduction~~ | ~~3~~ |
| ~~HIGH~~ | ~~Applied EDA report with visualizations~~ | ~~3~~ |
| ~~HIGH~~ | ~~Data dictionary~~ | ~~2~~ |
| ~~MEDIUM~~ | ~~SVM model experiment~~ | ~~4~~ |
| MEDIUM | Demo GIF/screencast | 8 |
| ~~LOW~~ | ~~Experiment tracking (MLflow)~~ | ~~8~~ |
| ~~LOW~~ | ~~Generative AI usage documentation~~ | ~~9~~ |
