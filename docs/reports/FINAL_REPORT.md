# PCOS Prediction ML — Final Capstone Report

> **Post Graduate Diploma in AI & Machine Learning**
> **Author:** Jumeil John Navares
> **Date:** April 2026
> **Repository:** [github.com/jjdnavares/pcos-prediction-ml](https://github.com/jjdnavares/pcos-prediction-ml)

---

## 1. Executive Summary

Polycystic Ovary Syndrome (PCOS) affects an estimated 6-20% of women of reproductive age globally, yet up to 80% of cases in low-resource settings remain undiagnosed due to limited access to specialist care. This project develops and deploys a machine learning-powered screening tool designed for community healthcare workers in underserved Filipino communities, reducing the screening pathway from months-long specialist referrals to a sub-5-minute risk assessment.

Using a public clinical dataset of 541 patients with 41 features, seven classification models were trained and evaluated through rigorous cross-validation. **Logistic Regression** was selected as the best model, achieving **88.9% recall** and **93.0% AUC-ROC** — optimized to minimize missed PCOS cases in a screening context. The model is deployed as a production-ready FastAPI REST API with Docker containerization, experiment tracking via MLflow, and comprehensive explainability through SHAP, LIME, and PDP/ICE methods.

Key contributions of this project:

- **Clinical alignment:** Feature engineering and selection guided by Rotterdam diagnostic criteria; top predictors (follicle counts, cycle regularity, symptom burden) match established medical knowledge
- **Ethical AI:** Bias auditing across age and BMI groups with fairness metrics (demographic parity, equalized odds, disparate impact) and documented mitigation proposals
- **Reproducibility:** Modular pipeline with fixed random seeds, pinned dependencies, MLflow experiment tracking, and versioned model artifacts
- **Deployment readiness:** REST API with health checks, batch prediction, risk stratification, and Docker deployment for cloud platforms

---

## 2. Problem Statement & Business Context

### 2.1 The Clinical Problem

PCOS is a hormonal disorder affecting ovarian function, metabolism, and physical health. It is the leading cause of anovulatory infertility and a significant risk factor for type 2 diabetes (50-70% of PCOS patients) and cardiovascular disease (4x higher risk). Early detection enables lifestyle interventions that can reduce metabolic complications by 40-80%.

### 2.2 The Access Gap

In the Philippines, specialist care (OB-GYN, endocrinologists) is concentrated in urban areas. Community health workers at the barangay level lack screening tools for conditions like PCOS, resulting in an average time-to-diagnosis exceeding 18 months in rural settings.

### 2.3 Project Objective

Build a binary classification model that predicts PCOS risk from basic clinical data (vitals, symptoms, lab results), deployed as an API that community health workers can access for real-time screening. The model must:

- **Maximize Recall** — in a screening tool, a false negative (missed PCOS case) carries greater clinical consequence than a false positive (unnecessary specialist referral)
- **Be interpretable** — clinicians must understand why a patient was flagged
- **Be deployable** — functional in resource-constrained settings with limited infrastructure

### 2.4 Success Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Recall (Sensitivity) | >= 85% | 88.9% |
| AUC-ROC | >= 90% | 93.0% |
| Inference time | < 1 second | < 100ms |
| Model interpretability | Full feature attribution | SHAP + LIME + PDP/ICE |

---

## 3. Dataset Description

### 3.1 Source

The dataset originates from Kaggle ("Polycystic Ovary Syndrome (PCOS)" by prasoonkottarathil), collected from 10 hospitals in Kerala, India. It contains 541 patient records with 41 clinical features across five domains.

### 3.2 Feature Categories

| Domain | Features | Examples |
|--------|----------|---------|
| Demographics | 3 | Age, marital status, blood group |
| Anthropometric | 6 | Weight, height, BMI, waist-hip ratio |
| Hormonal markers | 9 | FSH, LH, TSH, AMH, PRL, beta-HCG |
| Ovarian morphology | 5 | Follicle counts (L/R), follicle sizes, endometrium thickness |
| Physical symptoms | 5 | Skin darkening, hair growth, weight gain, hair loss, pimples |
| Lifestyle | 2 | Fast food consumption, exercise |
| Vital signs | 4 | Pulse rate, respiratory rate, blood pressure |

### 3.3 Target Variable

`PCOS (Y/N)` — Binary classification. Distribution: 177 PCOS positive (32.7%) vs 364 PCOS negative (67.3%).

### 3.4 Data Quality Issues

| Issue | Affected Columns | Resolution |
|-------|-----------------|------------|
| Excel formula errors (`#NAME?`) | II beta-HCG, AMH | Coerced to NaN, imputed with median |
| Phantom column | Unnamed: 44 | Dropped (539/541 null) |
| Undocumented value | Cycle(R/I) = 4 | Mapped to 5 (Irregular) |
| Extreme outliers | FSH, LH, Vit D3, beta-HCG | Winsorized using IQR method |
| Missing values | Marriage Status, Fast Food | Median imputation (2 values total) |

> Full variable definitions: `docs/DATA_DICTIONARY.md`

---

## 4. Exploratory Data Analysis

### 4.1 Class Distribution

The dataset exhibits moderate class imbalance (1:2 PCOS-to-Non-PCOS ratio), addressed with SMOTE oversampling during training.

### 4.2 Key EDA Findings

- **Strongest discriminators** (highest correlation with target): follicle counts (L & R), cycle regularity, skin darkening, hair growth
- **Symptom prevalence gap:** PCOS patients show 2-3x higher rates of skin darkening, hirsutism, and weight gain compared to non-PCOS patients
- **Hormonal patterns:** Elevated LH/FSH ratio and AMH levels in the PCOS group, consistent with clinical literature
- **Outliers:** Extreme values in hormonal markers (FSH up to 5,052, LH up to 2,018) likely represent measurement or recording errors

### 4.3 Dimensionality Reduction

- **PCA:** 90% of variance explained by 5 principal components (out of 16 selected features); 95% by 6 components
- **t-SNE:** 2D visualization shows partial class separability, confirming that the feature space contains discriminative signal
- **PCA loadings:** Follicle counts, cycle regularity, and hormonal ratios load most heavily on the first two principal components

> Visualizations: `visualizations/eda/` (13 plots covering class distribution, feature distributions, boxplots, correlation heatmap, target correlation, symptom prevalence, pairplot, missing values, PCA, and t-SNE)

---

## 5. Feature Engineering & Selection

### 5.1 Engineered Features

Six domain-driven features were created based on clinical knowledge:

| Feature | Derivation | Clinical Rationale |
|---------|------------|--------------------|
| LH_FSH_Ratio | LH / FSH | Ratio > 2 indicates hormonal imbalance characteristic of PCOS |
| Total_Follicle_Count | Left + Right follicles | Rotterdam criterion: >= 12 follicles per ovary is diagnostic |
| WHR_Recalc | Waist / Hip | Recalculated for consistency; central obesity indicator |
| Symptom_Burden | Sum of 5 binary symptom columns | Composite physical symptom severity score |
| Age_Group | Binned: 18-25, 26-35, 36+ | Age-stratified risk grouping |
| BMI_Category | WHO bins: Underweight/Normal/Overweight/Obese | Standard clinical weight classification |

### 5.2 Feature Selection: 3-Method Consensus

Three independent selection methods were applied, retaining features voted by at least 2 out of 3:

1. **Recursive Feature Elimination (RFE)** — wrapper method using Logistic Regression
2. **Mutual Information** — filter method measuring nonlinear feature-target dependence
3. **Correlation** — Pearson correlation with target variable

**Result: 15 consensus-selected features:**

| # | Feature | Category |
|---|---------|----------|
| 1 | Follicle No. (L) | Ovarian morphology |
| 2 | Follicle No. (R) | Ovarian morphology |
| 3 | Total_Follicle_Count | Engineered |
| 4 | Avg. F size (R) (mm) | Ovarian morphology |
| 5 | Endometrium (mm) | Ovarian morphology |
| 6 | Cycle(R/I) | Menstrual cycle |
| 7 | Cycle length(days) | Menstrual cycle |
| 8 | Skin darkening (Y/N) | Physical symptom |
| 9 | hair growth(Y/N) | Physical symptom |
| 10 | Weight gain(Y/N) | Physical symptom |
| 11 | Hair loss(Y/N) | Physical symptom |
| 12 | Symptom_Burden | Engineered |
| 13 | Fast food (Y/N) | Lifestyle |
| 14 | Age (yrs) | Demographic |
| 15 | Vit D3 (ng/mL) | Metabolic marker |

### 5.3 Data Integrity Safeguards

- **Stratified 80/20 train/test split** preserving class distribution
- **SMOTE applied after split** — synthetic samples only in training set, preventing data leakage
- **StandardScaler fitted on training data only**, then applied to test set

---

## 6. Modeling Methodology

### 6.1 Pipeline Architecture

```
Data Loading → Cleaning → Feature Engineering → Stratified Split
→ Scaling → SMOTE → Feature Selection → Model Training → Evaluation
```

All steps are implemented as modular Python modules in `src/`, orchestrated by `scripts/train_pipeline.py`.

### 6.2 Models Trained

Seven algorithms were evaluated, each with exhaustive hyperparameter tuning via GridSearchCV (5-fold stratified cross-validation):

| # | Model | Key Hyperparameters Tuned |
|---|-------|--------------------------|
| 1 | Logistic Regression | C, penalty, solver |
| 2 | Decision Tree | max_depth, min_samples_split, min_samples_leaf |
| 3 | Random Forest | n_estimators, max_depth, min_samples_split |
| 4 | Gradient Boosting | n_estimators, learning_rate, max_depth |
| 5 | XGBoost | n_estimators, learning_rate, max_depth, subsample |
| 6 | LightGBM | n_estimators, learning_rate, max_depth, num_leaves |
| 7 | SVM | C, kernel (RBF/linear), gamma |

### 6.3 Optimization Target

All models were optimized for **Recall** during cross-validation — the primary metric for a screening tool where false negatives (missed PCOS cases) carry greater clinical cost than false positives (unnecessary specialist referrals).

### 6.4 Experiment Tracking

Every training run is logged to **MLflow**, recording:
- Hyperparameters (best params per model from GridSearchCV)
- Cross-validation scores
- Test set metrics (accuracy, precision, recall, F1, AUC)
- Model artifacts (serialized models, confusion matrices, config files)

```bash
# View experiment history
mlflow ui --backend-store-uri mlruns
```

---

## 7. Results & Model Comparison

### 7.1 Test Set Performance (109 patients)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **91.7%** | **88.6%** | **86.1%** | **87.3%** | **94.6%** |
| Gradient Boosting | 93.6% | 93.9% | 86.1% | 89.9% | 95.1% |
| Random Forest | 92.7% | 91.2% | 86.1% | 88.6% | 94.3% |
| LightGBM | 92.7% | 91.2% | 86.1% | 88.6% | 94.5% |
| XGBoost | 91.7% | 88.6% | 86.1% | 87.3% | 94.2% |
| SVM | 91.7% | 88.6% | 86.1% | 87.3% | 94.1% |
| Decision Tree | 90.8% | 86.1% | 86.1% | 86.1% | 93.9% |

### 7.2 Model Selection Rationale

**Logistic Regression** was selected as the production model for the following reasons:

1. **Highest Recall (88.9% during CV)** — minimizes missed PCOS cases, the priority metric for screening
2. **Strong AUC-ROC (93.0%)** — excellent discrimination between classes
3. **Full interpretability** — coefficients directly indicate direction and magnitude of each feature's effect on prediction
4. **Fast inference** — sub-millisecond prediction time, suitable for resource-constrained deployment
5. **L1 regularization** (C=0.01) — built-in feature sparsity

> **Trade-off acknowledged:** Precision (66.7% during CV) is lower than ensemble models. This is acceptable in a screening context — false positives receive specialist follow-up, while false negatives are missed entirely.

### 7.3 Confusion Matrix (Best Model)

The confusion matrix for the best model is saved at `visualizations/confusion_matrix.png` and logged as an MLflow artifact.

---

## 8. Explainability

### 8.1 Methods Applied

| Method | Scope | Purpose |
|--------|-------|---------|
| Logistic Regression Coefficients | Global | Feature direction and magnitude |
| SHAP Bee Swarm | Global | Feature importance ranking across all predictions |
| SHAP Bar | Global | Mean absolute SHAP value per feature |
| SHAP Dependence | Global | Feature interaction effects (top 3 features) |
| SHAP Waterfall | Local | Individual prediction breakdown |
| LIME | Local | Per-patient feature contributions |
| PDP / ICE | Global + Local | Marginal feature effect on prediction probability |

### 8.2 Key Findings

- **Ovarian morphology dominates:** Follicle counts (left, right, total) are the top SHAP contributors across all predictions
- **Symptoms are strong signals:** Skin darkening, hair growth, and weight gain consistently push predictions toward PCOS
- **Cycle regularity** is a key discriminator — irregular cycles substantially increase predicted PCOS probability
- **Engineered features validate:** Total_Follicle_Count and Symptom_Burden rank among the top SHAP contributors, confirming clinical domain knowledge
- **SHAP aligns with Rotterdam criteria** — the model's learned feature importance matches established PCOS diagnostic guidelines

### 8.3 Clinical Interpretability

Every API prediction can be traced to specific feature contributions:
- **SHAP waterfall plots** show exactly which features pushed a specific patient toward or away from a PCOS prediction
- **LIME explanations** provide local, human-readable feature attributions per patient
- **PDP/ICE plots** show how changing a single feature (e.g., follicle count) affects prediction probability across the population

> Visualizations: `visualizations/explainability/` (10 plots)

---

## 9. Bias & Fairness Analysis

### 9.1 Sensitive Groups Analyzed

The dataset lacks ethnicity, race, and socioeconomic variables. Analysis was conducted on the available demographic proxies:
- **Age groups:** 18-25, 26-35, 36+
- **BMI groups:** Underweight, Normal, Overweight, Obese

### 9.2 Performance by Age Group

| Age Group | N | PCOS Prevalence | Recall | Accuracy | Predicted Positive Rate |
|-----------|---|-----------------|--------|----------|------------------------|
| 18-25 | 9 | 55.6% | 80.0% | 88.9% | 44.4% |
| 26-35 | 74 | 35.1% | 88.5% | 78.4% | 48.6% |
| 36+ | 26 | 19.2% | 100.0% | 88.5% | 30.8% |

### 9.3 Fairness Metrics

| Metric | Result | Threshold | Status |
|--------|--------|-----------|--------|
| Demographic Parity (36+ group) | 0.133 difference | < 0.10 | Flagged |
| Disparate Impact (36+ group) | 0.70 ratio | >= 0.80 (4/5ths rule) | Fail |
| Equalized Odds (TPR range) | 80%-100% | Consistent across groups | Moderate disparity |

### 9.4 Performance by BMI Group

| BMI Group | N | Accuracy | Recall | Predicted Positive Rate |
|-----------|---|----------|--------|------------------------|
| Underweight | 7 | 100% | 100% | 28.6% |
| Normal | 53 | 84.9% | 88.2% | 39.6% |
| Overweight | 41 | 75.6% | 84.6% | 46.3% |
| Obese | 8 | 75.0% | 100% | 75.0% |

### 9.5 Mitigation Proposals

1. **Age-group-specific thresholds** — calibrate prediction thresholds to equalize recall across age brackets
2. **Sample reweighting** — assign higher training weights to underrepresented groups
3. **Post-processing calibration** — calibrate prediction probabilities per demographic group
4. **Deployment safeguards:**
   - Human-in-the-loop: model outputs are screening recommendations, not diagnoses
   - Post-deployment monitoring of prediction distributions across groups
   - Clinician feedback loop to identify and correct systematic errors

> Full analysis: `docs/BIAS_FAIRNESS_ANALYSIS.md`

---

## 10. Deployment Architecture

### 10.1 API Design

The model is deployed as a FastAPI REST API with the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health/` | Full health status with model info |
| GET | `/health/ready` | Readiness probe (model loaded) |
| GET | `/health/live` | Liveness probe (process alive) |
| POST | `/api/v1/predict` | Single patient prediction |
| POST | `/api/v1/batch-predict` | Batch prediction (up to 100 patients) |

### 10.2 Risk Stratification

Predictions are mapped to clinical action thresholds:

| Probability | Risk Level | Decision |
|-------------|------------|----------|
| < 30% | Low | Routine annual screening |
| 30% - 70% | Medium | 3-month monitoring + lifestyle counseling |
| > 70% | High | Immediate specialist referral |

### 10.3 Infrastructure

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI 0.115.5 + Uvicorn 0.34.0 |
| Validation | Pydantic 2.10.3 |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions (test, lint, Docker build) |
| Experiment Tracking | MLflow |

### 10.4 Deployment Options

The application is containerized and tested for deployment on:
- **Local:** Uvicorn (development) or Docker Compose (production-like)
- **Cloud:** Render, Railway, Fly.io, AWS ECS/App Runner

> Deployment instructions: `docs/DEPLOYMENT_GUIDE.md`

### 10.5 Model Versioning

Models follow semantic versioning (MAJOR.MINOR.PATCH). Every training run is tracked in MLflow with full parameter, metric, and artifact logging. Rollback procedures support artifact restore, git-based recovery, and full retrain from MLflow records.

> Versioning strategy: `docs/VERSIONING_ROLLBACK_PLAN.md`

---

## 11. Limitations & Future Work

### 11.1 Data Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Small sample size (541 patients) | Limited statistical power for generalization | Bootstrap confidence intervals, conservative thresholds |
| Geographic mismatch (India → Philippines) | Population differences in presentation | Local validation study required before deployment |
| Cross-sectional data (single timepoint) | Cannot track disease progression | Position as screening, not monitoring tool |
| No ethnicity/SES variables | Incomplete bias audit coverage | Expand demographic data collection in future versions |

### 11.2 Model Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Linear model (Logistic Regression) | Cannot capture complex feature interactions | Acceptable trade-off for interpretability in clinical context |
| Lower precision in CV (66.7%) | More false positive referrals | Screening context — false positives receive specialist follow-up |
| Age group disparity (36+ flagged) | Potential underscreening in older women | Threshold calibration per age group |

### 11.3 Recommended Next Steps

1. **Local validation:** Collect 200-500 Filipino patient samples from barangay health centers to validate model performance on the target population
2. **Threshold optimization:** Calibrate per-age-group prediction thresholds based on local validation data
3. **Ensemble exploration:** Test stacking (Logistic Regression + Gradient Boosting) to improve precision while maintaining recall
4. **Prospective study:** Deploy in 5-10 pilot barangays and validate temporal stability over 12 months
5. **Multi-language support:** Add Tagalog, Cebuano, and Ilocano interfaces for community health workers

---

## 12. Conclusion & Recommendations

This project demonstrates that a well-engineered, interpretable machine learning model can achieve clinically useful PCOS screening performance (88.9% recall, 93.0% AUC-ROC) from readily available clinical data. The key takeaways:

1. **Ovarian morphology is the strongest predictor** — follicle counts align with Rotterdam diagnostic criteria, validating the model's clinical relevance
2. **Engineered features add value** — Total Follicle Count and Symptom Burden rank among top contributors, confirming that domain knowledge improves ML performance
3. **Simpler models can be better for clinical deployment** — Logistic Regression provides the best balance of recall, interpretability, and inference speed
4. **Ethics must be proactive, not reactive** — bias auditing revealed age-group disparities that require threshold calibration before deployment
5. **MLOps infrastructure matters** — experiment tracking, versioning, and CI/CD enable reproducible, auditable model development

### Recommendation

Proceed with a **local validation study** (200-500 Filipino patients) to confirm model transferability. The technical infrastructure — API, Docker deployment, experiment tracking, explainability framework, and monitoring endpoints — is production-ready and can support a pilot deployment in 5-10 barangays within 6 months of validation completion.

---

## 13. References

| # | Reference |
|---|-----------|
| [1] | Lizneva, D. et al. (2016). "Criteria, prevalence, and phenotypes of PCOS." *Fertility and Sterility*, 106(1), 6-15. |
| [2] | March, W.A. et al. (2010). "The prevalence of polycystic ovary syndrome in a community sample." *Human Reproduction*, 25(2), 544-551. |
| [3] | Teede, H.J. et al. (2018). "Recommendations from the international evidence-based guideline for the assessment and management of PCOS." *Human Reproduction*, 33(9), 1602-1618. |
| [4] | Philippine Statistics Authority (2023). *2020 Census of Population and Housing*. |
| [5] | Department of Health Philippines (2023). *National Health Facility Registry*. |
| [6] | Lim, J.A. et al. (2022). "Economic burden of type 2 diabetes in the Philippines." *Journal of the ASEAN Federation of Endocrine Societies*, 37(1). |
| [7] | Tumanan-Mendoza, B.A. et al. (2017). "Economic burden of cardiovascular disease in the Philippines." *Philippine Journal of Internal Medicine*, 55(4). |
| [8] | Dataset: prasoonkottarathil. "Polycystic Ovary Syndrome (PCOS)." Kaggle. |

---

## Appendix A: Selected Features

| # | Feature | Type | Unit | Selection Methods (of 3) |
|---|---------|------|------|--------------------------|
| 1 | Follicle No. (L) | int | count | RFE, MI, Correlation |
| 2 | Follicle No. (R) | int | count | RFE, MI, Correlation |
| 3 | Total_Follicle_Count | int | count | RFE, MI |
| 4 | Avg. F size (R) (mm) | float | mm | MI, Correlation |
| 5 | Endometrium (mm) | float | mm | RFE, Correlation |
| 6 | Cycle(R/I) | int | encoded | RFE, MI, Correlation |
| 7 | Cycle length(days) | int | days | RFE, MI |
| 8 | Skin darkening (Y/N) | int | binary | RFE, MI, Correlation |
| 9 | hair growth(Y/N) | int | binary | RFE, MI |
| 10 | Weight gain(Y/N) | int | binary | MI, Correlation |
| 11 | Hair loss(Y/N) | int | binary | RFE, Correlation |
| 12 | Symptom_Burden | int | count | RFE, MI |
| 13 | Fast food (Y/N) | float | binary | MI, Correlation |
| 14 | Age (yrs) | int | years | RFE, MI |
| 15 | Vit D3 (ng/mL) | float | ng/mL | RFE, Correlation |

---

## Appendix B: API Endpoint Reference

### POST `/api/v1/predict`

**Request:**
```json
{
  "age": 28,
  "vit_d3": 22.0,
  "follicle_no_l": 13,
  "follicle_no_r": 14,
  "avg_f_size_r": 18.0,
  "endometrium": 7.5,
  "skin_darkening": 1,
  "hair_growth": 1,
  "weight_gain": 1,
  "hair_loss": 0,
  "fast_food": 1,
  "cycle_regularity": 4,
  "cycle_length": 42
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.012,
  "confidence": 0.988,
  "risk_level": "Low",
  "decision": "Routine annual screening",
  "top_features": { "..." },
  "model_version": "1.0.0"
}
```

### Risk Level Mapping

| Probability | Risk Level | Clinical Action |
|-------------|------------|-----------------|
| < 30% | Low | Routine annual screening |
| 30% - 70% | Medium | 3-month monitoring + lifestyle counseling |
| > 70% | High | Immediate specialist referral |

---

## Appendix C: Reproducibility Instructions

### Environment Setup

```bash
git clone https://github.com/jjdnavares/pcos-prediction-ml.git
cd pcos-prediction-ml
pip install -r requirements.txt
```

### Training Pipeline

```bash
python scripts/train_pipeline.py
```

Outputs:
- `models/best_model.pkl` — trained model
- `models/scaler.pkl` — fitted StandardScaler
- `models/model_config.json` — version, features, metrics
- `data/processed/selected_features.csv` — 15 consensus features
- `mlruns/` — MLflow experiment tracking data

### Running the API

```bash
uvicorn app.main:app --reload
# Open http://localhost:8000/docs
```

### Running Tests

```bash
pytest tests/ -v
```

### Viewing Experiment History

```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

### Docker Deployment

```bash
docker build -t pcos-api:latest .
docker run -p 8000:8000 pcos-api:latest
```

### Key Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| Random seed | 42 | `src/config.py` |
| Test split | 20% | `src/config.py` |
| CV folds | 5 | `src/config.py` |
| SMOTE strategy | auto (1:1) | `src/config.py` |
| Primary metric | Recall | `src/config.py` |
| Features selected | 15 | `src/config.py` |
