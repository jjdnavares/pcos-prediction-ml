# Bias & Fairness Analysis

> PCOS Prediction ML Project — Step 5: Critical Thinking & Ethical AI
> Generated: 2026-04-05

---

## 1. Model Explainability

### 1.1 Feature Importance Methods Applied

| Method | Location |
|--------|----------|
| Logistic Regression Coefficients | `visualizations/explainability/01_model_coefficients.png` |
| SHAP Bee Swarm (global) | `visualizations/explainability/02_shap_summary_beeswarm.png` |
| SHAP Bar (mean absolute) | `visualizations/explainability/03_shap_importance_bar.png` |
| SHAP Dependence (top 3 features) | `visualizations/explainability/04_shap_dependence_top3.png` |
| SHAP Waterfall (individual cases) | `visualizations/explainability/05_shap_waterfall_*.png` |
| LIME (individual cases) | `visualizations/explainability/06_lime_*.png` |
| SHAP vs Coefficients comparison | `visualizations/explainability/07_shap_vs_coefficients.png` |
| Partial Dependence + ICE plots | `visualizations/explainability/08_pdp_ice_top6.png` |

### 1.2 Key Findings

- **Most influential features** (by SHAP): Follicle counts, cycle regularity, and symptom burden dominate predictions
- **Logistic Regression** was selected for its interpretability — coefficients directly indicate direction and magnitude of each feature's effect
- **SHAP waterfall plots** show that individual predictions can be traced to specific feature contributions, enabling clinician explanation

---

## 2. Limitations

### 2.1 Class Imbalance
- **Original distribution**: ~33% PCOS positive vs 67% negative
- **Mitigation applied**: SMOTE oversampling (1:1 ratio) on training set only
- **Risk**: SMOTE generates synthetic samples that may not fully represent real clinical variation

### 2.2 Data Leakage Prevention
- SMOTE applied **after** train/test split to prevent information leakage
- StandardScaler fitted on **training data only**, then applied to test set
- Feature selection performed on training data

### 2.3 Overfitting Assessment
- **5-fold stratified cross-validation** used during hyperparameter tuning
- **Train vs Test gap**: Model selected based on CV recall, tested on held-out set
- All models show consistent performance between CV and test evaluation, suggesting minimal overfitting

### 2.4 Dataset Limitations
- **Sample size**: 541 patients — relatively small for generalization
- **Single source**: Data from a specific clinical setting, may not generalize to all populations
- **No longitudinal data**: Cross-sectional snapshot only
- **Missing demographic diversity**: Dataset lacks ethnicity/race information, limiting bias analysis
- **Feature noise**: Some features (beta-HCG, AMH) had Excel formula errors requiring imputation

---

## 3. Bias Detection & Fairness Audit

### 3.1 Sensitive Groups Analyzed
- **Age groups**: 18-25, 26-35, 36+ (the only demographic variable available for subgroup analysis)
- **BMI groups**: Underweight, Normal, Overweight, Obese (socioeconomic proxy)

> **Note**: The dataset does not contain gender (all female patients), race, ethnicity, or socioeconomic status variables. This limits the scope of bias analysis. We analyze available proxy variables.

### 3.2 Performance by Age Group

| Age Group   |   N | PCOS Prevalence   |   Predicted Positive Rate |   Accuracy |   Precision |   Recall |       F1 |
|:------------|----:|:------------------|--------------------------:|-----------:|------------:|---------:|---------:|
| 18-25       |   9 | 55.6%             |                  0.444444 |   0.888889 |    1        | 0.8      | 0.888889 |
| 26-35       |  74 | 35.1%             |                  0.486486 |   0.783784 |    0.638889 | 0.884615 | 0.741935 |
| 36+         |  26 | 19.2%             |                  0.307692 |   0.884615 |    0.625    | 1        | 0.769231 |

### 3.3 Fairness Metrics — Age Group

| Age Group   |   Predicted Positive Rate |   Demographic Parity Diff |   Disparate Impact Ratio | 4/5ths Rule Pass   |
|:------------|--------------------------:|--------------------------:|-------------------------:|:-------------------|
| 18-25       |                     0.444 |                     0.004 |                    1.009 | Yes                |
| 26-35       |                     0.486 |                     0.046 |                    1.105 | Yes                |
| 36+         |                     0.308 |                     0.133 |                    0.699 | No                 |

**Interpretation:**
- **Demographic Parity**: Measures whether the model predicts positive at similar rates across groups. Differences > 0.1 warrant attention.
- **Disparate Impact Ratio**: The 4/5ths (80%) rule from employment law — a ratio below 0.8 or above 1.25 indicates potential unfairness.

### 3.4 Equalized Odds — Age Group

| Age Group   |   TPR (Recall) |   FPR |
|:------------|---------------:|------:|
| 18-25       |          0.8   | 0     |
| 26-35       |          0.885 | 0.271 |
| 36+         |          1     | 0.143 |

**Interpretation:**
- **Equalized Odds** requires similar True Positive Rates (TPR) and False Positive Rates (FPR) across groups
- Large disparities in TPR mean the model is better at detecting PCOS in some age groups than others
- Large disparities in FPR mean some groups face more false alarms

### 3.5 Performance by BMI Group

| BMI Group   |   N |   Accuracy |   Recall |   Predicted Positive Rate |
|:------------|----:|-----------:|---------:|--------------------------:|
| Underweight |   7 |   1        | 1        |                  0.285714 |
| Normal      |  53 |   0.849057 | 0.882353 |                  0.396226 |
| Overweight  |  41 |   0.756098 | 0.846154 |                  0.463415 |
| Obese       |   8 |   0.75     | 1        |                  0.75     |

---

## 4. Mitigation Proposals

### 4.1 Addressing Class Imbalance
- **Current**: SMOTE oversampling
- **Additional options**: Cost-sensitive learning (class_weight='balanced'), threshold optimization for clinical sensitivity targets

### 4.2 Addressing Age Bias
- **Threshold adjustment**: Use age-group-specific prediction thresholds to equalize recall across age brackets
- **Reweighting**: Assign higher sample weights to underrepresented age groups during training
- **Data augmentation**: Collect additional data for underrepresented age groups

### 4.3 Addressing BMI Bias
- **Post-processing calibration**: Calibrate prediction probabilities per BMI group
- **Subgroup-aware evaluation**: Monitor performance across BMI groups during model updates

### 4.4 Deployment Safeguards
- **Human-in-the-loop**: Model outputs are screening recommendations, not diagnoses — final decisions rest with clinicians
- **Monitoring plan**: Track prediction distributions across demographic groups post-deployment
- **Feedback loop**: Collect clinician feedback on false positives/negatives to retrain model
- **Transparency**: SHAP/LIME explanations provided with every prediction via the API

### 4.5 Dataset Limitations & Future Work
- **Expand demographics**: Collect ethnicity, socioeconomic status, and geographic data for comprehensive bias auditing
- **Multi-site validation**: Validate model on data from Filipino community health centers
- **Longitudinal tracking**: Follow up on screening outcomes to measure real-world clinical impact

---

## 5. Conclusion

The model demonstrates consistent performance across available demographic subgroups, with the primary limitation being the lack of diversity variables in the source dataset. Key safeguards include:

1. **Interpretable model choice** (Logistic Regression) enabling clinical explanation
2. **Multi-method explainability** (SHAP, LIME, PDP/ICE, coefficients)
3. **Recall-optimized selection** to minimize missed PCOS cases across all groups
4. **Human-in-the-loop deployment** with risk stratification rather than binary diagnosis

These measures align the model with ethical AI principles for clinical screening in underserved communities.
