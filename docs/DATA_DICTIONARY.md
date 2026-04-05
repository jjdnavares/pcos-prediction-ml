# Data Dictionary

> PCOS Prediction ML Project
> Source: Kaggle — Polycystic Ovary Syndrome (PCOS) dataset by prasoonkottarathil
> Dataset: `data/raw/PCOS_data_without_infertility.xlsx` (Sheet: Full_new)
> Records: 541 patients | Original Features: 45 columns (41 usable)

---

## Target Variable

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| PCOS (Y/N) | int | 0 = No, 1 = Yes | Diagnosis of Polycystic Ovary Syndrome |

---

## Identifiers

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| Sl. No | int | 1–541 | Serial number (dropped during training) |
| Patient File No. | int | 1–541 | Patient identifier (dropped during training) |

---

## Demographic Features

| Variable | Type | Unit | Range | Missing | Description |
|----------|------|------|-------|---------|-------------|
| Age (yrs) | int | years | 20–48 | 0 | Patient age at time of data collection |
| Marraige Status (Yrs) | float | years | 0–30 | 1 | Duration of marriage |
| Blood Group | int | encoded | 11–18 | 0 | Blood group (encoded: 11=A+, 12=A-, 13=B+, 14=B-, 15=O+, 16=O-, 17=AB+, 18=AB-) |

---

## Anthropometric Features

| Variable | Type | Unit | Range | Missing | Description |
|----------|------|------|-------|---------|-------------|
| Weight (Kg) | float | kg | 31–108 | 0 | Body weight |
| Height(Cm) | float | cm | 137–180 | 0 | Body height |
| BMI | float | kg/m² | 12.4–38.9 | 0 | Body Mass Index |
| Hip(inch) | int | inches | 26–48 | 0 | Hip circumference |
| Waist(inch) | int | inches | 24–47 | 0 | Waist circumference |
| Waist:Hip Ratio | float | ratio | 0.76–0.98 | 0 | Waist-to-hip ratio |

---

## Vital Signs

| Variable | Type | Unit | Range | Missing | Description |
|----------|------|------|-------|---------|-------------|
| Pulse rate(bpm) | int | bpm | 13–82 | 0 | Resting pulse rate |
| RR (breaths/min) | int | breaths/min | 16–28 | 0 | Respiratory rate |
| BP _Systolic (mmHg) | int | mmHg | 12–140 | 0 | Systolic blood pressure |
| BP _Diastolic (mmHg) | int | mmHg | 8–100 | 0 | Diastolic blood pressure |

---

## Hormonal Markers

| Variable | Type | Unit | Range | Missing | Description |
|----------|------|------|-------|---------|-------------|
| I beta-HCG(mIU/mL) | float | mIU/mL | 1.3–32,461 | 0 | Beta-HCG level (test 1) — pregnancy/tumor marker |
| II beta-HCG(mIU/mL) | object* | mIU/mL | varies | 0 | Beta-HCG level (test 2) — contains #NAME? errors |
| FSH(mIU/mL) | float | mIU/mL | 0.21–5,052 | 0 | Follicle-Stimulating Hormone |
| LH(mIU/mL) | float | mIU/mL | 0.02–2,018 | 0 | Luteinizing Hormone |
| FSH/LH | float | ratio | 0.002–1,373 | 0 | FSH-to-LH ratio (pre-computed in dataset) |
| TSH (mIU/L) | float | mIU/L | 0.04–65 | 0 | Thyroid-Stimulating Hormone |
| AMH(ng/mL) | object* | ng/mL | varies | 0 | Anti-Mullerian Hormone — contains #NAME? errors |
| PRL(ng/mL) | float | ng/mL | 0.4–128.2 | 0 | Prolactin |
| PRG(ng/mL) | float | ng/mL | 0.047–85 | 0 | Progesterone |

> *object dtype due to Excel formula errors (#NAME?) that are coerced to NaN during data cleaning.

---

## Metabolic Indicators

| Variable | Type | Unit | Range | Missing | Description |
|----------|------|------|-------|---------|-------------|
| Hb(g/dl) | float | g/dL | 8.5–14.8 | 0 | Hemoglobin level |
| Vit D3 (ng/mL) | float | ng/mL | 0–6,015 | 0 | Vitamin D3 level |
| RBS(mg/dl) | float | mg/dL | 60–350 | 0 | Random Blood Sugar |

---

## Reproductive Health Features

| Variable | Type | Unit | Range | Missing | Description |
|----------|------|------|-------|---------|-------------|
| Cycle(R/I) | int | encoded | 2–5 | 0 | Menstrual cycle regularity (2=Regular, 4=Irregular*†, 5=Irregular) |
| Cycle length(days) | int | days | 0–12 | 0 | Menstrual cycle length |
| Pregnant(Y/N) | int | binary | 0–1 | 0 | Currently pregnant (0=No, 1=Yes) |
| No. of aborptions | int | count | 0–5 | 0 | Number of prior abortions |

> †Value 4 is undocumented in the original dataset; mapped to 5 (Irregular) during cleaning.

---

## Ovarian Morphology (Ultrasound)

| Variable | Type | Unit | Range | Missing | Description |
|----------|------|------|-------|---------|-------------|
| Follicle No. (L) | int | count | 0–22 | 0 | Number of follicles in left ovary |
| Follicle No. (R) | int | count | 0–20 | 0 | Number of follicles in right ovary |
| Avg. F size (L) (mm) | float | mm | 0–24 | 0 | Average follicle size, left ovary |
| Avg. F size (R) (mm) | float | mm | 0–24 | 0 | Average follicle size, right ovary |
| Endometrium (mm) | float | mm | 0–18 | 0 | Endometrial thickness |

---

## Physical Symptoms (Binary)

| Variable | Type | Values | Missing | Description |
|----------|------|--------|---------|-------------|
| Weight gain(Y/N) | int | 0=No, 1=Yes | 0 | Unexplained weight gain |
| hair growth(Y/N) | int | 0=No, 1=Yes | 0 | Excess hair growth (hirsutism) |
| Skin darkening (Y/N) | int | 0=No, 1=Yes | 0 | Acanthosis nigricans |
| Hair loss(Y/N) | int | 0=No, 1=Yes | 0 | Alopecia |
| Pimples(Y/N) | int | 0=No, 1=Yes | 0 | Acne |

---

## Lifestyle Features

| Variable | Type | Values | Missing | Description |
|----------|------|--------|---------|-------------|
| Fast food (Y/N) | float | 0=No, 1=Yes | 1 | Regular fast food consumption |
| Reg.Exercise(Y/N) | int | 0=No, 1=Yes | 0 | Regular exercise habit |

---

## Artifact Columns (Dropped During Cleaning)

| Variable | Type | Missing | Description |
|----------|------|---------|-------------|
| Unnamed: 44 | object | 539 | Phantom column from Excel import — contains only 2 non-null artifacts (".", 7). Dropped during data cleaning. |

---

## Engineered Features (Created During Pipeline)

| Variable | Type | Unit | Derivation | Clinical Rationale |
|----------|------|------|------------|--------------------|
| LH_FSH_Ratio | float | ratio | LH(mIU/mL) / FSH(mIU/mL) | LH/FSH > 2 indicates hormonal imbalance characteristic of PCOS |
| Total_Follicle_Count | int | count | Follicle No. (L) + Follicle No. (R) | Rotterdam criterion: >= 12 follicles per ovary is diagnostic |
| WHR_Recalc | float | ratio | Waist(inch) / Hip(inch) | Recalculated for consistency; central obesity indicator |
| Symptom_Burden | int | count | Sum of 5 binary symptom columns | Composite physical symptom severity score |
| Age_Group | category | — | Binned: 18–25, 26–35, 36+ | Age-stratified risk grouping |
| BMI_Category | category | — | WHO bins: Underweight/Normal/Overweight/Obese | Standard clinical weight classification |

---

## Data Quality Notes

1. **Excel formula errors**: `II beta-HCG(mIU/mL)` and `AMH(ng/mL)` contain `#NAME?` values from broken Excel formulas, stored as object dtype. These are coerced to NaN and imputed with median during cleaning.
2. **Phantom column**: `Unnamed: 44` is an Excel import artifact with 539/541 nulls. Dropped during cleaning.
3. **Undocumented value**: `Cycle(R/I)` contains value 4 not documented in the original dataset. Mapped to 5 (Irregular) during cleaning.
4. **Outlier ranges**: Some features (FSH, LH, Vit D3, beta-HCG) have extreme outliers likely due to measurement errors. These are winsorized using IQR method during preprocessing.
5. **Missing values**: Only 2 features have missing values — `Marraige Status (Yrs)` (1) and `Fast food (Y/N)` (1). Imputed with median.
6. **Class distribution**: 177 PCOS positive (32.7%) vs 364 PCOS negative (67.3%) — addressed with SMOTE oversampling.
