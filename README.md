# PCOS Prediction ML

Machine learning-powered Polycystic Ovary Syndrome (PCOS) screening tool designed for community healthcare workers in underserved Filipino communities.

## Project Overview

| | |
|---|---|
| **Goal** | Early PCOS detection through accessible ML-powered screening |
| **Model** | Logistic Regression with consensus feature selection |
| **Features** | 15 selected from 41 via RFE + Mutual Information + Correlation consensus |
| **API** | FastAPI REST API with Swagger UI |
| **Deployment** | Docker-ready with health checks and CORS |

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/jjdnavares/pcos-prediction-ml.git
cd pcos-prediction-ml

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python scripts/train_pipeline.py
```

The pipeline runs 9 steps end-to-end:
1. Load raw data (541 patients from Excel)
2. Clean data (fix Excel errors, impute missing values, winsorize outliers)
3. Engineer features (LH/FSH ratio, total follicle count, symptom burden, etc.)
4. Stratified train/test split (80/20)
5. StandardScaler normalization
6. SMOTE oversampling for class balance
7. Consensus feature selection (RFE + Mutual Info + Correlation)
8. Train 6 models with GridSearchCV (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM)
9. Evaluate and save best model (optimized for Recall)

Output files:
- `models/best_model.pkl` - Trained model
- `models/scaler.pkl` - StandardScaler
- `models/model_config.json` - Feature list and metrics
- `data/processed/selected_features.csv` - 15 consensus-selected features

### Run the API

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI.

### Make a Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.012,
  "confidence": 0.988,
  "risk_level": "Low",
  "decision": "Routine annual screening",
  "top_features": { ... },
  "model_version": "1.0.0"
}
```

## Model Performance

Best model: **Logistic Regression** (selected by highest Recall)

| Metric | Score |
|--------|-------|
| Accuracy | 91.7% |
| Precision | 88.6% |
| Recall (Sensitivity) | 86.1% |
| F1-Score | 87.3% |
| ROC-AUC | 94.6% |

**Why Recall?** In a screening tool, missing a true PCOS case (false negative) is more harmful than a false alarm. The model is optimized to minimize missed diagnoses.

### All Models Compared

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 91.7% | 88.6% | 86.1% | 87.3% | 94.6% |
| Gradient Boosting | 93.6% | 93.9% | 86.1% | 89.9% | 95.1% |
| Random Forest | 92.7% | 91.2% | 86.1% | 88.6% | 94.3% |
| LightGBM | 92.7% | 91.2% | 86.1% | 88.6% | 94.5% |
| XGBoost | 91.7% | 88.6% | 86.1% | 87.3% | 94.2% |
| Decision Tree | 90.8% | 86.1% | 86.1% | 86.1% | 93.9% |

## Selected Features (15)

The consensus feature selection process uses three independent methods and retains features selected by at least 2 out of 3:

| # | Feature | Category |
|---|---------|----------|
| 1 | Follicle No. (L) | Ovarian morphology |
| 2 | Follicle No. (R) | Ovarian morphology |
| 3 | Total_Follicle_Count | Engineered (L + R) |
| 4 | Avg. F size (R) (mm) | Ovarian morphology |
| 5 | Endometrium (mm) | Ovarian morphology |
| 6 | Cycle(R/I) | Menstrual cycle |
| 7 | Cycle length(days) | Menstrual cycle |
| 8 | Skin darkening (Y/N) | Physical symptom |
| 9 | hair growth(Y/N) | Physical symptom |
| 10 | Weight gain(Y/N) | Physical symptom |
| 11 | Hair loss(Y/N) | Physical symptom |
| 12 | Symptom_Burden | Engineered (symptom count) |
| 13 | Fast food (Y/N) | Lifestyle |
| 14 | Age (yrs) | Demographic |
| 15 | Vit D3 (ng/mL) | Metabolic marker |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Redirect to Swagger docs |
| GET | `/health/` | Health check (model status) |
| GET | `/health/ready` | Readiness probe |
| GET | `/health/live` | Liveness probe |
| POST | `/api/v1/predict` | Single patient prediction |
| POST | `/api/v1/batch-predict` | Batch prediction (up to 100) |

### Risk Stratification

| Probability | Risk Level | Decision |
|-------------|------------|----------|
| < 30% | Low | Routine annual screening |
| 30% - 70% | Medium | 3-month monitoring + lifestyle counseling |
| > 70% | High | Immediate specialist referral |

## Project Architecture

```
pcos-prediction-ml/
├── src/                           # Core ML modules
│   ├── config.py                  # Constants, paths, settings
│   ├── data_loader.py             # Excel/CSV loading
│   ├── data_cleaning.py           # Missing values, outliers, Excel errors
│   ├── feature_engineering.py     # LH/FSH ratio, follicle count, symptom burden
│   ├── preprocessing.py           # Train/test split, scaling, SMOTE
│   ├── feature_selection.py       # RFE, Mutual Info, Correlation consensus
│   ├── model_training.py          # GridSearchCV for 6 models
│   └── model_evaluation.py        # Metrics, confusion matrix, ROC curves
├── app/                           # FastAPI application
│   ├── main.py                    # App entry point with CORS and routers
│   ├── config.py                  # API settings (Pydantic BaseSettings)
│   ├── models.py                  # Request/response Pydantic schemas
│   ├── dependencies.py            # Model/scaler/feature singleton loading
│   ├── routers/
│   │   ├── health.py              # Health check endpoints
│   │   └── predict.py             # Prediction + batch prediction
│   └── utils/
│       └── preprocessing.py       # API-side feature engineering
├── scripts/
│   └── train_pipeline.py          # End-to-end training orchestrator
├── tests/
│   ├── test_api.py                # API endpoint tests
│   └── test_data_processing.py    # Data cleaning/engineering tests
├── models/                        # Trained model artifacts
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── model_config.json
├── data/
│   ├── raw/                       # Original Excel dataset
│   └── processed/                 # Cleaned CSVs and feature list
├── notebooks/                     # Archived Jupyter notebook
├── visualizations/                # Generated plots
├── docs/presentations/            # Technical and business decks
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── requirements.txt
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Class Balancing | imbalanced-learn (SMOTE) |
| API | FastAPI + Uvicorn |
| Validation | Pydantic |
| Data | pandas, numpy, openpyxl |
| Visualization | matplotlib, seaborn |
| Testing | pytest, httpx |
| Containerization | Docker |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=app --cov-report=html
```

## Docker Deployment

```bash
# Build image
docker build -t pcos-api:latest .

# Run container
docker run -p 8000:8000 pcos-api:latest

# Or use docker-compose
docker-compose up -d
```

## Dataset

- **Source:** [Kaggle - Polycystic Ovary Syndrome (PCOS)](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)
- **Size:** 541 patients, 41 clinical features
- **Categories:** Demographics, hormonal markers (FSH, LH, TSH, AMH, PRL), metabolic indicators (BMI, RBS, Vit D3), ovarian morphology (follicle counts, follicle sizes, endometrium), physical symptoms (hair growth, skin darkening, weight gain, hair loss, pimples)
- **Target:** `PCOS (Y/N)` - Binary classification
- **Class Distribution:** ~33% PCOS positive (imbalanced, addressed with SMOTE)

## Documentation

- [Data Dictionary](docs/DATA_DICTIONARY.md)
- [Bias & Fairness Analysis](docs/BIAS_FAIRNESS_ANALYSIS.md)
- [Technical Presentation](docs/presentations/TECHNICAL_PRESENTATION_DECK.md)
- [Business Case](docs/presentations/BUSINESS_PRESENTATION_DECK.md)
- [API Documentation](http://localhost:8000/docs) (interactive, when server is running)

## Use of Generative AI

Generative AI tools were used throughout the development of this project. All outputs were reviewed, validated, and adapted by the author.

| Tool | How It Was Used |
|------|----------------|
| **Claude.ai** | Assisted with code generation, debugging, and iterating on ML pipeline logic. Also used to review and refine presentation drafts. |
| **Claude Code** | Used for code refactoring into modular `src/` architecture, production-readiness improvements (FastAPI, Docker, tests), and debugging across the full stack. |
| **ChatGPT** | Assisted with research, brainstorming project direction, and drafting documentation (README, data dictionary, presentation content). |
| **Gamma** | Used for creating visually polished presentation slides and visual summaries from the markdown presentation drafts. |

### What Was AI-Generated vs Human-Directed

- **Human-directed:** Problem framing, clinical domain decisions (recall optimization, feature engineering rationale), model selection criteria, deployment architecture, all final review and validation
- **AI-assisted:** Code scaffolding, boilerplate generation, debugging suggestions, documentation drafts, EDA/explainability script generation
- **Human-validated:** Every AI-generated output was reviewed for correctness, relevance, and alignment with project goals before inclusion

## Contributing

This is a capstone project for the PGDAIML program. For questions or collaboration:

- **Author:** Jumeil John Navares
- **Email:** jumeil.navares@gmail.com
- **LinkedIn:** [jjdnavares](https://www.linkedin.com/in/jjdnavares/)
- **GitHub:** [jjdnavares](https://github.com/jjdnavares)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **Dataset:** prasoonkottarathil (Kaggle)
- **Frameworks:** FastAPI, scikit-learn, XGBoost, LightGBM
- **Inspiration:** Philippine Department of Health NCDs Program
