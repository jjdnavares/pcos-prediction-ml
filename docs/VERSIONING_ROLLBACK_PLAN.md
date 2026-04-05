# Model Versioning & Rollback Plan

> Strategy for tracking model versions, managing deployments, and rolling back safely.

---

## 1. Versioning Scheme

Models follow **semantic versioning** (`MAJOR.MINOR.PATCH`):

| Component | When to Increment | Example |
|-----------|-------------------|---------|
| MAJOR | Breaking changes (new features, different input schema) | 1.0.0 → 2.0.0 |
| MINOR | Model retrained with improved performance or new algorithm | 1.0.0 → 1.1.0 |
| PATCH | Bug fix, config change, no retraining | 1.0.0 → 1.0.1 |

The current version is stored in `models/model_config.json` under the `version` field.

---

## 2. What Gets Versioned

Every model release consists of these artifacts:

| Artifact | Path | Description |
|----------|------|-------------|
| Trained model | `models/best_model.pkl` | Serialized scikit-learn model |
| Scaler | `models/scaler.pkl` | StandardScaler fitted on training data |
| Config | `models/model_config.json` | Version, feature list, metrics |
| Features | `data/processed/selected_features.csv` | 15 consensus-selected features |

All four artifacts must be kept in sync — deploying a new model without its matching scaler or feature list will produce incorrect predictions.

---

## 3. Version Tracking with MLflow

Each training pipeline run is automatically logged to MLflow:

```bash
# View all experiment runs
mlflow ui --backend-store-uri mlruns
```

MLflow records per run:
- **Parameters:** dataset size, SMOTE ratio, CV folds, feature count, selected features
- **Metrics:** accuracy, precision, recall, F1, AUC for all 7 models
- **Artifacts:** best model, model config, confusion matrix, selected features

This provides a complete audit trail — you can compare any two versions side-by-side in the MLflow UI.

---

## 4. Release Process

### 4.1 Before Releasing a New Version

1. **Train and evaluate** — run `python scripts/train_pipeline.py`
2. **Compare with current production** — check MLflow UI to confirm the new model improves on the metric that matters (Recall)
3. **Run all tests** — `pytest tests/ -v`
4. **Update version** in `models/model_config.json`

### 4.2 Creating a Release

```bash
# Tag the release in git
git tag -a v1.1.0 -m "Model v1.1.0: retrained with SVM, improved recall"
git push origin v1.1.0

# The CI pipeline will run tests and build Docker automatically
```

### 4.3 Artifact Backup

Before overwriting production artifacts, copy the current set:

```bash
# Create versioned backup
mkdir -p models/archive/v1.0.0
cp models/best_model.pkl models/archive/v1.0.0/
cp models/scaler.pkl models/archive/v1.0.0/
cp models/model_config.json models/archive/v1.0.0/
cp data/processed/selected_features.csv models/archive/v1.0.0/
```

---

## 5. Rollback Procedure

### 5.1 When to Roll Back

- New model has worse Recall on production traffic
- Prediction latency degrades significantly
- Unexpected errors in the `/api/v1/predict` endpoint after deployment
- Stakeholder or clinical review identifies problematic predictions

### 5.2 Quick Rollback (< 5 minutes)

If you have the archived artifacts:

```bash
# 1. Stop the running service
docker-compose down

# 2. Restore previous model artifacts
cp models/archive/v1.0.0/best_model.pkl models/
cp models/archive/v1.0.0/scaler.pkl models/
cp models/archive/v1.0.0/model_config.json models/
cp models/archive/v1.0.0/selected_features.csv data/processed/

# 3. Restart
docker-compose up -d

# 4. Verify health
curl http://localhost:8000/health/
```

### 5.3 Git-Based Rollback

If artifacts weren't archived but were committed:

```bash
# Find the commit with the previous model
git log --oneline -- models/best_model.pkl

# Restore artifacts from that commit
git checkout <commit-hash> -- models/best_model.pkl models/scaler.pkl models/model_config.json

# Restart the service
docker-compose restart
```

### 5.4 Full Retrain Rollback

If you need to reproduce an exact previous model:

1. Open MLflow UI and find the run with the desired metrics
2. Note the parameters (random state, features, SMOTE config)
3. Check out the git tag for that version
4. Re-run `python scripts/train_pipeline.py`

---

## 6. Version History

| Version | Date | Model | Recall | AUC | Notes |
|---------|------|-------|--------|-----|-------|
| 1.0.0 | 2026-04-05 | Logistic Regression | 86.1% | 94.6% | Initial release, 6 models compared |
| 1.1.0 | 2026-04-06 | Logistic Regression | 86.1% | 94.6% | Added SVM + MLflow experiment tracking |
