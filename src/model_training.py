"""
Model training and hyperparameter tuning
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from typing import Dict, Tuple
from src.config import *
import logging
import joblib
import mlflow

logger = logging.getLogger(__name__)

# Hyperparameter grids for each model
PARAM_GRIDS = {
    'logistic_regression': {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000]
    },
    'decision_tree': {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'lightgbm': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63]
    },
    'svm': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
}

def get_model(model_name: str):
    """
    Get model instance by name.

    Args:
        model_name: Name of the model

    Returns:
        Scikit-learn compatible model instance
    """

    models = {
        'logistic_regression': LogisticRegression(random_state=RANDOM_STATE),
        'decision_tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'random_forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'xgboost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
        'lightgbm': LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        'svm': SVC(random_state=RANDOM_STATE, probability=True)
    }

    return models.get(model_name)

def train_model_with_tuning(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = N_FOLDS,
    scoring: str = PRIMARY_METRIC
) -> Tuple[object, Dict]:
    """
    Train model with hyperparameter tuning.

    Uses GridSearchCV with stratified K-fold cross-validation.

    Args:
        model_name: Name of the model to train
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of CV folds
        scoring: Scoring metric for optimization

    Returns:
        (best_model, results_dict)
    """

    logger.info(f"=== Training {model_name} ===")

    # Get base model and param grid
    model = get_model(model_name)
    param_grid = PARAM_GRIDS.get(model_name, {})

    if not param_grid:
        logger.warning(f"No param grid for {model_name}, using default params")
        model.fit(X_train, y_train)
        return model, {}

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    # Grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    # Fit
    grid_search.fit(X_train, y_train)

    # Results
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }

    # Log to MLflow (nested run per model)
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric(f"cv_best_{scoring}", grid_search.best_score_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, artifact_path=model_name)

    logger.info(f"✓ Best {scoring}: {grid_search.best_score_:.4f}")
    logger.info(f"✓ Best params: {grid_search.best_params_}")

    return grid_search.best_estimator_, results

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Dict[str, Tuple[object, Dict]]:
    """
    Train all configured models.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Dictionary mapping model_name -> (model, results)
    """

    trained_models = {}

    for model_name in MODELS_TO_TRAIN:
        try:
            model, results = train_model_with_tuning(model_name, X_train, y_train)
            trained_models[model_name] = (model, results)
        except Exception as e:
            logger.error(f"✗ Failed to train {model_name}: {e}")

    logger.info(f"\n=== Training Complete: {len(trained_models)}/{len(MODELS_TO_TRAIN)} models ===")

    return trained_models

def save_model(
    model: object,
    filepath: str = BEST_MODEL_FILE
) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained scikit-learn model
        filepath: Output path
    """

    joblib.dump(model, filepath)
    logger.info(f"✓ Model saved to {filepath}")

def load_model(filepath: str = BEST_MODEL_FILE) -> object:
    """
    Load trained model from disk.

    Args:
        filepath: Model file path

    Returns:
        Loaded model
    """

    model = joblib.load(filepath)
    logger.info(f"✓ Model loaded from {filepath}")
    return model
