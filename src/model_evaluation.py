"""
Model evaluation utilities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict
from src.config import *
import logging

logger = logging.getLogger(__name__)

def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict:
    """
    Comprehensive model evaluation.

    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Name for logging

    Returns:
        Dictionary of metrics
    """

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    logger.info(f"\n=== {model_name} Performance ===")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    return metrics

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Model",
    save_path: str = None
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2x2 array)
        model_name: Model name for title
        save_path: Path to save plot (optional)
    """

    plt.figure(figsize=SMALL_FIG_SIZE)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No PCOS', 'PCOS'],
        yticklabels=['No PCOS', 'PCOS']
    )
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved confusion matrix to {save_path}")

    plt.show()

def plot_roc_curve(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    save_path: str = None
) -> None:
    """
    Plot ROC curve.

    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        model_name: Model name for legend
        save_path: Path to save plot (optional)
    """

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=SMALL_FIG_SIZE)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved ROC curve to {save_path}")

    plt.show()

def compare_models(
    models_dict: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Args:
        models_dict: Dictionary mapping model_name -> model
        X_test: Test features
        y_test: True labels

    Returns:
        DataFrame with comparison metrics
    """

    results = []

    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc']
        })

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Recall', ascending=False)

    logger.info("\n=== Model Comparison ===")
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    return comparison_df
