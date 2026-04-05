"""
Feature selection methods
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple
from src.config import *
import logging

logger = logging.getLogger(__name__)

def select_by_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = N_FEATURES_TO_SELECT
) -> List[str]:
    """
    Select features using Recursive Feature Elimination.

    Uses Logistic Regression as the base estimator.

    Args:
        X: Feature matrix
        y: Target vector
        n_features: Number of features to select

    Returns:
        List of selected feature names
    """

    estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)

    rfe.fit(X, y)

    selected = X.columns[rfe.support_].tolist()
    logger.info(f"✓ RFE selected {len(selected)} features")

    return selected

def select_by_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = N_FEATURES_TO_SELECT
) -> List[str]:
    """
    Select features using Mutual Information.

    Measures dependence between features and target.

    Args:
        X: Feature matrix
        y: Target vector
        n_features: Number of features to select

    Returns:
        List of selected feature names
    """

    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    # Create DataFrame for sorting
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'score': mi_scores
    }).sort_values('score', ascending=False)

    selected = mi_df.head(n_features)['feature'].tolist()
    logger.info(f"✓ Mutual Info selected {len(selected)} features")

    return selected

def select_by_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = N_FEATURES_TO_SELECT
) -> List[str]:
    """
    Select features by correlation with target.

    Args:
        X: Feature matrix
        y: Target vector
        n_features: Number of features to select

    Returns:
        List of selected feature names
    """

    # Combine X and y for correlation
    df = X.copy()
    df['target'] = y

    # Get absolute correlation with target
    corr_with_target = df.corr()['target'].abs().drop('target')

    selected = corr_with_target.nlargest(n_features).index.tolist()
    logger.info(f"✓ Correlation selected {len(selected)} features")

    return selected

def consensus_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = N_FEATURES_TO_SELECT
) -> List[str]:
    """
    Combine multiple feature selection methods via consensus.

    Features selected by at least 2 out of 3 methods are retained.

    Args:
        X: Feature matrix
        y: Target vector
        n_features: Number of features to select

    Returns:
        List of consensus-selected feature names
    """

    logger.info("=== Consensus Feature Selection ===")

    # Run all three methods
    rfe_features = select_by_rfe(X, y, n_features)
    mi_features = select_by_mutual_info(X, y, n_features)
    corr_features = select_by_correlation(X, y, n_features)

    # Count votes for each feature
    all_features = set(rfe_features + mi_features + corr_features)
    feature_votes = {}

    for feature in all_features:
        votes = 0
        if feature in rfe_features:
            votes += 1
        if feature in mi_features:
            votes += 1
        if feature in corr_features:
            votes += 1
        feature_votes[feature] = votes

    # Select features with at least 2 votes
    consensus_features = [f for f, v in feature_votes.items() if v >= 2]

    # If consensus gives too few features, take top N by vote count
    if len(consensus_features) < n_features:
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        consensus_features = [f for f, v in sorted_features[:n_features]]

    logger.info(f"✓ Consensus selected {len(consensus_features)} features:")
    for i, feat in enumerate(consensus_features, 1):
        logger.info(f"  {i:2d}. {feat} (votes: {feature_votes[feat]})")

    return consensus_features
