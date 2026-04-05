"""
Preprocessing utilities: splitting, scaling, SMOTE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple
from src.config import *
import logging

logger = logging.getLogger(__name__)

def split_data(
    df: pd.DataFrame,
    target_col: str = 'PCOS (Y/N)',
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets (stratified).

    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        (X_train, X_test, y_train, y_test)
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    logger.info(f"✓ Train set: {len(X_train)} samples ({y_train.sum()} PCOS)")
    logger.info(f"✓ Test set:  {len(X_test)} samples ({y_test.sum()} PCOS)")

    return X_train, X_test, y_train, y_test

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.

    Fit on train, transform both train and test.

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        (X_train_scaled, X_test_scaled, scaler)
    """

    scaler = StandardScaler()

    # Fit on train only
    scaler.fit(X_train)

    # Transform both
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    logger.info("✓ Features scaled using StandardScaler")

    return X_train_scaled, X_test_scaled, scaler

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = SMOTE_STRATEGY,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance training data.

    WARNING: Apply ONLY to training data after split to avoid data leakage.

    Args:
        X_train: Training features
        y_train: Training labels
        strategy: SMOTE sampling strategy ('auto' = balance to 1:1)
        k_neighbors: Number of nearest neighbors
        random_state: Random seed

    Returns:
        (X_train_balanced, y_train_balanced)
    """

    # Guard: SMOTE cannot handle NaN — fill any remaining before resampling
    nan_count = X_train.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"⚠ {nan_count} NaN in X_train before SMOTE — filling with median")
        X_train = X_train.fillna(X_train.median())

    logger.info("=== Applying SMOTE ===")
    logger.info(f"  Before: {len(y_train)} samples, {y_train.sum()} PCOS ({y_train.mean():.1%})")

    smote = SMOTE(
        sampling_strategy=strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series
    X_train_balanced = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_train_balanced = pd.Series(y_resampled)

    logger.info(f"  After:  {len(y_train_balanced)} samples, {y_train_balanced.sum()} PCOS ({y_train_balanced.mean():.1%})")
    logger.info("✓ SMOTE complete")

    return X_train_balanced, y_train_balanced
