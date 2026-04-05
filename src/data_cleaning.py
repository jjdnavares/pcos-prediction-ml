"""
Data cleaning utilities
"""

import pandas as pd
import numpy as np
from typing import List
from src.config import *
import logging

logger = logging.getLogger(__name__)

def remove_phantom_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove phantom 'Unnamed: 44' column from Excel import.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with phantom column removed
    """

    if 'Unnamed: 44' in df.columns:
        df = df.drop(columns=['Unnamed: 44'])
        logger.info("✓ Removed phantom column 'Unnamed: 44'")

    return df

def handle_excel_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix Excel #NAME? errors in object-type columns.

    Excel errors appear as strings "#NAME?" in numeric columns.
    Coerce these to numeric, setting errors to NaN.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with Excel errors fixed
    """

    object_cols = df.select_dtypes(include='object').columns.tolist()

    for col in object_cols:
        # Skip non-numeric columns (e.g., Y/N columns)
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.info(f"  Coerced {col} to numeric")
            except:
                pass  # Column is truly non-numeric

    logger.info("✓ Fixed Excel #NAME? errors")
    return df

def fix_undocumented_cycle_value(df: pd.DataFrame, column: str = 'Cycle(R/I)') -> pd.DataFrame:
    """
    Fix undocumented cycle length value (4.0).

    The dataset has an undocumented value 4.0 in Cycle(R/I) column.
    Based on domain knowledge and correlation analysis, 4.0 → irregular (5.0).

    Args:
        df: Input DataFrame
        column: Column name for cycle regularity

    Returns:
        DataFrame with fixed cycle values
    """

    if column in df.columns:
        # Replace 4.0 with 5.0 (irregular)
        df[column] = df[column].replace(4.0, 5.0)
        logger.info(f"✓ Fixed undocumented cycle value in {column}")

    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Handle missing values using median imputation.

    Only numeric columns are imputed. Missing rate is <5% so median is appropriate.

    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('median', 'mean', 'mode')

    Returns:
        DataFrame with missing values imputed
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    missing_before = df[numeric_cols].isnull().sum().sum()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == 'median':
                fill_value = df[col].median()
            elif strategy == 'mean':
                fill_value = df[col].mean()
            else:  # mode
                fill_value = df[col].mode()[0]

            df[col] = df[col].fillna(fill_value)

    missing_after = df[numeric_cols].isnull().sum().sum()

    logger.info(f"✓ Imputed {missing_before - missing_after} missing values using {strategy}")

    return df

def handle_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str] = None,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Handle outliers using IQR-based winsorization.

    Caps extreme values at Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    More conservative than removal (preserves data).

    Args:
        df: Input DataFrame
        columns: Columns to process (None = all numeric)
        multiplier: IQR multiplier (default: 1.5)

    Returns:
        DataFrame with outliers winsorized
    """

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers_count = 0

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Count outliers before winsorization
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers_count += outliers

        # Winsorize
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    logger.info(f"✓ Winsorized {outliers_count} outliers across {len(columns)} columns")

    return df
