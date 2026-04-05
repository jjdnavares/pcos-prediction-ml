"""
Feature engineering utilities
"""

import pandas as pd
import numpy as np
from src.config import *
import logging

logger = logging.getLogger(__name__)

def create_lh_fsh_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create LH/FSH ratio feature.

    Clinical significance: LH/FSH > 2 indicates hormonal imbalance characteristic of PCOS.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with LH_FSH_Ratio column added
    """

    df['LH_FSH_Ratio'] = df['LH(mIU/mL)'] / (df['FSH(mIU/mL)'] + 1e-6)  # Avoid division by zero
    logger.info("✓ Created LH_FSH_Ratio feature")

    return df

def create_total_follicle_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create total follicle count feature.

    Composite Rotterdam criterion: ≥12 follicles per ovary is diagnostic.
    Total count provides additional signal.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with Total_Follicle_Count column added
    """

    df['Total_Follicle_Count'] = df['Follicle No. (L)'] + df['Follicle No. (R)']
    logger.info("✓ Created Total_Follicle_Count feature")

    return df

def create_whr_recalc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalculate Waist-Hip Ratio.

    Dataset has WHR column but may have rounding errors.
    Recalculate for consistency.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with WHR_Recalc column added
    """

    df['WHR_Recalc'] = df['Waist(inch)'] / (df['Hip(inch)'] + 1e-6)
    logger.info("✓ Created WHR_Recalc feature")

    return df

def create_symptom_burden(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create symptom burden score.

    Count of androgenic/metabolic symptoms present.
    Higher score = more severe symptom presentation.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with Symptom_Burden column added
    """

    symptom_cols = [
        'Weight gain(Y/N)',
        'hair growth(Y/N)',
        'Skin darkening (Y/N)',
        'Hair loss(Y/N)',
        'Pimples(Y/N)'
    ]

    # Filter to columns that exist
    available_cols = [c for c in symptom_cols if c in df.columns]

    df['Symptom_Burden'] = df[available_cols].sum(axis=1)
    logger.info(f"✓ Created Symptom_Burden feature from {len(available_cols)} symptoms")

    return df

def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create age group bins.

    Bins: 18-25, 26-35, 36-44 (reproductive age spectrum)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with Age_Group column added
    """

    df['Age_Group'] = pd.cut(
        df['Age (yrs)'],
        bins=AGE_BINS,
        labels=False,  # Use integer codes (0, 1, 2) instead of strings
        include_lowest=True
    ).astype(float)

    logger.info("✓ Created Age_Group feature")

    return df

def create_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create BMI category.

    WHO categories: Underweight (<18.5), Normal (18.5-25),
                    Overweight (25-30), Obese (>30)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with BMI_Category column added
    """

    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=BMI_BINS,
        labels=False,  # Use integer codes (0, 1, 2, 3) instead of strings
        include_lowest=True
    ).astype(float)

    logger.info("✓ Created BMI_Category feature")

    return df

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with all engineered features
    """

    logger.info("=== Feature Engineering ===")

    df = create_lh_fsh_ratio(df)
    df = create_total_follicle_count(df)
    df = create_whr_recalc(df)
    df = create_symptom_burden(df)
    df = create_age_group(df)
    df = create_bmi_category(df)

    logger.info(f"✓ Total features after engineering: {len(df.columns)}")

    return df
