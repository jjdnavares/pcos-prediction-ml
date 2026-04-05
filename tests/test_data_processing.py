"""
Tests for data processing modules
"""

import pytest
import pandas as pd
import numpy as np
from src.data_cleaning import handle_missing_values, handle_outliers_iqr
from src.feature_engineering import create_lh_fsh_ratio

def test_missing_value_imputation():
    """Test median imputation"""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [10, 20, 30, 40]
    })

    df_clean = handle_missing_values(df, strategy='median')

    assert df_clean['A'].isnull().sum() == 0
    assert df_clean['A'].iloc[2] == 2.0  # Median of [1, 2, 4]

def test_lh_fsh_ratio():
    """Test LH/FSH ratio creation"""
    df = pd.DataFrame({
        'LH(mIU/mL)': [10, 20],
        'FSH(mIU/mL)': [5, 10]
    })

    df_eng = create_lh_fsh_ratio(df)

    assert 'LH_FSH_Ratio' in df_eng.columns
    assert abs(df_eng['LH_FSH_Ratio'].iloc[0] - 2.0) < 1e-5
