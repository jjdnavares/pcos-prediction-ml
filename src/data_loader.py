"""
Data loading utilities
"""

import pandas as pd
from pathlib import Path
from src.config import *
import logging

logger = logging.getLogger(__name__)

def load_raw_data(filepath: str = RAW_DATA_FILE, sheet_name: str = RAW_SHEET_NAME) -> pd.DataFrame:
    """
    Load raw PCOS dataset from Excel file.

    Args:
        filepath: Path to Excel file
        sheet_name: Name of the sheet to load (default: 'Full_new')

    Returns:
        Raw DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If sheet doesn't exist
    """

    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        # Strip leading/trailing whitespace from column names (Excel artifact)
        df.columns = df.columns.str.strip()
        logger.info(f"✓ Loaded {len(df)} rows from {filepath} (sheet: {sheet_name})")
        return df
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}")
        raise

def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save processed DataFrame to CSV.

    Args:
        df: DataFrame to save
        filepath: Output CSV path
    """

    df.to_csv(filepath, index=False)
    logger.info(f"✓ Saved {len(df)} rows to {filepath}")

def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load processed data from CSV.

    Args:
        filepath: CSV file path

    Returns:
        Loaded DataFrame
    """

    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"✓ Loaded {len(df)} rows from {filepath}")
    return df
