"""
Configuration file for PCOS prediction project.
All constants, paths, and settings in one place.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent.parent  # Points to project root
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================

# Input
RAW_DATA_FILE = RAW_DATA_DIR / "PCOS_data_without_infertility.xlsx"
RAW_SHEET_NAME = "Full_new"  # Important: not "PCOS_data" or "Sheet1"

# Output
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "pcos_step2_cleaned.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "pcos_step3_train_balanced.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "pcos_step3_test.csv"
SELECTED_FEATURES_FILE = PROCESSED_DATA_DIR / "selected_features.csv"

# ============================================================================
# MODEL FILES
# ============================================================================

BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
MODEL_CONFIG_FILE = MODELS_DIR / "model_config.json"

# ============================================================================
# MODEL TRAINING PARAMETERS
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5

# SMOTE parameters
SMOTE_STRATEGY = 'auto'  # Balance to 1:1 ratio
SMOTE_K_NEIGHBORS = 5

# Feature selection
N_FEATURES_TO_SELECT = 15  # Final feature count after selection

# ============================================================================
# EVALUATION METRICS
# ============================================================================

PRIMARY_METRIC = 'recall'  # Optimize for recall (minimize false negatives)
SECONDARY_METRIC = 'roc_auc'

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Plot style
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
COLOR_PALETTE = 'husl'

# Figure sizes
SMALL_FIG_SIZE = (10, 6)
MEDIUM_FIG_SIZE = (12, 8)
LARGE_FIG_SIZE = (14, 10)

# DPI for saved plots
PLOT_DPI = 300

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Age bins for age group feature (extended to 60 to cover all dataset values)
AGE_BINS = [0, 25, 35, 60]
AGE_LABELS = ['18-25', '26-35', '36+']

# BMI categories
BMI_BINS = [0, 18.5, 25, 30, 100]
BMI_LABELS = ['Underweight', 'Normal', 'Overweight', 'Obese']

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Models to train
MODELS_TO_TRAIN = [
    'logistic_regression',
    'decision_tree',
    'random_forest',
    'gradient_boosting',
    'xgboost',
    'lightgbm',
    'svm'
]

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
