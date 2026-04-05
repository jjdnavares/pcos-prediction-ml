"""
Dependency injection for model loading and shared resources
"""
import joblib
import json
import pandas as pd
from pathlib import Path
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Global variables (loaded once at startup)
_model = None
_scaler = None
_feature_names = None
_model_config = None

def load_model():
    """Load the trained ML model (singleton pattern)"""
    global _model
    if _model is None:
        try:
            _model = joblib.load(settings.MODEL_PATH)
            logger.info(f"✓ Model loaded from {settings.MODEL_PATH}")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise RuntimeError(f"Could not load model: {e}")
    return _model

def load_scaler():
    """Load the preprocessing scaler (singleton pattern)"""
    global _scaler
    if _scaler is None:
        try:
            _scaler = joblib.load(settings.SCALER_PATH)
            logger.info(f"✓ Scaler loaded from {settings.SCALER_PATH}")
        except Exception as e:
            logger.error(f"✗ Failed to load scaler: {e}")
            raise RuntimeError(f"Could not load scaler: {e}")
    return _scaler

def load_feature_names():
    """Load the list of selected features (singleton pattern)"""
    global _feature_names
    if _feature_names is None:
        try:
            features_df = pd.read_csv(settings.FEATURES_PATH)
            _feature_names = features_df['Feature'].tolist()
            logger.info(f"✓ Feature names loaded: {len(_feature_names)} features")
        except Exception as e:
            logger.error(f"✗ Failed to load feature names: {e}")
            raise RuntimeError(f"Could not load features: {e}")
    return _feature_names

def load_model_config():
    """Load model configuration metadata (singleton pattern)"""
    global _model_config
    if _model_config is None:
        try:
            with open(settings.CONFIG_PATH, 'r') as f:
                _model_config = json.load(f)
            logger.info(f"✓ Model config loaded")
        except Exception as e:
            logger.warning(f"⚠ Model config not found, using defaults: {e}")
            _model_config = {"version": "1.0.0"}
    return _model_config

def get_model():
    """Dependency for FastAPI routes"""
    return load_model()

def get_scaler():
    """Dependency for FastAPI routes"""
    return load_scaler()

def get_feature_names():
    """Dependency for FastAPI routes"""
    return load_feature_names()

def get_model_config():
    """Dependency for FastAPI routes"""
    return load_model_config()

def startup_check():
    """
    Run at application startup to ensure all resources load successfully.
    Fails fast if critical files are missing.
    """
    logger.info("=== Startup Check ===")
    load_model()
    load_scaler()
    load_feature_names()
    load_model_config()
    logger.info("=== All resources loaded successfully ===")
