"""
Configuration settings for the PCOS Prediction API
"""
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""

    # API Metadata
    APP_NAME: str = "PCOS Prediction API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "ML-powered PCOS screening for community health workers"

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"

    # Model files
    MODEL_PATH: Path = MODELS_DIR / "best_model.pkl"
    SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
    CONFIG_PATH: Path = MODELS_DIR / "model_config.json"
    FEATURES_PATH: Path = DATA_DIR / "processed" / "selected_features.csv"

    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True  # Set to False in production

    # CORS
    ALLOW_ORIGINS: list = ["*"]  # Restrict in production
    ALLOW_METHODS: list = ["*"]
    ALLOW_HEADERS: list = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"

    # Risk thresholds (probability cutoffs)
    LOW_RISK_THRESHOLD: float = 0.3
    HIGH_RISK_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()
