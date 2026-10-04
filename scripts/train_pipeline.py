"""
End-to-end training pipeline script.

Usage:
    python scripts/train_pipeline.py
"""

import os
import sys
import warnings
import multiprocessing
from pathlib import Path

# Handle loky/joblib warnings and multiprocessing on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
if sys.platform == 'win32':
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:joblib.externals.loky.backend.context'
    import joblib.externals.loky.backend.context as _loky_ctx
    _loky_ctx._count_physical_cores = lambda: (os.cpu_count() or 4, None)
multiprocessing.set_start_method('spawn', force=True)

# Suppress MLflow "Inferred schema contains integer column(s)" hint
warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import *
from src.data_loader import load_raw_data, save_processed_data
from src.data_cleaning import *
from src.feature_engineering import engineer_all_features
from src.preprocessing import split_data, scale_features, apply_smote
from src.feature_selection import consensus_feature_selection
from src.model_training import train_all_models, save_model
from src.model_evaluation import compare_models, plot_confusion_matrix
import logging
import joblib
import json
import os
import mlflow

# MLflow >= 3.x treats the ./mlruns file store as maintenance-mode and refuses it by default.
# Opt in explicitly to keep the existing file-based tracking layout.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the complete training pipeline."""

    logger.info("="*80)
    logger.info("PCOS PREDICTION MODEL - TRAINING PIPELINE")
    logger.info("="*80)

    # Set up MLflow experiment
    # Tracking URI must be set before the experiment (MLflow 3.x defaults to a different store)
    mlflow.set_tracking_uri((BASE_DIR / "mlruns").as_uri())
    mlflow.set_experiment("pcos-prediction")

    with mlflow.start_run(run_name="training-pipeline"):

        # Step 1: Load data
        logger.info("\n[Step 1/9] Loading raw data...")
        df = load_raw_data()

        # Step 2: Clean data
        logger.info("\n[Step 2/9] Cleaning data...")
        df = handle_excel_errors(df)
        df = remove_phantom_column(df)
        df = fix_undocumented_cycle_value(df)
        df = handle_missing_values(df)
        df = handle_outliers_iqr(df)

        # Save cleaned data
        save_processed_data(df, CLEANED_DATA_FILE)

        # Step 3: Feature engineering
        logger.info("\n[Step 3/9] Engineering features...")
        df = engineer_all_features(df)

        # Drop non-numeric columns (e.g. any remaining object/categorical) and fill NaN
        # that may have been introduced by feature engineering (e.g. out-of-range bins)
        df = df.select_dtypes(include=['number'])
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠ {nan_count} NaN values found after feature engineering — filling with median")
            df = handle_missing_values(df)

        # Log dataset parameters
        mlflow.log_params({
            "dataset_rows": df.shape[0],
            "dataset_features": df.shape[1],
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": N_FOLDS,
            "primary_metric": PRIMARY_METRIC,
            "smote_strategy": str(SMOTE_STRATEGY),
            "n_features_to_select": N_FEATURES_TO_SELECT,
        })

        # Step 4: Train/test split
        logger.info("\n[Step 4/9] Splitting data...")
        X_train, X_test, y_train, y_test = split_data(df)

        # Step 5: Scale features
        logger.info("\n[Step 5/9] Scaling features...")
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # Save scaler
        joblib.dump(scaler, SCALER_FILE)
        logger.info(f"✓ Scaler saved to {SCALER_FILE}")

        # Step 6: Apply SMOTE
        logger.info("\n[Step 6/9] Applying SMOTE...")
        X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)

        mlflow.log_params({
            "train_samples_before_smote": len(y_train),
            "train_samples_after_smote": len(y_train_balanced),
            "test_samples": len(y_test),
        })

        # Save balanced train data
        train_df = X_train_balanced.copy()
        train_df['PCOS (Y/N)'] = y_train_balanced
        save_processed_data(train_df, TRAIN_DATA_FILE)

        # Save test data
        test_df = X_test_scaled.copy()
        test_df['PCOS (Y/N)'] = y_test
        save_processed_data(test_df, TEST_DATA_FILE)

        # Step 7: Feature selection
        logger.info("\n[Step 7/9] Selecting features...")
        selected_features = consensus_feature_selection(X_train_balanced, y_train_balanced)

        # Save selected features
        import pandas as pd
        features_df = pd.DataFrame({'Feature': selected_features})
        save_processed_data(features_df, SELECTED_FEATURES_FILE)

        mlflow.log_param("selected_features", ", ".join(selected_features))

        # Filter to selected features
        X_train_final = X_train_balanced[selected_features]
        X_test_final = X_test_scaled[selected_features]

        # Step 8: Train models (each model logs as a nested run)
        logger.info("\n[Step 8/9] Training models...")
        trained_models = train_all_models(X_train_final, y_train_balanced)

        # Step 9: Evaluate and select best model
        logger.info("\n[Step 9/9] Evaluating models...")
        models_dict = {name: model for name, (model, results) in trained_models.items()}
        comparison_df = compare_models(models_dict, X_test_final, y_test)

        # Log test metrics for every model
        for _, row in comparison_df.iterrows():
            model_tag = row['Model']
            mlflow.log_metrics({
                f"{model_tag}_accuracy": row['Accuracy'],
                f"{model_tag}_precision": row['Precision'],
                f"{model_tag}_recall": row['Recall'],
                f"{model_tag}_f1": row['F1-Score'],
                f"{model_tag}_roc_auc": row['ROC-AUC'],
            })

        # Select best model (highest recall)
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = models_dict[best_model_name]

        logger.info(f"\n✓ Best model: {best_model_name}")

        # Log best model info
        best_row = comparison_df.iloc[0]
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metrics({
            "best_accuracy": best_row['Accuracy'],
            "best_precision": best_row['Precision'],
            "best_recall": best_row['Recall'],
            "best_f1": best_row['F1-Score'],
            "best_roc_auc": best_row['ROC-AUC'],
        })
        mlflow.sklearn.log_model(
            best_model, artifact_path="best_model",
            input_example=X_test_final.iloc[:1].astype(float),
        )

        # Save best model
        save_model(best_model, BEST_MODEL_FILE)

        # Save model config
        model_config = {
            'model_name': best_model_name,
            'version': '1.0.0',
            'features': selected_features,
            'metrics': comparison_df.iloc[0].to_dict()
        }

        with open(MODEL_CONFIG_FILE, 'w') as f:
            json.dump(model_config, f, indent=2)

        logger.info(f"✓ Model config saved to {MODEL_CONFIG_FILE}")

        # Log artifacts
        mlflow.log_artifact(str(MODEL_CONFIG_FILE))
        mlflow.log_artifact(str(SELECTED_FEATURES_FILE))

        # Plot confusion matrix for best model
        from sklearn.metrics import confusion_matrix
        y_pred = best_model.predict(X_test_final)
        cm = confusion_matrix(y_test, y_pred)
        cm_path = VISUALIZATIONS_DIR / 'confusion_matrix.png'
        plot_confusion_matrix(
            cm,
            model_name=best_model_name,
            save_path=cm_path
        )
        mlflow.log_artifact(str(cm_path))

        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info(f"MLflow UI: run 'mlflow ui --backend-store-uri {BASE_DIR / 'mlruns'}' to view results")
        logger.info("="*80)

if __name__ == "__main__":
    main()
