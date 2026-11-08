"""
Configuration module for the MLflow Fraud Detection Project.

Contains all global paths, dataset identifiers, model hyperparameters,
and MLflow tracking configuration settings.
"""

import os
from pathlib import Path

# Opt into MLflow file-based tracking for local storage in ./mlruns
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
MLRUNS_DIR = BASE_DIR / "mlruns"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configuration
OPENML_DATA_ID = 42397
DATASET_NAME = "CreditCardFraudDetection"
TARGET_CANDIDATE_COLUMNS = ["Class", "class", "target", "fraud"]
FEATURE_COLUMNS_TO_SCALE = ["Time", "Amount"]
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Model configurations
DUMMY_CLASSIFIER_PARAMS = {
    "strategy": "most_frequent",
    "random_state": RANDOM_STATE,
}

LOGISTIC_REGRESSION_PARAMS = {
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "Credit Card Fraud Detection - Model Comparison"
MLFLOW_TRACKING_URI = str(MLRUNS_DIR.as_uri())  # Use file URI for reliable cross-platform local tracking

MLFLOW_DEFAULT_TAGS = {
    "project": "fraud_detection",
    "dataset": "openml_credit_card_fraud",
    "problem_type": "binary_classification",
    "model_type": "classification",
}

