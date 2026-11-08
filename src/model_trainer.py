"""
Model Training Module.

Initializes, builds pipelines, and trains the three machine learning models:
1. Dummy Classifier (Baseline)
2. Logistic Regression (Linear with balanced weights)
3. Random Forest Classifier (Non-linear ensemble with balanced subsample weights)
"""

import time
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from config.config import (
    DUMMY_CLASSIFIER_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
)


def train_dummy_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[DummyClassifier, Dict[str, Any], float]:
    """
    Train a baseline DummyClassifier that always predicts the most frequent class.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    Tuple[DummyClassifier, Dict[str, Any], float]
        Fitted model, logged parameters dict, training duration in seconds.
    """
    print(f"\n--- Training Model 1: Dummy Classifier (Baseline) ---")
    print(f"Strategy: {DUMMY_CLASSIFIER_PARAMS['strategy']}")
    print("Purpose : Always predicts majority class (Normal: 0) to demonstrate the Accuracy Paradox.")

    model = DummyClassifier(
        strategy=DUMMY_CLASSIFIER_PARAMS["strategy"],
        random_state=DUMMY_CLASSIFIER_PARAMS["random_state"],
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0

    params = {
        "model_name": "Dummy Classifier",
        "model_type": "DummyClassifier",
        "strategy": DUMMY_CLASSIFIER_PARAMS["strategy"],
        "random_state": RANDOM_STATE,
        "n_features": X_train.shape[1],
        "training_samples": len(X_train),
        "preprocessing": "None (Majority Class Rule)",
    }

    print(f"✓ Dummy Classifier trained in {training_time:.3f} seconds.")
    return model, params, training_time


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train a Logistic Regression model with balanced class weights inside a preprocessing pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    preprocessor : ColumnTransformer
        Fitted or unfitted preprocessor.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any], float]
        Fitted pipeline, logged parameters dict, training duration in seconds.
    """
    print(f"\n--- Training Model 2: Logistic Regression (Balanced) ---")
    print("Configuration: class_weight='balanced', max_iter=1000, StandardScaler on [Time, Amount]")

    classifier = LogisticRegression(
        class_weight=LOGISTIC_REGRESSION_PARAMS["class_weight"],
        max_iter=LOGISTIC_REGRESSION_PARAMS["max_iter"],
        random_state=LOGISTIC_REGRESSION_PARAMS["random_state"],
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - t0

    params = {
        "model_name": "Logistic Regression",
        "model_type": "LogisticRegression",
        "class_weight": LOGISTIC_REGRESSION_PARAMS["class_weight"],
        "max_iter": LOGISTIC_REGRESSION_PARAMS["max_iter"],
        "random_state": RANDOM_STATE,
        "n_features": X_train.shape[1],
        "training_samples": len(X_train),
        "preprocessing": "StandardScaler(Time, Amount) + Passthrough(V1-V28)",
        "solver": "lbfgs",
    }

    print(f"✓ Logistic Regression pipeline trained in {training_time:.3f} seconds.")
    return pipeline, params, training_time


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train a Random Forest classifier with balanced_subsample class weights inside a pipeline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    preprocessor : ColumnTransformer
        Fitted or unfitted preprocessor.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any], float]
        Fitted pipeline, logged parameters dict, training duration in seconds.
    """
    print(f"\n--- Training Model 3: Random Forest Classifier ---")
    print("Configuration: n_estimators=100, max_depth=10, class_weight='balanced_subsample'")

    classifier = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_PARAMS["n_estimators"],
        max_depth=RANDOM_FOREST_PARAMS["max_depth"],
        class_weight=RANDOM_FOREST_PARAMS["class_weight"],
        random_state=RANDOM_FOREST_PARAMS["random_state"],
        n_jobs=RANDOM_FOREST_PARAMS["n_jobs"],
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - t0

    params = {
        "model_name": "Random Forest",
        "model_type": "RandomForestClassifier",
        "n_estimators": RANDOM_FOREST_PARAMS["n_estimators"],
        "max_depth": RANDOM_FOREST_PARAMS["max_depth"],
        "class_weight": RANDOM_FOREST_PARAMS["class_weight"],
        "random_state": RANDOM_STATE,
        "n_features": X_train.shape[1],
        "training_samples": len(X_train),
        "preprocessing": "StandardScaler(Time, Amount) + Passthrough(V1-V28)",
        "n_jobs": RANDOM_FOREST_PARAMS["n_jobs"],
    }

    print(f"✓ Random Forest pipeline trained in {training_time:.3f} seconds.")
    return pipeline, params, training_time

