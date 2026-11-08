"""
Data Preprocessing and Splitting Pipeline Module.

Handles stratified train-test splitting and feature transformations
using Scikit-learn's ColumnTransformer and StandardScaler.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from config.config import (
    FEATURE_COLUMNS_TO_SCALE,
    TEST_SIZE,
    RANDOM_STATE,
)


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate dataset into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Complete dataset.
    target_col : str
        Name of the target column.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features dataframe X and target series y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a reproducible, stratified train-test split.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target array.
    test_size : float
        Proportion of dataset to include in test split (default 0.2).
    random_state : int
        Random seed for reproducibility (default 42).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test.
    """
    print(f"\n{'='*70}")
    print(f"STEP 3: STRATIFIED TRAIN-TEST SPLIT ({int((1-test_size)*100)}:{int(test_size*100)})")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )

    n_train_total = len(y_train)
    n_train_fraud = int(y_train.sum())
    train_fraud_pct = (n_train_fraud / n_train_total) * 100

    n_test_total = len(y_test)
    n_test_fraud = int(y_test.sum())
    test_fraud_pct = (n_test_fraud / n_test_total) * 100

    print(f"✓ Training set : {n_train_total:,} samples | Fraud: {n_train_fraud:,} ({train_fraud_pct:.4f}%)")
    print(f"✓ Testing set  : {n_test_total:,} samples | Fraud: {n_test_fraud:,} ({test_fraud_pct:.4f}%)")
    print("✓ Stratification preserved exact minority class distribution across splits.")
    print("✓ Test set isolated to prevent data leakage.")

    return X_train, X_test, y_train, y_test


def build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer preprocessor.

    Applies StandardScaler to 'Time' and 'Amount' columns (if present),
    and passes through all other numerical features (V1-V28) untouched.

    Parameters
    ----------
    feature_columns : List[str]
        List of all feature column names.

    Returns
    -------
    ColumnTransformer
        Scikit-learn ColumnTransformer instance.
    """
    # Identify which of the scale candidates are actually in the feature columns
    cols_to_scale = [c for c in FEATURE_COLUMNS_TO_SCALE if c in feature_columns]
    remainder_cols = [c for c in feature_columns if c not in cols_to_scale]

    print(f"\n{'='*70}")
    print("STEP 4: BUILDING PREPROCESSING PIPELINE")
    print(f"{'='*70}")
    print(f"✓ Features to scale with StandardScaler: {cols_to_scale}")
    print(f"✓ Features passed through unchanged (PCA components): {len(remainder_cols)} columns")

    transformers = []
    if cols_to_scale:
        transformers.append(
            ("scaler", StandardScaler(), cols_to_scale)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return preprocessor

