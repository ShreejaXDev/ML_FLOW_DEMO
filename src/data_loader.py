"""
Data Loader and Validator Module.

Fetches the Credit Card Fraud Detection dataset from OpenML,
performs rigorous data validation, and outputs exploratory data analysis (EDA).
Ensures robust streaming and native Scikit-learn OpenML caching.
"""

import os
import gzip
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Optional
import requests
import pandas as pd
from sklearn.datasets import fetch_openml, get_data_home
from sklearn.datasets._openml import _get_data_description_by_id, _openml_path_from_url, _get_local_path

from config.config import (
    OPENML_DATA_ID,
    TARGET_CANDIDATE_COLUMNS,
)

logger = logging.getLogger(__name__)


def ensure_openml_cache(data_id: int = OPENML_DATA_ID) -> None:
    """
    Ensure the OpenML dataset file is present in Scikit-learn's local cache.

    If missing or corrupt, streams the official ARFF file from OpenML with connection
    pooling, chunks and compresses directly into Scikit-learn's native cache directory,
    verifying MD5 checksum integrity.
    """
    data_home = os.path.join(get_data_home(), "openml")
    try:
        desc = _get_data_description_by_id(data_id, data_home=data_home)
        url = desc.get("url")
        expected_md5 = desc.get("md5_checksum")

        openml_subpath = _openml_path_from_url(url)
        local_cache_path = os.path.normpath(_get_local_path(openml_subpath, data_home))

        # Check if already cached and intact
        if os.path.exists(local_cache_path) and os.path.getsize(local_cache_path) > 10_000_000:
            return  # Cache already present and populated

        print(f"Preparing Scikit-learn OpenML cache for dataset {data_id}...")
        os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
        tmp_path = local_cache_path + ".tmp"

        headers = {"Accept-Encoding": "gzip"}
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            r.raw.decode_content = False
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

        # Verify uncompressed MD5 checksum
        md5 = hashlib.md5()
        with gzip.GzipFile(tmp_path, "rb") as gz:
            for chunk in iter(lambda: gz.read(1024 * 1024), b""):
                md5.update(chunk)

        actual_md5 = md5.hexdigest()
        if expected_md5 and actual_md5 != expected_md5:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise ValueError(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")

        if os.path.exists(local_cache_path):
            os.remove(local_cache_path)
        os.rename(tmp_path, local_cache_path)
        print(f"\n✓ OpenML dataset verified and cached at: {local_cache_path}")
    except Exception as exc:
        logger.warning(f"Note on pre-caching: {exc}. Will attempt native fetch_openml directly.")


def load_and_validate_data(
    data_id: int = OPENML_DATA_ID,
    as_frame: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Fetch the Credit Card Fraud Detection dataset from OpenML and validate its integrity.

    Parameters
    ----------
    data_id : int
        OpenML dataset ID (default is 42397).
    as_frame : bool
        Whether to return the data as a pandas DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        Cleaned pandas DataFrame containing features and target, and the identified target column name.

    Raises
    ------
    RuntimeError
        If dataset fetching fails or network is unavailable.
    ValueError
        If validation checks fail.
    """
    print(f"\n{'='*70}")
    print(f"STEP 1: FETCHING DATASET FROM OPENML (ID: {data_id})")
    print(f"{'='*70}")

    # Ensure reliable cache
    ensure_openml_cache(data_id)

    try:
        print(f"Calling Scikit-learn fetch_openml(data_id={data_id}, as_frame=True)...")
        dataset = fetch_openml(data_id=data_id, as_frame=as_frame, parser="auto")
        print("✓ OpenML dataset loaded successfully via Scikit-learn fetch_openml.")
    except Exception as exc:
        error_msg = (
            f"ERROR: Failed to fetch dataset from OpenML (data_id={data_id}).\n"
            f"Details: {exc}\n\n"
            f"Troubleshooting:\n"
            f"1. Verify internet connectivity.\n"
            f"2. Ensure firewall/proxy permits HTTPS access to https://www.openml.org.\n"
            f"3. Verify that OpenML API is currently reachable."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from exc

    df: Optional[pd.DataFrame] = dataset.frame
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Fetched dataset frame is empty or not a valid DataFrame.")

    # 1. Dataset Dimensions Validation
    n_rows, n_cols = df.shape
    print(f"✓ Raw dataset dimensions: {n_rows:,} rows, {n_cols} columns.")

    if n_rows < 100_000:
        raise ValueError(
            f"Dataset row count ({n_rows:,}) is significantly lower than expected (~284,807)."
        )

    # 2. Identify Target Column
    target_col = None
    if hasattr(dataset, "target_names") and dataset.target_names:
        candidate = dataset.target_names[0]
        if candidate in df.columns:
            target_col = candidate

    if not target_col:
        for candidate in TARGET_CANDIDATE_COLUMNS:
            if candidate in df.columns:
                target_col = candidate
                break

    if not target_col:
        last_col = df.columns[-1]
        if df[last_col].nunique() == 2:
            target_col = last_col

    if not target_col:
        raise ValueError(
            f"Unable to identify target column. Columns available: {list(df.columns)}"
        )

    print(f"✓ Target column identified: '{target_col}'")

    # 3. Target Class Normalization & Validation
    unique_vals = df[target_col].dropna().unique()
    print(f"✓ Raw target values: {unique_vals}")

    if set(unique_vals).issubset({0, 1}):
        df[target_col] = df[target_col].astype(int)
    elif set(str(v).strip().lower() for v in unique_vals).issubset({"0", "1", "'0'", "'1'"}):
        df[target_col] = (
            df[target_col].astype(str).str.strip().str.replace("'", "").astype(int)
        )
    elif set(str(v).strip().lower() for v in unique_vals).issubset({"false", "true"}):
        df[target_col] = df[target_col].astype(str).str.lower().map({"true": 1, "false": 0})
    else:
        raise ValueError(
            f"Target column '{target_col}' does not have standard binary values: {unique_vals}"
        )

    # 4. Minority-Class Distribution Validation
    counts = df[target_col].value_counts().to_dict()
    normal_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    fraud_pct = (fraud_count / n_rows) * 100

    print(f"✓ Target distribution: Class 0 (Normal) = {normal_count:,}, "
          f"Class 1 (Fraud) = {fraud_count:,} ({fraud_pct:.4f}%)")

    if fraud_count == 0:
        raise ValueError("No fraudulent transactions (Class 1) found in target column.")
    if fraud_pct > 5.0:
        raise ValueError(
            f"Unexpectedly high fraud percentage ({fraud_pct:.2f}%). "
            f"Expected ~0.17% for Credit Card Fraud dataset."
        )

    # 5. Feature Structure Validation
    expected_v_features = [f"V{i}" for i in range(1, 29)]
    has_v_features = any(f in df.columns for f in expected_v_features)
    has_amount = "Amount" in df.columns
    has_time = "Time" in df.columns

    if not (has_v_features or (has_amount and has_time)):
        raise ValueError(
            "Dataset does not appear to be the genuine Credit Card Fraud Detection dataset. "
            f"Columns: {list(df.columns[:10])}..."
        )

    print("✓ Dataset schema validated successfully (genuine Credit Card Fraud dataset).")
    return df, target_col


def perform_eda(df: pd.DataFrame, target_col: str) -> dict:
    """
    Perform Exploratory Data Analysis (EDA) on the dataset and print key insights.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataset.
    target_col : str
        Target column name.

    Returns
    -------
    dict
        Dictionary containing EDA summary statistics.
    """
    print(f"\n{'='*70}")
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print(f"{'='*70}")

    n_rows, n_cols = df.shape
    print(f"Dataset Shape: {n_rows:,} rows × {n_cols} columns")

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    print(f"Total Missing Values: {total_nulls}")
    if total_nulls > 0:
        missing_cols = null_counts[null_counts > 0]
        print(f"Columns with missing values:\n{missing_cols}")

    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate Rows Count: {duplicate_count:,} ({(duplicate_count/n_rows)*100:.2f}%)")

    print("\n--- First 5 Rows (Preview of key columns) ---")
    preview_cols = [c for c in ["Time", "V1", "V2", "V3", "Amount", target_col] if c in df.columns]
    print(df[preview_cols].head())

    counts = df[target_col].value_counts().sort_index()
    percentages = (counts / n_rows) * 100

    print("\n--- Target Class Distribution ---")
    for cls_val in counts.index:
        label = "Fraudulent (1)" if cls_val == 1 else "Normal (0)"
        print(f"  • {label:15s}: {counts[cls_val]:>7,} transactions ({percentages[cls_val]:.4f}%)")

    imbalance_ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
    print(f"\n⚠️  CRITICAL TAKEAWAY FOR MODEL EVALUATION:")
    print(f"   The majority class (Normal) represents {percentages.get(0, 0):.2f}% of the data.")
    print(f"   Imbalance Ratio: {imbalance_ratio:.1f} to 1.")
    print(f"   A naive baseline model predicting EVERY transaction as 'Normal (0)'")
    print(f"   will achieve {percentages.get(0, 0):.2f}% ACCURACY while catching 0 FRAUD CASES!")
    print(f"{'='*70}\n")

    eda_summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "total_nulls": total_nulls,
        "duplicate_count": duplicate_count,
        "normal_count": int(counts.get(0, 0)),
        "fraud_count": int(counts.get(1, 0)),
        "fraud_percentage": float(percentages.get(1, 0)),
        "majority_percentage": float(percentages.get(0, 0)),
        "imbalance_ratio": float(imbalance_ratio),
    }
    return eda_summary
