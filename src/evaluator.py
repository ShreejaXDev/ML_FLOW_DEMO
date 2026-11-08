"""
Model Evaluation Module.

Calculates comprehensive classification metrics, handles probability scoring safely,
generates confusion matrix elements (TP, FP, TN, FN), and produces classification reports.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Tuple[Dict[str, float], Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a classification model across standard and imbalanced-learning metrics.

    Parameters
    ----------
    model : Any
        Trained model or pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Ground truth test labels.
    model_name : str
        Human-readable name of the model.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]
        - metrics: Dictionary of numeric metrics for MLflow logging
        - report_dict: Full classification report as a dictionary
        - y_pred: Array of class predictions
        - y_proba: Array of predicted probabilities for positive class (1)
        - cm: Confusion matrix array [[tn, fp], [fn, tp]]
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*70}")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Obtain predicted probabilities for the positive class (Fraud: 1)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)
            if probas.shape[1] == 2:
                y_proba = probas[:, 1]
            elif probas.shape[1] == 1:
                # In DummyClassifier with most_frequent, probas may have shape (N, 1)
                # If the single class predicted is 0, the probability of class 1 is 0.0
                classes_ = getattr(model, "classes_", [0])
                if classes_[0] == 0:
                    y_proba = np.zeros(len(X_test))
                else:
                    y_proba = np.ones(len(X_test))
        except Exception:
            y_proba = None

    if y_proba is None and hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(X_test)
            # Min-max normalize or sigmoid to get 0-1 scale
            y_proba = 1 / (1 + np.exp(-decision))
        except Exception:
            y_proba = None

    if y_proba is None:
        # Fallback to binary predictions
        y_proba = y_pred.astype(float)

    # Calculate Core Classification Metrics (Positive class = 1)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    # ROC-AUC calculation with safe handling for constant scores
    try:
        if len(np.unique(y_proba)) <= 1:
            # Constant prediction produces no discriminatory ranking; ROC-AUC is 0.5
            roc_auc = 0.5
        else:
            roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = 0.5

    # PR-AUC / Average Precision calculation
    try:
        if len(np.unique(y_proba)) <= 1:
            # Baseline PR-AUC for constant predictor is the positive class ratio (prevalence)
            roc_pr = float(np.mean(y_test == 1))
        else:
            roc_pr = float(average_precision_score(y_test, y_proba))
    except Exception:
        roc_pr = float(np.mean(y_test == 1))

    # Confusion Matrix Breakdown
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Classification report
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Normal (Class 0)", "Fraud (Class 1)"],
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Normal (Class 0)", "Fraud (Class 1)"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": roc_pr,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    # Print clean summary
    print(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (Fraud): {prec:.4f}")
    print(f"Recall (Fraud)   : {rec:.4f}  <-- [Crucial for fraud detection: catches {tp}/{tp+fn} frauds]")
    print(f"F1-Score (Fraud) : {f1:.4f}")
    print(f"ROC-AUC Score    : {roc_auc:.4f}")
    print(f"PR-AUC Score     : {roc_pr:.4f}  <-- [Gold standard metric for severe class imbalance]")
    print("\nConfusion Matrix:")
    print(f"  • True Negatives  (TN - Normal identified as Normal)   : {tn:>6,}")
    print(f"  • False Positives (FP - Normal misclassified as Fraud) : {fp:>6,}")
    print(f"  • False Negatives (FN - Fraud MISSED by model!)        : {fn:>6,}")
    print(f"  • True Positives  (TP - Fraud successfully caught)     : {tp:>6,}")
    print("\nClassification Report:\n" + report_text)

    return metrics, report_dict, y_pred, y_proba, cm

