"""
Interactive Inference Script: Load Best Model from MLflow and Predict.

Demonstrates how to dynamically query the MLflow Tracking Server,
load the trained Random Forest pipeline with its StandardScaler preprocessing,
and run inference on both normal and suspicious transactions.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Configure MLflow environment
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import mlflow.sklearn
from config.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def run_inference_demo():
    print("=" * 75)
    print("   CREDIT CARD FRAUD DETECTION: REAL-TIME INFERENCE DEMO")
    print("   Loading Best Trained Model from MLflow Tracking Store")
    print("=" * 75)

    # 1. Connect to MLflow store
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Connecting to MLflow Tracking Store: {MLFLOW_TRACKING_URI}...")

    # 2. Retrieve the experiment runs
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not exp:
        print(f"Error: Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")
        print("Please run 'python main.py' first to train and log models.")
        return

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    rf_runs = runs[runs["tags.mlflow.runName"] == "Random Forest"]
    if rf_runs.empty:
        print("Error: No 'Random Forest' run found in MLflow experiment.")
        return

    best_run = rf_runs.iloc[0]
    run_id = best_run["run_id"]
    pr_auc = best_run.get("metrics.pr_auc", 0.0)
    recall = best_run.get("metrics.recall", 0.0)

    print(f"✓ Found Best Model: Random Forest")
    print(f"  • MLflow Run ID : {run_id}")
    print(f"  • Test Recall   : {recall:.2%}")
    print(f"  • Test PR-AUC   : {pr_auc:.4f}")

    # 3. Load model artifact
    model_uri = f"runs:/{run_id}/model"
    print(f"\nLoading serialized model artifact from '{model_uri}'...")
    model = mlflow.sklearn.load_model(model_uri)
    print("✓ Model pipeline successfully loaded (includes StandardScaler + Classifier)!")

    # 4. Prepare sample transactions for demonstration
    # Feature columns: Time, V1-V28, Amount
    feature_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # Transaction A: Legitimate transaction (low amount, normal PCA coordinates)
    normal_tx = {
        "Time": 420.0,
        "Amount": 45.50,
        "V1": 1.15, "V2": -0.25, "V3": 0.55, "V4": 0.35, "V5": -0.45,
        "V6": 0.12, "V7": -0.35, "V8": 0.08, "V9": 0.52, "V10": -0.15,
        "V11": -0.22, "V12": 0.15, "V13": -0.40, "V14": -0.05, "V15": 0.85,
        "V16": 0.35, "V17": -0.15, "V18": -0.20, "V19": 0.10, "V20": -0.05,
        "V21": -0.05, "V22": 0.12, "V23": -0.08, "V24": 0.02, "V25": 0.45,
        "V26": -0.32, "V27": 0.02, "V28": 0.01
    }

    # Transaction B: Known fraud pattern (negative spikes in V10, V12, V14, V17)
    fraud_tx = {
        "Time": 1500.0,
        "Amount": 199.99,
        "V1": -3.04, "V2": 2.15, "V3": -3.55, "V4": 3.42, "V5": -2.85,
        "V6": -1.25, "V7": -4.10, "V8": 1.45, "V9": -2.15, "V10": -5.20,
        "V11": 3.85, "V12": -6.15, "V13": -0.85, "V14": -7.45, "V15": -0.35,
        "V16": -4.20, "V17": -7.15, "V18": -2.50, "V19": 1.15, "V20": 0.85,
        "V21": 0.85, "V22": -0.25, "V23": -0.15, "V24": -0.45, "V25": 0.35,
        "V26": 0.42, "V27": 0.65, "V28": 0.25
    }

    test_df = pd.DataFrame([normal_tx, fraud_tx])[feature_columns]

    # 5. Predict
    preds = model.predict(test_df)
    probs = model.predict_proba(test_df)

    print("\n" + "=" * 75)
    print("   INFERENCE TEST RESULTS")
    print("=" * 75)

    cases = [
        ("Transaction #1 (Everyday Grocery Purchase - $45.50)", normal_tx),
        ("Transaction #2 (Suspicious Offshore Card Swipe - $199.99)", fraud_tx)
    ]

    for idx, (label, tx_data) in enumerate(cases):
        pred_class = preds[idx]
        fraud_prob = probs[idx][1]
        normal_prob = probs[idx][0]

        status = "🔴 FRAUD DETECTED (ACTION: BLOCK & ALERT)" if pred_class == 1 else "🟢 NORMAL TRANSACTION (ACTION: APPROVE)"

        print(f"\n{label}:")
        print(f"  • Transaction Amount : ${tx_data['Amount']:.2f}")
        print(f"  • Normal Probability : {normal_prob:.2%}")
        print(f"  • Fraud Probability  : {fraud_prob:.2%}")
        print(f"  • Model Verdict      : {status}")

    print("\n" + "=" * 75)
    print("✓ Inference completed successfully using MLflow-managed model artifact!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_inference_demo()

