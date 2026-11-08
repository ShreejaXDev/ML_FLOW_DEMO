"""
MLflow Experiment Tracking Module.

Manages MLflow lifecycle:
- Initializing experiments with a persistent local tracking URI (mlruns/)
- Managing separate runs for Dummy Classifier, Logistic Regression, and Random Forest
- Logging parameters, metrics, artifacts (plots, reports, JSONs), model signatures, and models
- Applying metadata tags for discovery and comparison
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd

from config.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MLFLOW_DEFAULT_TAGS,
)


def setup_mlflow() -> str:
    """
    Configure MLflow tracking URI and create/retrieve the experiment.

    Returns
    -------
    str
        Active experiment ID.
    """
    print(f"\n{'='*70}")
    print("STEP 5: INITIALIZING MLFLOW TRACKING ENVIRONMENT")
    print(f"{'='*70}")
    print(f"✓ Tracking URI   : {MLFLOW_TRACKING_URI}")
    print(f"✓ Experiment Name: {MLFLOW_EXPERIMENT_NAME}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            tags={"project": "fraud_detection", "created_by": "mlflow_fraud_demo"},
        )
        print(f"✓ Created new MLflow experiment with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"✓ Found existing MLflow experiment with ID: {experiment_id}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return experiment_id


def log_model_run(
    run_name: str,
    model: Any,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    report_dict: Dict[str, Any],
    confusion_matrix_path: Path,
    roc_curve_path: Optional[Path] = None,
    pr_curve_path: Optional[Path] = None,
    input_sample: Optional[pd.DataFrame] = None,
    extra_tags: Optional[Dict[str, str]] = None,
) -> str:
    """
    Execute and log a complete model run inside the active MLflow experiment.

    Parameters
    ----------
    run_name : str
        Human-readable name of the run (e.g. 'Dummy Classifier').
    model : Any
        Trained Scikit-learn estimator or Pipeline.
    params : Dict[str, Any]
        Model parameters and metadata.
    metrics : Dict[str, float]
        Evaluation metrics (accuracy, recall, pr_auc, etc.).
    report_dict : Dict[str, Any]
        Classification report dictionary.
    confusion_matrix_path : Path
        Path to the confusion matrix image file.
    roc_curve_path : Optional[Path]
        Path to the ROC curve image.
    pr_curve_path : Optional[Path]
        Path to the Precision-Recall curve image.
    input_sample : Optional[pd.DataFrame]
        A sample of test input features for signature inference.
    extra_tags : Optional[Dict[str, str]]
        Run-specific tags.

    Returns
    -------
    str
        The generated MLflow Run ID.
    """
    print(f"\n--- Logging Run to MLflow: '{run_name}' ---")

    tags = dict(MLFLOW_DEFAULT_TAGS)
    tags["model_name"] = run_name
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"✓ Active Run ID: {run_id}")

        # 1. Log Tags
        mlflow.set_tags(tags)

        # 2. Log Parameters
        clean_params = {}
        for k, v in params.items():
            # MLflow parameters must be primitives or strings
            clean_params[k] = str(v) if not isinstance(v, (int, float, bool, str)) else v
        mlflow.log_params(clean_params)
        print(f"✓ Logged {len(clean_params)} parameters.")

        # 3. Log Metrics
        mlflow.log_metrics(metrics)
        print(f"✓ Logged {len(metrics)} metrics (Accuracy={metrics['accuracy']:.4f}, Recall={metrics['recall']:.4f}, PR-AUC={metrics['pr_auc']:.4f}).")

        # 4. Save and Log Artifacts
        # 4a. Classification Report as TXT
        report_file = Path(f"results/classification_report_{run_name.lower().replace(' ', '_')}.txt")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w") as f:
            f.write(f"Classification Report for {run_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(json.dumps(report_dict, indent=4))
        mlflow.log_artifact(str(report_file), artifact_path="reports")

        # 4b. Evaluation Summary as JSON
        summary_file = Path(f"results/summary_{run_name.lower().replace(' ', '_')}.json")
        with open(summary_file, "w") as f:
            json.dump({
                "model_name": run_name,
                "metrics": metrics,
                "parameters": clean_params,
            }, f, indent=4)
        mlflow.log_artifact(str(summary_file), artifact_path="reports")

        # 4c. Visualizations
        if confusion_matrix_path.exists():
            mlflow.log_artifact(str(confusion_matrix_path), artifact_path="visualizations")
        if roc_curve_path and roc_curve_path.exists():
            mlflow.log_artifact(str(roc_curve_path), artifact_path="visualizations")
        if pr_curve_path and pr_curve_path.exists():
            mlflow.log_artifact(str(pr_curve_path), artifact_path="visualizations")

        print("✓ Logged artifacts (confusion matrix, evaluation reports, curve comparisons).")

        # 5. Log Model with Signature & Input Example
        try:
            signature = None
            input_example = None
            if input_sample is not None and len(input_sample) > 0:
                input_example = input_sample.iloc[:5]
                # Try inferring signature
                try:
                    preds = model.predict(input_example)
                    signature = infer_signature(input_example, preds)
                except Exception:
                    signature = None

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )
            print(f"✓ Logged Scikit-learn model artifact with signature.")
        except Exception as exc:
            print(f"⚠️ Warning: Could not log model via mlflow.sklearn.log_model: {exc}")

    print(f"✓ Run '{run_name}' successfully completed in MLflow.")
    return run_id

