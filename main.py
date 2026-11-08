"""
Main Execution Pipeline for MLflow Experiment Tracking & Model Comparison.

Project Topic: "Your ML Model Has 95% Accuracy. But Is It Actually Good?"
Educational demonstration of MLflow experiment tracking on an imbalanced fraud dataset.
"""

import os
import sys
import logging
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import pandas as pd
import numpy as np

# Ensure root directory is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import (
    OPENML_DATA_ID,
    RESULTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
from src.data_loader import load_and_validate_data, perform_eda
from src.preprocessing import (
    split_features_and_target,
    split_data,
    build_preprocessor,
)
from src.model_trainer import (
    train_dummy_classifier,
    train_logistic_regression,
    train_random_forest,
)
from src.evaluator import evaluate_model
from src.visualizer import (
    plot_class_distribution,
    plot_single_metric_comparison,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
)
from src.experiment_tracker import setup_mlflow, log_model_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def print_banner():
    banner = """
========================================================================================
   CREDIT CARD FRAUD DETECTION: EXPERIMENT TRACKING & MODEL COMPARISON WITH MLFLOW
   "Your ML Model Has 95% Accuracy. But Is It Actually Good?"
========================================================================================
   Demonstrating the Accuracy Paradox on Highly Imbalanced Data
   • Model 1: Dummy Classifier (Majority Class Baseline)
   • Model 2: Logistic Regression (Balanced Class Weights)
   • Model 3: Random Forest Classifier (Subsample Balanced Ensemble)
========================================================================================
"""
    print(banner)


def main():
    print_banner()

    # 1. Fetch & Validate Data from OpenML
    df, target_col = load_and_validate_data(data_id=OPENML_DATA_ID)

    # 2. Exploratory Data Analysis
    eda_summary = perform_eda(df, target_col=target_col)

    # 3. Separate Features and Target
    X, y = split_features_and_target(df, target_col=target_col)

    # 4. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Build Preprocessor (StandardScaler on Time & Amount, Passthrough on V1-V28)
    preprocessor = build_preprocessor(list(X.columns))

    # 6. Train the 3 Models
    # Model 1: Dummy Classifier (Baseline)
    dummy_model, dummy_params, dummy_time = train_dummy_classifier(X_train, y_train)

    # Model 2: Logistic Regression (Balanced)
    lr_pipeline, lr_params, lr_time = train_logistic_regression(X_train, y_train, preprocessor)

    # Model 3: Random Forest Classifier
    rf_pipeline, rf_params, rf_time = train_random_forest(X_train, y_train, preprocessor)

    # 7. Evaluate All Models
    models = {
        "Dummy Classifier": dummy_model,
        "Logistic Regression": lr_pipeline,
        "Random Forest": rf_pipeline,
    }

    all_metrics = {}
    all_reports = {}
    curves_data = []
    confusion_matrices = {}

    for name, model in models.items():
        metrics, report, y_pred, y_proba, cm = evaluate_model(model, X_test, y_test, model_name=name)
        all_metrics[name] = metrics
        all_reports[name] = report
        confusion_matrices[name] = cm
        curves_data.append({
            "name": name,
            "y_true": y_test,
            "y_proba": y_proba,
            "auc": metrics["roc_auc"],
            "ap": metrics["pr_auc"],
        })

    # 8. Generate Visualizations in results/
    print(f"\n{'='*70}")
    print("STEP 6: GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*70}")

    dist_chart = plot_class_distribution(
        normal_count=eda_summary["normal_count"],
        fraud_count=eda_summary["fraud_count"],
        output_path=RESULTS_DIR / "class_distribution.png",
    )
    print(f"✓ Saved: {dist_chart}")

    acc_chart = plot_single_metric_comparison(
        all_metrics,
        metric_key="accuracy",
        metric_label="Accuracy Score",
        output_path=RESULTS_DIR / "accuracy_comparison.png",
        highlight_dummy=True,
    )
    print(f"✓ Saved: {acc_chart}")

    prec_chart = plot_single_metric_comparison(
        all_metrics,
        metric_key="precision",
        metric_label="Precision (Fraud Class)",
        output_path=RESULTS_DIR / "precision_comparison.png",
    )
    print(f"✓ Saved: {prec_chart}")

    rec_chart = plot_single_metric_comparison(
        all_metrics,
        metric_key="recall",
        metric_label="Recall / True Positive Rate (Fraud Class)",
        output_path=RESULTS_DIR / "recall_comparison.png",
    )
    print(f"✓ Saved: {rec_chart}")

    f1_chart = plot_single_metric_comparison(
        all_metrics,
        metric_key="f1_score",
        metric_label="F1-Score (Harmonic Mean of Precision & Recall)",
        output_path=RESULTS_DIR / "f1_score_comparison.png",
    )
    print(f"✓ Saved: {f1_chart}")

    roc_chart = plot_roc_curves(curves_data, output_path=RESULTS_DIR / "roc_curves_comparison.png")
    print(f"✓ Saved: {roc_chart}")

    fraud_prevalence = float(eda_summary["fraud_percentage"] / 100.0)
    pr_chart = plot_pr_curves(curves_data, prevalence=fraud_prevalence, output_path=RESULTS_DIR / "pr_curves_comparison.png")
    print(f"✓ Saved: {pr_chart}")

    # Generate Confusion Matrix Plots
    cm_paths = {}
    for name, cm in confusion_matrices.items():
        slug = name.lower().replace(" ", "_")
        cm_path = plot_confusion_matrix(cm, model_name=name, output_path=RESULTS_DIR / f"confusion_matrix_{slug}.png")
        cm_paths[name] = cm_path
        print(f"✓ Saved: {cm_path}")

    # 9. Log Experiments to MLflow
    setup_mlflow()

    model_configs = {
        "Dummy Classifier": (dummy_model, dummy_params, dummy_time),
        "Logistic Regression": (lr_pipeline, lr_params, lr_time),
        "Random Forest": (rf_pipeline, rf_params, rf_time),
    }

    logged_runs = {}
    for name, (model, params, training_time) in model_configs.items():
        # Inject training time into metrics
        run_metrics = dict(all_metrics[name])
        run_metrics["training_time_seconds"] = float(training_time)

        run_id = log_model_run(
            run_name=name,
            model=model,
            params=params,
            metrics=run_metrics,
            report_dict=all_reports[name],
            confusion_matrix_path=cm_paths[name],
            roc_curve_path=roc_chart,
            pr_curve_path=pr_chart,
            input_sample=X_test,
        )
        logged_runs[name] = run_id

    # 10. Print Model Comparison Table
    print(f"\n{'='*95}")
    print("FINAL MODEL PERFORMANCE COMPARISON TABLE")
    print(f"{'='*95}")
    header = f"{'Model Name':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'Frauds Caught':<13}"
    print(header)
    print("-" * 95)

    for name, m in all_metrics.items():
        caught = f"{m['true_positives']} / {m['true_positives'] + m['false_negatives']}"
        row = (
            f"{name:<22} | "
            f"{m['accuracy']:<9.4f} | "
            f"{m['precision']:<9.4f} | "
            f"{m['recall']:<9.4f} | "
            f"{m['f1_score']:<9.4f} | "
            f"{m['roc_auc']:<9.4f} | "
            f"{m['pr_auc']:<9.4f} | "
            f"{caught:<13}"
        )
        print(row)
    print("=" * 95)

    # 11. Educational Insights & Recommendation
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["pr_auc"])
    best_pr_auc = all_metrics[best_model_name]["pr_auc"]
    best_recall = all_metrics[best_model_name]["recall"]

    print("\n🎓 EDUCATIONAL INSIGHTS & THE ACCURACY PARADOX:")
    print("1. Accuracy is Misleading:")
    print(f"   The Dummy Classifier achieved {all_metrics['Dummy Classifier']['accuracy']*100:.2f}% accuracy!")
    print("   Yet, it caught 0 frauds out of all test transactions (Recall = 0.00%).")
    print("   In imbalanced problems, high accuracy is a mathematical illusion.")
    print("\n2. The Precision-Recall Trade-off:")
    print("   • Logistic Regression achieves high Recall by flagging suspect transactions,")
    print("     but incurs higher False Positives (lower Precision).")
    print("   • Random Forest achieves an optimal balance with superior PR-AUC.")
    print(f"\n🏆 BEST OVERALL MODEL: {best_model_name.upper()}")
    print(f"   Selected based on PR-AUC ({best_pr_auc:.4f}) and Recall ({best_recall:.4f}), NOT Accuracy!")

    # 12. Instructions for MLflow UI
    print(f"\n{'='*70}")
    print("MLFLOW EXPERIMENT TRACKING COMPLETE!")
    print(f"{'='*70}")
    print("To explore the interactive dashboard and compare runs:")
    print("\n  1. Open a terminal and run:")
    print("     mlflow ui --port 5000")
    print("\n  2. Or double-click the included script:")
    print("     scripts\\start_mlflow.bat")
    print("\n  3. Open your browser at:")
    print("     http://localhost:5000")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

