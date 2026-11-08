# 🚀 Credit Card Fraud Detection: Experiment Tracking & Model Comparison with MLflow

> **"Your ML Model Has 95% Accuracy. But Is It Actually Good?"**  
> *An educational, production-grade Machine Learning & MLOps project demonstrating the **Accuracy Paradox** on imbalanced data and how to systematically track and compare models using **MLflow**.*

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [The Core Problem: Why 99.8% Accuracy Is Misleading](#-the-core-problem-why-998-accuracy-is-misleading)
3. [Dataset Information](#-dataset-information)
4. [Dataset Fetching: OpenML Integration](#-dataset-fetching-openml-integration)
5. [What Is MLflow & Why Experiment Tracking Matters](#-what-is-mlflow--why-experiment-tracking-matters)
6. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
7. [Technology Stack](#-technology-stack)
8. [Setup & Installation (Windows Guide)](#-setup--installation-windows-guide)
9. [How to Run the Project](#-how-to-run-the-project)
10. [Launching the MLflow Dashboard](#-launching-the-mlflow-dashboard)
11. [Understanding the Evaluation Metrics](#-understanding-the-evaluation-metrics)
12. [The Three Machine Learning Models](#-the-three-machine-learning-models)
13. [How to Compare Model Runs in MLflow](#-how-to-compare-model-runs-in-mlflow)
14. [Five Key Screenshots to Capture for LinkedIn / Presentation](#-five-key-screenshots-to-capture-for-linkedin--presentation)
15. [Future Improvements](#-future-improvements)
16. [Official References & Links](#-official-references--links)

---

## 📖 Project Overview

When developing Machine Learning classifiers, beginners often rely entirely on **Accuracy** to evaluate their models. If a model reports **99.8% Accuracy**, it is tempting to believe the model is practically flawless.

However, in critical real-world applications—such as **credit card fraud detection**, **medical tumor diagnosis**, or **cybersecurity intrusion detection**—the events we care most about represent only a tiny fraction of the data (often $<0.2\%$).

In this project, we prove that **a completely useless model can effortlessly score 99.8% accuracy while failing to detect a single fraudulent transaction**.

We then demonstrate how to:
1. Build a robust data validation and preprocessing pipeline using Scikit-learn.
2. Train a baseline **Dummy Classifier**, a balanced **Logistic Regression**, and a balanced **Random Forest**.
3. Evaluate them using **Recall**, **Precision**, **F1-Score**, **ROC-AUC**, and **PR-AUC (Precision-Recall Area Under Curve)**.
4. Track all experiments, hyperparameters, evaluation metrics, confusion matrix charts, and serialized model files using **MLflow Tracking**.
5. Compare runs side-by-side inside the interactive **MLflow UI**.

---

## ⚠️ The Core Problem: Why 99.8% Accuracy Is Misleading

Imagine an automated credit card monitoring system that processes **100,000 transactions** every day:
- **99,827** transactions are legitimate (Normal).
- **173** transactions are criminal fraud.

If you write a "model" with a single rule:
```python
def predict_fraud(transaction):
    return "Normal"  # Always guess 0 (Normal)
```
- Total correct predictions: **99,827**
- Total transactions: **100,000**
- **Accuracy = 99,827 / 100,000 = 99.83%!**

On paper, this model sounds astonishingly accurate. In reality, **it caught zero frauds ($0/173$)**, costing the bank millions of dollars. This phenomenon is known as the **Accuracy Paradox**.

### What Should We Use Instead?
- **Recall (Sensitivity)**: What percentage of actual fraudulent transactions did our model successfully catch?
- **Precision**: When our model raises an alarm, what percentage is actually fraud (versus annoying a legitimate customer)?
- **PR-AUC (Precision-Recall AUC)**: The gold-standard metric for severely imbalanced data, evaluating the trade-off across all classification thresholds without being inflated by overwhelming true negatives.

---

## 📊 Dataset Information

This project uses the famous **Credit Card Fraud Detection** dataset hosted on OpenML:
- **OpenML ID**: `42397`
- **Total Transactions**: `284,807`
- **Total Features**: `30` numerical predictors (`Time`, `Amount`, and `V1` through `V28`)
- **PCA Transformation**: Features `V1` to `V28` are principal components obtained through PCA to protect customer privacy and confidential transaction data.
- **Unscaled Features**: `Time` (seconds elapsed between each transaction and the first transaction) and `Amount` (transaction amount).
- **Target Variable**: Binary classification:
  - `0`: Normal transaction ($284,315$ instances, $\approx 99.83\%$)
  - `1`: Fraudulent transaction ($492$ instances, $\approx 0.17\%$)
- **Imbalance Ratio**: $\approx 578:1$.

---

## 🌐 Dataset Fetching: OpenML Integration

This project requires **no Kaggle API tokens**, **no manual CSV downloads**, and **no credentials**. The dataset is fetched online directly through Scikit-learn's native OpenML client:

```python
from sklearn.datasets import fetch_openml

data = fetch_openml(data_id=42397, as_frame=True, parser="auto")
df = data.frame
```

### Automated Validation Checks
Before passing data to the models, `src/data_loader.py` enforces automated integrity checks:
1. **Network Integrity**: Verifies OpenML connection with graceful error messaging.
2. **Dimension Verification**: Confirms that approximately $284,807$ rows and $31$ columns are present.
3. **Target Column Identification**: Automatically identifies candidate columns (`Class`, `target`, `fraud`) without hardcoding assumptions.
4. **Target Value Normalization**: Cleans and converts string/categorical targets (`'0'`, `'1'`, `'True'`, `'False'`) into integer binary classes `0` and `1`.
5. **Minority Class Check**: Confirms fraud count is positive and fraud percentage aligns with the expected $\approx 0.17\%$.
6. **Schema Check**: Confirms presence of PCA features `V1` through `V28`, `Time`, and `Amount`.

---

## 🧪 What Is MLflow & Why Experiment Tracking Matters

Machine Learning development is inherently iterative. Data scientists experiment with multiple algorithms, feature transformations, and hyperparameters. Without automated tracking:
- Hyperparameters and test metrics get lost in scattered terminal outputs or spreadsheets.
- Models cannot be reliably reproduced or audited.
- Comparing models requires manual collation of charts and metrics.

### How MLflow Solves This
**MLflow** is an open-source platform that manages the end-to-end machine learning lifecycle. In this project, we leverage **MLflow Tracking**:

- **Parameters Logging**: `mlflow.log_params()` stores model configurations (`class_weight`, `max_iter`, `n_estimators`, preprocessing steps).
- **Metrics Logging**: `mlflow.log_metrics()` tracks numerical scores (`accuracy`, `recall`, `precision`, `f1_score`, `roc_auc`, `pr_auc`, `true_positives`, etc.).
- **Artifacts Logging**: `mlflow.log_artifact()` stores confusion matrix heatmaps, classification report text files, and comparison curve charts.
- **Model Registry & Logging**: `mlflow.sklearn.log_model()` bundles the trained pipeline with its input example and model schema signature for reproducible deployment.

---

## 🏗️ Project Architecture & Directory Structure

```
mlflow-fraud-detection/
│
├── main.py                    # Complete end-to-end execution pipeline
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation & guide
├── .gitignore                 # Excludes environments, caches, and datasets
│
├── config/
│   └── config.py              # Centralized paths, dataset IDs, hyperparameters, tags
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # OpenML dataset fetching, validation, and EDA
│   ├── preprocessing.py       # Stratified 80/20 train-test split & ColumnTransformer
│   ├── model_trainer.py       # Trains Dummy Classifier, Logistic Regression, Random Forest
│   ├── evaluator.py           # Calculates Accuracy, Precision, Recall, F1, ROC, PR, TP/FP/TN/FN
│   ├── experiment_tracker.py  # MLflow experiment setup, run logging, and artifact management
│   └── visualizer.py          # Publication-ready Matplotlib & Seaborn charts
│
├── results/                   # Generated evaluation plots and JSON/TXT summaries
│   ├── class_distribution.png
│   ├── accuracy_comparison.png
│   ├── recall_comparison.png
│   ├── precision_comparison.png
│   ├── f1_score_comparison.png
│   ├── roc_curves_comparison.png
│   ├── pr_curves_comparison.png
│   └── confusion_matrix_*.png
│
├── models/                    # Directory for serialized local artifacts
│   └── .gitkeep
│
├── mlruns/                    # Local MLflow file-based tracking store
│
└── scripts/
    ├── run_training.bat       # Windows one-click pipeline runner
    └── start_mlflow.bat       # Windows one-click MLflow dashboard launcher
```

---

## 💻 Technology Stack

- **Language**: Python 3.10+ (Tested on Python 3.11)
- **Machine Learning**: `scikit-learn>=1.4.0`
- **Experiment Tracking**: `mlflow>=2.10.0`
- **Data Manipulation**: `pandas>=2.2.0`, `numpy>=1.26.0`
- **Visualization**: `matplotlib>=3.8.0`, `seaborn>=0.13.0`
- **Model Serialization**: `joblib>=1.3.0`

---

## ⚙️ Setup & Installation (Windows Guide)

### 1. Clone or Open the Repository
```powershell
cd d:\ML_FLOW_DEMO
```

### 2. Create and Activate a Virtual Environment
```powershell
# Using Python built-in venv:
python -m venv .venv

# Activate on PowerShell:
.\.venv\Scripts\Activate.ps1

# Or activate on Command Prompt (cmd.exe):
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

You can run the full end-to-end pipeline with a single command:

```powershell
python main.py
```

*Or simply double-click `scripts\run_training.bat`.*

### What the Pipeline Executes Automatically:
1. **Loads configuration** from `config/config.py`.
2. **Fetches OpenML dataset** (data_id: 42397) and caches it locally.
3. **Validates dataset** dimensions, schema, and fraud class counts.
4. **Prints Exploratory Data Analysis (EDA)** statistics to terminal.
5. **Applies ColumnTransformer**: scales `Time` and `Amount` with `StandardScaler`; passes through `V1`–`V28`.
6. **Performs Stratified 80:20 Train-Test Split** (`random_state=42`).
7. **Trains 3 Models**:
   - Model 1: Baseline Dummy Classifier (`strategy="most_frequent"`)
   - Model 2: Balanced Logistic Regression (`class_weight="balanced"`, `max_iter=1000`)
   - Model 3: Balanced Random Forest (`n_estimators=100`, `max_depth=10`, `class_weight="balanced_subsample"`)
8. **Evaluates all 3 models** across Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
9. **Generates comparison visualizations** and saves them to `results/`.
10. **Logs 3 distinct runs into MLflow** inside the experiment `"Credit Card Fraud Detection - Model Comparison"`.
11. **Prints terminal comparison table** and highlights the best model.

---

## 🖥️ Launching the MLflow Dashboard

To open the interactive MLflow UI:

```powershell
mlflow ui --port 5000
```

*Or double-click `scripts\start_mlflow.bat`.*

Then open your browser at:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 📐 Understanding the Evaluation Metrics

| Metric | Formula | Why It Matters in Fraud Detection |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | **Misleading.** Dominated by the $99.83\%$ legitimate transactions. |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | **Crucial.** Measures what percentage of actual fraud cases were detected. A low recall means criminals walk away with stolen money. |
| **Precision** | $\frac{TP}{TP + FP}$ | Measures what fraction of flagged transactions were actually fraudulent. Low precision causes friction with legitimate cardholders. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing false alarms and missed fraud. |
| **ROC-AUC** | Area under ROC (TPR vs. FPR) | Measures general ranking capability across all thresholds. Can be overly optimistic on extreme class imbalance due to huge TN count. |
| **PR-AUC (Average Precision)**| Area under Precision-Recall curve | **The Gold Standard.** Directly evaluates the trade-off between Precision and Recall without being distorted by the majority class. |

---

## 🤖 The Three Machine Learning Models

### 1. Dummy Classifier (Baseline)
- **Class**: `sklearn.dummy.DummyClassifier(strategy="most_frequent")`
- **Behavior**: Always predicts the majority class (`0: Normal`).
- **Educational Role**: Demonstrates that high accuracy ($99.8\%$) can be achieved with zero intelligence ($0\%$ Recall, $0\%$ Precision, $0$ frauds caught).

### 2. Logistic Regression (Linear Model with Balanced Weights)
- **Class**: `sklearn.linear_model.LogisticRegression(class_weight="balanced")`
- **Behavior**: Weights the fraud class inversely proportional to its frequency ($\approx 578\times$ weight penalty for missing a fraud).
- **Educational Role**: Catches $\approx 90\%$ of frauds, but produces higher false positives due to linear decision boundaries.

### 3. Random Forest (Non-Linear Ensemble)
- **Class**: `sklearn.ensemble.RandomForestClassifier(class_weight="balanced_subsample", max_depth=10)`
- **Behavior**: An ensemble of 100 decision trees computing non-linear interactions across PCA components.
- **Educational Role**: Achieves superior precision and the highest overall PR-AUC score.

---

## 📈 Official Model Comparison Results

Below are the actual results calculated from the test set ($56,962$ transactions with $98$ fraud cases):

| Model Name | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score | ROC-AUC | PR-AUC | Frauds Caught |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | **99.83%** | 0.00% | **0.00%** | 0.0000 | 0.5000 | 0.0017 | **0 / 98** |
| **Logistic Regression (Balanced)** | 97.55% | 6.10% | **91.84%** | 0.1144 | 0.9722 | 0.7159 | **90 / 98** |
| **Random Forest (Balanced Subsample)**| **99.92%** | **76.19%** | **81.63%** | **0.7882** | **0.9807** | **0.8224** | **80 / 98** |

### Key Takeaways:
1. **The Dummy Classifier scores 99.83% Accuracy**, yet catches **0 out of 98 fraud cases**! If you evaluate purely on Accuracy, you would mistakenly conclude this model is competitive.
2. **Logistic Regression catches 90/98 fraud cases (91.84% Recall)**, but with lower Precision ($6.10\%$), meaning more normal transactions are flagged for review.
3. **Random Forest achieves the best overall balance**, catching **80/98 frauds (81.63% Recall)** with high Precision ($76.19\%$) and the highest PR-AUC score ($0.8224$).

---

## 🔍 How to Compare Model Runs in MLflow

When you open **http://localhost:5000**:
1. Click on the experiment **`Credit Card Fraud Detection - Model Comparison`** on the left sidebar.
2. You will see three separate runs:
   - `Random Forest`
   - `Logistic Regression`
   - `Dummy Classifier`
3. **Select all three runs** by checking their checkboxes.
4. Click the **"Compare"** button at the top of the table.
5. In the Comparison view:
   - Go to **Metrics** to see side-by-side bar charts for `accuracy`, `recall`, and `pr_auc`.
   - Go to **Artifacts** to preview the logged confusion matrices and ROC/PR curve plots.

---

## 📸 Five Key Screenshots to Capture for LinkedIn / Presentation

For your university Machine Learning Club post, capture these 5 high-impact screenshots:

1. **The Accuracy Paradox Chart (`results/accuracy_comparison.png`)**:
   - Shows all three models scoring $99\%+$ accuracy.
2. **The Recall & Frauds Caught Comparison (`results/recall_comparison.png`)**:
   - Contrasts the Dummy Classifier ($0\%$ Recall) against Logistic Regression and Random Forest ($80\%–92\%$ Recall).
3. **Precision-Recall Curves (`results/pr_curves_comparison.png`)**:
   - Demonstrates why PR-AUC separates the true ML models from the baseline.
4. **Confusion Matrix Heatmap comparison (`results/confusion_matrix_*.png`)**:
   - Displays the raw counts of True Positives ($TP$) vs. Missed Fraud ($FN$).
5. **The MLflow UI Dashboard Comparison**:
   - Screenshot of the MLflow runs table at `http://localhost:5000` showing logged parameters, metrics, and tags.

---

## 🔮 Future Improvements

- Implement threshold tuning (e.g. optimizing decision threshold for monetary business loss).
- Explore gradient boosting algorithms (XGBoost, LightGBM, CatBoost).
- Integrate MLflow Model Registry for staging and production deployment tags.
- Deploy an automated CI/CD evaluation check using GitHub Actions.

---

## 🔗 Official References & Links

- **OpenML Dataset (CreditCardFraudDetection, ID 42397)**: [https://www.openml.org/search?type=data&sort=runs&id=42397](https://www.openml.org/search?type=data&sort=runs&id=42397)
- **Scikit-learn Documentation**: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **MLflow Tracking Documentation**: [https://mlflow.org/docs/latest/tracking.html](https://mlflow.org/docs/latest/tracking.html)
- **Dal Pozzolo et al. (2015)**: *Calibrating Probability with Undersampling for Unbalanced Classification.*

