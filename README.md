# 💳 MLflow Fraud Detection & Experiment Tracking

> **"Your ML Model Has 95% Accuracy. But Is It Actually Good?"**  
> *A beginner-friendly Machine Learning & MLOps project demonstrating why **Accuracy** fails on imbalanced data, and how to track and compare models using **MLflow**.*

---

## ⚡ Quick Run Guide (3 Steps)

All dependencies and Python 3.11 are pre-configured in `.venv`. Run these commands from the project root (`D:\ML_FLOW_DEMO`):

### 1️⃣ Run the ML Pipeline (Train, Evaluate & Track)
```powershell
.\.venv\Scripts\python.exe main.py
```
*Or double-click:* `scripts\run_training.bat`  
*Fetches OpenML dataset, trains 3 models, generates 10 plots in `results/`, and logs all runs to MLflow.*

---

### 2️⃣ Launch the MLflow Web Dashboard
```powershell
.\.venv\Scripts\python.exe launch_ui.py
```
*Or run:* `$env:MLFLOW_ALLOW_FILE_STORE="true"; .\.venv\Scripts\mlflow.exe ui --port 5000`  
*Or double-click:* `scripts\start_mlflow.bat`

👉 Open your browser at: **[http://localhost:5000](http://localhost:5000)**  
*Click the experiment **`Credit Card Fraud Detection - Model Comparison`**, select all 3 runs, and click **Compare**.*

---

### 3️⃣ Test Real-Time Fraud Predictions
```powershell
.\.venv\Scripts\python.exe predict_sample.py
```
*Loads the best Random Forest model directly from MLflow and tests legitimate vs. fraudulent transactions in real time.*

---

## 📊 Model Performance Comparison

Evaluated on **56,962 test transactions** ($98$ actual frauds):

| Model | Accuracy | Precision | Recall (Frauds Caught) | PR-AUC | False Alarms |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🤖 **Dummy Classifier (Baseline)** | **99.83%** | 0.00% | **0.00% (0 / 98)** | 0.0017 | **0** |
| 📈 **Logistic Regression (Balanced)** | 97.55% | 6.10% | **91.84% (90 / 98)** | 0.7159 | 1,385 |
| 🌲 **Random Forest (Balanced)** | **99.92%** | **76.19%** | **81.63% (80 / 98)** | **0.8224** | **25** |

---

## 🧠 The Accuracy Paradox in 3 Bullet Points

1. **Accuracy is Misleading:** The Dummy Classifier scores **99.83% accuracy** by simply predicting that *every transaction is normal*. It caught **0 frauds**!
2. **Recall Measures Protection:** Logistic Regression caught **91.8% of frauds**, but triggered 1,385 false alarms.
3. **PR-AUC is the Gold Standard:** Standard ROC-AUC is distorted by millions of legitimate transactions. **PR-AUC (0.8224)** proves that **Random Forest** provides the optimal balance of fraud detection with minimal false alarms.

---

## 📁 Key Project Files

* [`main.py`](file:///d:/ML_FLOW_DEMO/main.py) — End-to-end pipeline orchestrator.
* [`launch_ui.py`](file:///d:/ML_FLOW_DEMO/launch_ui.py) — One-click launcher for the MLflow UI.
* [`predict_sample.py`](file:///d:/ML_FLOW_DEMO/predict_sample.py) — Live inference script loading model from MLflow.
* [`results/`](file:///d:/ML_FLOW_DEMO/results/) — Saved plots (`accuracy_comparison.png`, `pr_curves_comparison.png`, confusion matrices).
* [`mlruns/`](file:///d:/ML_FLOW_DEMO/mlruns/) — Local MLflow tracking store containing all 3 runs.
* [`PRESENTATION_GUIDE.md`](file:///d:/ML_FLOW_DEMO/PRESENTATION_GUIDE.md) — 15-minute presentation script and Q&A notes for ML Club meetings.
* [`DOCUMENTATION.md`](file:///d:/ML_FLOW_DEMO/DOCUMENTATION.md) — Comprehensive technical reference manual.
