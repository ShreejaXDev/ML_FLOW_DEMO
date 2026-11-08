# 🎤 University Machine Learning Club: Workshop & Presentation Guide

### **Session Title: "Your ML Model Has 95% Accuracy. But Is It Actually Good?"**
*A 15-to-20-Minute Educational Interactive Workshop on the Accuracy Paradox & MLOps with MLflow.*

---

## 🎯 Presentation Objectives
1. Dispel the common student misconception that **Accuracy** is the primary metric for ML classification.
2. Demonstrate the **Accuracy Paradox** mathematically and visually using real fraud data.
3. Contrast a naive baseline against **Logistic Regression** and **Random Forest**.
4. Teach beginner-friendly **MLOps best practices** (experiment tracking, artifact management, and model signatures) using **MLflow**.
5. Perform a **live real-time inference demo** approving a legitimate transaction and blocking fraud.

---

## ⏱️ Recommended Slide Deck & Time Allocation (15–20 Mins)

### **Part 1: The Hook (3 Mins)**
* **Slide 1**: Title Slide: *"Your ML Model Has 95% Accuracy. But Is It Actually Good?"*
* **Speaker Script**: 
  > *"Imagine you build a machine learning model for a major fintech bank. You test it on 100,000 transactions, and your model boasts a staggering 99.8% accuracy. You excitedly deploy it to production. The next morning, the fraud prevention manager calls you into their office: criminals just stole $2.4 million through 100 fraudulent transactions, and your model flagged NONE of them. How could a 99.8% accurate model fail so catastrophically? Today, we will explore the infamous Accuracy Paradox and how MLOps tools like MLflow save us from deploying useless models."*
* **Slide 2**: The Dataset Reality
  * OpenML Dataset 42397: 284,807 transactions.
  * Legitimate (Normal): 284,315 (99.83%).
  * Fraudulent: 492 (0.17%).
  * Imbalance Ratio: **578 to 1**.
  * Visual: Show `results/class_distribution.png`.

---

### **Part 2: The Accuracy Paradox in Code (4 Mins)**
* **Slide 3**: The World's "Best" Dumb Classifier
  * Show the code of `DummyClassifier(strategy="most_frequent")`:
    ```python
    def predict(transaction):
        return 0  # Always guess 'Normal'
    ```
  * Math breakdown:
    $$\text{Accuracy} = \frac{56,864 \text{ (correct normal)}}{56,864 + 98} = 99.83\%$$
  * But what about Recall?
    $$\text{Recall} = \frac{\text{Fraud Caught}}{\text{Actual Fraud}} = \frac{0}{98} = 0.00\%$$
* **Slide 4**: Visualizing the Illusion
  * Show `results/accuracy_comparison.png` side-by-side with `results/recall_comparison.png`.
  * Point out how the Dummy Classifier looks competitive on accuracy, but completely collapses on Recall.

---

### **Part 3: Balancing the Models & The PR-AUC Metric (5 Mins)**
* **Slide 5**: Fixing the Imbalance
  * Explain `class_weight="balanced"`: How penalizing minority class misclassifications forces models to pay attention to fraud.
* **Slide 6**: The Trade-Off: Logistic Regression vs. Random Forest
  * **Logistic Regression**: High Recall (91.84%, 90/98 caught), but low Precision (6.10%, 1,385 false alarms).
  * **Random Forest**: Balanced Recall (81.63%, 80/98 caught) with high Precision (76.19%, only 25 false alarms).
* **Slide 7**: Why PR-AUC Over ROC-AUC?
  * Standard ROC-AUC is flattered by the 56,864 True Negatives ($FPR \approx 0$).
  * PR-AUC isolates Precision vs. Recall directly. Show `results/pr_curves_comparison.png`.

---

### **Part 4: Live MLflow Interactive Demo (5 Mins)**
1. **Open MLflow UI**:
   * Run in terminal: `.\.venv\Scripts\python.exe launch_ui.py`
   * Open `http://localhost:5000`.
2. **Show the Experiment View**:
   * Point out the 3 separate runs: *Dummy Classifier*, *Logistic Regression*, and *Random Forest*.
   * Show that all hyperparameters, train/test sample counts, and metrics are permanently recorded.
3. **Run Comparison**:
   * Check all 3 runs and click **Compare**.
   * Switch between **Metrics** (bar charts of Accuracy vs. PR-AUC) and **Artifacts** (Confusion Matrix heatmaps).
4. **Show Model Governance**:
   * Click on the `Random Forest` run.
   * Expand the `model/` artifact: show the **Model Signature** (inputs, data types, and outputs) and input examples.

---

### **Part 5: Live Real-Time Inference Demo (2 Mins)**
Open the terminal and run:
```powershell
.\.venv\Scripts\python.exe predict_sample.py
```
Explain to the audience:
> *"This script doesn't just re-run Python code. It connects directly to MLflow, queries the best Random Forest run, pulls the serialized model artifact from disk, and evaluates two transactions in real-time."*
* **Transaction 1 (Everyday Grocery Purchase - $45.50)**:
  * Model Verdict: `🟢 NORMAL TRANSACTION (Approved, 0.29% fraud risk)`
* **Transaction 2 (Suspicious Offshore Swipe - $199.99)**:
  * Model Verdict: `🔴 FRAUD DETECTED (Blocked & Alerted, 71.97% fraud risk)`

---

## ❓ Frequently Asked Questions (Q&A Prep)

**Q1: Why didn't you use SMOTE (Synthetic Minority Over-sampling Technique)?**  
*A: SMOTE is an effective technique, but in ultra-high dimensional and extreme imbalance settings ($578:1$), synthetic interpolation can introduce noisy artifacts across decision boundaries. Algorithmic balancing (`class_weight='balanced_subsample'`) is faster, avoids expanding memory, and preserves the authentic empirical distribution.*

**Q2: In production, should we pick Logistic Regression (91.8% Recall) or Random Forest (81.6% Recall)?**  
*A: It depends on the business cost function: if blocking a transaction costs $2 in SMS verification while missing a fraud costs $1,500, you prioritize Recall (Logistic Regression). If high false positives cause churn and customer outrage, Random Forest’s 76.2% precision is vastly superior. In practice, teams use threshold tuning on Random Forest probabilities to achieve both high Recall and manageable Precision.*

**Q3: Why did we scale only `Time` and `Amount` and leave `V1`–`V28` unscaled?**  
*A: `V1` through `V28` are the outputs of PCA (Principal Component Analysis), meaning they are already centered and normalized orthogonal components. Only `Time` (raw seconds up to 172,800) and `Amount` (dollars from $0 to thousands) require `StandardScaler` to prevent dominating linear gradients.*

