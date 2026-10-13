# Leakage-Safe Model Evaluation Benchmark Report

**Generated:** 2026-10-13T05:10:33Z  
**Dataset:** Elliptic Bitcoin Transaction Benchmark  
**Protocol:** Strict Chronological Inductive Temporal Evaluation  

---

## 1. Experimental Protocol & Leakage Safeguards

- **Temporal Partitioning:**
  - **Training Split:** Steps 1 to 29 (26,381 labeled nodes; Licit: 23,510, Illicit: 2,871; Imbalance: 8.19:1)
  - **Validation Split:** Steps 30 to 34 (3,513 labeled nodes; Licit: 2,922, Illicit: 591)
  - **Test Split:** Steps 35 to 49 (16,670 labeled nodes; Licit: 15,587, Illicit: 1,083; Prevalence: 6.50%)
  - **Background Unknowns:** 157,205 transactions with label `-1` isolated from loss and metric evaluation.
- **Leakage Prevention Guarantees:**
  1. `StandardScaler` fitted **strictly on training partition** (zero exposure to validation or test statistics).
  2. Class weight ratio calculated **strictly from training labels** (`8.19:1`).
  3. Decision thresholds tuned **strictly on validation partition** to maximize illicit F1 score.
  4. Test partition evaluated **exactly once** after model decisions and thresholds are frozen.

---

## 2. Supervised Benchmark Results on Unseen Test Partition (Steps 35–49)

| Model | Decision Thresh | Illicit Precision | Illicit Recall | Illicit F1 | Macro F1 | ROC-AUC | PR-AUC (AP) | Latency / Tx |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cost-Sensitive Logistic Regression** | 0.92 | **0.1778** | **0.8476** | **0.2939** | 0.5655 | 0.8503 | 0.2025 | 0.0001 ms |
| **Cost-Sensitive Random Forest** | 0.67 | **0.8266** | **0.6251** | **0.7119** | 0.8472 | 0.9269 | 0.7618 | 0.0024 ms |
| **Cost-Sensitive Gradient Boosted Trees (HistGBDT)** | 0.56 | **0.7238** | **0.7258** | **0.7248** | 0.8528 | 0.9376 | 0.7916 | 0.0015 ms |

---

## 3. Confusion Matrix Breakdown (Test Set, 16,670 Transactions)

### Cost-Sensitive Logistic Regression (Threshold = 0.92)

| | Predicted Licit (0) | Predicted Illicit (1) |
|---|---|---|
| **Actual Licit (0)** | **TN = 11,341** | **FP = 4,246** |
| **Actual Illicit (1)** | **FN = 165** | **TP = 918** |

### Cost-Sensitive Random Forest (Threshold = 0.67)

| | Predicted Licit (0) | Predicted Illicit (1) |
|---|---|---|
| **Actual Licit (0)** | **TN = 15,445** | **FP = 142** |
| **Actual Illicit (1)** | **FN = 406** | **TP = 677** |

### Cost-Sensitive Gradient Boosted Trees (HistGBDT) (Threshold = 0.56)

| | Predicted Licit (0) | Predicted Illicit (1) |
|---|---|---|
| **Actual Licit (0)** | **TN = 15,287** | **FP = 300** |
| **Actual Illicit (1)** | **FN = 297** | **TP = 786** |


---

## 4. Hardware and Runtime Methodology

- **Inference Timing:** Measured over all 16,670 test transactions sequentially through Python runtime.
- **Environment:** Windows, Python 3.12, scikit-learn 1.9.1.
- **PyG / PyTorch Status:** PyTorch ('torch') is not installed in the active environment..
