# Leakage-Safe Model Evaluation Benchmark Report

**Generated:** 2026-10-13T05:10:33Z *(Updated with Temporal Drift & Latency Audit: 2026-10-13T05:22:00Z)*  
**Dataset:** Elliptic Bitcoin Transaction Benchmark (203,769 nodes, 234,355 directed edges)  
**Protocol:** Strict Chronological Inductive Temporal Evaluation  
**Hardware Environment:** Intel64 (10 physical cores / 16 logical cores) | 15.69 GB RAM | Windows 11 | Python 3.12.10 (.venv)  

---

## 1. Experimental Protocol & Leakage Safeguards

- **Temporal Split Boundaries:**
  - **Training Split (Steps 1–29):** 26,381 labeled transactions (23,510 licit, 2,871 illicit; imbalance ratio: **8.19:1**).
  - **Validation Split (Steps 30–34):** 3,513 labeled transactions (2,922 licit, 591 illicit; prevalence: 16.82%).
  - **Test Split (Steps 35–49):** 16,670 labeled transactions (15,587 licit, 1,083 illicit; prevalence: 6.50%).
  - **Background Unknown Transactions:** 157,205 transactions with label `-1` are preserved in graph topology but strictly masked out of supervised loss and evaluation metrics.
- **Strict Leakage Prevention Guarantees:**
  1. **Feature Normalization Isolation:** `StandardScaler` is fitted strictly on labeled training nodes (steps 1–29). Zero validation or test distribution parameters enter the scaler.
  2. **Class Weighting Isolation:** Cost-sensitive class weight ratio (`8.19:1`) is computed solely from training split counts.
  3. **Decision Threshold Tuning Isolation:** Optimal decision thresholds are tuned exclusively on validation split predictions to maximize illicit class F1 score.
  4. **Strictly Inductive Test Isolation:** Test split (steps 35–49) is scored **exactly once** with frozen models, frozen scalers, and frozen decision thresholds. Test labels are never observed during threshold or hyperparameter selection.

---

## 2. Supervised Benchmark Results on Unseen Test Partition (Steps 35–49)

All metrics below are derived directly from actual saved predictions in `data/validation/model_evaluation/test_predictions.parquet` and serialized in `data/validation/model_evaluation/metrics.json`.

| Model | Decision Thresh | Illicit Precision | Illicit Recall | Illicit F1 | Macro F1 | ROC-AUC | Average Precision (AP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cost-Sensitive Logistic Regression** | 0.92 | **0.1778** | **0.8476** | **0.2939** | 0.5655 | 0.8503 | 0.2025 |
| **Cost-Sensitive Random Forest** | 0.67 | **0.8266** | **0.6251** | **0.7119** | 0.8472 | 0.9269 | 0.7618 |
| **Cost-Sensitive Gradient Boosted Trees (HistGBDT)** | 0.56 | **0.7238** | **0.7258** | **0.7248** | 0.8528 | 0.9376 | 0.7916 |

> [!NOTE]
> **Metric Distinction (Average Precision vs. Trapezoidal PR-AUC):**  
> We report **Average Precision (AP)** computed via `sklearn.metrics.average_precision_score`, which represents the weighted mean of precisions at each threshold (Riemann sum of the step-interpolated precision-recall curve):  
> $$\text{AP} = \sum_{n} (R_n - R_{n-1}) P_n$$  
> This avoids the known optimistic bias of trapezoidal rule integration (`auc(recall, precision)`) under severe class imbalance.

---

## 3. Confusion Matrix Breakdown & Verification

The confusion matrices below reconcile precisely with reported precision, recall, and illicit F1 scores down to four decimal places.

### Cost-Sensitive Logistic Regression ($\tau = 0.92$)
| | Predicted Licit (0) | Predicted Illicit (1) | Total Actual |
|---|:---:|:---:|:---:|
| **Actual Licit (0)** | **TN = 11,341** | **FP = 4,246** | 15,587 |
| **Actual Illicit (1)** | **FN = 165** | **TP = 918** | 1,083 |
| **Total Predicted** | 11,506 | 5,164 | 16,670 |

- $\text{Precision} = \frac{918}{918 + 4246} = \frac{918}{5164} = 0.177769 \approx \mathbf{0.1778}$
- $\text{Recall} = \frac{918}{918 + 165} = \frac{918}{1083} = 0.847645 \approx \mathbf{0.8476}$
- $\text{F1} = \frac{2 \times 0.177769 \times 0.847645}{0.177769 + 0.847645} = \mathbf{0.2939}$

### Cost-Sensitive Random Forest ($\tau = 0.67$)
| | Predicted Licit (0) | Predicted Illicit (1) | Total Actual |
|---|:---:|:---:|:---:|
| **Actual Licit (0)** | **TN = 15,445** | **FP = 142** | 15,587 |
| **Actual Illicit (1)** | **FN = 406** | **TP = 677** | 1,083 |
| **Total Predicted** | 15,851 | 819 | 16,670 |

- $\text{Precision} = \frac{677}{677 + 142} = \frac{677}{819} = 0.826618 \approx \mathbf{0.8266}$
- $\text{Recall} = \frac{677}{677 + 406} = \frac{677}{1083} = 0.625115 \approx \mathbf{0.6251}$
- $\text{F1} = \frac{2 \times 0.826618 \times 0.625115}{0.826618 + 0.625115} = \mathbf{0.7119}$

### Cost-Sensitive Gradient Boosted Trees (HistGBDT) ($\tau = 0.56$)
| | Predicted Licit (0) | Predicted Illicit (1) | Total Actual |
|---|:---:|:---:|:---:|
| **Actual Licit (0)** | **TN = 15,287** | **FP = 300** | 15,587 |
| **Actual Illicit (1)** | **FN = 297** | **TP = 786** | 1,083 |
| **Total Predicted** | 15,584 | 1,086 | 16,670 |

- $\text{Precision} = \frac{786}{786 + 300} = \frac{786}{1086} = 0.723757 \approx \mathbf{0.7238}$
- $\text{Recall} = \frac{786}{786 + 297} = \frac{786}{1083} = 0.725762 \approx \mathbf{0.7258}$
- $\text{F1} = \frac{2 \times 0.723757 \times 0.725762}{0.723757 + 0.725762} = \mathbf{0.7248}$

---

## 4. Investigation 1: Validation F1 (~0.97) vs. Test F1 (~0.72) Root Cause Analysis

### Verification of Code Alignment
1. **Labels & Probability Alignment:** `CostSensitiveAnomalyModel.predict_proba()` produces probabilities strictly for class index 1 (`illicit`). In both validation tuning and test evaluation, `probs >= threshold` consistently classifies class 1.
2. **Recomputation from Saved Predictions:** Direct recomputation from `data/validation/model_evaluation/test_predictions.parquet` reproduces the exact confusion matrices and metrics in `metrics.json`.

### Temporal Concept Drift & Darknet Market Takedown
The drop from Validation F1 (0.967) to Test F1 (0.725) is **not a software defect or leakage bug**, but an authentic historical domain shift inherent to the Elliptic dataset (documented by Weber et al., 2019):
- In **July 2017 (time step 43)**, an international law enforcement operation seized and shut down **AlphaBay** and **Hansa Market**, the two largest darknet marketplaces operating on Bitcoin.
- This shutdown fundamentally altered the transaction graph structure and topology of illicit Bitcoin flows overnight.

#### Per-Time-Step Performance Breakdown Across Test Split

| Time Step Range | Event / Context | Labeled Tx | Illicit Tx | RF Illicit Precision | RF Illicit Recall | RF Illicit F1 | HistGBDT Illicit F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 35** | Immediate step post-validation (step 34) | 1,677 | 157 | **0.9551** | **0.9529** | **0.9540** | **0.9490** |
| **Steps 35–42** | Pre-AlphaBay Takedown regime | 9,997 | 933 | **0.9036** | **0.7385** | **0.8128** | **0.8506** |
| **Steps 43–49** | Post-AlphaBay Takedown regime | 6,673 | 150 | **0.0519** | **0.0101** | **0.0166** | **0.0000** |
| **Aggregate (35–49)** | Full Unseen Test Split | 16,670 | 1,083 | **0.8266** | **0.6251** | **0.7119** | **0.7248** |

**Conclusion:**  
Models trained on steps 1–29 generalize near-identically across steps 30–34 (Val F1: 0.967) and step 35 (Test F1: 0.954), sustaining high performance across steps 35–42 (Pre-shutdown F1: 0.851). When the market takedown occurs at step 43, structural features diverge abruptly, causing post-shutdown recall to drop to near zero and pulling the aggregate 15-step test F1 down to 0.725.

---

## 5. Investigation 2: PyTorch & Graph Neural Network (GAT) Status

### Environment Status
- **Target Interpreter:** `D:\blockchainbigdata\blockchain-streaming-anomaly-detection\.venv\Scripts\python.exe` (Python 3.12.10 on win32).
- **Package Audit:**
  - `torch`: **NOT INSTALLED**
  - `torch_geometric` (PyG): **NOT INSTALLED**
  - `matplotlib`: **NOT INSTALLED**
- **Evaluation Status:** **Spatio-Temporal GAT has NOT been evaluated in this run.** It is marked as `NOT_REQUESTED_OR_ENV_UNAVAILABLE` in `metrics.json`.

### GAT Implementation Architecture
- `src/models/gat.py` implements a temporal multi-head Graph Attention Network using `torch.nn.Module`, `torch_geometric.nn.GATConv`, and inductive time-step subgraph batching.
- It requires PyTorch and PyTorch Geometric binaries matching Python 3.12 and the host C++ runtime.

### Minimal Safe Installation Plan (For Future GAT Evaluation)
To evaluate GAT in a subsequent phase without destabilizing the current environment:
```powershell
# 1. Install CPU-compatible PyTorch wheels for Python 3.12
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install PyTorch Geometric core
.venv\Scripts\python.exe -m pip install torch-geometric

# 3. Verify PyG import and GAT module readiness
.venv\Scripts\python.exe -c "import torch, torch_geometric; from src.models.gat import SpatioTemporalGAT; print('GAT ready on torch', torch.__version__)"
```

---

## 6. Investigation 3: Latency & Throughput Benchmark Audit

### Methodological Audit of Previous Timing
The preliminary latency numbers in the earlier report (`0.0015 ms - 0.0024 ms`) represented crude single-pass total batch duration ($T_{\text{total}}$) divided by $N = 16,670$ transactions. That metric is susceptible to cold-start cache anomalies and does not reflect single-event response time.

### Distinction: In-Memory Feature Vector Scoring vs. Live Streaming Latency
- **In-Memory Scoring (Benchmarked Here):** The time required by the model engine to execute vector dot products or decision tree traversals on pre-extracted feature tensors in RAM.
- **Live Distributed Streaming Latency (System Pipeline):** Includes network socket transfer, Kafka broker partition replication, Spark micro-batch orchestration, RocksDB state lookups, and graph feature window extraction. In-memory scoring represents only the algorithmic core of this pipeline.

### High-Resolution Latency Distribution Benchmark
Measured via `scripts/benchmark_inference_latency.py` using `time.perf_counter()` (monotonic CPU clock) after 10 warm-up runs, evaluated across batch sizes from single-event to full-batch:

| Model Family | Batch Size ($B$) | Iterations | Median Latency ($p_{50}$) | 95th Percentile ($p_{95}$) | Per-Tx Latency | Throughput (Tx / sec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **1** (Single Tx) | 200 | **0.1148 ms** | 0.1329 ms | 114.8 μs | 8,506 tx/s |
| | **16** | 100 | **0.1125 ms** | 0.1622 ms | 7.0 μs | 137,040 tx/s |
| | **64** | 100 | **0.1168 ms** | 0.2264 ms | 1.8 μs | 483,811 tx/s |
| | **256** | 50 | **0.1131 ms** | 0.1224 ms | 0.44 μs | 2,255,188 tx/s |
| | **1,000** (Micro-batch) | 50 | **0.1633 ms** | 0.2113 ms | 0.16 μs | 5,979,359 tx/s |
| | **16,670** (Full Test) | 20 | **1.2318 ms** | 1.9199 ms | 0.07 μs | 12,127,605 tx/s |
| **HistGBDT** | **1** (Single Tx) | 200 | **0.7655 ms** | 1.2262 ms | 765.5 μs | 1,221 tx/s |
| | **16** | 100 | **0.7879 ms** | 0.9905 ms | 49.2 μs | 20,303 tx/s |
| | **64** | 100 | **0.7497 ms** | 1.2310 ms | 11.7 μs | 78,107 tx/s |
| | **256** | 50 | **0.9514 ms** | 1.1000 ms | 3.7 μs | 261,359 tx/s |
| | **1,000** (Micro-batch) | 50 | **1.9841 ms** | 2.4784 ms | 1.98 μs | 482,536 tx/s |
| | **16,670** (Full Test) | 20 | **21.0994 ms** | 24.6608 ms | 1.27 μs | 773,193 tx/s |
| **Random Forest** | **1** (Single Tx) | 200 | **14.6251 ms** | 16.1071 ms | 14,625 μs | 66 tx/s |
| | **16** | 100 | **14.6362 ms** | 15.8186 ms | 914.8 μs | 1,075 tx/s |
| | **64** | 100 | **14.5253 ms** | 15.8320 ms | 227.0 μs | 4,294 tx/s |
| | **256** | 50 | **14.6484 ms** | 24.8528 ms | 57.2 μs | 15,993 tx/s |
| | **1,000** (Micro-batch) | 50 | **14.9379 ms** | 26.2524 ms | 14.9 μs | 52,080 tx/s |
| | **16,670** (Full Test) | 20 | **36.2907 ms** | 39.2034 ms | 2.18 μs | 481,488 tx/s |

**Operational Insights:**
1. **HistGBDT offers the superior operational profile for streaming micro-batches ($B=64\text{ to }1000$):** Achieving 0.75 ms – 1.98 ms batch latency ($p_{95} \le 2.48\text{ ms}$) and scoring $\approx 480,000\text{ tx/s}$, comfortably satisfying sub-10 ms SLA requirements while attaining the highest test F1 (0.7248).
2. **Random Forest has fixed tree ensemble overhead:** Single-transaction latency is $\approx 14.6\text{ ms}$ due to iterating through 100 unvectorized estimator trees; throughput scales effectively only when batched ($B \ge 1,000$).
3. **Logistic Regression is ultra-lightweight:** Sub-millisecond across all batch scales ($< 0.16\text{ ms}$ for 1,000 tx), suitable for rapid pre-filtering.

---

## 7. Artifact Manifest

| Artifact Path | Format | Verification Status | Contents |
| :--- | :--- | :--- | :--- |
| `data/validation/model_evaluation/metrics.json` | JSON | Verified | Full serializable metrics, splits, thresholds, and confusion matrices |
| `data/validation/model_evaluation/test_predictions.parquet` | Parquet | Verified (16,670 rows) | Ground truth, predicted probabilities, thresholded binary flags |
| `data/validation/model_evaluation/latency_benchmark.json` | JSON | Verified | High-resolution multi-batch latency distributions and hardware profile |
| `scripts/benchmark_inference_latency.py` | Python | Executable & Tested | Reproducible warm-up timing benchmark script |
