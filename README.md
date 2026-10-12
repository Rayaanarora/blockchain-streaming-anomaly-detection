# Distributed Streaming Anomaly Detection on Blockchain Transaction Graphs

**Course:** 21CSC314P – Big Data Essentials  
**Institution:** SRM Institute of Science and Technology (SRMIST)  
**Faculty Guide:** Dr. Banu Priya Prathaban  

### 👥 Team Members
- **Rayaan Arora** (`RA2411031010073`)
- **Faheem Mushtaq** (`RA2411031010079`)
- **Lakshita Kumarkar** (`RA2411031010091`)
- **Aarushi Sarkar** (`RA2411031010099`)

---

## 🌟 Overview & Novel Hybrid Architecture: `TGAT-BoostNet`
This repository contains the end-to-end distributed Big Data pipeline for detecting illicit cryptocurrency transactions (anti-money laundering / AML) across continuous blockchain streams.

Instead of relying solely on static graph convolutions or tabular classifiers, we propose **TGAT-BoostNet (Spatio-Temporal Graph Attention Boosting Network)**:
1. **Stage 1 (Deep Graph Attention)**: 4-Head Graph Attention Network (`GATConv`) in PyTorch Geometric dynamically computing edge weights over peel-chains and continuous temporal embeddings.
2. **Stage 2 (Manifold Fusion)**: Concatenating 32-dimensional deep graph attention representations with the 165 raw transaction features into a 197-dimensional augmented manifold.
3. **Stage 3 (Cost-Weighted Boosting)**: Distributed Gradient Boosted Decision Forests operating on the augmented manifold with cost-sensitive class rebalancing ($w_{\text{illicit}} = 4.32\times$).

---

## 📊 Benchmark Results (Holdout Test Steps 35–49)
- **Illicit Class Recall:** **0.739** (higher fraud capture rate than GCN's 0.600 and EvolveGCN's 0.710)
- **AUC-ROC:** **0.919**
- **Average Precision (AP):** **0.770**
- **Average Streaming Latency:** **1.69 ms** per micro-batch
- **Sustained Streaming Throughput:** **399,525 transactions/sec**
- **Parallel Speedup (Amdahl's Law):** $S_P = 3.12\times$ on 4 workers ($78.1\%$ efficiency)

---

## 📁 Repository Structure
- `src/spark/` - Apache Spark distributed batch preprocessing pipeline and data contracts:
  - `schemas.py`: Formal schemas for raw records, assembled transactions, and 18-column feature tables.
  - `zenodo_batch.py`: Distributed batch transformer and bounded PyArrow Parquet writer.
- `src/data/` - Dataset boundary handling, Elliptic audit, preprocessing, and validation.
- `src/graph/` - Directed bipartite address-transaction graph builders (Cartesian-free).
- `src/features/` - Chronological transaction and graph feature extractors.
- `scripts/smoke_test_spark.py` - Reproducible Spark batch smoke test validating causality and PyArrow output.
- `tests/` - Comprehensive unit and integration test suite (34 tests).
- `Review2_Blockchain_AML_Colab_Notebook.ipynb` - Full interactive Jupyter/Colab notebook with all 14 executable sections
- `Distributed_Blockchain_Anomaly_Review2.pptx` - Official 15-slide presentation for Review 2
- `Report_Review2_Blockchain_21CSC314P.docx` - Complete academic report in Times New Roman 12pt format
- `Report_Review2_Blockchain_21CSC314P.md` - Complete markdown report
- `train_sota_dl.py` - PyTorch Geometric implementation script
- `*.png` - All 7 high-resolution diagnostic figure panels

---

## ⚙️ Distributed Preprocessing Architecture: Spark vs. Bounded PyArrow

To scale to multi-million Zenodo Bitcoin transactions on Windows without unverified native binary dependencies (`winutils.exe` / `hadoop.dll`), the preprocessing pipeline divides responsibilities:

```
[ Raw Zenodo Data on D: ]
           │
           ▼
[ Apache Spark Catalyst Engine ]
  ├─ Raw text parsing & hash validation (64-char hex)
  ├─ Address disambiguation (unableToDecode / coinbase masking)
  ├─ Completed transaction grouping & fee aggregation
  └─ Leakage-safe chronological window features (strictly rowsBetween(unboundedPreceding, -1))
           │
           ▼  (Partition-by-partition toLocalIterator streaming; bounded <50 MB driver RAM)
[ Native PyArrow Parquet Writer ]
  ├─ Snappy compression
  ├─ Chunk-partitioned directory layout (block_chunk={start}-{end}/part-0.parquet)
  ├─ Atomic staging directory replacement (retry-safe & idempotent)
  └─ Parquet _SUCCESS marker generation
           │
           ▼
[ Verified Parquet Feature Store on D: ]
```

### Key Architectural Safeguards:
1. **Zero Full Driver Collection:** Neither `collect()` nor `toPandas()` is executed on the feature dataset. Partitions are streamed chunk-by-chunk using `toLocalIterator()` in bounded micro-batches (default 1,000 rows).
2. **Drive C: Isolation:** All Spark local scratch space (`spark.local.dir`), temporary files (`java.io.tmpdir`, `TEMP`, `TMP`), Spark warehouse files, pip caches, and Parquet outputs are configured strictly on Drive `D:`.
3. **Temporal Causality Guarantee:** Window operations strictly exclude current and future transactions (`rowsBetween(Window.unboundedPreceding, -1)`), ensuring zero label or feature leakage.
4. **Windows NativeIO Immunity:** By using native C++ PyArrow for Parquet output writing, the system operates deterministically on Windows without requiring `winutils.exe` or `hadoop.dll`.

