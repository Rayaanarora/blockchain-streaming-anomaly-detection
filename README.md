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

## 📊 Experimental Evaluation & Evidence Status

To maintain scientific integrity, experimental results in this repository are divided into two distinct tiers:

### Tier 1: Reproducible Local Engine Benchmarks (Zenodo Dataset on Windows)
*Evidence Status:* **Directly Reproducible.** Verified locally using Apache Spark 3.5.0 + PyArrow on Drive `D:`, validated with 100% concordance against an independent pure-Python causal reference implementation:

- **100,000 Raw Lines Benchmark (`spark_scale_benchmark_100000.json`):**
  - **Throughput:** **3,709.46 lines/sec** (**674.12 completed transactions/sec**).
  - **Wall-Clock Runtime:** 26.96 s (10.46 s Catalyst transformations, 12.15 s streaming Parquet write).
  - **Memory Bounds:** Peak process RSS strictly bounded at **100.32 MB** (driver memory growth $\le$ 50.8 MB).
  - **Disk Footprint:** 2.20 MB Snappy-compressed Parquet partitioned by `block_chunk=480000-480999`.
  - **Drive Isolation:** Exactly **0 bytes written to Drive C:** (all scratch, warehouse, and outputs on Drive D:).
  - **Correctness:** 60/60 representative sample transactions matched 100% against Python ground truth across all 18 feature columns.
- **50,000 Raw Lines Benchmark (`spark_scale_benchmark_50000.json`):**
  - **Throughput:** **2,523.74 lines/sec** (**411.32 transactions/sec**).
  - **Wall-Clock Runtime:** 19.81 s (7.51 s transformations, 8.30 s streaming write).
  - **Peak RSS:** **100.32 MB**.

### Tier 2: Academic Course Simulation Baseline (Elliptic Dataset, Review 2 Submission)
*Evidence Status:* **Academic Milestone Presentation / In-Memory Prototype.** These metrics reflect the Review 2 academic submission for 21CSC314P Big Data Essentials evaluated on the 49-timestep Kaggle Elliptic transaction graph dataset in Google Colab (`Review2_Blockchain_AML_Colab_Notebook.ipynb`, `Report_Review2_Blockchain_21CSC314P.md`):

- **Model Detection Metrics (Holdout Timesteps 35–49):**
  - **Illicit Class Recall:** **0.739** (vs. baseline GCN 0.600, EvolveGCN 0.710)
  - **AUC-ROC:** **0.919** | **Average Precision (AP):** **0.770**
  - *Note:* Trained on static pre-extracted 165-dimensional Elliptic node features with GATConv + GBDT. Full retraining on the Zenodo feature store is a downstream integration task.
- **Synthetic Streaming & Scaling Simulation:**
  - **Synthetic Streaming Latency:** **1.69 ms** per micro-batch (measured as in-memory inference latency on pre-vectorized feature batches; does not include raw socket ingestion or Kafka network overhead).
  - **Theoretical Throughput:** **399,525 tx/sec** (in-memory batch inference rate without graph rebuild).
  - **Simulated Speedup (Amdahl's Law):** $S_P = 3.12\times$ on 4 workers ($f = 0.92$, $78.1\%$ efficiency).

---

## 📁 Repository Structure
- `src/spark/` - Apache Spark distributed batch preprocessing pipeline and data contracts:
  - `schemas.py`: Formal schemas for raw records, assembled transactions, and 18-column feature tables.
  - `zenodo_batch.py`: Distributed batch transformer with strict causal windows and bounded PyArrow Parquet writer.
- `src/data/` - Dataset boundary handling, Elliptic audit, preprocessing, and validation.
- `src/graph/` - Directed bipartite address-transaction graph builders (Cartesian-free).
- `src/features/` - Chronological transaction and graph feature extractors (deduplicated participation, no change double-counting).
- `scripts/smoke_test_spark.py` - Reproducible Spark batch smoke test validating causality and PyArrow output.
- `scripts/run_spark_scale_benchmark.py` - Automated bounded scale benchmark with streaming dataset validation.
- `tests/` - Comprehensive unit and integration test suite (36 tests).
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
           ▼  (Partition-by-partition toLocalIterator streaming; bounded < 1 MB batch RAM)
[ Native PyArrow Parquet Writer ]
  ├─ Snappy compression
  ├─ Chunk-partitioned directory layout (block_chunk={start}-{end}/part-0.parquet)
  ├─ Safe two-phase replacement with automatic rollback on failure
  └─ Parquet _SUCCESS marker generation
           │
           ▼
[ Verified Parquet Feature Store on D: ]
```

### Architectural Guarantees & Memory Bounds:
1. **Bounded Driver RAM Streaming:** Neither `collect()` nor `toPandas()` is executed on the feature dataset. Partitions are streamed chunk-by-chunk using `toLocalIterator()` in bounded micro-batches (default 1,000 rows, ~1 MB per batch). The operating driver RSS remains bounded around ~100–150 MB.
2. **Drive C: Strict Isolation:** All Spark local scratch space (`spark.local.dir`), temporary files (`java.io.tmpdir`, `TEMP`, `TMP`), Spark warehouse files, pip caches, and Parquet outputs are configured strictly on Drive `D:`.
3. **Temporal Causality & Change Isolation:** Window operations strictly exclude current and future transactions (`rowsBetween(Window.unboundedPreceding, -1)`). An address participating as both input and change output in transaction $T$ is grouped into a single historical participation event with transferred volume $\max(\text{in}, \text{out})$, preventing change outputs from leaking into transaction $T$'s features or double-counting historical volume.
4. **Safe Two-Phase Output Replacement:** Output writes are staged in `.staging_{uuid}`. If destination exists, it is backed up to `.backup_{uuid}` before replacement. Any failure during staging or replacement automatically rolls back and cleans up, guaranteeing that existing valid datasets are never deleted or corrupted.
5. **Memory-Bounded Dataset Validation:** Parquet validation uses `pyarrow.dataset` metadata scanners and micro-batched streams, verifying millions of rows without loading the dataset into driver memory.


