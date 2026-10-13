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

> [!WARNING]
> **Scope & Caveat:** These benchmarks evaluate bounded local batch preprocessing of static Zenodo text slices on Windows using a local SparkSession. They **do NOT** represent live streaming ingestion, production distributed cluster throughput, or real-time Kafka streaming latency.

#### Benchmark Versions & Evolution:
1. **Initial Baseline (`spark_scale_benchmark_*.json`):**
   - Evaluated initial Spark implementation with full Parquet read-back validation.
   - Point-in-time discrete RSS sampling at transformation and writer boundaries.
2. **Streaming Validation Revision (`*_v2.json`):**
   - Corrected address-history volume methodology ($\max(\text{in}, \text{out})$ per transaction event).
   - Replaced full Parquet table loads with bounded `pyarrow.dataset` micro-batched validation.
   - Identified measurement artifact: Python driver ending RSS (170.56 MB) exceeded discrete peak sample (100.22 MB) due to post-write verification allocations.
3. **Continuous High-Frequency Monitoring Revision (`*_v3.json`):**
   - Integrated continuous 50ms background daemon polling tracker capturing true peak memory across all execution phases.
   - Separated Python Driver Spark-phase peak from overall peak, JVM child process RSS, and total process tree RSS.
   - Replaced full-dataset key aggregation and full-dataset ground truth parsing with genuinely bounded $O(\text{sample\_size})$ streaming key selection and two-pass address history extraction.

#### Measured Results (Latest `_v3` Reports with Bounded Validation & Continuous Polling):
- **100,000 Raw Lines Benchmark (`spark_scale_benchmark_100000_v3.json`):**
  - **Throughput:** **4,055.64 lines/sec** (**737.03 completed transactions/sec**).
  - **Wall-Clock Runtime:** 24.66 s (9.35 s Spark Catalyst transforms, 10.93 s streaming Parquet write).
  - **Python Driver Memory:** Start: **49.88 MB** | Spark Write Peak: **101.80 MB** | Overall Peak: **125.64 MB** | End: **125.64 MB** | Driver Peak Growth: **75.76 MB**.
  - **JVM Child Subprocess Peak:** **2,098.39 MB** (Spark Java executor/driver heap).
  - **Total Process Tree Peak:** **2,155.93 MB** (Combined Python + Java memory).
  - **Disk Footprint:** 2.20 MB Snappy-compressed Parquet partitioned by `block_chunk=480000-480999`.
  - **Drive Isolation:** Exactly **0 bytes written to Drive C:** (9.66 GB free before and after).
  - **Correctness:** 60/60 representative sample transactions matched 100% against Python ground truth across all 18 feature columns.
- **50,000 Raw Lines Benchmark (`spark_scale_benchmark_50000_v3.json`):**
  - **Throughput:** **2,525.68 lines/sec** (**411.63 completed transactions/sec**).
  - **Wall-Clock Runtime:** 19.80 s (8.04 s transforms, 8.44 s streaming Parquet write).
  - **Python Driver Memory:** Start: **49.34 MB** | Spark Write Peak: **95.08 MB** | Overall Peak: **117.07 MB** | End: **117.07 MB** | Driver Peak Growth: **67.73 MB**.
  - **JVM Child Subprocess Peak:** **1,946.18 MB**.
  - **Total Process Tree Peak:** **2,036.46 MB**.
  - **Drive Isolation:** 0 bytes delta on Drive C: (9.66 GB free).
- **Full ~10.86 GB Zenodo Dataset Preprocessing Run (`zenodo_preprocessing_report.json`):**
  - **Dataset Scope:** Full Zenodo Record 7157854 (Part 2 of 4), Block Range 480000–489999 (all 18 archives `aa` through `ar`).
  - **Raw Input:** **10,860,618,992 bytes** (~10.115 GiB uncompressed text, 85,721,542 records).
  - **Completed Transactions Assembled:** **16,889,271** (0 dropped boundary transactions).
  - **Throughput:** **6,132.89 raw lines/sec** across 86 distributed Spark splits.
  - **Wall-Clock Runtime:** 13,977.35 s (~3.88 hours).
  - **Output Parquet:** **2,132.83 MB** (~2.13 GB Snappy Parquet, 10 Hive partitions `block_chunk=480000-480999` through `489000-489999`).
  - **Python Driver Memory:** Start: **90.37 MB** | Spark Write Peak: **90.23 MB** | Overall Peak: **940.46 MB** (bounded streaming via 1,000-row microbatches).
  - **JVM Child Peak:** **4,645.73 MB** (~4.65 GB) | **Total Process Tree Peak:** **5,277.75 MB** (~5.28 GB).
  - **Validation:** 100% row match (16,889,271 rows), 0 null primary keys, 0 negative fees, verified with `status: PASS`.

---

### Tier 2: Academic Course Simulation Baseline (Elliptic Dataset, Review 2 Submission)
*Evidence Status:* **Academic Milestone Presentation / In-Memory Prototype.** These metrics reflect the Review 2 academic submission for 21CSC314P Big Data Essentials evaluated on the 49-timestep Kaggle Elliptic transaction graph dataset in Google Colab (`Review2_Blockchain_AML_Colab_Notebook.ipynb`, `Report_Review2_Blockchain_21CSC314P.md`):

- **Model Detection Metrics (Holdout Timesteps 35–49):**
  - **Illicit Class Recall:** **0.739** (vs. baseline GCN 0.600, EvolveGCN 0.710)
  - **AUC-ROC:** **0.919** | **Average Precision (AP):** **0.770**
  - *Note:* Evaluated on pre-extracted 165-dimensional Elliptic node features with GATConv + GBDT. Retraining on the Zenodo feature store is a downstream integration task.
- **Synthetic Streaming & Scaling Simulation:**
  - **Synthetic Streaming Latency:** **1.69 ms** per micro-batch (measured as in-memory inference latency on pre-vectorized feature batches; does not include raw socket ingestion or Kafka network overhead).
  - **Theoretical Throughput:** **399,525 tx/sec** (in-memory batch inference rate without graph rebuild).
  - **Simulated Speedup (Amdahl's Law):** $S_P = 3.12\times$ on 4 workers ($f = 0.92$, $78.1\%$ efficiency).

---

## 📐 Data Contracts & Schema Specification

The pipeline enforces a strict four-layer column taxonomy across preprocessing, storage, and modeling:

| Schema Layer | Column Count | Description & Columns Included |
| :--- | :---: | :--- |
| **Physical Parquet Columns** | **18** | Columns physically encoded in `part-0.parquet` files: 3 identifiers (`block_number`, `tx_index`, `tx_hash`) + 15 numerical features. |
| **Hive Partition Directory Key** | **1** | Directory hierarchy path (`block_chunk={start}-{end}`). Enables filesystem pruning without bloating individual Parquet files. |
| **Partitioned Dataset Columns** | **19** | Total columns exposed when reading via `pyarrow.dataset.dataset(..., partitioning="hive")` or `spark.read.parquet()` (18 physical + 1 virtual `block_chunk`). |
| **Downstream Model Features** | **15** | Feature vector consumed by anomaly detection algorithms: `num_inputs`, `num_outputs`, `total_input_satoshi`, `total_output_satoshi`, `fee_satoshi`, `fee_rate`, `is_coinbase`, `avg_input_val`, `avg_output_val`, `input_output_count_ratio`, `input_output_val_ratio`, `max_input_prior_tx_count`, `avg_input_prior_tx_count`, `max_input_prior_volume_satoshi`, `new_address_fraction`. |

---

## 📁 Repository Structure
- `src/spark/` - Apache Spark distributed batch preprocessing pipeline and data contracts:
  - `schemas.py`: Formal schemas for raw records, assembled transactions, and 18-column feature tables.
  - `zenodo_batch.py`: Distributed batch transformer with strict causal windows and bounded PyArrow Parquet writer.
- `src/data/` - Dataset boundary handling, Elliptic audit, preprocessing, and validation.
- `src/graph/` - Directed bipartite address-transaction graph builders (Cartesian-free).
- `src/features/` - Chronological transaction and graph feature extractors (deduplicated participation, no change double-counting).
- `scripts/smoke_test_spark.py` - Reproducible Spark batch smoke test validating causality and PyArrow output.
- `scripts/run_spark_scale_benchmark.py` - Automated bounded scale benchmark with continuous memory tracking and dataset validation.
- `scripts/run_zenodo_preprocessing.py` - Dedicated multi-chunk batch preprocessor with cross-boundary reconstruction, preflight test, and bounded validation.
- `tests/` - Comprehensive unit and integration test suite (41 tests).
- `Review2_Blockchain_AML_Colab_Notebook.ipynb` - Full interactive Jupyter/Colab notebook with all 14 executable sections
- `Distributed_Blockchain_Anomaly_Review2.pptx` - Official 15-slide presentation for Review 2
- `Report_Review2_Blockchain_21CSC314P.docx` - Complete academic report in Times New Roman 12pt format
- `Report_Review2_Blockchain_21CSC314P.md` - Complete markdown report
- `train_sota_dl.py` - PyTorch Geometric implementation script
- `*.png` - All 7 high-resolution diagnostic figure panels

---

## 🚀 Zenodo Batch Preprocessing Results (Multi-Chunk Real Dataset)

The dedicated batch preprocessor (`scripts/run_zenodo_preprocessing.py`) executed across all 4 contiguous real Zenodo chunks on Drive `D:`, verifying boundary continuity and generating the partitioned feature store:

| Metric | Measured Value | Methodology & Caveats |
|---|---|---|
| **Deposit Context** | Zenodo Record 7157854 (Block 480000–489999) | Full deposit range comprises 18 chunk parts (`aa`–`ar`, ~10.6 GB uncompressed). |
| **Chunks Processed** | 4 contiguous chunks (`aa`, `ab`, `ac`, `ad`) | Processed parts present on disk; parts `ae`–`ar` (2.58 GB compressed) remain on Zenodo. |
| **Raw Input Data** | **2,535,662,744 bytes (2.536 GB)** | 4 uncompressed text slices in chronological block order (blocks 480000 to ~482080). |
| **Raw Lines Ingested** | **20,000,000 lines** | Exactly 20M lines parsed; 0 malformed lines. |
| **Completed Transactions** | **4,045,123 transactions** | Grouped by `(block_number, tx_index, tx_hash)` across chunks; 1 incomplete end-boundary transaction dropped. |
| **Output Parquet Size** | **507.69 MB** (532,347,696 bytes) | Snappy-compressed, written via bounded PyArrow streaming writer. |
| **Hive Partitions** | **3 partitions** | `block_chunk=480000-480999`, `481000-481999`, `482000-482999`. |
| **Preflight Boundary Check** | **PASSED** | Verified cross-chunk transaction `(480521, 839)` assembled with 2 inputs from `aa` and 2 outputs from `ab`. |
| **Bounded PyArrow Validation** | **PASSED** | Verified all 4,045,123 rows, 18 physical columns, zero null primary keys, zero negative fee violations. |
| **Spark Processing Time** | **1,384.58 s (~23.08 min)** | Measured via `ZenodoSparkBatchPipeline` end-to-end execution. |
| **Peak Process Tree Memory** | **4,630.75 MB (~4.63 GB)** | Combined footprint of Python driver (324.68 MB peak) and JVM Catalyst worker (4,554.38 MB peak). |
| **Drive C: Containment** | **0.23 GB delta (9.42 GB free)** | Driver and JVM temp directories strictly pinned to Drive `D:` scratch space. |
| **Drive D: Free Space** | **130.64 GB free** | All output Parquet, unpacked inputs, and Spark scratch directories reside on Drive `D:`. |
| **Reproducible Report** | `data/validation/zenodo_preprocessing_report.json` | Fully verifiable JSON report generated at run completion. |

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

### Architectural Guarantees & Memory Realities:
1. **Bounded PyArrow Buffer vs. Process Footprint:**
   - **What is strictly bounded:** The PyArrow writer streams records partition-by-partition via `toLocalIterator()` in discrete micro-batches (`batch_size=1000`, ~1 MB RAM buffer), preventing full dataset materialization in Python.
   - **Process RSS Reality:** Total process RSS is **not** strictly capped at 100 MB. The Python driver operates between ~50 MB and ~170–240 MB due to Py4J socket buffers, object allocations, and verification data structures. Simultaneously, the JVM child process utilizes ~1.9–2.0 GB of memory for Spark's Catalyst engine, internal shuffle buffers, and Java heap. Total process tree footprint peaks around ~2.1–2.2 GB.
2. **Drive C: Strict Storage Isolation:** All Spark local scratch space (`spark.local.dir`), temporary files (`java.io.tmpdir`, `TEMP`, `TMP`), Spark warehouse files, pip caches, and Parquet outputs are configured strictly on Drive `D:`. Exactly 0 bytes are written to Drive `C:`.
3. **Temporal Causality & Change Isolation:** Window operations strictly exclude current and future transactions (`rowsBetween(Window.unboundedPreceding, -1)`). An address participating as both input and change output in transaction $T$ is grouped into a single historical participation event with transferred volume $\max(\text{in}, \text{out})$, preventing change outputs from leaking into transaction $T$'s features or double-counting historical volume.
4. **Safe Two-Phase Output Replacement:** Output writes are staged in `.staging_{uuid}`. If destination exists, it is backed up to `.backup_{uuid}` before replacement. Any failure during staging or replacement automatically rolls back and cleans up. If automated rollback also encounters a filesystem failure, the backup directory is preserved safely on disk, raising a descriptive `RuntimeError`.
5. **Memory-Bounded Dataset Validation:** Parquet validation uses `pyarrow.dataset` metadata scanners and micro-batched streams, verifying millions of rows without loading the dataset into driver memory.


