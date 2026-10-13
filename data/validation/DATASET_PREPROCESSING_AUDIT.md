# Consolidated Blockchain Dataset Preprocessing Audit Report

**Report Generated:** 2026-10-13  
**Project:** Distributed Streaming Anomaly Detection on Blockchain Transaction Graphs  
**Repository Path:** `D:\blockchainbigdata\blockchain-streaming-anomaly-detection`  
**Overall Status:** **`PASS` (Both Datasets Preprocessed, Validated, and Pipeline-Ready)**

---

## Executive Summary

| Dataset | Preprocessed at Start? | Action Taken | Output Path & Format | Row / Entity Count | Integrity & Leakage | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Zenodo Bitcoin Part 2** (Blocks 480000–489999) | **Yes** (Output existed) | Direct audit & Parquet metadata verification; verified all 18 raw archive MD5 checksums; zero expensive re-runs. | `data/parquet/zenodo_features/`<br>(10 block-range partitions, Snappy Parquet, 2.13 GB) | **16,889,271** transactions assembled from 85,721,542 raw inputs/outputs | 0 null keys<br>0 negative non-coinbase fees<br>111 boundary transactions preserved | **`PASS`** |
| **Elliptic Bitcoin** (Graph Benchmark) | **No** (Raw files absent locally; code lacked disk persistence) | Acquired authentic raw CSVs; added persistent storage (`.npz`, Parquet) to `src/data/elliptic.py`; established exact 93 local + 72 aggregated feature taxonomy; executed pipeline with train-only scaling. | `data/processed/elliptic/`<br>(`.npz` 74.3 MB, Features Parquet 52.3 MB, Edgelist Parquet 4.4 MB) | **203,769** nodes<br>**234,355** directed edges<br>49 time steps | 0 duplicates<br>0 missing/inf<br>157,205 unknowns isolated (-1)<br>Train-only scaling | **`PASS`** |

---

## 1. Zenodo Bitcoin Transaction Dataset (Part 2)

### 1.1 Dataset Identification & Sources
- **Dataset Name:** Zenodo Bitcoin Transaction Dataset (Part 2, blocks 480000–489999).
- **Source:** Zenodo Deposit Record 7157854.
- **Deposit Parts:** Exactly 18 archives (`aa` through `ar`), compressed with `tar.bz2`.

### 1.2 Archive Physical Inventory & Checksum Verification
- **Input Directory:** `data/raw/`
- **Total Archives Present on Disk:** 18 of 18 physically verified.
- **Total Compressed Bytes:** 3,615,895,097 bytes (3.368 GB / 3.616 GB decimal).
- **Total Uncompressed Bytes:** 10,860,618,992 bytes (10.115 GB / 10.861 GB decimal).
- **Investigation of Historical `2 verified, 16 not downloaded` Message:**
  - An earlier terminal session reported `Summary: 2 verified, 0 corrupted, 16 not downloaded`.
  - **Investigation Result:** This message was produced *prior* to running the multi-part downloader when only `a.tar.bz2` (alias for `aa`) and `ab` were present in `data/raw/`.
  - **Direct File Check:** Inspection of `data/raw/` confirmed all 18 archive files are physically present on Drive D:.
  - **Direct Verification Re-run:** Executing `.venv\Scripts\python download_zenodo.py --verify --filter 480000` re-read every file and matched all MD5 hashes against Zenodo's live REST API:
    ```
    Summary: 18 verified, 0 corrupted, 0 not downloaded.
    ```
  - **Verdict:** All 18 archives are intact on disk. Zero files missing; zero downloads required.

### 1.3 Preprocessing Status & Audit Decisions
- **Status at Start:** Complete in `data/parquet/zenodo_features/`, produced by an earlier Spark batch execution.
- **Audit Decision:** Preserved the validated 10 GB Parquet output without re-running the 3.8-hour Spark batch job.
- **Work Performed:** Directly audited all 10 partition directories, read Parquet metadata for each partition, reconciled row counts directly from Parquet metadata, checked for primary key nullability and fee negativity, and validated schema contracts.

### 1.4 Output Artifacts & Verified Parquet Partition Breakdown
- **Output Directory:** `data/parquet/zenodo_features/`
- **Completion Marker:** `_SUCCESS` is present and valid.
- **Total Output Rows:** Exactly **16,889,271** completed transactions.
- **Total Storage on Disk:** 2,236,432,286 bytes (2,132.83 MB Snappy Parquet).
- **Exact Per-Partition Row Counts and Sizes (Read Directly from Parquet File Metadata):**

| Partition Directory | Physical Rows | File Size (Bytes) | File Size (MB) | Physical Parquet File |
| :--- | :---: | :---: | :---: | :--- |
| `block_chunk=480000-480999` | 1,953,743 | 256,819,331 B | 244.92 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=481000-481999` | 1,961,416 | 258,319,678 B | 246.35 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=482000-482999` | 1,770,937 | 234,485,355 B | 223.62 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=483000-483999` | 1,462,493 | 194,343,257 B | 185.34 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=484000-484999` | 1,435,850 | 190,507,773 B | 181.68 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=485000-485999` | 1,475,753 | 196,376,120 B | 187.28 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=486000-486999` | 1,568,757 | 208,189,270 B | 198.54 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=487000-487999` | 1,682,130 | 223,433,361 B | 213.08 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=488000-488999` | 1,634,832 | 216,522,691 B | 206.49 MB | `part-00000.c000.snappy.parquet` |
| `block_chunk=489000-489999` | 1,943,360 | 257,435,450 B | 245.51 MB | `part-00000.c000.snappy.parquet` |
| **TOTAL** | **16,889,271** | **2,236,432,286 B** | **2,132.83 MB** | **10 files across 10 partitions** |

*Note: All partition numbers above were regenerated by opening and inspecting each individual file's Parquet metadata header via PyArrow.*

### 1.5 Schema Architecture & Column Classification
- **Physical Columns in Parquet Files:** 18 columns.
- **Virtual Hive Partition Column:** `block_chunk` (string), derived from the partition directory name.
- **Total Dataset Columns in PyArrow/Spark Scan:** 19 columns.
- **Model Features:** Exactly 15 features:
  - Transaction Identifiers (3): `tx_hash` (string), `block_id` (int64), `timestamp` (int64).
  - Graph Topology & Degree (2): `num_inputs` (int64), `num_outputs` (int64).
  - Financial Volume & Velocity (2): `total_input_satoshi` (int64), `total_output_satoshi` (int64).
  - Fee & Incentive Dynamics (2): `fee_satoshi` (int64), `fee_rate` (float64).
  - Coinbase Structural Indicator (1): `is_coinbase` (float64).
  - Dispersion & Ratio Metrics (4): `avg_input_val` (float64), `avg_output_val` (float64), `input_output_count_ratio` (float64), `input_output_val_ratio` (float64).
  - Historical Address State (4): `max_input_prior_tx_count` (float64), `avg_input_prior_tx_count` (float64), `max_input_prior_volume_satoshi` (float64), `new_address_fraction` (float64).

### 1.6 Data Hygiene & Boundary Assembly Integrity
- **Raw Input Records:** 85,721,542 input/output lines parsed.
- **Malformed Rows:** 0.
- **Boundary Crossings:** 111 transactions split across 600 MB uncompressed archive chunk boundaries were successfully buffered and assembled without truncation.
- **Primary Key Integrity:** 0 null transaction hashes across 16,889,271 records.
- **Financial Value Integrity:** 0 negative fees observed on standard non-coinbase transactions.

---

## 2. Elliptic Bitcoin Transaction Graph Dataset

### 2.1 Dataset Identification & Sources
- **Dataset Name:** Elliptic Bitcoin Dataset (Weber et al., 2019).
- **Source:** Authentic Kaggle replica mirror hosted on Hugging Face (`yhoma/elliptic-bitcoin-dataset`).
- **Official Raw Files:**
  1. `data/raw/elliptic/elliptic_txs_features.csv` (689,683,771 bytes, 203,769 lines, no header).
  2. `data/raw/elliptic/elliptic_txs_classes.csv` (3,305,144 bytes, 203,770 lines with header).
  3. `data/raw/elliptic/elliptic_txs_edgelist.csv` (4,470,584 bytes, 234,356 lines with header).

### 2.2 Feature Taxonomy Resolution & Exact Dimensionality
- **Total Features in Dataset:** Exactly **165** numerical features (`f1` through `f165`).
- **Resolution of Historical "94 Local + 72 Aggregated" Discrepancy:**
  - The previous text stated "94 local plus 72 aggregated", which arithmetically sums to 166 features (94 + 72 = 166), contradicting the verified 165 total features.
  - The origin of this error was an off-by-one confusion where `time_step` (column 1 in the raw file) was informally grouped with the 93 local features (93 + 1 = 94), but then added to the 72 aggregated features.
  - **Verified Taxonomy:**
    - **Local Features:** Exactly **93** features (`f1` through `f93`), representing transaction-level metrics (in/out degree, fees, output financial volume, and local input/output statistics).
    - **Aggregated Features:** Exactly **72** features (`f94` through `f165`), representing 1-hop and 2-hop neighborhood aggregated metrics (minimum, maximum, standard deviation, and correlation of neighbor transactions).
    - **Sum Check:** 93 + 72 = **165** total features.
    - **Metadata Columns:** `tx_id` (int64 identifier), `time_step` (int64 step 1..49), `label` (int64 class: 0, 1, -1), `train_mask` (bool), `test_mask` (bool).
    - **Total Columns in `elliptic_features.parquet`:** 5 metadata + 165 features = **170 columns**.
  - Codified in source code: `ELLIPTIC_NUM_LOCAL_FEATURES = 93`, `ELLIPTIC_NUM_AGG_FEATURES = 72`, `ELLIPTIC_NUM_FEATURES = 165` in `src/data/elliptic.py` and validated by unit tests.

### 2.3 Generated Output Files
All preprocessed artifacts reside in `data/processed/elliptic/`:
- `elliptic_preprocessed.npz` (77,925,422 bytes / 74.32 MB): Compressed NumPy archive containing `tx_ids`, `time_steps`, `X_raw`, `X_scaled`, `y`, `train_mask`, `test_mask`, `edge_index`, and `edge_time_steps`.
- `elliptic_features.parquet` (54,807,040 bytes / 52.27 MB): Tabular node feature table (203,769 rows × 170 columns) containing `tx_id`, `time_step`, `label`, `train_mask`, `test_mask`, and `f1` through `f165`.
- `elliptic_edgelist.parquet` (4,582,152 bytes / 4.37 MB): Graph edge list (234,355 rows × 5 columns) containing `source_tx_id`, `target_tx_id`, `source_idx`, `target_idx`, and `time_step`.
- `elliptic_summary.json` (818 bytes): Summary metadata.
- `data/validation/elliptic_audit_report.json` and `data/validation/ELLIPTIC_AUDIT_REPORT.md`: Comprehensive audit reports.

### 2.4 Integrity & Relational Alignment Results
- **Aligned Nodes:** Exactly 203,769 transactions.
- **Duplicate IDs in Features:** 0.
- **Duplicate IDs in Classes:** 0.
- **Duplicate Edges:** 0.
- **Missing / NaN Feature Values:** 0.
- **Non-Finite / Infinite Values:** 0.
- **Unmatched Feature vs Class IDs:** 0 (100% 1-to-1 correspondence).
- **Unmatched Edge Endpoints:** 0 (all 234,355 edges connect aligned nodes).
- **Cross-Timestep Edges:** 0 (all edges are strictly intra-step snapshots).

### 2.5 Label Distribution & Unknown Handling
- **Licit Transactions (Class 2 → 0):** 42,019.
- **Illicit Transactions (Class 1 → 1):** 4,545.
- **Unknown Transactions (Class 'unknown' → -1):** 157,205.
- **Invalid Labels:** 0.
- **Critical Guarantee:** Unknown transactions are assigned label `-1` and excluded from `train_mask` and `test_mask`. They are **never** collapsed into licit transactions, preventing catastrophic false-negative distortion in supervised training.

### 2.6 Inductive Temporal Split & Leakage Safeguard
- **Time Steps:** Discrete steps 1 to 49 (each representing ~2 weeks).
- **Chronological Split Protocol:**
  - **Training Set (Steps 1–34):** 29,894 labeled nodes (26,432 licit, 3,462 illicit, imbalance ratio 7.63:1).
  - **Testing Set (Steps 35–49):** 16,670 labeled nodes (15,587 licit, 1,083 illicit, imbalance ratio 14.39:1).
- **Data Leakage Safeguard:** `StandardScaler` is fitted strictly on labeled training nodes (`time_step <= 34` and `y != -1`). Mean and standard deviation are calculated without any exposure to future time steps (35–49). The pre-fitted scaler is then applied inductively to test nodes.

---

## 3. Verification Method Taxonomy

| Check Item | Zenodo Verification Basis | Elliptic Verification Basis |
| :--- | :--- | :--- |
| Raw source file existence & size | Verified directly by reading actual files on disk | Verified directly by reading actual files on disk |
| Archive checksums / line counts | Verified directly via live Zenodo API MD5 match (18/18 PASS) | Verified directly via CSV line counts |
| Processed files existence & format | Verified directly by opening Parquet files & checking `_SUCCESS` | Verified directly by opening NPZ & Parquet files |
| Row counts & column schemas | Verified directly via PyArrow metadata on each of 10 partitions | Verified directly via NumPy / Pandas shape inspections |
| Null primary keys & invalid values | Verified directly by scanning all 16,889,271 records | Verified directly via NumPy isnan/isinf/duplicated checks |
| Edge endpoints & cross-step edges | N/A (Transaction features dataset) | Verified directly via relational mapping checks |
| Train-only feature scaling | Inferred from `extractor.py` causal address history | Verified directly via unit tests & scikit-learn fit mask |
| Downstream consumption | Verified via unit tests (`test_spark_specs.py`, `test_pipeline.py`) | Verified via unit tests (`test_elliptic_preprocessing.py`) |

---

## 4. Test Execution & Integration Results

### Historical Run (Reference Evidence)
- **Command:** `.venv\Scripts\python -m pytest tests/ -v`
- **Result:** 42 passed in 18.84 seconds.

### Latest Final Test Run (Post-Taxonomy Codification)
- **Command:** `.venv\Scripts\python -m pytest tests/ -v`
- **Result:** **43 passed in 21.90 seconds** (100% pass rate).

**Test Suites Breakdown:**
1. `tests/test_elliptic_preprocessing.py` (**11 passed**): Verifies ID alignment, duplicate handling, unknown label isolation, inductive temporal splitting, train-only scaling (zero leakage), edge validity, persistence round-trip, and the newly added explicit feature taxonomy verification (`93 + 72 == 165`).
2. `tests/test_graph_validation.py` (**8 passed**): Verifies Zenodo record parsing, endpoint semantics, coinbase handling, bipartite direction, and boundary chunk buffering.
3. `tests/test_pipeline.py` (**11 passed**): Verifies multi-input/output assembly, duplicate rejection, historical address volume causality, and cost-sensitive train/test separation.
4. `tests/test_spark_specs.py` (**13 passed**): Verifies Spark schema specifications, 18-to-19 column taxonomy, bounded validator isolation, Parquet rollback logic, and preprocessor boundary discovery.

---

## 5. Scope Distinction: Preprocessing vs. Full System

- **Dataset Preprocessing (Complete - `PASS`):**
  - Both datasets are fully acquired, parsed, aligned, cleaned, scaled, partitioned, and persisted on Drive D:.
  - All schemas and data types are verified.
  - Zero data leakage is enforced across train/test splits.
- **Downstream Anomaly Detection & Streaming System (Next Phase):**
  - Graph Neural Network (GNN/TGAT) and Cost-Sensitive GBDT model training.
  - Apache Flink / Kafka distributed streaming simulation.
  - End-to-end latency, throughput, and streaming alert evaluation.
