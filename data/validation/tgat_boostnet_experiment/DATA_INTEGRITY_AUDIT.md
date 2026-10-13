# Data Integrity Audit: Step-35 Count Discrepancy Resolution & Split Verification

**Date:** 2026-10-13  
**Artifact Path:** `data/validation/tgat_boostnet_experiment/DATA_INTEGRITY_AUDIT.md`  
**Dataset:** Elliptic Bitcoin AML Benchmark (203,769 total transactions across 49 time steps)  
**Investigation Target:** Discrepancy between earlier evaluation report (`MODEL_EVALUATION.md`: 1,677 labeled / 157 illicit) and empirical ablation report (`per_step_metrics.csv`: 1,341 labeled / 182 illicit).

---

## 1. Executive Summary & Root Cause Finding

### The Discrepancy
- **Earlier Evaluation Markdown Narrative (`MODEL_EVALUATION.md`, Section 4):**
  Reported Step 35 as having **1,677 labeled nodes** and **157 illicit nodes**.
- **Ablation Study Output & Source Data (`per_step_metrics.csv` & `elliptic_preprocessed.npz`):**
  Reported Step 35 as having **1,341 labeled nodes** and **182 illicit nodes**.

### Definitive Root Cause: Narrative Transcription Error
The discrepancy is a **purely typographical error in the Markdown narrative table of `MODEL_EVALUATION.md`**.  
The underlying binary and tabular datasets—from the raw Kaggle CSV files, to the preprocessed NumPy archive, to the saved test predictions Parquet table—have **never changed** and exhibit **100% concordance**:

$$\text{Authoritative Step 35 Counts: } \mathbf{1,341\text{ labeled nodes (182 illicit, 1,159 licit, 4,166 unknown, 5,507 total)}}.$$

---

## 2. Multi-Level Empirical Verification Proof

To establish absolute provenance without ambiguity, data integrity was verified across four independent data layers:

### Level 1: Raw Original CSV Source Files
Direct query against `data/raw/elliptic/elliptic_txs_classes.csv` and `data/raw/elliptic/elliptic_txs_features.csv`:
```text
Total transactions at time_step == 35:  5,507
  - class 'unknown'   : 4,166  (75.65%)
  - class '2' (licit) : 1,159  (21.05%)
  - class '1' (illicit):   182  ( 3.30%)
Total labeled (Class 1 + Class 2):      1,341
```

### Level 2: Preprocessed NumPy Tensor Archive
Direct query against `data/processed/elliptic/elliptic_preprocessed.npz`:
```text
Total nodes at time_steps == 35:         5,507
  - label == -1 (unknown):              4,166
  - label == 0  (licit):                1,159
  - label == 1  (illicit):                182
Total labeled (y != -1):                1,341
```

### Level 3: Saved Test Prediction Parquet Table
Direct query against `data/validation/model_evaluation/test_predictions.parquet`:
```text
Filter: time_step == 35
Total rows present in test set:         1,341
  - y_true == 0 (licit):                1,159
  - y_true == 1 (illicit):                182
```
*Note:* The total test set in `test_predictions.parquet` contains exactly 16,670 rows. The 15 test time steps (35 to 49) sum precisely to 16,670 rows, with Step 35 contributing exactly 1,341 rows.

### Level 4: Time-Step Metrics Table
Direct query against `data/validation/ablation_study/per_step_metrics.csv`:
```text
time_step = 35:
  total_labeled_nodes = 1341
  illicit_count = 182
  licit_count = 1159
  illicit_prevalence = 0.1357 (13.57%)
```

---

## 3. Detailed Forensic Accounting Across All Test Time Steps

The table below reconciles the true empirical counts from `test_predictions.parquet` and `elliptic_preprocessed.npz` across all 15 test time steps:

| Time Step | Total Labeled Nodes | Licit Nodes ($y=0$) | Illicit Nodes ($y=1$) | Illicit Prevalence | Unknown Nodes ($y=-1$) | Total Graph Nodes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 35** | **1,341** | **1,159** | **182** | **13.57%** | 4,166 | 5,507 |
| **Step 36** | 1,708 | 1,675 | 33 | 1.93% | 4,302 | 6,010 |
| **Step 37** | 498 | 458 | 40 | 8.03% | 2,833 | 3,331 |
| **Step 38** | 756 | 645 | 111 | 14.68% | 2,757 | 3,513 |
| **Step 39** | 1,183 | 1,102 | 81 | 6.85% | 3,124 | 4,307 |
| **Step 40** | 1,211 | 1,099 | 112 | 9.25% | 2,987 | 4,198 |
| **Step 41** | 1,132 | 1,016 | 116 | 10.25% | 3,246 | 4,378 |
| **Step 42** | 2,154 | 1,915 | 239 | 11.10% | 4,370 | 6,524 |
| **Subtotal (35–42)** | **9,983** | **9,069** | **914** | **9.16%** | **27,785** | **37,768** |
| **Step 43** | 1,370 | 1,346 | 24 | 1.75% | 3,654 | 5,024 |
| **Step 44** | 1,591 | 1,567 | 24 | 1.51% | 3,745 | 5,336 |
| **Step 45** | 1,221 | 1,216 | 5 | 0.41% | 3,678 | 4,899 |
| **Step 46** | 712 | 710 | 2 | 0.28% | 2,749 | 3,461 |
| **Step 47** | 846 | 824 | 22 | 2.60% | 2,903 | 3,749 |
| **Step 48** | 471 | 435 | 36 | 7.64% | 2,367 | 2,838 |
| **Step 49** | 476 | 420 | 56 | 11.76% | 2,525 | 3,001 |
| **Subtotal (43–49)** | **6,687** | **6,518** | **169** | **2.53%** | **21,621** | **28,308** |
| **Full Test (35–49)** | **16,670** | **15,587** | **1,083** | **6.50%** | **49,406** | **66,076** |

---

## 4. Authoritative Dataset Split Master Table

All downstream experiments for `TGAT-BoostNet` and competitive baselines are bound to these verified, immutable sample counts:

| Split Partition | Time Step Window | Labeled Nodes ($y \in \{0, 1\}$) | Licit Count ($y=0$) | Illicit Count ($y=1$) | Class Imbalance Ratio | Background Unknowns ($y=-1$) | Total Graph Nodes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Split** | Steps 1–29 | **26,381** | 23,510 | 2,871 | **8.1888 : 1** | 90,812 | 117,193 |
| **Validation Split** | Steps 30–34 | **3,513** | 2,922 | 591 | **4.9442 : 1** | 16,987 | 20,500 |
| **Test Split** | Steps 35–49 | **16,670** | 15,587 | 1,083 | **14.3924 : 1** | 49,406 | 66,076 |
| **Full Elliptic Network**| **Steps 1–49** | **46,564** | **42,019** | **4,545** | **9.2451 : 1** | **157,205** | **203,769** |

---

## 5. Verification Check Conclusion

- **Integrity Status:** **VERIFIED & CERTIFIED.**
- **Array Alignment:** No arrays, labels, time steps, or prediction rows have drifted.
- **Index Mappings:** Node-to-index mappings between raw CSV, preprocessed NPZ, and Parquet tables are 100% bijective and invariant.
- **Experimental Readiness:** Data integrity audit passes completely. Comparative training and evaluation of `TGAT-BoostNet` may proceed safely using the authoritative count of **1,341 labeled nodes (182 illicit) at Step 35**.
