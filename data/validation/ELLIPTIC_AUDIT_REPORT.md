# Elliptic Benchmark Preprocessing Audit Report

## 1. Raw Dataset Ingestion Summary
- **Raw Feature Rows:** 203,769
- **Raw Class Rows:** 203,769
- **Raw Edge Rows:** 234,355
- **Feature Dimension:** 167 columns detected (165 features + tx_id + time_step)

## 2. Data Hygiene & Integrity Checks
- **Duplicate Tx IDs in Features:** 0
- **Duplicate Tx IDs in Classes:** 0
- **Duplicate Edges:** 0
- **Missing / NaN Feature Values:** 0
- **Non-Finite / Inf Feature Values:** 0
- **Unmatched Feature IDs:** 0
- **Unmatched Class IDs:** 0
- **Unmatched Edge Endpoints:** 0
- **Cross-Timestep Edges:** 0

## 3. Ground Truth Label Distribution
- **Licit Transactions (Class 2 -> 0):** 42,019
- **Illicit Transactions (Class 1 -> 1):** 4,545
- **Unknown Transactions (-1):** 157,205 (Preserved separately; **NEVER** collapsed into licit)
- **Invalid Labels Found:** 0

## 4. Chronological Inductive Partitioning
- **Time Step Range:** Steps 1 through 49
- **Training Set (Steps 1-34):**
  - Total Labeled Nodes: 29,894
  - Licit: 26,432 | Illicit: 3,462
  - Class Imbalance Ratio: 7.63:1
- **Testing Set (Steps 35-49):**
  - Total Labeled Nodes: 16,670
  - Licit: 15,587 | Illicit: 1,083
  - Class Imbalance Ratio: 14.39:1

## 5. Methodological & Leakage Prevention Guarantees
- **Train-Only Scaling:** `StandardScaler` fitted strictly on training partition (`time_step <= 34` and `labeled`), preventing future-to-past feature leakage.
- **Explicit Relational Alignment:** Rows aligned strictly on `tx_id` join rather than positional index.
- **Unknown Isolation:** 157k background transactions retained for graph topology but excluded from supervised loss.

## 6. Exclusions and Anomalies Log
```json
[]
```
