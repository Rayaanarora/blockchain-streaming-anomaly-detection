# Model cards

Two datasets, two different score meanings. The Elliptic model served by default is **SnapShift-Boost 2.0.0**. Bundle 1.0.1 is its baseline and stays available via `BSAD_ELLIPTIC_BUNDLE`. The scoring service routes by `source_dataset` and never feeds one dataset's events to the other model (`src/scoring/service.py`).

---

## 1a. `elliptic-supervised` 2.0.0: SnapShift-Boost (served)

| | |
|---|---|
| Location | `artifacts/models/elliptic/2.0.0/` (`manifest.json` with `combine: two_stage`, `scaler.joblib`, `snapshift_stage1.joblib`, `snapshift_stage2.joblib`; SHA-256 per file, checked on load) |
| Design | Two-stage HistGradientBoosting (same hyper-parameters as 1.0.1). Stage 1 = the 1.0.1 feature set (165 + degree + neighbour means of f1–f93). Stage 2 = stage-1 features + 8 score-propagation features: own stage-1 probability; mean and max over in-, out- and undirected neighbours; undirected 2-hop mean. Training rows get out-of-fold stage-1 scores (4 time-step groups); see `docs/SNAPSHIFT_BOOST.md` |
| Inputs | Same as 1.0.1: one complete time-step snapshot (raw `f1…f165` + edges). `predict` raises without `edges` |
| Output | `score` = illicit probability (not calibrated); `flagged` = score ≥ **0.90** (validation-F1-optimal, steps 30–34) |
| Training data | Labelled steps 1–29 (Spark output) |
| Selection | Component choice on rolling folds inside steps 1–34 with a pre-registered rule (DECISIONS D-23); test steps never seen during development |
| Test performance (steps 35–49) | precision 0.8883, recall 0.7267, **illicit F1 0.7994**, ROC-AUC 0.9315, AP 0.8004; TN 15,488 / FP 99 / FN 296 / TP 787 |
| vs 1.0.1 | +0.0026 F1; 95 % paired-bootstrap interval [−0.005, +0.011]: **not significant** |
| Known weaknesses | Recall after step 43 is 0.024 (unchanged collapse). Threshold set on a near-saturated validation set. Deterministic, so no seed variance |
| Order invariance / streaming | Bit-identical scores under node reordering and between per-snapshot and full-graph scoring (`bundle_checks.json`, `tests/test_snapshift.py`) |
| Inference cost | ≈63 ms per ~5,000-node snapshot (13 µs per transaction) + ≈23 ms graph-context construction (`data/validation/scoring_benchmark.json`) |
| Intended use | Research benchmark on Elliptic only (same restrictions as 1.0.1) |

---

## 1b. `elliptic-supervised` 1.0.1 (baseline)

| | |
|---|---|
| Location | `artifacts/models/elliptic/1.0.1/` (`manifest.json`, `scaler.joblib`, `hgb_165_nbr.joblib`; SHA-256 per file, checked on load) |
| Config | `hgb_165_nbr`: scikit-learn `HistGradientBoostingClassifier` (learning_rate 0.1, max_leaf_nodes 31, l2 1.0, 200 iterations, early stopping off, class weight illicit = 8.19) |
| Inputs | One complete Elliptic **time-step snapshot**: `tx_id` + raw (unscaled) `f1…f165` for every transaction in the step, and the step's edges (`source_tx_id`, `target_tx_id`). The bundle standardises with the train-only scaler, then adds 95 graph features: in-degree, out-degree, mean of the standardised local features f1–f93 over undirected neighbours |
| Output | `score` = `illicit_probability` (probability-like score of the Elliptic illicit class, not calibrated); `flagged` = score ≥ **0.82** (validation-F1-optimal threshold, steps 30–34); `graph_context` ∈ {complete, partial} |
| Training data | Elliptic labelled steps 1–29 (26,381 tx, 2,871 illicit), prepared by Spark (`data/processed/elliptic_spark`) |
| Selection | Pre-registered validation rule over 12 configurations × 5 seeds (DECISIONS D-11, D-19) |
| Test performance (steps 35–49, 16,670 labelled) | precision 0.8831, recall 0.7258, **illicit F1 0.7968**, ROC-AUC 0.9336, AP 0.7977; TN 15,483 / FP 104 / FN 297 / TP 786 |
| Known weaknesses | Recall 0.85 on steps 35–42 but **0.03 on steps 43–49** (illicit behaviour changes; VERIFIED_RESULTS §1.4). Threshold is fragile (≈±0.03 F1, §1.3). Deterministic model, so seed std = 0 does not measure uncertainty. |
| Graph semantics | Elliptic edges are intra-step, so a transaction's correct context is its entire step. `predict` raises if `edges` is omitted; edges pointing outside the snapshot are dropped and the affected nodes are reported as `graph_context="partial"`. The streaming consumer buffers events and scores only complete snapshots, otherwise emits `rejected_incomplete_graph_context`. Scores are node-order invariant (D-18). |
| Not temporal | Uses one snapshot at a time; no temporal state. |
| Inference cost | ≈33 ms per ~5,000-node snapshot (≈7 µs per transaction) + ≈24 ms graph-context construction |
| Intended use | Research benchmark on the Elliptic dataset. Not valid for other chains, other feature definitions or Zenodo transactions. Elliptic's features are anonymised; the model cannot score arbitrary Bitcoin transactions. |

Previous version: **1.0.0** (same config, pre-D-18 features, threshold 0.61, test F1 0.7683). Kept on disk for the record; the service loads the highest version unless `BSAD_ELLIPTIC_BUNDLE` pins one.

---

## 2. `zenodo-isolation-forest` 1.0.0

| | |
|---|---|
| Location | `artifacts/models/zenodo/1.0.0/` (`manifest.json`, `detector.joblib`) |
| Model | scikit-learn `IsolationForest` (200 trees, max_samples 256, random_state 42) + robust-z reason codes |
| Inputs | One reconstructed Bitcoin transaction with the attributes stored in `data/parquet/zenodo_features` (counts, totals, fee, address-history aggregates). 13 log-scaled features, `zenodo-features-v1` |
| Output | `score` = `isolation_forest_anomaly_score` (higher = more unusual relative to the fit window); `score_percentile` within the fit sample; `flagged` = score ≥ 0.6707 (99.9th percentile of fit scores, ≈0.1 % flag rate by construction); `reason_codes` (the three features with the largest robust z = \|x − median\| / (1.4826·MAD) on the fit sample, with direction) |
| Fit data | Deterministic 2 % hash sample (132,402 tx) of blocks 481000–484999, non-coinbase, data-quality OK. Blocks 480000–480999 excluded (address-history cold start); 485000–489999 never used for fitting |
| Evaluation | **No labels exist.** Stability (PSI < 0.1 per chunk), robustness to seeds and sample size, synthetic fault injection, inspection of top-ranked transactions: `data/validation/zenodo_anomaly/ZENODO_ANOMALY_REPORT.md` |
| What a flag means | Statistically unusual on-chain shape/value/fee/address-history behaviour. **Not** evidence of illicit activity. Use as a ranked review queue; top-set membership changes between refits (Jaccard 0.22–0.63) |
| Blind spots | Subtle fee manipulation (synthetic AUROC 0.49). No addresses or transaction→transaction edges in the stored data, so no graph-structure features beyond address-history aggregates. Address-history features for new blocks would need streaming address state, which is not implemented |
| Inference cost | ≈12 ms single event through the full service; ≈17–20k events/s in batches of 1,000–5,000 |
