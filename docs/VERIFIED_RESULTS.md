# Verified results

Every number below was produced by a command in this repository on the development workstation (Windows 11, 16 logical CPUs, 15.7 GB RAM, Python 3.12.10, `.venv`) and can be traced to the file named next to it. Dates: 2026-10-13/14.

## 1. Elliptic supervised benchmark

**Protocol.** Labelled transactions only for loss and metrics. Train = steps 1–29 (26,381 labelled, 2,871 illicit), validation = steps 30–34 (3,513; 591 illicit), test = steps 35–49 (16,670; 1,083 illicit). 157,205 unknown-label nodes stay in the graph for neighbour features / message passing and never enter a loss, threshold search or metric. Hyper-parameter grids use validation AP (seed 42); thresholds maximise validation F1 on a 0.01 grid; the final configuration is chosen by a rule fixed in code before the run (highest mean validation F1 over seeds 42/123/456/789/2024; within 0.005 → highest mean validation AP → fewer components). Test labels are read once per fitted model after its threshold is fixed. Data source: the Spark output `data/processed/elliptic_spark/`.

Source: `data/validation/final_model_selection/summary.csv`, `seed_results.csv`, `selection.json`, `predictions_all_seeds.parquet` (val + test probabilities for every seed and config).

### 1.1 Five-seed results (mean over seeds; test = steps 35–49)

| Config | What it is | Val F1 | Test precision | Test recall | **Test illicit F1** (± std) | ROC-AUC | AP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **hgb_165_nbr** ✓ selected (baseline for §1.6) | HistGBDT, 165 features + 95 non-learned neighbour aggregates | **0.9743** | 0.8831 | 0.7258 | **0.7968 ± 0.0000**¹ | 0.9336 | 0.7977 | 104 | 297 |
| hgb_165_tuned | HistGBDT, 165 features, grid-tuned, early stopping off | 0.9735 | 0.8870 | 0.7175 | 0.7933 ± 0.0000¹ | 0.9376 | 0.7990 | 99 | 306 |
| ens_hgb_hgbgat | mean of hgb_165_tuned and hgb_165_gat_cf | 0.9721 | 0.8571 | 0.7204 | 0.7817 ± 0.0224 | 0.9399 | 0.7993 | 135 | 303 |
| rf_165 | Random forest | 0.9695 | 0.8454 | 0.7158 | 0.7746 ± 0.0169 | 0.9055 | 0.7744 | 144 | 308 |
| xgb_165_tuned | XGBoost, grid-tuned | 0.9669 | 0.7980 | 0.7346 | 0.7643 ± 0.0183 | 0.9226 | 0.7917 | 205 | 287 |
| hgb_165_gat_cf | HistGBDT, 165 + 32 **cross-fitted** GAT embeddings | 0.9655 | 0.8212 | 0.7184 | 0.7647 ± 0.0277 | 0.9390 | 0.7953 | 177 | 305 |
| hgb_165_legacy | HistGBDT as in the earlier ablation | 0.9650 | 0.8017 | 0.7221 | 0.7579 ± 0.0292 | 0.9404 | 0.7941 | 202 | 301 |
| hgb_165_gat_insample | as hgb_165_gat_cf but in-sample embeddings (leakage control) | 0.9486 | 0.7615 | 0.6493 | 0.6965 ± 0.0317 | 0.9048 | 0.7336 | 238 | 380 |
| hgb_lf93_tuned | HistGBDT, local features f1–f93 only | 0.9465 | 0.6790 | 0.6971 | 0.6879 ± 0.0000¹ | 0.9195 | 0.7640 | 357 | 328 |
| xgb_165_legacy | XGBoost as in the earlier ablation | 0.9298 | 0.6304 | 0.7433 | 0.6822 ± 0.0000¹ | 0.9276 | 0.7890 | 472 | 278 |
| xgb_165_gat_cf | XGBoost, 165 + 32 cross-fitted GAT embeddings | 0.9099 | 0.5751 | 0.7476 | 0.6438 ± 0.0710 | 0.9368 | 0.7976 | 651 | 273 |
| gat_alone | GraphAttentionEncoder classifier (static GAT) | 0.8293 | 0.4900 | 0.4944 | 0.4915 ± 0.0195 | 0.8678 | 0.4679 | 562 | 548 |

¹ **Zero standard deviation means the model is deterministic, not stable.** HistGBDT with early stopping off and XGBoost without row/column subsampling do not use the seed, so their five "seeds" are five identical runs. Their uncertainty is not measured by this table; see §1.3.

Selected model, seed 42 test confusion matrix (threshold 0.82): TN 15,483 · FP 104 · FN 297 · TP 786.

### 1.2 Findings

1. **The supervised baseline is reproducible.** `hgb_165_legacy` reproduces the earlier report's five-seed HistGBDT numbers to four decimals (precision 0.8017, recall 0.7221, F1 0.7579, ROC-AUC 0.9404, AP 0.7941), now trained from the Spark-prepared data.
2. **Learned GAT embeddings do not beat tabular HistGBDT.** Once stacking leakage is removed (cross-fitted embeddings), HistGBDT + GAT reaches 0.7647 vs 0.7933 for tuned HistGBDT alone. The in-sample variant (what the earlier "TGAT-BoostNet" did) is clearly worse on test (0.6965), as expected from leakage: embeddings are optimistic on training rows. GAT alone: 0.4915.
3. **The selected model uses graph information, but not a learned GNN**: degree and mean-of-neighbour features. Its advantage over hgb_165_tuned is within noise (§1.3). Test F1 was *not* used to choose it.
4. **The graph model is not temporal.** Elliptic edges never cross time steps (verified); the GAT and the neighbour features see one time-step snapshot at a time. Nothing here models temporal evolution, so we do not use the term "temporal graph attention" for any new model (DECISIONS D-09).

### 1.3 Uncertainty that the seed table hides

* **Threshold fragility.** Validation F1 is ≈0.97 for most models while test F1 is ≈0.76–0.80, so validation barely separates the strong models. A bit-level change to the neighbour features (the order-invariance fix, DECISIONS D-18) moved hgb_165_nbr's validation-optimal threshold from 0.61 to 0.82 and its test F1 from 0.7683 to 0.7968. Treat differences below ~0.03 test F1 between the top models as not established.
* **Bundle 1.0.0 vs 1.0.1.** 1.0.0 (exported before the fix) is kept unchanged on disk for the record; its aggregate outputs are in `data/validation/final_model_selection/backup_before_refit_hgb_165_nbr_*`. The scorer uses 1.0.1.

### 1.4 Distribution shift at time step 43

Source: `data/validation/final_model_selection/step43_shift/` (`scripts/analyze_step43_shift.py`; post-hoc diagnostic, test labels used only to describe the population).

| | Steps 35–42 | Steps 43–49 |
|---|---:|---:|
| Labelled / illicit | 9,983 / 914 | 6,687 / 169 |
| Illicit prevalence | 9.2 % | 2.5 % |
| Selected model recall (illicit) | **0.854** | **0.030** |
| Illicit tx: share of 10 nearest training neighbours that are illicit | 0.82 | **0.06** |
| Licit tx: same measure | 0.05 | 0.06 |

After step 43 the labelled illicit transactions are, in feature space, surrounded by training-period *licit* transactions; the licit population does not move. So the failure is a change in what illicit activity looks like (concept shift of the positive class), not a generic covariate drift, and no model trained on steps 1–29 in this study recovers it. The dataset has no timestamps, so the data cannot establish a cause. The original Elliptic paper (Weber et al., 2019) associates the period with a dark-market shutdown; we report that only as an external hypothesis. (A domain classifier was tried first; it separated every pair of periods at AUC ≈ 1.0, including licit vs licit, so it is not informative here.)

Per-step metrics for every config and seed: `per_step_metrics.csv`.

### 1.5 Verification of earlier reported numbers

`scripts/verify_saved_metrics.py` → `data/validation/metric_verification/metric_verification.json`: all saved seed-42 predictions recompute to the reported precision/recall/F1/ROC-AUC/AP and confusion matrices (same 16,670 rows, positive class = illicit = 1, thresholds from validation). The earlier five-seed table (HistGBDT 0.8017/0.7221/0.7579…, TGAT-BoostNet 0.8718/0.6643/0.7534…) equals the means of `tgat_boostnet_experiment/seed_results.csv`. Limits of that verification: only seed 42's probabilities were saved in the earlier runs, so seeds 123–2024 could not be recomputed from predictions; `6_Tuned_XGBoost_165` probabilities were never saved; legacy XGBoost's five seeds are identical runs. The new run saves probabilities for every seed and config.

### 1.6 SnapShift-Boost (the project's own model; served as bundle 2.0.0)

Design, ablation and discussion: `docs/SNAPSHIFT_BOOST.md`. Sources: `data/validation/snapshift/dev/summary.json` (component selection on rolling folds inside steps 1–34, test steps removed from the data), `data/validation/snapshift/final/seed_42.json` (single test read), `bundle_checks.json` (export reproduction checks).

| | hgb_165_nbr (1.0.1) | **SnapShift-Boost (2.0.0)** |
|---|---:|---:|
| Mean F1 over the 3 rolling development folds | 0.9305 | **0.9364** |
| Validation F1 (steps 30–34) / threshold | 0.9743 / 0.82 | 0.9796 / 0.90 |
| Test precision / recall | 0.8831 / 0.7258 | **0.8883 / 0.7267** |
| **Test illicit F1** | 0.7968 | **0.7994** |
| Test ROC-AUC / AP | **0.9336** / 0.7977 | 0.9315 / **0.8004** |
| FP / FN | 104 / 297 | 99 / 296 |
| Recall steps 35–42 / 43–49 | 0.854 / 0.030 | 0.857 / 0.024 |

Paired bootstrap of the test F1 difference: +0.0026, 95 % interval [−0.0048, +0.0107] resampling transactions and [−0.0050, +0.0104] resampling time steps; positive in 76 % of resamples. **A small improvement that is not statistically significant.** It does not change the post-step-43 collapse (§1.4). Deterministic: a second seed gives bit-identical scores on all 203,769 nodes.

## 2. Elliptic through Spark

| Item | Result | Source |
|---|---|---|
| Spark preprocessing (`scripts/process_elliptic_spark.py`, `local[4]`) | 218.3 s; 203,769 nodes, 234,355 edges, 49 steps | `data/processed/elliptic_spark/manifest.json` |
| Integrity | 0 duplicate ids/edges, 0 nulls, 0 non-finite values, 0 dangling endpoints, 0 cross-step edges, 0 self-loops | `validation_report.json` |
| Parity vs legacy pandas pipeline | 33/33 checks pass; raw features bit-identical; scaled features ≤ 1 float32 ULP | `data/validation/elliptic_spark/parity_report.json` |
| Downstream | all model selection above trains from the Spark output | `selection.json` (`"source": "spark"`) |

## 3. Zenodo unsupervised anomaly scoring

No reliable transaction-level labels exist, so there is **no** precision/recall/F1/illicit-ROC-AUC for Zenodo. Full report: `data/validation/zenodo_anomaly/ZENODO_ANOMALY_REPORT.md`.

| Item | Result |
|---|---|
| Input reused | `data/parquet/zenodo_features` (16,889,271 tx, blocks 480000–489999); raw reconstruction not rerun |
| Feature stage (Spark) | 16,889,271 in → out, 325.5 s, 0 data-quality violations |
| Detector | Isolation Forest, 13 log-scaled features, fit on a deterministic 2 % hash sample (132,402 rows) of blocks 481000–484999 in 10.3 s; threshold = 99.9th percentile of fit scores |
| Full scoring | 16,889,271 rows in 2,319 s (7,284 rows/s incl. Parquet IO) |
| Stability | PSI < 0.1 for every 1,000-block chunk; out-of-time flag rate 0.6–1.0 ‰ (design 1 ‰) |
| Robustness | Spearman ≥ 0.99 across seeds / sample sizes; top-0.1 % Jaccard only 0.22–0.63 (flag membership is sensitive to refits) |
| Synthetic fault injection (artificial) | fan-in AUROC 1.000, fan-out 0.986, value×10⁴ 0.833, fee 30 % 0.783, fee×2 0.485 (not detected) |

## 4. Scoring and streaming performance

### 4.1 In-memory (no Kafka/Spark) — `data/validation/scoring_benchmark.json`

| Measurement | Value |
|---|---|
| Model load (incl. SHA-256 checks): Elliptic bundle 1.0.1 / Zenodo 1.0.0 | 16 ms / 28 ms |
| Elliptic graph-context construction per snapshot (median 4,975 nodes) | p50 23.6 ms, p95 32.3 ms |
| Elliptic model inference per snapshot | p50 32.9 ms, p95 59.3 ms (≈7 µs per transaction) |
| Elliptic full service call incl. event validation | p50 213 ms, p95 282 ms per snapshot |
| Event path vs offline evaluation | test F1 0.7968 through events/adapters = offline value |
| Zenodo single event through `ScoringService` | p50 12.2 ms |
| Zenodo batches of 1,000 / 5,000 | 19,876 / 16,787 events/s |

### 4.2 End-to-end replay (producer → Kafka → Spark Structured Streaming → scoring → Parquet)

`e2e latency = scored_at − produced_at`. Single workstation, one broker, Spark `local[2]`, 3 partitions, 1–2 s trigger.

| Run | Events | Producer rate | Sustained throughput | E2E p50 / p95 | Notes | Source |
|---|---:|---:|---:|---:|---|---|
| bench_rate200 (Zenodo) | 6,000 | 200/s | 217 /s | 2.5 s / 4.8 s | 0 errors | `data/validation/streaming_replay/bench_rate200.json` |
| bench_unthrottled (Zenodo) | 20,000 | 2,844/s | 937 /s | 14.8 s / 21.2 s | consumer is the bottleneck; latency = queueing | `bench_unthrottled.json` |
| demo (Elliptic steps 42–44 + Zenodo) | 21,178 + 3 malformed + 5 dup | 300/s + 200/s | 378 /s over run | Zenodo mean 1.3 s; Elliptic mean 11.5 s, p95 22 s | Elliptic waits for its complete snapshot by design | `data/streaming/demo/` |

Consumer startup (JVM + Spark + Kafka connector) ≈ 65 s. Peak process-tree RSS ≈ 1.2 GB. These are local measurements on one machine; they are not a statement about cluster throughput.

### 4.3 Correctness under failure — `tests/integration/test_streaming_e2e.py`

* Restart from checkpoint processes only new offsets; hard kill (`taskkill /F`) mid-run then recovery → 0 lost and 0 duplicated predictions; malformed → dead letter; injected duplicates removed (watermark + sink index). Passed 6 of 7 runs; one failure on 2026-10-14 whose assertion message was not captured (cause unknown; the most timing-sensitive step is the "kill lands mid-replay" check).
* Mixed Elliptic + Zenodo topic: explicit routing, snapshots buffered across micro-batches, streamed Elliptic scores **bit-identical** to direct batch inference on the same snapshots. This test found the node-order bug (DECISIONS D-18).

## 5. Tests

| Suite | Command | Result (2026-10-14) |
|---|---|---|
| Unit (no infrastructure) | `.venv\Scripts\python.exe -m pytest` | **92 passed** |
| Integration (Kafka + Spark) | `.venv\Scripts\python.exe -m pytest tests\integration -m integration` | **2 passed** (latest two full runs) |
