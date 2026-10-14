# Zenodo anomaly scoring: evaluation report

**Scope.** Unsupervised anomaly scoring of 16,889,271 reconstructed Bitcoin transactions (blocks 480000–489999, Zenodo record 7157854 part 2). **No reliable transaction-level illicit labels exist for this data**, so this report contains no precision, recall, F1 or ROC-AUC against "illicit". A high score means *statistically unusual relative to the fit window*, not wrongdoing.

Sources: `data/processed/zenodo_anomaly/features_report.json`, `score_summary.json`, `top_ranked.parquet`, `data/validation/zenodo_anomaly/evaluation.json`, model `artifacts/models/zenodo/1.0.0/manifest.json`.

## 1. Pipeline and data quality

| Stage | Result |
|---|---|
| Input | existing `data/parquet/zenodo_features` (reused; raw reconstruction not rerun) |
| Spark feature stage | 16,889,271 rows in → 16,889,271 out, 325.5 s, `local[4]` |
| Data-quality rules (null keys, hash format, ≥1 output, inputs for non-coinbase, fee ≥ 0, fee = in − out, non-negative values) | 0 violations |
| Duplicate tx_hash | 0 |
| Non-finite model features | 0 |
| Coinbase | 10,000 (1 per block) → status `skipped_coinbase` |
| Model | Isolation Forest, 200 trees, `max_samples=256`, 13 log-scaled features |
| Fit sample | 132,402 rows = deterministic 2 % hash sample (`int(tx_hash[:8],16) % 10000 < 200`) of chunks 481000–484999, non-coinbase. Fit time 10.3 s |
| Flag threshold | 0.67065 = 99.9th percentile of fit-sample scores (≈0.1 % flagged by design) |
| Full scoring | 16,889,271 rows in 2,319 s (7,284 rows/s incl. Parquet IO; slowed by a concurrent training job); in-memory 19,063 rows/s |

Chunk 480000–480999 is excluded from fitting because address history starts at block 480000 in this dataset (cold start: address-history features are biased toward "new"). Chunks 485000–489999 are never used for fitting (out-of-time).

## 2. Stability across block chunks

PSI of the score histogram vs the fit window (50 bins, 0.30–0.85):

| Chunk | Role | PSI | Flag rate | Median | p99 |
|---|---|---:|---:|---:|---:|
| 480000–480999 | cold start (excluded) | 0.055 | 0.75 ‰ | 0.396 | 0.621 |
| 481000–481999 | fit | 0.018 | 0.81 ‰ | 0.395 | 0.614 |
| 482000–482999 | fit | 0.010 | 0.78 ‰ | 0.399 | 0.627 |
| 483000–483999 | fit | 0.007 | 1.10 ‰ | 0.408 | 0.631 |
| 484000–484999 | fit | 0.033 | 0.97 ‰ | 0.416 | 0.632 |
| 485000–485999 | out-of-time | 0.075 | 0.96 ‰ | 0.411 | 0.630 |
| 486000–486999 | out-of-time | 0.056 | 0.61 ‰ | 0.411 | 0.628 |
| 487000–487999 | out-of-time | 0.084 | 0.67 ‰ | 0.414 | 0.630 |
| 488000–488999 | out-of-time | 0.071 | 0.98 ‰ | 0.413 | 0.622 |
| 489000–489999 | out-of-time | 0.023 | 0.61 ‰ | 0.407 | 0.621 |

All PSI < 0.1 (conventionally "no significant shift"); out-of-time flag rates 0.6–1.0 ‰ against the 1 ‰ design rate.

## 3. Robustness of the ranking

Evaluation rows: chunk 489000–489999, hash buckets 9000–9199 (38,921 transactions, never used for fitting).

| Change | Spearman vs production | Top-0.1 % Jaccard |
|---|---:|---:|
| seed 1 / 2 / 3 / 4 | 0.993 / 0.990 / 0.991 / 0.993 | 0.63 / 0.34 / 0.37 / 0.42 |
| fit sample 0.5 % (32,994 rows) | 0.994 | 0.42 |
| fit sample 5 % (330,898 rows) | 0.991 | 0.22 |
| without address-history features | 0.766 | 0.18 |
| without fee features | 0.895 | 0.37 |
| without shape features (in/out counts) | 0.921 | 0.03 |
| without value features | 0.965 | 0.30 |

**Reading.** The overall ordering is very stable (Spearman ≥ 0.99 across seeds and sample sizes), but *which* transactions land in the top 0.1 % changes noticeably between refits (Jaccard 0.22–0.63). Individual flags should therefore be treated as a ranked review queue rather than stable verdicts. Address-history and shape features carry most of the signal.

## 4. Synthetic fault injection (artificial anomalies, NOT real-world detection)

5,000 held-out real transactions were modified, with all derived fields recomputed consistently:

| Injected pattern | Flagged at threshold | AUROC injected vs clean | Median score |
|---|---:|---:|---:|
| clean baseline | 0.06 % | — | — |
| fan-in: 150–400 inputs → 1 output, fresh addresses | 99.8 % | 1.000 | 0.687 |
| fan-out: 1 input → 200–1000 outputs | 7.9 % | 0.986 | 0.610 |
| value × 10,000 | 0.44 % | 0.833 | 0.498 |
| fee = 30 % of input value | 0.26 % | 0.783 | 0.473 |
| fee × 2 (subtle) | 0.10 % | 0.485 | 0.406 |

The detector reliably separates structural extremes (fan-in/fan-out) by ranking. At the 0.1 % operating point it catches fan-in only, and it cannot detect subtle fee manipulation (AUROC ≈ 0.5).

## 5. Inspection of top-ranked real transactions

* Of the top 200 transactions, 95.5 % lie in blocks 488000–488099. The top 1,000 contain 249 transactions from chunk 488000–488999 against 44–126 for the other chunks.
* These are near-identical **150-input → 1-output** transactions in which the fee is almost the entire input. Example: block 488003, tx_index 2065, `47fdea1b…8dbf`: input 5,241,629 sat, fee 5,238,899 sat. Pattern consistent with **dust consolidation where the output is just above dust**. This is unusual on-chain behaviour; nothing in the data indicates whether it is benign or not.
* Other top-ranked examples are large consolidations from entirely new addresses (e.g. block 481660 tx 2: 403 inputs → 1 output, 756.47 BTC, all input addresses unseen in the processed window).
* Most frequent primary reason codes in the top 1,000: `log_num_inputs` (617), `log_fee_per_io` (116), `new_addr_fraction` (92).

## 6. Limitations

* No labels, so precision of the flagged set is unknown.
* Address-history features only cover blocks ≥ 480000 (cold start in the first chunk).
* No addresses or transaction graph edges are available in the stored Parquet, so no graph-based anomaly features beyond the address-history aggregates.
* The 0.1 % threshold is a capacity choice, not a calibrated risk level.
