# Data schemas

Machine-readable JSON Schemas: `docs/schemas/transaction_event.v1.schema.json` and `docs/schemas/prediction_result.v1.schema.json` (generated from `src/contracts/schemas.py`).

## 1. Canonical event (`schema_version` "1.0")

| Field | Type | Notes |
|---|---|---|
| `schema_version` | "1.0" | |
| `source_dataset` | `"elliptic"` \| `"zenodo"` | discriminator; selects payload and model |
| `event_id` | string | must equal `<source_dataset>:<transaction_id>`; also the Kafka key |
| `transaction_id` | string | Elliptic txId, or Zenodo tx_hash |
| `event_time` | `{kind, value}` | `elliptic_time_step` (1–49) or `block_height`; **ordinal, not wall-clock** |
| `produced_at` | ISO-8601 UTC | producer wall clock (set at send) |
| `replay_sequence` | int \| null | deterministic replay order |
| `data_quality` | `{status, issues[]}` | |
| `attributes` | dataset-specific | see below; extra fields are rejected |
| `graph` | Elliptic: `{snapshot_id, snapshot_size, out_neighbors[]}`; Zenodo: null | |
| `tx_hash` | Zenodo only | `^[0-9a-f]{64}$` |

**Elliptic `attributes`**: `features` = 165 raw floats f1..f165 (anonymised by Elliptic; f1..f93 local, f94..f165 neighbourhood aggregates). **Labels are never part of an event.**

**Zenodo `attributes`**: `block_number, tx_index, num_inputs, num_outputs, total_input_satoshi, total_output_satoshi, fee_satoshi, fee_rate` (fee ÷ (inputs+outputs), *not* per byte), `is_coinbase, avg_input_val, avg_output_val, input_output_count_ratio, input_output_val_ratio, max_input_prior_tx_count, avg_input_prior_tx_count, max_input_prior_volume_satoshi, new_address_fraction`. History features cover only blocks ≥ 480000 (the processed window).

## 2. Prediction result (`schema_version` "1.0")

| Field | Notes |
|---|---|
| `event_id`, `transaction_id`, `source_dataset`, `event_time` | copied from the event (null if the event could not be parsed) |
| `model_id`, `model_version` | `elliptic-supervised` / `zenodo-isolation-forest` + semantic version |
| `score` | float, meaning depends on `score_type` |
| `score_type` | `illicit_probability` (Elliptic) or `isolation_forest_anomaly_score` (Zenodo). **Not comparable with each other.** |
| `score_percentile` | Zenodo only: rank within fit-window scores |
| `threshold`, `flagged` | model's decision threshold and `score >= threshold` |
| `reason_codes` | Zenodo: top-3 robust z-scores `feature:high/low(z)`; Elliptic: none (no faithful explanation implemented) |
| `graph_context` | `complete` / `partial` / `not_required` / `not_available` |
| `status` | `scored`, `skipped_coinbase`, `skipped_invalid_input`, `rejected_incomplete_graph_context`, `rejected_invalid_event`, `error` |
| `error` | message for non-scored statuses |
| `produced_at`, `processing_time`, `scored_at` | wall-clock pipeline times |

The streaming sink stores results as Parquet with additional `kafka_partition`, `kafka_offset`, `kafka_timestamp`, `batch_id`, `e2e_latency_ms` (= scored_at − produced_at).

## 3. Elliptic Spark output (`data/processed/elliptic_spark/`)

`nodes/` (partitioned by `time_step`): `node_idx` long (0..N-1, order = time_step, tx_id), `tx_id` long, `label` int (1 illicit, 0 licit, −1 unknown), `label_status`, `split` (`train` 1–29 / `val` 30–34 / `test` 35–49 for labelled; `unlabelled` otherwise), `f1..f165` float32 raw.
`nodes_scaled/`: same keys, features standardised with train-only mean/population std.
`edges/`: `edge_id` (CSV row order), `source_tx_id`, `target_tx_id`, `source_idx`, `target_idx`, `time_step`.
Plus `scaler_stats.json`, `validation_report.json`, `manifest.json` (input SHA-256s, counts, Spark version).

## 4. Zenodo stores

* `data/parquet/zenodo_features/` (existing, read-only): 18 physical columns + Hive key `block_chunk` (see `src/spark/schemas.py`).
* `data/processed/zenodo_anomaly/features/`: 18 input columns + `dq_status`, `sample_bucket` (0–9999 from tx_hash), 13 model features (`MODEL_FEATURES` in `src/anomaly/zenodo_features.py`), partitioned by `block_chunk`.
* `data/processed/zenodo_anomaly/scores/`: `block_number, tx_index, tx_hash, is_coinbase, dq_status, score_status, score, percentile, flagged, reason_codes` (reasons only for percentile ≥ 0.99).

## 5. Streaming run directory (`data/streaming/<run>/`)

`checkpoint/` (Spark), `state/` (Elliptic snapshot buffer, scored event-id index), `predictions/batch=<id>.parquet`, `dead_letter/batch=<id>.parquet` (`kafka_partition, kafka_offset, key, value, reason, batch_id, detected_at`), `monitoring/batches.jsonl`, `monitoring/progress.jsonl`, `monitoring/run_info.json`, `monitoring/consumer.log`.
