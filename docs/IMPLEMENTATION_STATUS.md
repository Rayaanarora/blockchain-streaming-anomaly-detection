# Implementation status

Legend: ✅ implemented and verified · 🟡 implemented with a stated limitation · ⛔ not implemented (reason given). Evidence paths are relative to the repository root. Last updated 2026-10-14.

## Definition of done

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Existing datasets and artifacts preserved | ✅ | Raw data, `data/parquet/zenodo_features`, legacy `data/processed/elliptic`, earlier validation reports untouched. Changed outputs were backed up first (`final_model_selection/backup_before_refit_*`); bundle 1.0.0 kept beside 1.0.1. Only stray JVM temp files were deleted (D-22) |
| 2 | Supervised Elliptic baseline reproducible | ✅ | `hgb_165_legacy` reproduces the earlier five-seed HistGBDT means to 4 decimals; saved seed-42 predictions recompute exactly (`metric_verification.json`) |
| 3 | Selected model saved and loadable for inference | ✅ | `artifacts/models/elliptic/1.0.1`; checksum-verified load; event-path test F1 = offline value (`scoring_benchmark.json`) |
| 4 | Elliptic processed through Spark with verified integrity | ✅ | `scripts/process_elliptic_spark.py` (218 s); 33/33 parity checks; `tests/test_elliptic_spark.py` |
| 5 | Existing Zenodo Parquet reused | ✅ | All Zenodo stages read `data/parquet/zenodo_features`; raw reconstruction not rerun |
| 6 | Zenodo anomaly-scoring pipeline, appropriately evaluated | ✅ | 16,889,271 tx scored; unsupervised evaluation without fake labels (`ZENODO_ANOMALY_REPORT.md`) |
| 7 | Explicit model-routing rules | ✅ | `ScoringService` routes on the validated `source_dataset`; tests `test_routing_and_score_semantics`, mixed integration test |
| 8 | Event schema and scoring-result schema | ✅ | `src/contracts/schemas.py`, `docs/schemas/*.json`, `docs/DATA_SCHEMAS.md` |
| 9 | Unified scoring interface works | ✅ | `src/scoring/service.py`, `/api/score`; contract tests |
| 10 | Deterministic Kafka → Spark replay runs | ✅ | `scripts/replay_producer.py` → `scripts/streaming_consumer.py`; integration tests; demo run `data/streaming/demo` |
| 11 | Scoring results persisted | ✅ | `<run>/predictions/batch=*.parquet`, `dead_letter/`, `monitoring/` |
| 12 | Dashboard displays actual results | ✅ | `src/api/app.py` + `static/index.html` read the persisted run; checked against the demo run (21,178 predictions) |
| 13 | Important unit and integration tests pass | 🟡 | 92/92 unit; integration 2/2 on the latest runs, but the restart/kill test failed once in 7 runs with an uncaptured cause |
| 14 | End-to-end latency and throughput measured | ✅ | `data/validation/streaming_replay/*.json`, demo run, `scoring_benchmark.json` (single workstation only) |
| 15 | README and architecture docs match the implementation | ✅ | `README.md`, `docs/END_TO_END_ARCHITECTURE.md` |
| 16 | Remaining limitations documented | ✅ | below, `docs/DECISIONS.md` |
| 17 | Nothing pushed to GitHub | ✅ | no commits or pushes were made |

## Capability matrix

| Capability | Status | Notes |
|---|---|---|
| Spark batch preprocessing (Elliptic, Zenodo features) | ✅ | local mode, JVM-only transforms |
| Supervised Elliptic model (12 configs × 5 seeds) | ✅ | validation-only selection |
| Graph-aware Elliptic inference (complete snapshots) | ✅ | buffered per time step in streaming |
| Temporal graph modelling | ⛔ | no model in the repository models temporal evolution (D-09) |
| Zenodo unsupervised anomaly scoring | ✅ | Isolation Forest + reason codes |
| Zenodo supervised evaluation | ⛔ | no reliable labels |
| Kafka replay of stored data | ✅ | deterministic, rate-limited, stable ids |
| Spark Structured Streaming scoring | ✅ | checkpoints, dedup, DLQ, restart recovery |
| Exactly-once end-to-end | ⛔ not claimed | at-least-once + idempotent sinks (D-15) |
| Live blockchain ingestion | ⛔ | no node/API source exists |
| Distributed training / multi-node Spark | ⛔ | single workstation |
| Streaming address-history state for new Zenodo blocks | ⛔ | features come from the batch store |
| Docker deployment | ⛔ | Docker daemon unavailable and C: nearly full; native Kafka on D: instead (D-06) |

## Known limitations

1. Elliptic recall collapses after time step 43 (0.85 → 0.03). No configuration tested here fixes it.
2. Threshold choice on the small, near-saturated validation set is fragile (≈±0.03 test F1).
3. GAT inference sums in float32 via torch scatter, so it is not guaranteed bit-identical under node reordering. The exported model does not use the GAT.
4. Spark Python workers crash on this host (D-05), so streaming scoring runs in the driver inside `foreachBatch`.
5. Kafka is a single plaintext broker on localhost with topic deletion disabled (D-08).
6. All performance figures come from one Windows workstation.
7. One unexplained failure of the restart/kill integration test (1 of 7 runs).
