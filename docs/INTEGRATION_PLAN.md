# Integration plan

How the pieces were connected, in execution order, and what remains. Status markers: ✅ done and verified, 🟡 partial, ⛔ blocked or out of scope.

| # | Step | Status | Verification |
|---|---|---|---|
| 1 | Audit repository, run existing tests (66 passed at audit time; 92 unit tests now), recompute saved metrics | ✅ | `scripts/verify_saved_metrics.py` → `data/validation/metric_verification/` |
| 2 | Make Spark usable for native writes and checkpoints on Windows (Java resolution, pure-Java FS) | ✅ | probe: batch Parquet write and rate-source checkpoint succeed |
| 3 | Elliptic → Spark preprocessing → `data/processed/elliptic_spark` | ✅ | 33/33 parity checks; 6 Spark tests |
| 4 | Train/select the supervised model from the Spark output (5 seeds, validation-only) and export a bundle | ✅ baseline `hgb_165_nbr` bundle 1.0.1; SnapShift-Boost bundle 2.0.0 served (D-23, D-24) | `data/validation/final_model_selection/`, `docs/VERIFIED_RESULTS.md` |
| 5 | Zenodo features (Spark) → detector fit → full scoring → unsupervised evaluation | ✅ | 16,889,271 rows scored; `data/validation/zenodo_anomaly/evaluation.json`; 5 tests |
| 6 | Versioned event/result contracts, adapters, unified scorer with routing | ✅ | 10 tests (routing, invalid inputs, graph context, checksums) |
| 7 | Local Kafka (KRaft) on D:, replay producer | ✅ | delivered counts in producer manifests |
| 8 | Spark Structured Streaming consumer: parse, dedup, DLQ, state, scoring, durable outputs | ✅ Zenodo and Elliptic (complete-snapshot buffering, stream = batch scores) | integration test (restart + kill/recover), 4 batch-logic unit tests |
| 9 | API + dashboard over persisted outputs | ✅ | API tests; screenshot check |
| 10 | Benchmarks: Spark preprocessing, model load, inference, Kafka→output latency, throughput, memory | ✅ | `data/validation/streaming_replay/*.json`, `scoring_benchmark.json` |
| 11 | Docs, README, final diff review | ✅ | this folder, `IMPLEMENTATION_STATUS.md`, `IMPLEMENTATION_LOG.md` |

## Not in scope / blocked

* ⛔ **Live blockchain ingestion**: no node or API source is available or implemented; the stream is a replay of stored data.
* ⛔ **Supervised evaluation on Zenodo**: no reliable labels (the earlier address-overlap study found insufficient overlap).
* ⛔ **Distributed model training / multi-node Spark**: single workstation only.
* ⛔ **Python UDFs in Spark**: Spark Python workers crash on this host (DECISIONS D-05); scoring runs in the driver.
* ⛔ **Live Zenodo address-history state**: history features come from the batch preprocessing; extending them to unseen blocks needs address-level streaming state (not implemented).

## How to extend safely

1. New model version: run the training script with `--model-version X.Y.Z` (export refuses to overwrite). The scorer picks the highest semantic version under `artifacts/models/<dataset>/`, or pin it with `BSAD_ELLIPTIC_BUNDLE` / `BSAD_ZENODO_MODEL`.
2. New dataset: add a payload model and a `Literal` to the event union in `src/contracts/schemas.py`, an adapter, and a `_score_<dataset>` branch in `ScoringService`. Routing stays explicit.
3. Cluster deployment: drop the Windows-specific options in `get_spark` (bare FS, Java path), move scoring from `foreachBatch` driver code to `mapInPandas` once Python workers are available, and replace the local Kafka with a managed cluster.
