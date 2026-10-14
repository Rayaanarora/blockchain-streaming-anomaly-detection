# Implementation log

Concise record of what was changed, run and observed. Commands are PowerShell from the repository root with `.venv\Scripts\python.exe` abbreviated as `py`.

## Session 1 — 2026-10-13 (reconstructed from that session's outputs: `docs/INTEGRATION_PLAN.md`, `docs/DECISIONS.md` D-01…D-17, validation reports)

* Audit: existing tests 66 passed; `scripts/verify_saved_metrics.py` recomputed saved seed-42 metrics (all match; issues listed in `metric_verification.json`).
* Spark on Windows made usable for native writes/checkpoints: `src/spark/session.py` (Java resolution, pure-Java local FS jar via `scripts/fetch_spark_jars.py`).
* Elliptic via Spark: `src/spark/elliptic_spark.py`, `scripts/process_elliptic_spark.py` (218 s) → `data/processed/elliptic_spark`; `scripts/validate_elliptic_spark_parity.py` 33/33.
* Zenodo anomaly pipeline: `src/anomaly/`, `scripts/zenodo_anomaly_pipeline.py` (features 325 s, fit 10 s, full scoring 2,319 s), `scripts/evaluate_zenodo_anomaly.py`; model `artifacts/models/zenodo/1.0.0`.
* Contracts/scoring: `src/contracts/`, `src/scoring/service.py`, JSON schemas in `docs/schemas/`.
* Kafka 3.7.2 native KRaft broker on D: (`scripts/local_kafka.py`), producer `src/streaming_replay/producer.py`, consumer `src/streaming_replay/consumer.py`, integration test, streaming benchmarks (`data/validation/streaming_replay/`).
* API + dashboard: `src/api/`.
* Final model selection (`scripts/run_final_model_selection.py`) completed its grid search (`grid_search.json`) and was then interrupted at the start of the seed loop; no Elliptic bundle existed when session 1 ended.

## Session 2 — 2026-10-14

### Changes

| File | Change |
|---|---|
| `scripts/run_final_model_selection.py` | `--train-only` flag for parallel seed workers (D-21) |
| `src/models/elliptic_training.py` | `neighbor_aggregate_features` made node-order invariant (D-18) |
| `scripts/refit_selection_config.py` (new) | refit one config after a fix, splice into per-seed outputs with backups |
| `scripts/analyze_step43_shift.py` (new) | step-43 shift diagnostic (D-20) |
| `src/streaming_replay/consumer.py` | late duplicates of already-scored Elliptic events are dropped before buffering (previously they opened a snapshot that later expired as a spurious rejection) |
| `src/spark/session.py` | forward slashes in `-Djava.io.tmpdir` (D-22) |
| `tests/test_consumer_batch_and_api.py` | regression assertions for the late-duplicate case |
| `tests/test_contracts_scoring.py` | order-invariance regression test (fails on the old code) |
| `tests/integration/test_streaming_e2e.py` | new mixed Elliptic+Zenodo test with stream-vs-batch score parity |
| `docs/` | `VERIFIED_RESULTS.md`, `MODEL_CARDS.md`, `IMPLEMENTATION_STATUS.md`, this log (new); `DECISIONS.md` D-18…D-22, `INTEGRATION_PLAN.md`, `END_TO_END_ARCHITECTURE.md` (updated) |
| `README.md` | rewritten to match the implementation; earlier results kept with corrections |

Removed: three stray directories in the repository root (`blockchainbigdatablockchain-streaming-anomaly-detectiondata…`) containing only JVM native-library temp copies and faulthandler files (D-22).

### Commands and outcomes

| Command | Outcome |
|---|---|
| `py -m pytest` (before changes) | 91 passed, 1 skipped (the skip needed an Elliptic bundle) |
| 3 parallel workers: `py scripts\run_final_model_selection.py --train-only --torch-threads 5 --seeds 42 789` / `123 2024` / `456` (OMP_NUM_THREADS=5) | all seeds finished; ≈5.5 min tabular + ≈35 min GAT/cross-fit per seed; 19:14 → 20:24 |
| `py scripts\run_final_model_selection.py --aggregate-only` | selected `hgb_165_nbr`; exported bundle 1.0.0 (test F1 0.7683) |
| `py -m pytest tests\integration -m integration` | Zenodo e2e passed; new mixed test **failed**: 17/5,408 streamed Elliptic scores ≠ batch scores |
| diagnosis | node-order dependence of float32 neighbour means (shuffle changed ≤37 scores by ≤0.07); fixed; 0 differing rows over 49 steps × 3 permutations |
| `py scripts\refit_selection_config.py --config hgb_165_nbr` | backups written; val F1 0.9743, threshold 0.82, test F1 0.7968 (identical for every seed: deterministic) |
| `py scripts\run_final_model_selection.py --aggregate-only --model-version 1.0.1` | selection unchanged (`hgb_165_nbr`); bundle 1.0.1 exported |
| `py scripts\analyze_step43_shift.py` | recall 0.854 (35–42) vs 0.030 (43–49); kNN illicit share 0.82 vs 0.06 |
| `py scripts\benchmark_scoring.py` | event-path test F1 0.7968 = offline; snapshot inference p50 32.9 ms |
| `py -m pytest tests\integration -m integration` (×1 full, ×3 restart only, ×2 full) | mixed test passes; restart/kill test 6 passes, 1 failure (first run after the fix; assertion not captured) |
| demo: `py scripts\streaming_consumer.py --topic bsad.demo.202610142043 --run-dir data\streaming\demo --trigger "2 seconds" --max-seconds 420 --max-offsets-per-trigger 2000`; producers: Elliptic steps 42–44 at 300/s, Zenodo 4,000 at 200/s with 3 malformed + 5 duplicates | 21,178 unique predictions, 3 dead-lettered, 5 duplicates dropped by Spark, 0 errors |
| `py -m uvicorn src.api.app:app --port 8000` + headless Edge screenshot | dashboard shows the demo run's real outputs |
| `py -m pytest` (after changes) | **92 passed** |

### Remaining tasks / open issues

* Capture and diagnose the intermittent restart/kill integration failure (1 of 7).
* Make GAT inference order-invariant if a GAT-based model is ever exported.
* Elliptic post-43 concept shift: needs a different approach (e.g. periodic retraining with newer labels); out of scope here.
* Live ingestion, streaming address-history state for Zenodo, cluster deployment: not implemented (see `IMPLEMENTATION_STATUS.md`).
