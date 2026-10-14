# Architectural decisions

Each entry: context → decision → consequence. Dates are 2026-10-13 unless stated.

## D-01 Reuse the existing Zenodo Parquet store; never rerun raw reconstruction
**Context.** `data/parquet/zenodo_features/` (16,889,271 rows, 10 `block_chunk` partitions, 2.13 GB) took ~3.9 h to build. Inspection found no correctness problem: 0 duplicate `tx_hash`, 0 null keys, 0 rows failing the data-quality rules (fee = inputs − outputs, ≥1 output, …), 0 non-finite derived features, exactly 1 coinbase per block.
**Decision.** All Zenodo work (features, fitting, scoring, replay) reads this store read-only.
**Consequence.** Only what the store contains is available: no addresses, no per-output values, no block timestamps, no transaction→transaction edges. Graph information for Zenodo is limited to the address-history aggregates computed during preprocessing.

## D-02 The Spark Elliptic output is the training source; the legacy npz is kept as the parity reference
`scripts/process_elliptic_spark.py` writes `data/processed/elliptic_spark/`. `scripts/validate_elliptic_spark_parity.py` proves it is identical to the legacy pandas output (33/33 checks; raw features bit-identical; scaled features ≤ 1 float32 ULP). `scripts/run_final_model_selection.py --source spark` (default) trains from the Spark output. The legacy `data/processed/elliptic/` is untouched.

## D-03 Resolve the Java runtime explicitly
The shell's `JAVA_HOME` points to a non-existent `C:\Program Files\Java\jdk\bin`. The existing code silently fell back to the Java 8 JRE. `src/spark/session.py::resolve_java_home` now checks candidates (`BSAD_JAVA_HOME`, `JAVA_HOME`, the Java 8 JRE, JDK 17) for an existing `bin\java.exe`. Spark 3.5.0 and Kafka 3.7.2 both run on Java 8 here.

## D-04 Pure-Java local filesystem instead of `winutils.exe`
**Context.** pyspark 3.5.0 bundles Hadoop 3.3.4. On Windows its local filesystem shells out to `winutils.exe` for permissions, so `df.write.parquet` and Structured Streaming checkpoints failed ("HADOOP_HOME and hadoop.home.dir are unset"). This is why the original Zenodo pipeline writes via PyArrow.
**Decision.** Put `com.globalmentor:hadoop-bare-naked-local-fs:0.1.0` (Apache-2.0, Maven Central, pure Java) on the driver classpath and set `fs.file.impl` / `fs.AbstractFileSystem.file.impl` to it, with the FileSystem-based checkpoint manager. Jars are fetched by `scripts/fetch_spark_jars.py` with SHA-1 verification; no native binaries are downloaded.
**Consequence.** Native Spark Parquet writes and streaming checkpoints work (verified). Linux/cluster deployments would simply not set these options.

## D-05 No Python workers inside Spark on this host
**Context.** Every Python-worker task (pandas UDF, RDD `map`, `zipWithIndex`) crashes with "Python worker exited unexpectedly (crashed)" and no Python traceback, even with faulthandler on. JVM-only DataFrame operations work.
**Decision.** All Spark transformations are JVM expressions (Elliptic preprocessing, Zenodo features, Kafka parsing, de-duplication). Model scoring in streaming runs in the driver inside `foreachBatch` on the micro-batch collected to pandas.
**Consequence.** Scoring is single-process per micro-batch (fine for the measured local replay rates; it would be the first thing to distribute on a cluster with working Python workers). Root cause is unresolved and documented as a limitation.

## D-06 Native single-node Kafka (KRaft) on D:, not Docker
Docker Desktop's daemon was not running, its storage is on C: (7.5 GB free, 99 % full), and the user asked for everything—including Kafka data—to stay on D:. Kafka 3.7.2 (SHA-512 verified) lives in `D:\blockchainbigdata\tools\`, data in `data/kafka_data/`, logs in `data/kafka_run/`. `scripts/local_kafka.py` manages it. Plaintext, localhost only, not a production configuration.

## D-07 Launch Kafka's JVM directly; bind to 127.0.0.1
Kafka's Windows `.bat` launchers exceed cmd.exe's line-length limit ("The input line is too long"); `local_kafka.py` starts `kafka.Kafka` with `java -cp libs\*`. `localhost` resolved to `::1` first and every client connection paid a 2 s IPv6 connect timeout (the producer measured 94 events/s); binding and advertising `127.0.0.1` raised it to ~1,260–2,840 events/s.

## D-08 Topic deletion disabled
Deleting a topic on Windows renames its directory while segment files are open → `AccessDeniedException` → the broker marks its only log dir failed and shuts down (observed twice: once at runtime, again at startup). `delete.topic.enable=false`; tests and benchmarks use fresh uniquely named topics and leave them. The directories of the two topics whose deletion crashed the broker were **moved** (not deleted) to `data/kafka_quarantine/<timestamp>/`.

## D-09 The graph model is a static per-snapshot GAT, not a temporal model
The legacy `SpatioTemporalGAT` concatenates a linear projection of the scalar `t/49` to the graph representation. That is not temporal modelling, and because test steps (35–49) lie outside the training range (1–29) the time input must extrapolate. Elliptic edges never cross time steps (verified), so message passing only sees the current snapshot. The new `GraphAttentionEncoder` (`src/models/graph_encoder.py`) has no time input and is described as a static GAT. "TGAT" naming is retained only for the legacy artefacts it describes.

## D-10 Cross-fitted GAT embeddings for stacking
The legacy hybrid trained the booster on embeddings of training nodes produced by a GAT that had been fitted on those same labels (stacking leakage: embeddings are optimistic on train, not on test). `cross_fitted_train_embeddings` trains a GAT per contiguous block of training steps and embeds the held-out block; validation/test embeddings come from a GAT trained on all of steps 1–29. An in-sample variant (`hgb_165_gat_insample`) is kept as a control.

## D-11 Pre-registered, validation-only model selection
Rule (in code before the run): highest mean validation F1 over seeds 42/123/456/789/2024; within 0.005 → highest mean validation AP → fewer components. Hyper-parameter grids use validation AP on seed 42. Thresholds: F1-maximising on validation (0.01 grid). Test labels are used once per fitted model after its threshold is fixed. Finding: validation F1 is ~0.95–0.97 for most models while test F1 is ~0.70–0.80, so validation discriminates weakly between strong models — reported, not hidden.

## D-12 Zenodo detector: Isolation Forest + robust-z reason codes
Transparent, cheap, unsupervised, robust to heavy tails after log transforms. Fit window = chunks 481000–484999 (chunk 480000 excluded: address history starts at block 480000, so almost every address looks new there). Deterministic 2 % sample by `int(tx_hash[:8],16) % 10000 < 200` (132,402 rows); chunks 485000–489999 are never used for fitting (out-of-time). Flag threshold = 99.9th percentile of fit scores. Sampling impact measured (0.5 % / 2 % / 5 %: Spearman ≥ 0.99, top-0.1 % Jaccard 0.22–0.42 — top-set membership is sensitive).

## D-13 Event time is an ordinal, never an invented timestamp
Neither dataset has wall-clock transaction times. `event_time` = `{kind: block_height}` (Zenodo) or `{kind: elliptic_time_step}` (Elliptic). `produced_at`, `processing_time`, `scored_at` are pipeline wall-clock times.

## D-14 Graph context in streaming = complete time-step snapshots
A transaction's correct graph context for the GAT is its whole time-step snapshot. Events carry `graph.snapshot_id`, `snapshot_size`, `out_neighbors`. If the exported model needs the graph, the consumer buffers events per snapshot in durable driver-side state and scores only complete snapshots; snapshots incomplete after `--snapshot-ttl-batches` are emitted as `rejected_incomplete_graph_context`. A tabular-only exported model scores events immediately (`graph_context = not_required`).

## D-15 Delivery semantics: at-least-once with idempotent sinks, not claimed exactly-once
Kafka offsets + Spark checkpoint → batches re-execute after a crash. Prediction files are named per batch id and written atomically (replace), and a sink-side `event_id` index drops duplicates that escape the 30-minute streaming watermark. The crash/restart integration test found 0 lost and 0 duplicated predictions, but end-to-end exactly-once is not formally verified (e.g. the sink index append and the Parquet write are not one transaction).

## D-16 FastAPI + static HTML dashboard
No frontend existed (the team context mentions a planned Streamlit demo). One FastAPI process serves the JSON API, the unified `/api/score` endpoint and a dependency-free HTML/SVG dashboard, all reading persisted outputs. Fewer moving parts than Streamlit + a separate API.

## D-17 Generated artefacts are not for Git
`artifacts/`, `data/streaming/`, `data/kafka_*`, `data/jars/`, Spark scratch and large prediction files are git-ignored. Nothing was committed or pushed.

## D-18 Graph features must be node-order invariant (found by the stream/batch parity test) — 2026-10-14
**Context.** The mixed Elliptic+Zenodo integration test compares streamed scores with direct batch inference on the same snapshot. 17 of 5,408 scores differed (up to 0.028). Cause: `neighbor_aggregate_features` scaled each neighbour term by 1/deg in float32 before summing, so rounding depended on node order; Kafka partitions interleave a snapshot's events, so the order differs from the training order, and a last-bit change flips tree splits for borderline nodes (shuffling a snapshot offline changed up to 37 scores by up to 0.07).
**Decision.** Sum neighbour values in float64 (exact for float32 inputs of similar magnitude) and divide by the degree once. Verified exactly order-invariant on all 49 steps × 3 permutations; regression test `test_neighbor_features_and_graph_bundle_are_node_order_invariant` fails on the old code. Only `hgb_165_nbr` uses these features: it was refit for all five seeds (`scripts/refit_selection_config.py`, originals backed up) and re-exported as bundle **1.0.1**. Bundle 1.0.0 is kept, unchanged, as the record of the pre-fix model.
**Consequence.** Streamed and batch scores are now bit-identical. The GAT path (torch scatter sums) has the same theoretical sensitivity; it is not used by the exported model and is listed as a limitation.

## D-19 Keep the pre-registered selection even when another model scores higher on test
`hgb_165_nbr` wins by the validation rule (val F1 0.9743; tie band includes hgb_165_tuned 0.9735 and the ensemble 0.9721; tie-break validation AP 0.9933). Before the D-18 fix the same rule also chose it while hgb_165_tuned had the higher test F1 (0.7933 vs 0.7683). Switching on test results would be test-set selection, so the rule was never overridden. The test gap between the top models (< 0.03) is reported as not established (VERIFIED_RESULTS §1.3).

## D-20 No causal attribution of the step-43 shift
The earlier ablation report attributed the post-43 degradation to the AlphaBay/Hansa takedown. Elliptic has no timestamps, so this cannot be checked against the data. `scripts/analyze_step43_shift.py` measures what changed (illicit prevalence 9.2 % → 2.5 %, recall 0.85 → 0.03, post-43 illicit transactions' training neighbours are 6 % illicit vs 82 % before, licit unchanged) and cites the dark-market explanation only as an external hypothesis from the literature.

## D-21 Parallel seed workers, single aggregation
The five-seed run takes ~40 min per seed (three GAT trainings for cross-fitting dominate). `--train-only` lets independent processes write `seed_<s>.json` / `seed_<s>_probs.parquet`; one `--aggregate-only` call then selects and exports, so no worker can export a partial-seed model.

## D-22 Forward slashes in Spark JVM options on Windows
Spark tokenises `spark.driver.extraJavaOptions` with backslash as an escape, so `-Djava.io.tmpdir=D:\a\b` became the drive-relative `D:ab` and JVM temp files (native lz4/snappy copies) piled up in oddly named folders in the repository root. `get_spark` now passes the path with forward slashes (verified via `System.getProperty("java.io.tmpdir")`); the stray folders contained only those temp files and were removed.

## D-23 The project's own model: SnapShift-Boost, chosen on rolling folds inside steps 1–34 — 2026-10-14
**Context.** The project needs an original model; the leakage-free GAT hybrids did not beat the tree baseline. Elliptic validation (steps 30–34) is near-saturated (val F1 ≈ 0.97 for many models), so it separates candidates poorly.
**Decision.** Three components (R within-step ranks, I per-step Isolation Forest score, P two-stage score propagation with time-grouped cross-fitting) were compared in all 8 combinations on three rolling-origin folds (1–19→20–24, 1–24→25–29, 1–29→30–34). Test steps were removed from the data inside the development script. The selection rule (mean fold F1, ≥ 0.005 over the baseline, ties → fewer components then AP) was written into `scripts/snapshift_dev.py` before the first run. It selected **P**. The test set was then read once (`scripts/snapshift_final.py`). Details: `docs/SNAPSHIFT_BOOST.md`.
**Consequence.** Test F1 0.7994 vs 0.7968 for the baseline (+0.0026; 95 % bootstrap interval −0.005 to +0.011, not significant). This is reported as a small, non-significant improvement. No component or threshold was changed after the test read.

## D-24 Bundle 2.0.0 replaces 1.0.1 as the served Elliptic model
SnapShift-Boost is exported as `artifacts/models/elliptic/2.0.0` with `combine: two_stage` in the manifest (stage 1 → propagation features → stage 2; older bundles keep the default `mean` combination). The major version changes because the scoring function and threshold (0.90) change. The service picks the highest version. Bundles 1.0.0 and 1.0.1 stay on disk, and `BSAD_ELLIPTIC_BUNDLE` pins any of them. Export is gated on three exact-reproduction checks (refit, second seed, per-snapshot scoring).

## D-25 Run the whole streaming demo with one command
`scripts/run_pipeline.py` starts Kafka if needed, a fresh topic and run directory, the consumer, the dashboard and the producers, and stops what it started on Ctrl+C. It reruns itself with `.venv` if launched by another interpreter (a user hit `ModuleNotFoundError: confluent_kafka` by running plain `python`). The dashboard honours `?run=<name>` so the launcher can open the new run directly.
