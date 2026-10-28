# Troubleshooting (Windows workstation)

| Symptom | Cause | Fix |
|---|---|---|
| `JAVA_GATEWAY_EXITED`, "The system cannot find the path specified" | `JAVA_HOME` points to a missing directory, or `TEMP` points to a deleted directory | Set `BSAD_JAVA_HOME` to a JRE containing `bin\java.exe` (Java 8/11/17). `get_spark` sets `TEMP`/`TMP` to `data\spark_temp`. |
| `Invalid switch - "blockchainbigdata"` while starting Spark | `TEMP` set with forward slashes; cmd.exe parses `/x` as a switch | Use backslash paths (`os.path.abspath`) |
| `HADOOP_HOME and hadoop.home.dir are unset` on `df.write` / streaming start | Hadoop needs `winutils.exe` | Run `scripts\fetch_spark_jars.py` and create sessions with `src.spark.session.get_spark` (pure-Java local FS). The WARN line about winutils at startup is harmless. |
| `ClassNotFoundException: ...BareLocalFileSystem` | jars passed via `spark.jars` (resolved after the FS is needed) | They must be on `spark.driver.extraClassPath` (what `get_spark` does) |
| `Python worker exited unexpectedly (crashed)` | Spark Python workers crash on this host (root cause unknown) | Avoid UDFs/RDD lambdas; use JVM column expressions, do Python work in the driver (`foreachBatch`) |
| `[download failed: ...hadoop-client-api]` | `spark.jars.packages` resolves transitive Hadoop jars | Use the pinned local jars from `fetch_spark_jars.py` |
| Kafka `.bat` scripts: "The input line is too long" | classpath exceeds cmd.exe limit | `scripts\local_kafka.py` starts the JVM directly |
| Every Kafka client call takes ~2 s | `localhost` → `::1` first, broker on IPv4 | Use `127.0.0.1:29092` (default everywhere now) |
| Broker exits: "all log dirs ... have failed", `AccessDeniedException ... -delete` | Topic deletion on Windows | `delete.topic.enable=false` is set; never delete topics. If it happens: stop broker, **move** the named `<topic>-<n>` dirs from `data\kafka_data` to `data\kafka_quarantine\`, start again |
| Consumer finds nothing / re-reads everything | Wrong `--run-dir` (the checkpoint lives there) | Reuse the same run dir to resume; use a new one to replay from the earliest offset |
| Model training extremely slow | Thread oversubscription (HistGBDT/torch/XGBoost/Spark all using every core) | `$env:OMP_NUM_THREADS=6`, `--torch-threads 6`, avoid concurrent heavy jobs |
| Dashboard shows "No predictions in this run yet" | Run dir empty | Pick a run in the selector or run producer + consumer (README §6) |
| `FileExistsError` from a pipeline script | Output already exists; scripts never overwrite silently | Pass `--overwrite` (old output is moved to `*.backup_<timestamp>`) or bump `--model-version` |
| C: drive filling up | Tools writing to `%TEMP%`/`~\.ivy2` | All scripts pin scratch to D:. Keep `PIP_CACHE_DIR` on D: (`.env.example`) |
| `run_pipeline.py` prints nothing for a while after "starting Kafka" | Broker start-up; after a forced stop Kafka re-checks every topic log (~15–60 s) | Wait for `kafka up at ... after Ns`; the script now prints progress every 10 s. Stop Kafka with `local_kafka.py stop` (clean shutdown) rather than killing java.exe, so the next start skips recovery. `kafka stopped (forced shutdown)` means the Ctrl+C event was not delivered; check `data\kafka_run/broker.out` |
