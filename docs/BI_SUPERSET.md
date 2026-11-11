# BI layer: Apache Superset

The console's chart views (Overview, Detection quality, Model evaluation, Data & preprocessing, Zenodo anomalies) are
Apache Superset dashboards embedded in the FastAPI console. Investigate and Event log stay as console tools.

```
Kafka → Spark Structured Streaming → predictions (Parquet) ─┐
Zenodo 16.9 M tx / Elliptic graph → Spark → gold tables ────┼→ DuckDB (views only) → Superset → embedded in :8000
tables/, results/runs/, data/validation/ → gold tables ─────┘
```

* **Spark processes, DuckDB serves, Superset visualizes.** The heavy aggregation (16.9 M Zenodo transactions joined to
  their features, the Elliptic graph) runs in Spark: `scripts/build_bi_gold.py` → `data/bi/gold/<table>/` (Parquet).
  Small existing result tables (notebooks 01-04, the served bundle's checks) are normalised with pandas.
* **DuckDB holds no data.** `scripts/bi/build_duckdb.py` creates views over the gold Parquet and over
  `data/streaming/run_*/predictions` and `monitoring/batches.jsonl`, so a running pipeline appears in the next query.
* **Superset 6.1.0** with the DuckDB driver, 5 dashboards, 62 charts, built by `scripts/bi/build_dashboards.py`:
  datasets, the chart types it supports (line/bar/area/scatter, pie, big number, table) and the dashboards go through
  Superset's **official MCP service**; chart types the MCP generator does not cover (heatmap, Sankey, radar, box plot,
  treemap, sunburst, funnel, gauge, bubble, histogram), the grid layout and embedding go through Superset's REST API.
  Every chart's axis titles, units and description are in `scripts/bi/chart_labels.py`; the build fails if one is missing.

## Where it runs (everything on D:)

Superset runs inside the Ubuntu WSL distribution at `D:\WSL\Ubuntu` (no Docker: Docker's disk image is on C:).

| Item | Location |
|---|---|
| Superset environment | `~/bi/superset-venv` (Python 3.12), metadata `~/bi/superset_meta.db`, DuckDB file `~/bi/bsad.duckdb` |
| Config | `infra/superset/superset_config.py` (secret key read from `~/bi/secret_key`, never in git) |
| Python wheels | `D:\blockchainbigdata\tools\wheels\superset` (downloaded on Windows, installed offline: WSL downloads were ~45× slower) |
| Pins | Superset 6.1.0's own `requirements/base.txt` as constraints (`tools/superset_6.1.0_constraints.txt`) |

## Commands

`scriptsun_pipeline.py` starts Superset automatically (in the background, in parallel with Kafka and Spark) and stops it on exit if it started it; `--no-superset` skips it, `--keep-superset` leaves it running. The commands below are for building or running the BI layer on its own.

```powershell
.venv\Scripts\python.exe scripts\build_bi_gold.py                          # gold tables (Spark, ~2 min)
powershell -ExecutionPolicy Bypass -File scripts\bi\start_bi.ps1           # Superset :8088 + MCP :5008 (-Stop to stop)
.venv\Scripts\python.exe scripts\bi\build_dashboards.py                    # datasets, charts, dashboards, embedding
D:\blockchainbigdata\tools\pwvenv\Scripts\python.exe scripts\bi\save_query_contexts.py   # see below
D:\blockchainbigdata\tools\pwvenv\Scripts\python.exe scripts\bi\verify_embeds.py --out <dir>   # render check + screenshots
```

First-time setup: `wsl -d Ubuntu -- bash /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/scripts/bi/setup_superset.sh`
(metadata DB, admin user, DuckDB connection, read-only `EmbedViewer` role, `embed_service` user; passwords in `~/bi/*.secret`).

## Embedding and security

* The browser asks `POST /api/bi/guest-token?dashboard=<key>&run=<run>`; the API logs in as `embed_service` and returns
  only a short-lived Superset guest token. No Superset credential reaches the browser.
* Streaming datasets get a row-level-security clause `run_id = '<run>'`, so embedded charts follow the console's run
  selector. The run name must match `run_YYYYMMDD_HHMMSS` (anything else → HTTP 400).
* Superset rejects guest queries that differ from a chart's stored *query context* ("Guest user cannot modify chart
  payload"). Charts created through the MCP/REST API have none, so `save_query_contexts.py` opens each chart in
  Superset's Explore view with the local Chrome (Playwright) and saves it, letting Superset's own frontend store the
  exact query it sends. The security check is not disabled.
* Superset and the MCP service bind locally; `MCP_AUTH_ENABLED = False` is a development setting for this single-user
  workstation only.

## Known limitations / workarounds

| Issue | Handling |
|---|---|
| Superset 6.1 MCP `generate_chart` supports 7 chart families | other types via REST (`rest_chart` in the builder) |
| MCP stores raw-table sort columns in a format the table plugin rejects ("Found invalid orderby options") | `fix_table_sorting()` rewrites them |
| DuckDB engine has no 5-second time grain | 5-second buckets computed in SQL (`time_bucket`), grain = 1 s |
| Unpinned dependency resolution broke Superset (flask-caching 2.5) | install against Superset's own pins |
| Default metastore caches fail with the resolved flask-caching | in-memory caches in `superset_config.py` |
| Line charts cannot log-scale the x axis | log-x charts use the bubble type |
