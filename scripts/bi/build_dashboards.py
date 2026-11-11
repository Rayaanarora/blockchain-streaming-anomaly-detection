"""
Build the five BSAD Superset dashboards (idempotent: removes the previous BSAD objects first).

How the work is split:
  * Superset's official MCP service (http://127.0.0.1:5008/mcp): virtual datasets
    (create_virtual_dataset), every chart family it can generate (generate_chart: xy line/bar/area/
    scatter, pie/donut, big number, table) and the dashboards themselves (generate_dashboard).
  * Superset's REST API (/api/v1): chart types the MCP chart generator does not support
    (heatmap, Sankey, radar, box plot, treemap, sunburst, funnel, gauge, bubble, histogram),
    the dashboard grid layout, colours, auto-refresh and the embedding configuration.

Data: DuckDB views over Parquet on D: (gold_* from scripts/build_bi_gold.py, stream_* read live).

    .venv\\Scripts\\python.exe scripts\\bi\\build_dashboards.py
Writes data/bi/embed.json (dashboard -> embedded UUID) for the FastAPI dashboard.
"""
import json
import os
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from mcp_call import Client  # noqa: E402
from chart_labels import L as LABELS  # noqa: E402

SUPERSET = "http://127.0.0.1:8088"
PREFIX = "bsad_"
EMBED_DOMAINS = ["http://127.0.0.1:8000", "http://localhost:8000"]
LABEL_COLORS = {
    "illicit": "#e66767", "licit": "#8a93a0", "unknown": "#3a414c",
    "TP": "#199e70", "TN": "#4b5563", "FP": "#c98500", "FN": "#e66767",
    "elliptic": "#3987e5", "zenodo": "#d95926", "flagged": "#e66767", "not flagged": "#4b5563", "not flagged (0.3 % sample)": "#6f7884",
    "TGAT-BoostNet (ours)": "#3987e5", "TGAT-BoostNet (tuned, ours)": "#3987e5", "TGAT-BoostNet (a-priori)": "#3987e5",
    "4.0.0 TGAT-BoostNet (served)": "#3987e5", "3.0.0 GAT-BoostNet (previous)": "#8a93a0",
    "tuned": "#3987e5", "untuned": "#8a93a0", "precision": "#199e70", "recall": "#c98500", "f1": "#3987e5",
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------ REST client
class Rest:
    def __init__(self):
        pw = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "cat", "/home/katherine/bi/admin.secret"],
                            capture_output=True, text=True, check=True).stdout.strip()
        self.s = requests.Session()
        tok = self.s.post(f"{SUPERSET}/api/v1/security/login",
                          json={"username": "admin", "password": pw, "provider": "db", "refresh": True}).json()
        self.s.headers["Authorization"] = f"Bearer {tok['access_token']}"
        csrf = self.s.get(f"{SUPERSET}/api/v1/security/csrf_token/").json()["result"]
        self.s.headers.update({"X-CSRFToken": csrf, "Referer": SUPERSET})

    def _req(self, method, path, **kw):
        # Superset (single gunicorn worker) occasionally drops a connection under load: retry a few times
        for attempt in range(4):
            try:
                return self.s.request(method, SUPERSET + path, timeout=120, **kw)
            except requests.ConnectionError:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))

    def get(self, path, **kw):
        r = self._req("GET", path, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path, body):
        r = self._req("POST", path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:800]}")
        return r.json()

    def put(self, path, body):
        r = self._req("PUT", path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"PUT {path} -> {r.status_code}: {r.text[:800]}")
        return r.json()

    def delete(self, path):
        self._req("DELETE", path)

    def all(self, kind):
        q = json.dumps({"page_size": 100, "page": 0}).replace('"', "'")
        out, page = [], 0
        while True:
            q = "(page:%d,page_size:100)" % page
            res = self.get(f"/api/v1/{kind}/?q={q}")["result"]
            out += res
            if len(res) < 100:
                return out
            page += 1


# ------------------------------------------------------------------------------ helpers
class Builder:
    def __init__(self):
        self.mcp = Client()
        self.rest = Rest()
        self.db_id = self.tool("list_databases", {})["databases"][0]["id"]
        self.datasets = {}
        self.charts = []  # (dashboard_key, chart_id, width, height, title)
        self.n_mcp = 0

    def tool(self, name, req):
        res, err = self.mcp.call("call_tool", {"name": name, "arguments": {"request": req}})
        if err or (isinstance(res, str) and res.startswith("Error")):
            raise RuntimeError(f"MCP {name}: {str(res)[:800]}")
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(f"MCP {name}: {res['error']}")
        return res

    def clean(self):
        for d in self.rest.all("dashboard"):
            if d["dashboard_title"].startswith("BSAD"):
                self.rest.delete(f"/api/v1/dashboard/{d['id']}")
        for c in self.rest.all("chart"):
            self.rest.delete(f"/api/v1/chart/{c['id']}")
        for d in self.rest.all("dataset"):
            if d["table_name"].startswith(PREFIX):
                self.rest.delete(f"/api/v1/dataset/{d['id']}")

    def ds(self, name, sql, desc=""):
        if name not in self.datasets:
            r = self.tool("create_virtual_dataset", {"database_id": self.db_id, "dataset_name": PREFIX + name,
                                                      "sql": sql, "description": desc})
            self.datasets[name] = r["id"]
        return self.datasets[name]

    def mcp_chart(self, dash, ds, title, config, w=4, h=50):
        r = self.tool("generate_chart", {"dataset_id": self.datasets[ds], "save_chart": True, "chart_name": title,
                                         "config": config, "generate_preview": False})
        cid = r["chart"]["id"]
        self.n_mcp += 1
        self.charts.append((dash, cid, w, h, title))
        log(f"  [MCP ] {title} ({config['chart_type']}{'/' + config.get('kind', '') if config.get('kind') else ''})")
        return cid

    def rest_chart(self, dash, ds, title, viz, form, w=4, h=50, desc=None):
        form = dict(form, viz_type=viz, datasource=f"{self.datasets[ds]}__table")
        r = self.rest.post("/api/v1/chart/", {"slice_name": title, "viz_type": viz, "datasource_id": self.datasets[ds],
                                              "datasource_type": "table", "params": json.dumps(form),
                                              "description": desc})
        cid = r["id"]
        self.charts.append((dash, cid, w, h, title))
        log(f"  [REST] {title} ({viz})")
        return cid


def M(sql, label):
    return {"expressionType": "SQL", "sqlExpression": sql, "label": label}


def col(name, agg=None, label=None):
    d = {"name": name}
    if agg:
        d["aggregate"] = agg
    if label:
        d["label"] = label
    return d


# ------------------------------------------------------------------------------ datasets
DATASETS = {
    # live streaming (read straight from the Spark consumer's output on every query)
    "stream": ("SELECT run_id, source_dataset, status, is_flagged, CAST(is_flagged AS INT) AS flagged_int, "
               "CASE WHEN is_flagged THEN 'flagged' ELSE 'not flagged' END AS flag_label, score, threshold, "
               "e2e_latency_ms, CAST(kafka_partition AS INT) AS kafka_partition, batch_id, time_step, scored_ts, "
               "model_version, transaction_id, event_id, reason_codes, time_bucket(INTERVAL 5 SECOND, scored_ts) AS scored_5s "
               "FROM stream_predictions WHERE run_id LIKE 'run_%'", "Every scored event of every pipeline run"),
    "stream_batches": ("SELECT * FROM stream_batches WHERE run_id LIKE 'run_%' AND input_rows > 0", "Spark micro-batches"),
    "stream_flow": ("SELECT run_id, source_dataset AS src, status AS tgt, count(*) AS n FROM stream_predictions "
                    "WHERE run_id LIKE 'run_%' GROUP BY ALL UNION ALL "
                    "SELECT run_id, status AS src, CASE WHEN is_flagged THEN 'flagged' ELSE 'not flagged' END AS tgt, count(*) "
                    "FROM stream_predictions WHERE run_id LIKE 'run_%' GROUP BY ALL", "dataset -> status -> decision"),
    "stream_labelled": ("SELECT run_id, 'step ' || CAST(time_step AS VARCHAR) AS step_label, time_step, class, outcome, "
                        "score, is_flagged FROM stream_elliptic_labelled WHERE run_id LIKE 'run_%' AND class IN ('illicit', 'licit')",
                        "Live Elliptic predictions joined to ground truth (labels never reach the scorer)"),
    "stream_runs": ("SELECT * FROM stream_runs WHERE run_id LIKE 'run_%'", "One row per pipeline run"),
    # served model (bundle 4.0.0)
    "served_confusion": ("SELECT * FROM gold_served_confusion", ""),
    "served_radar": ("SELECT bundle, max(value) FILTER (WHERE metric='Precision') AS precision, "
                     "max(value) FILTER (WHERE metric='Recall') AS recall, max(value) FILTER (WHERE metric='F1') AS f1, "
                     "max(value) FILTER (WHERE metric='ROC-AUC') AS roc_auc, max(value) FILTER (WHERE metric='PR-AUC') AS pr_auc "
                     "FROM gold_served_vs_previous GROUP BY bundle", ""),
    "served_hist": ("SELECT round(bin_mid, 3) AS bin_mid, class, n, n / sum(n) OVER (PARTITION BY class) AS share FROM gold_served_score_hist", ""),
    "served_per_step": ("SELECT *, CAST(time_step AS VARCHAR) AS step FROM gold_served_per_step", ""),
    "served_sweep": ("SELECT * FROM gold_served_threshold_sweep", ""),
    "served_funnel": ("SELECT * FROM (VALUES (1,'Test transactions (steps 35-49)', (SELECT sum(n) FROM gold_served_confusion)), "
                      "(2,'Actually illicit', (SELECT sum(n) FROM gold_served_confusion WHERE actual='illicit')), "
                      "(3,'Caught (true positives)', (SELECT n FROM gold_served_confusion WHERE cell='TP'))) t(ord, stage, n)", ""),
    "served_checks": ("SELECT \"check\" AS export_check, CASE WHEN passed THEN 'passed' ELSE 'FAILED' END AS result FROM gold_served_checks", ""),
    # model comparison (notebook 03/04 reproduced results; ours = a-priori config unless marked tuned)
    "models": ("SELECT replace(m.model, 'TGAT-BoostNet (ours)', 'TGAT-BoostNet (a-priori)') AS model, m.family, m.f1, m.pr_auc, "
               "m.roc_auc, m.precision, m.recall, m.mcc, m.balanced_accuracy, e.latency_ms, e.throughput_tx_s, e.size_kb, "
               "e.train_s FROM gold_model_metrics_wide m JOIN gold_model_efficiency e USING (model)", ""),
    "per_step": ("SELECT replace(model, 'TGAT-BoostNet (ours)', 'TGAT-BoostNet (a-priori)') AS model, time_step, "
                 "'step ' || lpad(CAST(time_step AS VARCHAR), 2, '0') AS step, f1, family, period FROM gold_per_step_f1_long", ""),
    "curves_pr": ("SELECT * FROM gold_curve_pr", ""),
    "curves_roc": ("SELECT * FROM gold_curve_roc", ""),
    "calibration": ("SELECT * FROM gold_curve_calibration", ""),
    "tuned": ("SELECT model_family, kind, f1_illicit, pr_auc_ap, mcc FROM gold_tuned_vs_untuned", ""),
    "significance": ("SELECT baseline, f1 AS delta_f1, ci_lo, ci_hi, p_f1_boot_holm, verdict FROM gold_significance_ci", ""),
    "ablation": ("SELECT item AS variant, f1, f1_vs_full AS delta_f1 FROM gold_t03_ablation", ""),
    "reported": ("SELECT model, source, reported_f1_paper, reproduced_f1_ours_mean_sd_std AS reproduced_f1 "
                 "FROM gold_t03_reproduced_vs_reported", ""),
    "multi_metric": ("SELECT replace(model, 'TGAT-BoostNet (ours)', 'TGAT-BoostNet (a-priori)') AS model, precision, recall, f1, "
                     "mcc, roc_auc, pr_auc FROM gold_model_metrics_wide WHERE model IN "
                     "('TGAT-BoostNet (ours)','LightGBM','CatBoost','Random Forest','GCN + XGBoost','GAT (alone)')", ""),
    # data & preprocessing
    "class_total": ("SELECT class, sum(n) AS n FROM gold_ell_class_by_step GROUP BY class", ""),
    "class_step": ("SELECT time_step, class, sum(n) AS n FROM gold_ell_class_by_step GROUP BY ALL", ""),
    "prep_funnel": ("SELECT * FROM (VALUES (1,'All transactions', (SELECT sum(n) FROM gold_ell_class_by_step)), "
                    "(2,'Labelled (unknown excluded from training)', (SELECT sum(n) FROM gold_ell_class_by_step WHERE class<>'unknown')), "
                    "(3,'Labelled, train steps 1-34', (SELECT sum(n) FROM gold_ell_class_by_step WHERE class<>'unknown' AND time_step<=34)), "
                    "(4,'Illicit, train steps 1-34', (SELECT sum(n) FROM gold_ell_class_by_step WHERE class='illicit' AND time_step<=34))) "
                    "t(ord, stage, n)", ""),
    "graph_step": ("SELECT * FROM gold_ell_graph_by_step", ""),
    "degree": ("SELECT * FROM gold_ell_degree_dist", ""),
    "feature_profile": ("SELECT class, feature, value FROM gold_ell_feature_profile UNPIVOT (value FOR feature IN (f2, f3, f4, f5, f6, f7, f8, f9))", ""),
    "feature_sample": ("SELECT row_number() OVER () AS rid, * FROM gold_ell_feature_sample", ""),
    "spark_scaling": ("SELECT max_step, n_nodes, n_edges, wall_seconds, tx_per_s FROM gold_t01_spark_scaling", ""),
    "shap": ("SELECT item AS feature, mean_shap FROM gold_t02_shap_top30 ORDER BY mean_shap DESC LIMIT 20", ""),
    "step43": ("SELECT item AS metric, 'steps 35-42' AS period, CAST(steps_35_42 AS DOUBLE) AS value FROM gold_t02_before_after_step43 WHERE item IS NOT NULL "
               "UNION ALL SELECT item, 'steps 43-49', CAST(steps_43_49 AS DOUBLE) FROM gold_t02_before_after_step43 WHERE item IS NOT NULL", ""),
    "learning": ("SELECT * FROM gold_t02_learning_curve", ""),
    # Zenodo (16.9 M transactions, aggregated by Spark)
    "zen_status": ("SELECT * FROM gold_zen_status", ""),
    "zen_hist": ("SELECT * FROM gold_zen_score_hist", ""),
    "zen_chunk": ("SELECT * FROM gold_zen_by_chunk", ""),
    "zen_bucket": ("SELECT * FROM gold_zen_by_block_bucket", ""),
    "zen_reasons": ("SELECT feature, direction, feature || ' (' || direction || ')' AS reason, n, mean_abs_z FROM gold_zen_reasons", ""),
    "zen_decile": ("SELECT 'decile ' || CAST(score_decile AS VARCHAR) AS decile, * FROM gold_zen_decile_profile", ""),
    "zen_sample": ("SELECT row_number() OVER () AS rid, block_chunk, score, flagged, "
                   "CASE WHEN flagged THEN 'flagged' ELSE 'not flagged' END AS flag_label, log_inputs, log_outputs, "
                   "log_output_btc, fee_rate, new_address_fraction FROM gold_zen_sample "
                   # unflagged rows are a 0.3 % Spark sample; take a matching 0.3 % of flagged rows so the mix is unbiased
                   "WHERE NOT flagged OR (hash(block_number, tx_index) % 1000) < 3", "~0.3 % uniform sample of scored Zenodo tx"),
    "zen_flagged_sample": ("SELECT row_number() OVER () AS rid, CASE WHEN flagged THEN 'flagged' ELSE 'not flagged (0.3 % sample)' END AS flag_label, "
                           "log_inputs, log_outputs, score FROM gold_zen_sample", "All flagged tx + 0.3 % of the rest"),
    "zen_flow": ("SELECT dq_status AS src, score_status AS tgt, sum(n) AS n FROM gold_zen_status GROUP BY ALL "
                 "UNION ALL SELECT 'scored', 'flagged', (SELECT sum(n_flagged) FROM gold_zen_by_chunk) "
                 "UNION ALL SELECT 'scored', 'not flagged', (SELECT sum(n) - sum(n_flagged) FROM gold_zen_by_chunk)", ""),
    "zen_totals": ("SELECT sum(n) AS scored, sum(n_flagged) AS flagged, 1000.0 * sum(n_flagged) / sum(n) AS per_mille FROM gold_zen_by_chunk", ""),
}

DASHBOARDS = {
    "live": ("BSAD · Live pipeline", "Kafka → Spark Structured Streaming → scoring, read live from the run's Parquet output", 10),
    "detect": ("BSAD · Detection quality", "Served model TGAT-BoostNet 4.0.0 on Elliptic test steps 35-49, plus live outcomes", 0),
    "models": ("BSAD · Model comparison", "Reproduced results, 5 seeds; train 1-34 / test 35-49", 0),
    "data": ("BSAD · Data & preprocessing", "Elliptic through Spark: classes, graph structure, features", 0),
    "zenodo": ("BSAD · Zenodo anomalies", "Isolation Forest over 16.9 M transactions (unsupervised: 'unusual', not 'illicit')", 0),
}


def build(b: Builder):
    for name, (sql, desc) in DATASETS.items():
        b.ds(name, sql, desc)
    log(f"{len(b.datasets)} datasets via MCP")

    # ======================= LIVE PIPELINE
    D = "live"
    b.mcp_chart(D, "stream", "Events scored (run total)", {"chart_type": "big_number", "metric": col("event_id", "COUNT"),
                "subheader": "events scored in the selected run"}, w=3, h=36)
    b.mcp_chart(D, "stream", "Flagged (run total)", {"chart_type": "big_number", "metric": col("flagged_int", "SUM"),
                "subheader": "events with score ≥ model threshold"}, w=3, h=36)
    b.mcp_chart(D, "stream_batches", "p95 end-to-end latency (ms)", {"chart_type": "big_number",
                "metric": col("e2e_latency_ms_p95", "MAX"), "subheader": "worst micro-batch p95, producer → Parquet",
                "y_axis_format": ",.0f"}, w=3, h=36)
    b.mcp_chart(D, "stream_batches", "Dead-lettered", {"chart_type": "big_number", "metric": col("dead_lettered", "SUM"),
                "subheader": "malformed events routed to the DLQ"}, w=3, h=36)
    b.mcp_chart(D, "stream", "Throughput by dataset (events per 5 s)", {"chart_type": "xy", "kind": "area", "stacked": True,
                "x": col("scored_5s"), "time_grain": "PT1S", "y": [col("event_id", "COUNT", "events")],
                "group_by": [col("source_dataset")], "y_axis": {"title": "events / 5 s"}}, w=8, h=56)
    b.rest_chart(D, "stream_flow", "Routing: dataset → status → decision", "sankey_v2",
                 {"source": "src", "target": "tgt", "metric": M("SUM(n)", "events"), "row_limit": 1000}, w=4, h=56)
    b.mcp_chart(D, "stream_batches", "Micro-batch latency p50 / p95 (ms)", {"chart_type": "xy", "kind": "line",
                "x": col("processing_ts"), "time_grain": "PT1S",
                "y": [col("e2e_latency_ms_p50", "MAX", "p50"), col("e2e_latency_ms_p95", "MAX", "p95")],
                "y_axis": {"title": "ms"}}, w=5, h=50)
    b.rest_chart(D, "stream", "End-to-end latency by dataset (per micro-batch)", "box_plot",
                 {"columns": ["batch_id"], "groupby": ["source_dataset"], "metrics": [M("AVG(e2e_latency_ms)", "latency ms")],
                  "whiskerOptions": "Tukey", "number_format": ",.0f", "row_limit": 50000}, w=4, h=50)
    b.rest_chart(D, "stream", "Events per Kafka partition (rows) × micro-batch (columns)", "heatmap_v2",
                 {"x_axis": "batch_id", "groupby": "kafka_partition", "metric": M("COUNT(*)", "events"),
                  "linear_color_scheme": "bsad_blue", "normalize_across": "heatmap", "show_values": False,
                  "show_legend": True, "sort_x_axis": "alpha_asc", "sort_y_axis": "alpha_asc", "xscale_interval": -1,
                  "yscale_interval": 1, "row_limit": 10000, "adhoc_filters": [
                      {"expressionType": "SQL", "sqlExpression": "kafka_partition IS NOT NULL", "clause": "WHERE"}]},
                 w=3, h=50)
    b.mcp_chart(D, "stream_batches", "Rows written per micro-batch", {"chart_type": "xy", "kind": "bar",
                "x": col("batch_id"), "y": [col("results_written", "SUM", "results written"),
                                            col("buffered_for_graph_context", "SUM", "buffered for snapshot")],
                "stacked": False, "y_axis": {"title": "rows"}}, w=6, h=50)
    b.rest_chart(D, "stream", "Live Elliptic scores: flagged vs not (threshold 0.37)", "histogram_v2",
                 {"column": "score", "groupby": ["flag_label"], "bins": 40, "normalize": False, "cumulative": False,
                  "x_axis_title": "illicit probability", "y_axis_title": "events", "show_legend": True, "row_limit": 100000,
                  "adhoc_filters": [{"expressionType": "SQL", "sqlExpression": "source_dataset = 'elliptic' AND score IS NOT NULL",
                                     "clause": "WHERE"}]}, w=6, h=50)
    b.mcp_chart(D, "stream", "Latest flagged events", {"chart_type": "table", "query_mode": "raw",
                "columns": [col("scored_ts"), col("source_dataset"), col("transaction_id"), col("score"), col("threshold"),
                            col("time_step"), col("reason_codes")],
                "filters": [{"column": "is_flagged", "op": "=", "value": True}],
                "sort_by": ["scored_ts"], "row_limit": 200}, w=8, h=60)
    b.mcp_chart(D, "stream_runs", "Pipeline runs", {"chart_type": "table", "query_mode": "raw",
                "columns": [col("run_id"), col("events"), col("flagged"), col("elliptic_events"), col("zenodo_events"),
                            col("elliptic_model")], "row_limit": 50}, w=4, h=60)

    # ======================= DETECTION QUALITY (served bundle 4.0.0)
    D = "detect"
    b.rest_chart(D, "served_confusion", "Confusion matrix: actual (rows) × predicted (columns), test steps 35-49", "heatmap_v2",
                 {"x_axis": "predicted", "groupby": "actual", "metric": M("SUM(n)", "transactions"),
                  "linear_color_scheme": "bsad_blue", "normalize_across": "heatmap", "show_values": True,
                  "show_percentage": True, "show_legend": False, "sort_x_axis": "alpha_asc", "sort_y_axis": "alpha_asc",
                  "xscale_interval": 1, "yscale_interval": 1, "value_bounds": [None, None]}, w=4, h=52)
    b.rest_chart(D, "served_radar", "Served 4.0.0 vs previous 3.0.0", "radar",
                 {"groupby": ["bundle"], "metrics": [M("MAX(precision)", "Precision"), M("MAX(recall)", "Recall"),
                                                      M("MAX(f1)", "F1"), M("MAX(roc_auc)", "ROC-AUC"), M("MAX(pr_auc)", "PR-AUC")],
                  "column_config": {k: {"radarMetricMaxValue": 1} for k in ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]},
                  "show_legend": True, "legendOrientation": "bottom", "show_labels": True, "label_type": "value",
                  "number_format": ".3f", "is_circle": False, "row_limit": 10}, w=4, h=52)
    b.rest_chart(D, "served_radar", "Served 4.0.0 test F1 (illicit, steps 35-49)", "gauge_chart",
                 {"metric": M("MAX(f1) FILTER (WHERE bundle LIKE '4.0.0%')", "F1"), "min_val": 0, "max_val": 1,
                  "start_angle": 225, "end_angle": -45, "number_format": ".3f", "show_progress": True, "overlap": True,
                  "round_cap": True, "split_number": 5, "show_pointer": False, "animation": True}, w=4, h=52)
    b.mcp_chart(D, "served_hist", "Score distribution by true class (share of each class per bin)", {"chart_type": "xy",
                "kind": "bar", "x": col("bin_mid"), "y": [col("share", "SUM", "share of class")], "group_by": [col("class")],
                "stacked": False}, w=6, h=52)
    b.mcp_chart(D, "served_sweep", "Threshold sweep (served model, test steps)", {"chart_type": "xy", "kind": "line",
                "x": col("threshold"), "y": [col("value", "MAX", "value")], "group_by": [col("metric")],
                "x_axis": {"title": "decision threshold"}, "y_axis": {"title": "score"}}, w=6, h=52)
    b.mcp_chart(D, "served_per_step", "Precision / recall / F1 per test step", {"chart_type": "xy", "kind": "line",
                "x": col("step"), "y": [col("precision", "MAX", "precision"), col("recall", "MAX", "recall"),
                                              col("f1", "MAX", "f1")], "x_axis": {"title": "time step"}}, w=8, h=52)
    b.rest_chart(D, "served_funnel", "From test set to caught", "funnel",
                 {"groupby": ["stage"], "metric": M("MAX(n)", "transactions"), "sort_by_metric": True, "show_labels": True,
                  "label_type": "key_value_percent", "number_format": "SMART_NUMBER", "show_legend": False,
                  "percent_calculation_type": "first_step", "row_limit": 10}, w=4, h=52)
    b.rest_chart(D, "stream_labelled", "Live outcomes by time step (stream ⨝ labels)", "sunburst_v2",
                 {"columns": ["step_label", "outcome"], "metric": M("COUNT(*)", "events"), "show_labels": True,
                  "label_type": "key", "number_format": "SMART_NUMBER", "show_total": True, "row_limit": 1000}, w=4, h=56)
    b.mcp_chart(D, "stream_labelled", "Live outcome share", {"chart_type": "pie", "dimension": col("outcome"),
                "metric": col("score", "COUNT", "events"), "donut": True, "label_type": "key_percent", "show_total": True},
                w=4, h=56)
    b.rest_chart(D, "stream_labelled", "Live score distribution by true class", "histogram_v2",
                 {"column": "score", "groupby": ["class"], "bins": 40, "normalize": True, "cumulative": False,
                  "x_axis_title": "illicit probability", "y_axis_title": "share", "show_legend": True,
                  "row_limit": 100000}, w=4, h=56)
    b.mcp_chart(D, "served_per_step", "Illicit transactions per test step: caught vs missed", {"chart_type": "xy", "kind": "bar",
                "x": col("time_step"), "y": [col("tp", "SUM", "caught (TP)"), col("fn", "SUM", "missed (FN)")],
                "stacked": True, "x_axis": {"title": "time step"}, "y_axis": {"title": "illicit transactions"}}, w=8, h=50)
    b.mcp_chart(D, "served_checks", "Export checks (bundle 4.0.0)", {"chart_type": "table", "query_mode": "raw",
                "columns": [col("export_check"), col("result")], "row_limit": 20}, w=4, h=50)

    # ======================= MODEL COMPARISON
    D = "models"
    b.rest_chart(D, "models", "F1 vs PR-AUC, untuned, 5-seed mean (bubble = throughput tx/s)", "bubble_v2",
                 {"entity": "model", "series": "family", "x": M("MAX(pr_auc)", "PR-AUC"), "y": M("MAX(f1)", "F1"),
                  "size": M("MAX(throughput_tx_s)", "throughput"), "max_bubble_size": "25", "x_axis_label": "PR-AUC",
                  "y_axis_label": "F1 (illicit)", "xAxisFormat": ".2f", "yAxisFormat": ".2f", "truncateXAxis": True,
                  "truncateYAxis": True, "show_legend": True, "legendOrientation": "bottom", "opacity": 0.75,
                  "row_limit": 100}, w=6, h=60)
    b.mcp_chart(D, "models", "Illicit F1 leaderboard, untuned (notebook 03, 5-seed mean)", {"chart_type": "xy", "kind": "bar",
                "orientation": "horizontal", "x": col("model"), "y": [col("f1", "MAX", "F1")], "x_axis": {"title": ""},
                "y_axis": {"title": "F1 (illicit)"}, "legend": {"show": False}}, w=6, h=60)
    b.rest_chart(D, "per_step", "F1 per model (rows) × test step (columns), untuned, 5-seed mean", "heatmap_v2",
                 {"x_axis": "step", "groupby": "model", "metric": M("MAX(f1)", "F1"), "linear_color_scheme": "bsad_blue",
                  "normalize_across": "heatmap", "show_values": False, "show_legend": True, "sort_x_axis": "alpha_asc",
                  "sort_y_axis": "value_desc", "xscale_interval": 1, "yscale_interval": 1, "row_limit": 10000,
                  "value_bounds": [0, 1]}, w=12, h=64)
    b.mcp_chart(D, "curves_pr", "Precision-recall curves (seed 42)", {"chart_type": "xy", "kind": "line",
                "x": col("recall"), "y": [col("precision", "MAX", "precision")], "group_by": [col("model")],
                "x_axis": {"title": "recall"}, "y_axis": {"title": "precision"}, "row_limit": 10000}, w=6, h=60)
    b.mcp_chart(D, "curves_roc", "ROC curves (seed 42)", {"chart_type": "xy", "kind": "line",
                "x": col("fpr"), "y": [col("tpr", "MAX", "TPR")], "group_by": [col("model")],
                "x_axis": {"title": "false-positive rate"}, "y_axis": {"title": "true-positive rate"}, "row_limit": 10000},
                w=6, h=60)
    b.rest_chart(D, "multi_metric", "Six metrics, six models (untuned, 5-seed mean)", "radar",
                 {"groupby": ["model"], "metrics": [M("MAX(precision)", "Precision"), M("MAX(recall)", "Recall"),
                                                     M("MAX(f1)", "F1"), M("MAX(mcc)", "MCC"), M("MAX(roc_auc)", "ROC-AUC"),
                                                     M("MAX(pr_auc)", "PR-AUC")],
                  "column_config": {k: {"radarMetricMaxValue": 1} for k in ["Precision", "Recall", "F1", "MCC", "ROC-AUC", "PR-AUC"]},
                  "show_legend": True, "legendOrientation": "bottom", "number_format": ".2f", "row_limit": 20}, w=4, h=60)
    b.rest_chart(D, "per_step", "Spread of per-step F1, steps 35-49 (untuned)", "box_plot",
                 {"columns": ["time_step"], "groupby": ["model"], "metrics": [M("MAX(f1)", "F1 per step")],
                  "whiskerOptions": "Min/max (no outliers)", "number_format": ".2f", "row_limit": 10000}, w=8, h=60)
    b.mcp_chart(D, "tuned", "Tuning effect: untuned vs tuned F1 (notebook 04, 5 seeds)", {"chart_type": "xy", "kind": "bar",
                "x": col("model_family"), "y": [col("f1_illicit", "MAX", "F1")], "group_by": [col("kind")],
                "y_axis": {"title": "F1 (illicit)"}}, w=4, h=52)
    b.mcp_chart(D, "calibration", "Calibration, seed 42 (predicted vs observed illicit rate)", {"chart_type": "xy", "kind": "line",
                "x": col("bin_centre"), "y": [col("observed_illicit_rate", "MAX", "observed")],
                "group_by": [col("model")], "x_axis": {"title": "mean predicted probability"},
                "y_axis": {"title": "observed illicit rate"}}, w=4, h=52)
    b.rest_chart(D, "models", "F1 vs inference latency (median ms per test snapshot, log)", "bubble_v2",
                 {"entity": "model", "series": "family", "x": M("MAX(latency_ms)", "latency ms"), "y": M("MAX(f1)", "F1"),
                  "size": M("MAX(size_kb)", "model size kB"), "max_bubble_size": "12", "xAxisFormat": ",.0f",
                  "yAxisFormat": ".2f", "truncateXAxis": True, "truncateYAxis": True, "show_legend": True,
                  "legendOrientation": "bottom", "opacity": 0.8, "row_limit": 100}, w=4, h=52)
    b.mcp_chart(D, "significance", "ΔF1 of TGAT-BoostNet (a-priori) vs each baseline (bootstrap)", {"chart_type": "xy",
                "kind": "bar", "orientation": "horizontal", "x": col("baseline"), "y": [col("delta_f1", "MAX", "ΔF1")],
                "y_axis": {"title": "ΔF1 (negative = baseline better)"}, "legend": {"show": False}}, w=6, h=56)
    b.mcp_chart(D, "ablation", "Ablations: ΔF1 vs the a-priori configuration", {"chart_type": "xy", "kind": "bar",
                "orientation": "horizontal", "x": col("variant"), "y": [col("delta_f1", "MAX", "ΔF1")],
                "y_axis": {"title": "ΔF1 (positive = variant scores higher)"}, "legend": {"show": False}}, w=6, h=56)
    b.rest_chart(D, "models", "Model size by family (kB)", "treemap_v2",
                 {"groupby": ["family", "model"], "metric": M("MAX(size_kb)", "size kB"), "show_labels": True,
                  "show_upper_labels": True, "label_type": "key_value", "number_format": ",.0f", "row_limit": 100}, w=6, h=56)
    b.mcp_chart(D, "reported", "Reproduced vs reported in paper", {"chart_type": "table", "query_mode": "raw",
                "columns": [col("model"), col("source"), col("reported_f1_paper"), col("reproduced_f1")], "row_limit": 50},
                w=6, h=56)

    # ======================= DATA & PREPROCESSING
    D = "data"
    b.mcp_chart(D, "class_total", "Elliptic class balance (203,769 tx)", {"chart_type": "pie", "dimension": col("class"),
                "metric": col("n", "SUM", "transactions"), "donut": True, "label_type": "key_percent", "show_total": True},
                w=4, h=52)
    b.rest_chart(D, "prep_funnel", "From raw graph to training labels", "funnel",
                 {"groupby": ["stage"], "metric": M("MAX(n)", "transactions"), "sort_by_metric": True, "show_labels": True,
                  "label_type": "key_value", "number_format": ",d", "show_legend": False, "row_limit": 10}, w=4, h=52)
    b.mcp_chart(D, "feature_profile", "Mean of features f2-f9: illicit vs licit", {"chart_type": "xy", "kind": "bar",
                "x": col("feature"), "y": [col("value", "MAX", "class mean")], "group_by": [col("class")]}, w=4, h=52)
    b.mcp_chart(D, "class_step", "Class composition over the 49 time steps", {"chart_type": "xy", "kind": "area",
                "stacked": True, "x": col("time_step"), "y": [col("n", "SUM", "transactions")], "group_by": [col("class")],
                "x_axis": {"title": "time step (train 1-34 | test 35-49)"}}, w=8, h=52)
    b.mcp_chart(D, "graph_step", "Illicit share of labelled tx per step", {"chart_type": "xy", "kind": "line",
                "x": col("time_step"), "y": [col("illicit_share", "MAX", "illicit share")],
                "x_axis": {"title": "time step"}, "y_axis": {"title": "share", "format": ".0%"}}, w=4, h=52)
    b.mcp_chart(D, "graph_step", "Elliptic graph size per time step (nodes, edges)", {"chart_type": "xy", "kind": "bar",
                "x": col("time_step"), "y": [col("n_nodes", "MAX", "nodes"), col("n_edges", "MAX", "edges")],
                "x_axis": {"title": "time step"}}, w=6, h=50)
    b.mcp_chart(D, "degree", "Degree distribution by class", {"chart_type": "xy", "kind": "bar", "x": col("degree_bucket"),
                "y": [col("n", "SUM", "transactions")], "group_by": [col("class")], "y_axis": {"title": "transactions", "scale": "log"},
                "x_axis": {"title": "degree (in + out)"}}, w=6, h=50)
    b.rest_chart(D, "feature_sample", "Feature space f2 × f3 (15 % labelled sample, size = degree)", "bubble_v2",
                 {"entity": "rid", "series": "class", "x": M("MAX(f2)", "f2"), "y": M("MAX(f3)", "f3"),
                  "size": M("MAX(degree) + 1", "degree"), "max_bubble_size": "8", "x_axis_label": "f2 (standardised)",
                  "y_axis_label": "f3 (standardised)", "show_legend": True, "legendOrientation": "bottom", "opacity": 0.5,
                  "row_limit": 10000}, w=6, h=56)
    b.mcp_chart(D, "shap", "Top-20 features by mean |SHAP| (TGAT-BoostNet a-priori, seed 42)", {"chart_type": "xy", "kind": "bar", "orientation": "horizontal",
                "x": col("feature"), "y": [col("mean_shap", "MAX", "mean |SHAP|")], "legend": {"show": False}}, w=6, h=72)
    b.mcp_chart(D, "spark_scaling", "Spark Elliptic preprocessing: throughput vs input size", {"chart_type": "xy", "kind": "line",
                "x": col("n_nodes"), "y": [col("tx_per_s", "MAX", "tx / s")], "x_axis": {"title": "transactions processed"},
                "y_axis": {"title": "tx per second"}}, w=4, h=50)
    b.mcp_chart(D, "step43", "TGAT-BoostNet (a-priori) before vs after the step-43 shift", {"chart_type": "xy", "kind": "bar", "x": col("metric"),
                "y": [col("value", "MAX", "value")], "group_by": [col("period")], "y_axis": {"title": "mean over seeds"}},
                w=4, h=50)
    b.mcp_chart(D, "learning", "Learning curve, TGAT-BoostNet a-priori (test F1, 5-seed mean)", {"chart_type": "xy", "kind": "line",
                "x": col("last_train_step"), "y": [col("f1", "MAX", "F1"), col("pr_auc", "MAX", "PR-AUC")],
                "x_axis": {"title": "last training step"}}, w=4, h=50)

    # ======================= ZENODO
    D = "zenodo"
    b.mcp_chart(D, "zen_totals", "Transactions scored", {"chart_type": "big_number", "metric": col("scored", "MAX"),
                "subheader": "Isolation Forest, blocks 480,000-489,999", "y_axis_format": ",d"}, w=4, h=34)
    b.mcp_chart(D, "zen_totals", "Flagged as unusual", {"chart_type": "big_number", "metric": col("flagged", "MAX"),
                "subheader": "score ≥ 0.6707 (unsupervised; not 'illicit')", "y_axis_format": ",d"}, w=4, h=34)
    b.mcp_chart(D, "zen_totals", "Flag rate (per mille)", {"chart_type": "big_number", "metric": col("per_mille", "MAX"),
                "subheader": "flagged per 1,000 scored", "y_axis_format": ".2f"}, w=4, h=34)
    b.mcp_chart(D, "zen_hist", "Anomaly score histogram, 16.9 M tx (log count)", {"chart_type": "xy", "kind": "bar",
                "x": col("bin_mid"), "y": [col("n", "SUM", "transactions")], "x_axis": {"title": "anomaly score"},
                "y_axis": {"title": "transactions (log)", "scale": "log"}, "legend": {"show": False}}, w=6, h=52)
    b.rest_chart(D, "zen_reasons", "Why flagged: reason codes (size = count)", "treemap_v2",
                 {"groupby": ["feature", "direction"], "metric": M("SUM(n)", "flags"), "show_labels": True,
                  "show_upper_labels": True, "label_type": "key_value", "number_format": ",d", "row_limit": 100}, w=6, h=52)
    b.mcp_chart(D, "zen_bucket", "Flag rate along the chain (per 100 blocks)", {"chart_type": "xy", "kind": "line",
                "x": col("block_bucket"), "y": [col("flag_rate_per_mille", "MAX", "‰ flagged")],
                "x_axis": {"title": "block height"}, "y_axis": {"title": "flagged per 1,000"}}, w=8, h=50)
    b.rest_chart(D, "zen_flow", "Data quality → scoring → decision", "sankey_v2",
                 {"source": "src", "target": "tgt", "metric": M("SUM(n)", "transactions"), "row_limit": 100}, w=4, h=50)
    b.rest_chart(D, "zen_sample", "Score distribution per 1,000-block chunk (0.3 % sample)", "box_plot",
                 {"columns": ["rid"], "groupby": ["block_chunk"], "metrics": [M("MAX(score)", "score")],
                  "whiskerOptions": "Tukey", "number_format": ".2f", "row_limit": 100000}, w=8, h=56)
    b.mcp_chart(D, "zen_chunk", "Flag rate per chunk (‰)", {"chart_type": "xy", "kind": "bar", "x": col("block_chunk"),
                "y": [col("flag_rate_per_mille", "MAX", "‰")], "legend": {"show": False}}, w=4, h=56)
    b.rest_chart(D, "zen_flagged_sample", "Inputs × outputs: flagged vs typical (log scale)", "bubble_v2",
                 {"entity": "rid", "series": "flag_label", "x": M("MAX(log_inputs)", "log(1+inputs)"),
                  "y": M("MAX(log_outputs)", "log(1+outputs)"), "size": M("MAX(score)", "score"), "max_bubble_size": "6",
                  "x_axis_label": "log(1 + inputs)", "y_axis_label": "log(1 + outputs)", "show_legend": True,
                  "legendOrientation": "bottom", "opacity": 0.45, "row_limit": 70000}, w=6, h=56)
    b.rest_chart(D, "zen_decile", "Zenodo score deciles: inputs vs new-address share (bubble = flagged count)", "bubble_v2",
                 {"entity": "decile", "x": M("MAX(mean_inputs)", "mean inputs"), "y": M("MAX(mean_new_address_fraction)", "new-address share"),
                  "size": M("MAX(n_flagged) + 1", "flagged"), "max_bubble_size": "30", "x_axis_label": "mean inputs per tx",
                  "y_axis_label": "mean new-address fraction", "show_legend": False, "opacity": 0.8, "row_limit": 20}, w=6, h=56)


def layout(b: Builder, rest: Rest, dash_ids):
    out = {}
    for key, (title, desc, refresh) in DASHBOARDS.items():
        did = dash_ids[key]
        items = [c for c in b.charts if c[0] == key]
        pos = {"DASHBOARD_VERSION_KEY": "v2",
               "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
               "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [], "parents": ["ROOT_ID"]},
               "HEADER_ID": {"id": "HEADER_ID", "type": "HEADER", "meta": {"text": title}}}
        row, used, rn = None, 12, 0
        for _, cid, w, h, name in items:
            if used + w > 12:
                rn += 1
                row = f"ROW-{rn}"
                pos[row] = {"type": "ROW", "id": row, "children": [], "parents": ["ROOT_ID", "GRID_ID"],
                            "meta": {"background": "BACKGROUND_TRANSPARENT"}}
                pos["GRID_ID"]["children"].append(row)
                used = 0
            cn = f"CHART-{cid}"
            pos[cn] = {"type": "CHART", "id": cn, "children": [], "parents": ["ROOT_ID", "GRID_ID", row],
                       "meta": {"chartId": cid, "width": w, "height": h, "sliceName": name}}
            pos[row]["children"].append(cn)
            used += w
        meta = {"color_scheme": "bsad", "label_colors": LABEL_COLORS, "refresh_frequency": refresh,
                "cross_filters_enabled": True, "color_scheme_domain": [], "shared_label_colors": [],
                "expanded_slices": {}, "timed_refresh_immune_slices": [], "positions": pos}
        rest.put(f"/api/v1/dashboard/{did}", {"dashboard_title": title, "position_json": json.dumps(pos),
                                              "json_metadata": json.dumps(meta), "published": True,
                                              "css": ".dashboard-header .header-title { font-weight: 600; }"})
        emb = rest.post(f"/api/v1/dashboard/{did}/embedded", {"allowed_domains": EMBED_DOMAINS})["result"]
        out[key] = {"id": did, "title": title, "description": desc, "embed_uuid": emb["uuid"],
                    "charts": len(items), "refresh_seconds": refresh}
        log(f"  layout + embedding: {title} ({len(items)} charts)")
    return out


TIMESERIES = {"echarts_timeseries", "echarts_timeseries_line", "echarts_timeseries_bar", "echarts_area",
              "echarts_timeseries_scatter", "echarts_timeseries_smooth", "echarts_timeseries_step"}


def apply_labels(b: Builder):
    """Axis titles (with units), legends and a description on every chart (scripts/bi/chart_labels.py)."""
    missing = [t for *_, t in b.charts if t not in LABELS]
    if missing:
        raise SystemExit(f"charts without labels in chart_labels.py: {missing}")
    for _, cid, _, _, title in b.charts:
        xt, yt, desc = LABELS[title]
        res = b.rest.get(f"/api/v1/chart/{cid}")["result"]
        p = json.loads(res["params"] or "{}")
        v = p.get("viz_type")
        multi = bool(p.get("groupby")) or len(p.get("metrics") or []) > 1
        if v in TIMESERIES:
            p.update(x_axis_title=xt or "", x_axis_title_margin=30, y_axis_title=yt or "", y_axis_title_margin=50,
                     y_axis_title_position="Left", show_legend=multi, legendOrientation="top", legendType="scroll",
                     rich_tooltip=True, truncateXAxis=True, truncateYAxis=False)
            if yt and "log" in yt.lower():
                p["logAxis"] = True
            if p.get("orientation") == "horizontal":  # category axis is vertical: room for labels, sorted
                metric = p["metrics"][0]
                p.update(x_axis_title_margin=150, y_axis_title_margin=30, y_axis_title_position="Top",
                         x_axis_sort=metric.get("label") if isinstance(metric, dict) else metric, x_axis_sort_asc=True)
        elif v == "table":
            p["column_config"] = {"score": {"d3NumberFormat": ".4f"}, "threshold": {"d3NumberFormat": ".4~f"}}
        elif v == "box_plot":
            p.update(x_axis_title=xt, x_axis_title_margin=30, y_axis_title=yt, y_axis_title_margin=50,
                     y_axis_title_position="Left")
        elif v == "bubble_v2":
            p.update(x_axis_label=xt, y_axis_label=yt, x_axis_title_margin=30, y_axis_title_margin=50,
                     logXAxis=bool(xt and "log scale" in xt.lower()))
        elif v == "histogram_v2":
            p.update(x_axis_title=xt, y_axis_title=yt, x_axis_title_margin=30, y_axis_title_margin=50)
        elif v in ("pie", "radar", "sunburst_v2", "treemap_v2", "funnel"):
            p.update(show_legend=v in ("pie", "radar"), legendOrientation="bottom", legendType="plain")
        b.rest.put(f"/api/v1/chart/{cid}", {"params": json.dumps(p), "description": desc})
    log(f"  labels applied to {len(b.charts)} charts")


def fix_table_sorting(b: Builder):
    """generate_chart (Superset 6.1 MCP) stores raw-table sort columns as bare names, but the table
    plugin expects JSON pairs '["col", ascending]' and otherwise fails with 'Found invalid orderby
    options'. Rewrite them as descending sorts (newest / highest first)."""
    for _, cid, *_ in b.charts:
        res = b.rest.get(f"/api/v1/chart/{cid}")["result"]
        p = json.loads(res["params"] or "{}")
        cols = p.get("order_by_cols") or []
        if cols and any(not str(c).startswith("[") for c in cols):
            p["order_by_cols"] = [c if str(c).startswith("[") else json.dumps([c, False]) for c in cols]
            b.rest.put(f"/api/v1/chart/{cid}", {"params": json.dumps(p)})
            log(f"  fixed table sort: {res['slice_name']}")


def main():
    b = Builder()
    log("removing previous BSAD objects")
    b.clean()
    build(b)
    dash_ids = {}
    for key, (title, desc, _) in DASHBOARDS.items():
        ids = [cid for d, cid, *_ in b.charts if d == key]
        r = b.tool("generate_dashboard", {"chart_ids": ids, "dashboard_title": title, "description": desc, "published": True})
        dash_ids[key] = r["dashboard"]["id"]
        log(f"[MCP ] dashboard {title}: {len(ids)} charts")
    apply_labels(b)
    fix_table_sorting(b)
    emb = layout(b, b.rest, dash_ids)
    stream_ds = [b.datasets[k] for k in ("stream", "stream_batches", "stream_flow", "stream_labelled", "stream_runs")]
    path = os.path.join(ROOT, "data", "bi", "embed.json")
    json.dump({"superset_url": SUPERSET, "dashboards": emb, "stream_dataset_ids": stream_ds,
               "n_charts": len(b.charts), "via_mcp": b.n_mcp, "via_rest": len(b.charts) - b.n_mcp, "built": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(path, "w"), indent=2)
    log(f"{len(b.charts)} charts on {len(emb)} dashboards -> {path}")


if __name__ == "__main__":
    main()
