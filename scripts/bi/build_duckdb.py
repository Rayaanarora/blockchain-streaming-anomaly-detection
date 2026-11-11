"""
Create the DuckDB serving database that Superset queries (run inside the Superset environment in WSL).

DuckDB holds no data of its own: every object is a VIEW over Parquet/JSON files on D:.
  * gold_*    -> data/bi/gold/<table>/*.parquet          (built by scripts/build_bi_gold.py with Spark)
  * stream_*  -> data/streaming/run_*/predictions, monitoring/*.jsonl  (read live: new micro-batch files
                 written by the Spark consumer appear in the next query, no rebuild needed)

    ~/bi/superset-venv/bin/python scripts/bi/build_duckdb.py \
        --repo /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection --db ~/bi/bsad.duckdb
"""
import argparse
import glob
import os

import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection")
    ap.add_argument("--db", default=os.path.expanduser("~/bi/bsad.duckdb"))
    a = ap.parse_args()
    gold = os.path.join(a.repo, "data/bi/gold")
    stream = os.path.join(a.repo, "data/streaming")
    tmp = a.db + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = duckdb.connect(tmp)

    names = sorted(d for d in os.listdir(gold) if glob.glob(os.path.join(gold, d, "*.parquet")))
    for n in names:
        con.execute(f"CREATE VIEW gold_{n} AS SELECT * FROM read_parquet('{gold}/{n}/*.parquet')")

    preds = f"{stream}/run_*/predictions/*.parquet"
    con.execute(f"""
        CREATE VIEW stream_predictions AS
        SELECT regexp_extract(filename, 'streaming/([^/]+)/predictions', 1) AS run_id,
               p.* EXCLUDE (filename),
               CAST(p.flagged AS BOOLEAN) AS is_flagged,
               CASE WHEN p.source_dataset = 'elliptic' THEN p.event_time_value END AS time_step,
               try_cast(p.scored_at AS TIMESTAMP) AS scored_ts,
               try_cast(p.produced_at AS TIMESTAMP) AS produced_ts
        FROM read_parquet('{preds}', filename = true, union_by_name = true) p
    """)
    con.execute(f"""
        CREATE VIEW stream_elliptic_labelled AS
        SELECT s.*, l.class,
               CASE WHEN l.class = 'illicit' AND s.is_flagged THEN 'TP'
                    WHEN l.class = 'licit' AND s.is_flagged THEN 'FP'
                    WHEN l.class = 'illicit' AND NOT s.is_flagged THEN 'FN'
                    WHEN l.class = 'licit' AND NOT s.is_flagged THEN 'TN' END AS outcome
        FROM stream_predictions s
        JOIN gold_ell_labels l ON CAST(l.tx_id AS VARCHAR) = s.transaction_id
        WHERE s.source_dataset = 'elliptic' AND s.status = 'scored'
    """)
    con.execute(f"""
        CREATE VIEW stream_batches AS
        SELECT regexp_extract(filename, 'streaming/([^/]+)/monitoring', 1) AS run_id,
               batch_id, try_cast(processing_time AS TIMESTAMP) AS processing_ts, input_rows, results_written,
               dead_lettered, buffered_for_graph_context, flagged, batch_seconds,
               e2e_latency_ms_p50, e2e_latency_ms_p95, e2e_latency_ms_max
        FROM read_json('{stream}/run_*/monitoring/batches.jsonl', format = 'newline_delimited',
                       filename = true, union_by_name = true, ignore_errors = true)
    """)
    con.execute("""
        CREATE VIEW stream_runs AS
        SELECT run_id, count(*) AS events, sum(CAST(is_flagged AS INT)) AS flagged,
               min(scored_ts) AS started, max(scored_ts) AS finished,
               count(*) FILTER (WHERE source_dataset = 'elliptic') AS elliptic_events,
               count(*) FILTER (WHERE source_dataset = 'zenodo') AS zenodo_events,
               max(model_version) FILTER (WHERE source_dataset = 'elliptic') AS elliptic_model
        FROM stream_predictions GROUP BY run_id
    """)
    for v in ["stream_predictions", "stream_batches", "stream_elliptic_labelled", "stream_runs"]:
        print(v, con.execute(f"SELECT count(*) FROM {v}").fetchone()[0])
    print(len(names), "gold views")
    con.close()
    os.replace(tmp, a.db)
    print("wrote", a.db)


if __name__ == "__main__":
    main()
