"""
Replay existing data into Kafka as canonical events.

    # 2,000 Zenodo transactions from an existing partition at 500 events/s
    .venv\\Scripts\\python.exe scripts\\replay_producer.py zenodo --limit 2000 --rate 500
    # complete Elliptic snapshots (time steps 35 and 36) from the Spark output
    .venv\\Scripts\\python.exe scripts\\replay_producer.py elliptic --steps 35 36

Requires a running broker (scripts/local_kafka.py start). Writes a run manifest JSON.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.streaming_replay.producer import (ReplayProducer, elliptic_events, zenodo_events,  # noqa: E402
                                           zenodo_source)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["zenodo", "elliptic"])
    ap.add_argument("--bootstrap", default=os.environ.get("BSAD_KAFKA_BOOTSTRAP", "127.0.0.1:29092"))
    ap.add_argument("--topic", default=os.environ.get("BSAD_EVENTS_TOPIC", "bsad.events.v1"))
    ap.add_argument("--zenodo-dir", default=os.path.join(ROOT, "data/parquet/zenodo_features"))
    ap.add_argument("--block-chunk", default="485000-485999")
    ap.add_argument("--start-block", type=int, default=None)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N rows (for multi-part replays)")
    ap.add_argument("--elliptic-dir", default=os.path.join(ROOT, "data/processed/elliptic_spark"))
    ap.add_argument("--steps", type=int, nargs="+", default=[35])
    ap.add_argument("--rate", type=float, default=0.0, help="events per second (0 = unthrottled)")
    ap.add_argument("--inject-malformed", type=int, default=0, help="TEST ONLY")
    ap.add_argument("--inject-duplicates", type=int, default=0, help="TEST ONLY")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    if args.dataset == "zenodo":
        df = zenodo_source(args.zenodo_dir, args.block_chunk, args.limit + args.offset, args.start_block)
        df = df.iloc[args.offset:]
        events = zenodo_events(df, start_seq=args.offset)
        source = {"dataset": "zenodo", "block_chunk": args.block_chunk, "rows": len(df), "offset": args.offset}
    else:
        events = elliptic_events(args.elliptic_dir, args.steps)
        source = {"dataset": "elliptic", "steps": args.steps}
    prod = ReplayProducer(args.bootstrap, args.topic)
    report = prod.replay(events, rate=args.rate, inject_malformed=args.inject_malformed,
                         inject_duplicates=args.inject_duplicates)
    report.update({"source": source, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    path = args.manifest or os.path.join(ROOT, "data/streaming/producer_runs",
                                         f"{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(report, open(path, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
