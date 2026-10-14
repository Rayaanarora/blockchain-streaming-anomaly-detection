"""
Spark Structured Streaming consumer (Kafka -> scoring -> durable outputs).

    # process everything currently in the topic, then stop (deterministic; used by tests/benchmarks)
    .venv\\Scripts\\python.exe scripts\\streaming_consumer.py --run-dir data\\streaming\\demo --trigger availableNow
    # continuous micro-batches every 2 s until Ctrl+C or --max-seconds
    .venv\\Scripts\\python.exe scripts\\streaming_consumer.py --run-dir data\\streaming\\demo --trigger "2 seconds"

Restart recovery: rerun with the same --run-dir; Kafka offsets, dedup state and buffered Elliptic
snapshots resume from <run-dir>/checkpoint and <run-dir>/state.
"""
import argparse
import json
import logging
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", default=os.environ.get("BSAD_KAFKA_BOOTSTRAP", "127.0.0.1:29092"))
    ap.add_argument("--topic", default=os.environ.get("BSAD_EVENTS_TOPIC", "bsad.events.v1"))
    ap.add_argument("--run-dir", default=os.path.join(ROOT, "data/streaming/default"))
    ap.add_argument("--trigger", default="availableNow", help="'availableNow' or a processing-time interval")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop a continuous query after N seconds")
    ap.add_argument("--max-offsets-per-trigger", type=int, default=5000)
    ap.add_argument("--dedup-watermark", default="30 minutes")
    ap.add_argument("--snapshot-ttl-batches", type=int, default=20)
    ap.add_argument("--master", default="local[2]")
    ap.add_argument("--no-elliptic", action="store_true", help="do not load the Elliptic model")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.run_dir, "monitoring"), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.run_dir, "monitoring", "consumer.log")),
                                  logging.StreamHandler()])
    from src.scoring.service import ScoringService
    from src.spark.session import get_spark
    from src.streaming_replay.consumer import BatchProcessor, build_query, progress_listener

    t0 = time.perf_counter()
    service = ScoringService(load_elliptic=not args.no_elliptic)
    load_s = time.perf_counter() - t0
    logging.info("models: %s (load %.2fs)", json.dumps(service.models()), load_s)
    processor = BatchProcessor(service, args.run_dir, args.snapshot_ttl_batches)
    spark = get_spark("BSADStreamingConsumer", master=args.master, with_kafka=True, shuffle_partitions=4,
                      scratch_dir=os.path.join(ROOT, "data/spark_temp/streaming"))
    spark.streams.addListener(progress_listener(os.path.join(args.run_dir, "monitoring", "progress.jsonl")))
    run_info = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(args),
                "models": service.models(), "model_load_seconds": service.load_seconds, "spark_version": spark.version}
    json.dump(run_info, open(os.path.join(args.run_dir, "monitoring", "run_info.json"), "w"), indent=2)
    q = build_query(spark, args.bootstrap, args.topic, args.run_dir, os.path.join(args.run_dir, "checkpoint"),
                    processor, trigger=args.trigger, max_offsets_per_trigger=args.max_offsets_per_trigger,
                    dedup_watermark=args.dedup_watermark)
    try:
        if args.trigger == "availableNow":
            q.awaitTermination()
        elif args.max_seconds:
            q.awaitTermination(args.max_seconds)
        else:
            q.awaitTermination()
    except KeyboardInterrupt:
        pass
    finally:
        if q.isActive:
            q.stop()
        exc = q.exception()
        spark.stop()
    if exc:
        raise SystemExit(f"streaming query failed: {exc}")
    logging.info("consumer finished in %.1fs", time.perf_counter() - t0)


if __name__ == "__main__":
    main()
