"""
Reproducible Historical Streaming Replay Benchmark Runner.

Benchmarks:
1. Isolated raw parser throughput (lines/second, zero graph overhead).
2. End-to-end streaming replay throughput across configurable micro-batch sizes (100, 500, 1000).
3. Micro-batch latency distribution profiling (p50, p95, p99, mean, min, max).
4. Memory consumption (RSS start, peak, end, delta).
"""
import os
import sys
import argparse
import json
from typing import List

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.streaming.replay import HistoricalReplayEngine
from src.evaluation.metrics import save_metrics_report, format_streaming_table


def parse_args():
    parser = argparse.ArgumentParser(description="Run blockchain streaming replay benchmark.")
    parser.add_argument(
        "--input-paths",
        nargs="+",
        default=["data/raw/sample_a/btc-tx-480000-489999-aa"],
        help="Input plain text or tar.bz2 files to replay."
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=100000,
        help="Number of raw records to ingest in the benchmark run."
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[100, 500, 1000],
        help="Micro-batch sizes to benchmark."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/validation/streaming_benchmark.json",
        help="Path to save benchmark JSON report."
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="data/validation/STREAMING_BENCHMARK.md",
        help="Path to save benchmark Markdown table."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = [p for p in args.input_paths if os.path.exists(p)]
    if not input_paths:
        # Fallback to tar.bz2 archive if plain sample is not found
        fallback = "data/raw/a.tar.bz2"
        if os.path.exists(fallback):
            input_paths = [fallback]
        else:
            raise FileNotFoundError(f"None of the specified input paths exist: {args.input_paths}")

    print("=" * 80)
    print("BLOCKCHAIN STREAMING ANOMALY DETECTION - REPLAY ENGINE BENCHMARK")
    print(f"Input path(s): {input_paths}")
    print(f"Max records:   {args.max_records:,}")
    print(f"Batch sizes:   {args.batch_sizes}")
    print("=" * 80)

    # 1. Isolated Parser Benchmark
    print("\n[Step 1/3] Benchmarking isolated parser throughput...")
    engine_parser = HistoricalReplayEngine(batch_size=500)
    parser_results = engine_parser.benchmark_parser_throughput(input_paths, max_records=args.max_records)
    print(f"  Records Parsed:     {parser_results['records_parsed']:,}")
    print(f"  Elapsed Time:       {parser_results['elapsed_seconds']:.3f} s")
    print(f"  Parser Throughput:  {parser_results['throughput_records_per_sec']:,} records/sec")
    print(f"  Memory Delta (RSS): {parser_results['memory_delta_mb']} MB")

    # 2. End-to-End Replay Benchmarks across batch sizes
    print("\n[Step 2/3] Benchmarking end-to-end replay across micro-batch sizes...")
    e2e_results = []
    for b_size in args.batch_sizes:
        print(f"\n  --> Running Micro-Batch Size = {b_size} ...")
        engine_e2e = HistoricalReplayEngine(batch_size=b_size)
        metrics = engine_e2e.run_end_to_end_replay(
            input_paths,
            max_records=args.max_records,
            enable_features=True,
            enable_graph=True
        )
        print(f"      Completed Transactions: {metrics['completed_transactions']:,}")
        print(f"      Tx Throughput:          {metrics['throughput_transactions_per_sec']:,} tx/s")
        print(f"      Raw Record Throughput:  {metrics['throughput_raw_records_per_sec']:,} rec/s")
        print(f"      Latency p50:            {metrics['latency_ms_p50']:.3f} ms")
        print(f"      Latency p95:            {metrics['latency_ms_p95']:.3f} ms")
        print(f"      Latency p99:            {metrics['latency_ms_p99']:.3f} ms")
        print(f"      Graph Nodes Tracked:    {metrics['graph_addresses_tracked']:,} addresses, {metrics['graph_transactions_tracked']:,} txs")
        print(f"      Peak Memory (RSS):      {metrics['memory_end_mb']} MB (Delta: {metrics['memory_delta_mb']} MB)")
        e2e_results.append(metrics)

    # 3. Save Reproducible Benchmark Artifacts
    print("\n[Step 3/3] Saving benchmark reports...")
    full_report = {
        "benchmark_config": {
            "input_paths": input_paths,
            "max_records": args.max_records,
            "batch_sizes": args.batch_sizes
        },
        "isolated_parser_benchmark": parser_results,
        "end_to_end_microbatch_benchmarks": e2e_results
    }

    save_metrics_report(full_report, args.output_json)

    # Format Markdown Report
    primary_metrics = e2e_results[1] if len(e2e_results) > 1 else e2e_results[0]
    md_content = f"""# Historical Streaming Replay Benchmark Report

**Dataset Source:** `{input_paths}`  
**Raw Records Processed:** `{args.max_records:,}` records  
**Measurement Methodology:** Single-process offline replay on local Windows host (`python 3.12`, `psutil` RSS memory, high-resolution `time.perf_counter()`).

## 1. Isolated Parser vs End-to-End Throughput

| Engine Mode | Micro-Batch Size | Throughput (Raw Records/s) | Throughput (Tx/s) | Latency p50 | Latency p95 | Memory Delta |
|---|---|---|---|---|---|---|
| **Isolated Parser** | N/A | **{parser_results['throughput_records_per_sec']:,} rec/s** | N/A | N/A | N/A | {parser_results['memory_delta_mb']} MB |
"""
    for res in e2e_results:
        md_content += f"| **End-to-End Pipeline** | {res['batch_size']} | {res['throughput_raw_records_per_sec']:,} rec/s | **{res['throughput_transactions_per_sec']:,} tx/s** | {res['latency_ms_p50']:.3f} ms | {res['latency_ms_p95']:.3f} ms | {res['memory_delta_mb']} MB |\n"

    md_content += f"""
## 2. Detailed Performance Profile (Batch Size {primary_metrics['batch_size']})

{format_streaming_table(primary_metrics)}

## 3. Methodological Notes
- **Correctness Guarantees:** 
  - Directed bipartite graph construction without Cartesian product explosion.
  - Zero future-information leakage: address transaction counts and cumulative volume are calculated strictly from past transactions and updated after feature extraction.
  - Pending transactions maintained across chunk boundaries in state buffer.
  - Special address isolation: `unableToDecode` and `0` pseudo-addresses are isolated per transaction.
- **Scope Disclaimer:** These measurements represent a local Python single-process baseline replay engine. Distributed streaming (e.g. Apache Spark / Flink) and live RPC ingestion are not claimed.
"""

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report saved to: {args.output_md}")


if __name__ == "__main__":
    main()
