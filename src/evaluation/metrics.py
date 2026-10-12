"""
Evaluation metrics, experiment logger, and report generation utilities.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any


def save_metrics_report(metrics: Dict[str, Any], output_path: str) -> None:
    """Save experiment metrics to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def format_streaming_table(metrics: Dict[str, Any]) -> str:
    """Format streaming benchmark metrics into GitHub markdown table."""
    return f"""| Metric | Value |
|---|---|
| **Raw Records Ingested** | {metrics.get('raw_records_read', 0):,} |
| **Completed Transactions** | {metrics.get('completed_transactions', 0):,} |
| **Micro-Batch Size** | {metrics.get('batch_size', 0)} |
| **Throughput (Transactions/sec)** | **{metrics.get('throughput_transactions_per_sec', 0):,} tx/s** |
| **Throughput (Raw Records/sec)** | **{metrics.get('throughput_raw_records_per_sec', 0):,} rec/s** |
| **Micro-Batch Latency p50** | **{metrics.get('latency_ms_p50', 0):.3f} ms** |
| **Micro-Batch Latency p95** | **{metrics.get('latency_ms_p95', 0):.3f} ms** |
| **Micro-Batch Latency p99** | **{metrics.get('latency_ms_p99', 0):.3f} ms** |
| **Micro-Batch Latency Mean** | {metrics.get('latency_ms_mean', 0):.3f} ms |
| **Graph Address Nodes Tracked** | {metrics.get('graph_addresses_tracked', 0):,} |
| **Graph Transaction Nodes** | {metrics.get('graph_transactions_tracked', 0):,} |
| **Memory Delta (RSS)** | {metrics.get('memory_delta_mb', 0)} MB (Peak: {metrics.get('memory_end_mb', 0)} MB) |
"""
