"""
Reproducible Inference Latency and Throughput Benchmark.

Measures:
1. Single-transaction latency (batch size = 1) for real-time transaction scoring.
2. Micro-batch latency (batch sizes = 16, 64, 256, 1000) for streaming ingestion.
3. Monolithic test-set latency (batch size = 16670).
4. Statistical distributions: Mean, Median (p50), 90th percentile (p90), 95th percentile (p95), Min, Max, and Throughput (tx/s).
5. Hardware profile: CPU model, core counts, OS, Python runtime, and scikit-learn version.

Explicit Methodology Note:
This benchmark measures strictly in-memory model inference (feature matrix -> probability predictions).
It does NOT measure live distributed streaming latency (which includes broker network RTT, Kafka consumer poll, and serialization).
"""
import os
import sys
import time
import json
import platform
import psutil
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure repository root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.baselines import CostSensitiveAnomalyModel


def benchmark_model_latency(
    model: CostSensitiveAnomalyModel,
    X_test: np.ndarray,
    batch_sizes: List[int] = [1, 16, 64, 256, 1000, 16670],
    warmup_runs: int = 10,
    timed_runs_map: Dict[int, int] = {1: 200, 16: 100, 64: 100, 256: 50, 1000: 50, 16670: 20}
) -> Dict[str, Any]:
    """Execute rigorous, multi-iteration latency benchmark across batch sizes."""
    results_by_batch: Dict[str, Any] = {}

    # Initial warm-up with full batch
    for _ in range(warmup_runs):
        _ = model.predict_proba(X_test[:min(100, len(X_test))])

    for b in batch_sizes:
        effective_b = min(b, len(X_test))
        sample_batch = X_test[:effective_b]
        num_runs = timed_runs_map.get(b, 30)

        # Warm-up for specific batch size
        for _ in range(5):
            _ = model.predict_proba(sample_batch)

        latencies_ms: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model.predict_proba(sample_batch)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

        lat_arr = np.array(latencies_ms)
        median_ms = float(np.median(lat_arr))
        mean_ms = float(np.mean(lat_arr))
        p90_ms = float(np.percentile(lat_arr, 90))
        p95_ms = float(np.percentile(lat_arr, 95))
        min_ms = float(np.min(lat_arr))
        max_ms = float(np.max(lat_arr))

        # Per-transaction latency in microseconds or milliseconds
        per_tx_us = (median_ms / effective_b) * 1000.0
        throughput_tx_per_sec = float(effective_b / (mean_ms / 1000.0))

        results_by_batch[f"batch_{b}"] = {
            "batch_size": effective_b,
            "timed_iterations": num_runs,
            "batch_latency_ms": {
                "median_p50": round(median_ms, 4),
                "mean": round(mean_ms, 4),
                "p90": round(p90_ms, 4),
                "p95": round(p95_ms, 4),
                "min": round(min_ms, 4),
                "max": round(max_ms, 4)
            },
            "per_transaction_latency_us": round(per_tx_us, 4),
            "throughput_tx_per_sec": round(throughput_tx_per_sec, 1)
        }

    return results_by_batch


def main():
    print("=" * 80)
    print("IN-MEMORY MODEL INFERENCE LATENCY & THROUGHPUT BENCHMARK")
    print("=" * 80)

    # 1. Environment & Hardware Metadata
    hardware_info = {
        "cpu_processor": platform.processor(),
        "cpu_architecture": platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "timer": "time.perf_counter() (high-resolution monotonic hardware clock)"
    }

    print(f"Hardware: {hardware_info['cpu_processor']} | {hardware_info['logical_cores']} logical cores | {hardware_info['total_memory_gb']} GB RAM")
    print(f"OS:       {hardware_info['os']}")
    print(f"Python:   {hardware_info['python_version']}")

    # 2. Load Data & Fit Models
    data_path = "data/processed/elliptic/elliptic_preprocessed.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed Elliptic dataset not found at '{data_path}'.")

    data = np.load(data_path)
    X_raw = data["X_raw"]
    ts = data["time_steps"]
    y = data["y"]

    train_mask = (ts <= 29) & (y != -1)
    test_mask = (ts >= 35) & (y != -1)

    scaler = StandardScaler()
    scaler.fit(X_raw[train_mask])
    X_train = scaler.transform(X_raw[train_mask])
    y_train = y[train_mask]
    X_test = scaler.transform(X_raw[test_mask])

    ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    print(f"\nTraining models on {X_train.shape[0]:,} samples (imbalance ratio: {ratio:.2f}:1)...")
    models = {
        "lr": CostSensitiveAnomalyModel("lr", class_weight_ratio=ratio, scale_input=False, random_state=42).fit(X_train, y_train),
        "rf": CostSensitiveAnomalyModel("rf", class_weight_ratio=ratio, scale_input=False, random_state=42).fit(X_train, y_train),
        "gbdt": CostSensitiveAnomalyModel("hgbdt", class_weight_ratio=ratio, scale_input=False, random_state=42).fit(X_train, y_train)
    }

    # 3. Benchmark Each Model
    benchmark_results: Dict[str, Any] = {
        "benchmark_title": "Reproducible In-Memory Model Inference Latency Benchmark",
        "methodology": "Repeated timed batches using time.perf_counter() with warm-up passes",
        "scope_distinction": "In-memory feature vector scoring only; does NOT include distributed network or broker latency",
        "hardware_environment": hardware_info,
        "models": {}
    }

    print("\nBenchmarking latency distributions...")
    for model_key, model in models.items():
        print(f"\n  --- Benchmarking {model_key.upper()} ---")
        model_latencies = benchmark_model_latency(model, X_test)
        benchmark_results["models"][model_key] = {
            "model_family": model.model_family,
            "latency_metrics": model_latencies
        }

        b1 = model_latencies["batch_1"]["batch_latency_ms"]
        b1000 = model_latencies["batch_1000"]["batch_latency_ms"]
        bfull = model_latencies["batch_16670"]["batch_latency_ms"]

        print(f"      Single Tx (B=1):    Median = {b1['median_p50']:.4f} ms | p95 = {b1['p95']:.4f} ms")
        print(f"      Micro-Batch (B=1000): Median = {b1000['median_p50']:.3f} ms | per-tx = {model_latencies['batch_1000']['per_transaction_latency_us']:.3f} us | {model_latencies['batch_1000']['throughput_tx_per_sec']:,.0f} tx/s")
        print(f"      Full Batch (16.6k): Median = {bfull['median_p50']:.2f} ms | p95 = {bfull['p95']:.2f} ms")

    # 4. Save JSON Report
    out_dir = "data/validation/model_evaluation"
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "latency_benchmark.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nBenchmark complete! Saved report: {out_json}")


if __name__ == "__main__":
    main()
