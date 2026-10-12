# Historical Streaming Replay Benchmark Report

**Dataset Source:** `['data/raw/a.tar.bz2']`  
**Raw Records Processed:** `10,000` records  
**Measurement Methodology:** Single-process offline replay on local Windows host (`python 3.12`, `psutil` RSS memory, high-resolution `time.perf_counter()`).

## 1. Isolated Parser vs End-to-End Throughput

| Engine Mode | Micro-Batch Size | Throughput (Raw Records/s) | Throughput (Tx/s) | Latency p50 | Latency p95 | Memory Delta |
|---|---|---|---|---|---|---|
| **Isolated Parser** | N/A | **80,435.32 rec/s** | N/A | N/A | N/A | 3.53 MB |
| **End-to-End Pipeline** | 500 | 57,221.13 rec/s | **12,616.0 tx/s** | 7.519 ms | 11.377 ms | 7.54 MB |

## 2. Detailed Performance Profile (Batch Size 500)

| Metric | Value |
|---|---|
| **Raw Records Ingested** | 10,001 |
| **Completed Transactions** | 2,205 |
| **Micro-Batch Size** | 500 |
| **Throughput (Transactions/sec)** | **12,616.0 tx/s** |
| **Throughput (Raw Records/sec)** | **57,221.13 rec/s** |
| **Micro-Batch Latency p50** | **7.519 ms** |
| **Micro-Batch Latency p95** | **11.377 ms** |
| **Micro-Batch Latency p99** | **12.010 ms** |
| **Micro-Batch Latency Mean** | 8.069 ms |
| **Graph Address Nodes Tracked** | 8,112 |
| **Graph Transaction Nodes** | 2,205 |
| **Memory Delta (RSS)** | 7.54 MB (Peak: 38.33 MB) |


## 3. Methodological Notes
- **Correctness Guarantees:** 
  - Directed bipartite graph construction without Cartesian product explosion.
  - Zero future-information leakage: address transaction counts and cumulative volume are calculated strictly from past transactions and updated after feature extraction.
  - Pending transactions maintained across chunk boundaries in state buffer.
  - Special address isolation: `unableToDecode` and `0` pseudo-addresses are isolated per transaction.
- **Scope Disclaimer:** These measurements represent a local Python single-process baseline replay engine. Distributed streaming (e.g. Apache Spark / Flink) and live RPC ingestion are not claimed.
