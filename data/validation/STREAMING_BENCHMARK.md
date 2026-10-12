# Historical Streaming Replay Benchmark Report

**Dataset Source:** `['data/raw/sample_a/btc-tx-480000-489999-aa']`  
**Raw Records Processed:** `100,000` records  
**Measurement Methodology:** Single-process offline replay on local Windows host (`python 3.12`, `psutil` RSS memory, high-resolution `time.perf_counter()`).

## 1. Isolated Parser vs End-to-End Throughput

| Engine Mode | Micro-Batch Size | Throughput (Raw Records/s) | Throughput (Tx/s) | Latency p50 | Latency p95 | Memory Delta |
|---|---|---|---|---|---|---|
| **Isolated Parser** | N/A | **347,516.89 rec/s** | N/A | N/A | N/A | 0.02 MB |
| **End-to-End Pipeline** | 100 | 97,981.05 rec/s | **17,805.92 tx/s** | 2.057 ms | 6.627 ms | 62.52 MB |
| **End-to-End Pipeline** | 500 | 110,450.89 rec/s | **20,072.04 tx/s** | 8.636 ms | 30.857 ms | 14.63 MB |
| **End-to-End Pipeline** | 1000 | 102,370.69 rec/s | **18,603.64 tx/s** | 26.853 ms | 49.131 ms | 14.18 MB |

## 2. Detailed Performance Profile (Batch Size 500)

| Metric | Value |
|---|---|
| **Raw Records Ingested** | 100,001 |
| **Completed Transactions** | 18,173 |
| **Micro-Batch Size** | 500 |
| **Throughput (Transactions/sec)** | **20,072.04 tx/s** |
| **Throughput (Raw Records/sec)** | **110,450.89 rec/s** |
| **Micro-Batch Latency p50** | **8.636 ms** |
| **Micro-Batch Latency p95** | **30.857 ms** |
| **Micro-Batch Latency p99** | **44.930 ms** |
| **Micro-Batch Latency Mean** | 12.440 ms |
| **Graph Address Nodes Tracked** | 74,196 |
| **Graph Transaction Nodes** | 18,173 |
| **Memory Delta (RSS)** | 14.63 MB (Peak: 95.51 MB) |


## 3. Methodological Notes
- **Correctness Guarantees:** 
  - Directed bipartite graph construction without Cartesian product explosion.
  - Zero future-information leakage: address transaction counts and cumulative volume are calculated strictly from past transactions and updated after feature extraction.
  - Pending transactions maintained across chunk boundaries in state buffer.
  - Special address isolation: `unableToDecode` and `0` pseudo-addresses are isolated per transaction.
- **Scope Disclaimer:** These measurements represent a local Python single-process baseline replay engine. Distributed streaming (e.g. Apache Spark / Flink) and live RPC ingestion are not claimed.
