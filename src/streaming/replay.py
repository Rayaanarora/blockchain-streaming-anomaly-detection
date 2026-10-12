"""
Deterministic historical event replay engine with end-to-end profiling.
Measures parsing throughput, micro-batch processing latencies, and memory consumption.
"""
import time
from typing import List, Dict, Optional, Iterator
import psutil
import numpy as np
from src.data.reader import ZenodoStreamReader
from src.data.boundary import CompletedTransaction
from src.graph.bipartite import BipartiteTransactionGraph
from src.features.extractor import HistoricalFeatureExtractor


class HistoricalReplayEngine:
    """
    Offline historical streaming replay engine.
    Replays blockchain transactions in deterministic block/transaction sequence.
    """
    def __init__(self, batch_size: int = 500, sanitize_endpoints: bool = True):
        self.batch_size = batch_size
        self.sanitize_endpoints = sanitize_endpoints
        self.graph = BipartiteTransactionGraph()
        self.feature_extractor = HistoricalFeatureExtractor()
        self.proc = psutil.Process()

    def benchmark_parser_throughput(
        self,
        paths: List[str],
        max_records: int = 100000
    ) -> Dict[str, float]:
        """
        Measure isolated parser throughput (raw record parsing only).
        """
        reader = ZenodoStreamReader(sanitize_endpoints=self.sanitize_endpoints)
        mem_start_mb = self.proc.memory_info().rss / (1024 * 1024)
        
        t0 = time.perf_counter()
        count = 0
        for path in paths:
            if path.endswith(".tar.bz2") or path.endswith(".bz2"):
                stream = reader.stream_records_from_archive(path, max_records=max_records - count)
            else:
                stream = reader.stream_records_from_file(path, max_records=max_records - count)
            for _ in stream:
                count += 1
                if count >= max_records:
                    break
            if count >= max_records:
                break
        t1 = time.perf_counter()
        
        elapsed = t1 - t0
        mem_end_mb = self.proc.memory_info().rss / (1024 * 1024)
        throughput = count / max(elapsed, 1e-6)
        
        return {
            "mode": "isolated_parser",
            "records_parsed": count,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_records_per_sec": round(throughput, 2),
            "memory_start_mb": round(mem_start_mb, 2),
            "memory_end_mb": round(mem_end_mb, 2),
            "memory_delta_mb": round(mem_end_mb - mem_start_mb, 2)
        }

    def run_end_to_end_replay(
        self,
        paths: List[str],
        max_records: Optional[int] = 100000,
        enable_features: bool = True,
        enable_graph: bool = True
    ) -> Dict[str, object]:
        """
        Replay transactions chronologically, maintain graph, extract features,
        and measure micro-batch processing latencies (p50, p95, p99) and sustained throughput.
        """
        reader = ZenodoStreamReader(sanitize_endpoints=self.sanitize_endpoints)
        mem_start_mb = self.proc.memory_info().rss / (1024 * 1024)
        
        batch_latencies_ms: List[float] = []
        current_batch: List[CompletedTransaction] = []
        total_txs_emitted = 0
        total_features_extracted = 0
        
        t_global_start = time.perf_counter()
        
        tx_stream = reader.stream_transactions_from_paths(paths, max_records_total=max_records)
        
        for tx in tx_stream:
            total_txs_emitted += 1
            current_batch.append(tx)
            
            if len(current_batch) >= self.batch_size:
                # Process micro-batch and record latency
                t_b0 = time.perf_counter()
                
                for b_tx in current_batch:
                    if enable_graph:
                        self.graph.add_transaction(b_tx)
                    if enable_features:
                        _, _ = self.feature_extractor.extract_features(b_tx)
                        total_features_extracted += 1
                        
                t_b1 = time.perf_counter()
                batch_latency_ms = (t_b1 - t_b0) * 1000.0
                batch_latencies_ms.append(batch_latency_ms)
                current_batch.clear()

        # Final residual batch
        if current_batch:
            t_b0 = time.perf_counter()
            for b_tx in current_batch:
                if enable_graph:
                    self.graph.add_transaction(b_tx)
                if enable_features:
                    _, _ = self.feature_extractor.extract_features(b_tx)
                    total_features_extracted += 1
            t_b1 = time.perf_counter()
            batch_latencies_ms.append((t_b1 - t_b0) * 1000.0)
            current_batch.clear()
            
        t_global_end = time.perf_counter()
        total_elapsed = t_global_end - t_global_start
        mem_end_mb = self.proc.memory_info().rss / (1024 * 1024)
        
        l_arr = np.array(batch_latencies_ms) if batch_latencies_ms else np.array([0.0])
        
        return {
            "mode": "end_to_end_replay",
            "batch_size": self.batch_size,
            "raw_records_read": reader.total_records_read,
            "completed_transactions": total_txs_emitted,
            "features_extracted": total_features_extracted,
            "total_elapsed_seconds": round(total_elapsed, 4),
            "throughput_transactions_per_sec": round(total_txs_emitted / max(total_elapsed, 1e-6), 2),
            "throughput_raw_records_per_sec": round(reader.total_records_read / max(total_elapsed, 1e-6), 2),
            "micro_batches_processed": len(batch_latencies_ms),
            "latency_ms_p50": round(float(np.percentile(l_arr, 50)), 3),
            "latency_ms_p95": round(float(np.percentile(l_arr, 95)), 3),
            "latency_ms_p99": round(float(np.percentile(l_arr, 99)), 3),
            "latency_ms_mean": round(float(np.mean(l_arr)), 3),
            "latency_ms_min": round(float(np.min(l_arr)), 3),
            "latency_ms_max": round(float(np.max(l_arr)), 3),
            "graph_addresses_tracked": self.graph.num_addresses,
            "graph_transactions_tracked": self.graph.num_transactions,
            "graph_edges_tracked": self.graph.num_edges,
            "memory_start_mb": round(mem_start_mb, 2),
            "memory_end_mb": round(mem_end_mb, 2),
            "memory_delta_mb": round(mem_end_mb - mem_start_mb, 2)
        }
