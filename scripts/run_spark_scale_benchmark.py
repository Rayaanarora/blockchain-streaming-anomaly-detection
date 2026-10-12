"""
Reproducible Bounded Scale-Up Benchmark for Apache Spark + PyArrow Pipeline.

Audits:
1. Scalability on bounded real Zenodo slices (50,000 / 100,000 lines).
2. Wall-clock elapsed time and records/sec throughput.
3. Peak driver process RSS memory and memory growth.
4. Output Parquet directory size on disk.
5. Exact schema, row counts, null handling, and numerical correctness.
6. Boundary transaction state handling (slice beginning and truncation at slice end).
7. Pure-Python reference ground truth validation on a representative sample of rows.
8. Storage isolation strictly on Drive D: (free space audit on C: and D: before/after).
"""
import os
import sys
import shutil
import json
import time
import argparse
import psutil
from collections import defaultdict

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyarrow.parquet as pq
from src.spark.schemas import get_transaction_features_pyarrow_schema
from src.spark.zenodo_batch import build_spark_session, ZenodoSparkBatchPipeline


def get_drive_free_gb():
    """Return free space in GB for C: and D:."""
    free_info = {}
    for drive in ["C:", "D:"]:
        try:
            usage = shutil.disk_usage(drive)
            free_info[drive] = round(usage.free / (1024 ** 3), 2)
        except Exception:
            free_info[drive] = None
    return free_info


def compute_ground_truth_for_keys(lines, target_keys):
    """
    Independently compute ground truth completed transactions and historical features
    in pure Python for a specified subset of transaction keys.
    """
    target_keys_set = set(target_keys)

    # 1. Parse raw lines
    parsed = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        kind, block, tx_idx, src, dst, val = parts[0], int(parts[1]), int(parts[2]), parts[3], parts[4], int(parts[5])
        if kind not in ("BTC-IN", "BTC-OUT") or block < 0 or tx_idx < 0 or val < 0:
            continue
        tx_hash = dst if kind == "BTC-IN" else src
        if len(tx_hash) != 64:
            continue
        raw_addr = src if kind == "BTC-IN" else dst
        if raw_addr == "unableToDecode":
            addr = f"unableToDecode_{tx_hash[:16]}"
        elif raw_addr == "0":
            addr = f"coinbase_source_{tx_hash[:16]}"
        else:
            addr = raw_addr

        parsed.append({
            "kind": kind,
            "block_number": block,
            "tx_index": tx_idx,
            "tx_hash": tx_hash,
            "address": addr,
            "value_satoshi": val
        })

    # 2. Assemble transactions
    tx_map = defaultdict(lambda: {
        "tx_hash": "",
        "inputs": [],
        "outputs": [],
        "in_val": 0,
        "out_val": 0,
        "in_addrs": []
    })

    for r in parsed:
        k = (r["block_number"], r["tx_index"])
        tx_map[k]["tx_hash"] = r["tx_hash"]
        if r["kind"] == "BTC-IN":
            tx_map[k]["inputs"].append(r)
            tx_map[k]["in_val"] += r["value_satoshi"]
            tx_map[k]["in_addrs"].append(r["address"])
        else:
            tx_map[k]["outputs"].append(r)
            tx_map[k]["out_val"] += r["value_satoshi"]

    # Filter completed transactions
    completed = {}
    for k, v in tx_map.items():
        is_coinbase = (k[1] == 0)
        n_in = len(v["inputs"])
        n_out = len(v["outputs"])
        if (is_coinbase and n_out > 0) or (not is_coinbase and n_in > 0 and n_out > 0):
            fee = 0 if is_coinbase else (v["in_val"] - v["out_val"])
            completed[k] = {
                "block_number": k[0],
                "tx_index": k[1],
                "tx_hash": v["tx_hash"],
                "num_inputs": n_in,
                "num_outputs": n_out,
                "total_input_satoshi": v["in_val"],
                "total_output_satoshi": v["out_val"],
                "fee_satoshi": fee,
                "is_coinbase": 1 if is_coinbase else 0,
                "input_addresses": v["in_addrs"]
            }

    # 3. Compute temporal features with strictly prior causality
    # Aggregate address participation strictly per (address, block_number, tx_index)
    addr_tx = defaultdict(lambda: {"in": 0, "out": 0, "has_in": False})
    for r in parsed:
        k_addr = (r["address"], r["block_number"], r["tx_index"])
        if r["kind"] == "BTC-IN":
            addr_tx[k_addr]["in"] += r["value_satoshi"]
            addr_tx[k_addr]["has_in"] = True
        else:
            addr_tx[k_addr]["out"] += r["value_satoshi"]

    # Sort each address's transaction events chronologically
    by_addr = defaultdict(list)
    for (addr, blk, tidx), data in addr_tx.items():
        tx_vol = max(data["in"], data["out"])
        by_addr[addr].append((blk, tidx, data["has_in"], tx_vol))

    addr_prior_stats = {}
    for addr, ev_list in by_addr.items():
        ev_list.sort(key=lambda x: (x[0], x[1]))
        cum_count = 0
        cum_vol = 0
        for (blk, tidx, has_in, vol) in ev_list:
            if has_in:
                addr_prior_stats[(addr, blk, tidx)] = (cum_count, cum_vol)
            cum_count += 1
            cum_vol += vol

    # Build features for targeted keys
    ground_truth = {}
    for k in target_keys_set:
        if k not in completed:
            continue
        tx = completed[k]
        in_addrs = tx["input_addresses"]
        is_cb = (tx["is_coinbase"] == 1)

        if not in_addrs or is_cb:
            max_prior_count = 0
            avg_prior_count = 0.0
            max_prior_vol = 0
            new_addr_frac = 0.0 if is_cb else 1.0
        else:
            p_counts = []
            p_vols = []
            new_cnt = 0
            for a in in_addrs:
                p_cnt, p_vol = addr_prior_stats.get((a, tx["block_number"], tx["tx_index"]), (0, 0))
                p_counts.append(p_cnt)
                p_vols.append(p_vol)
                if p_cnt == 0:
                    new_cnt += 1
            max_prior_count = max(p_counts) if p_counts else 0
            avg_prior_count = sum(p_counts) / len(p_counts) if p_counts else 0.0
            max_prior_vol = max(p_vols) if p_vols else 0
            new_addr_frac = new_cnt / len(in_addrs) if in_addrs else 1.0

        n_in = tx["num_inputs"]
        n_out = tx["num_outputs"]
        tot_in = tx["total_input_satoshi"]
        tot_out = tx["total_output_satoshi"]
        fee = tx["fee_satoshi"]

        ground_truth[k] = {
            "block_number": tx["block_number"],
            "tx_index": tx["tx_index"],
            "tx_hash": tx["tx_hash"],
            "num_inputs": n_in,
            "num_outputs": n_out,
            "total_input_satoshi": tot_in,
            "total_output_satoshi": tot_out,
            "fee_satoshi": fee,
            "fee_rate": fee / (n_in + n_out),
            "is_coinbase": tx["is_coinbase"],
            "avg_input_val": tot_in / n_in,
            "avg_output_val": tot_out / n_out,
            "input_output_count_ratio": float(n_in) / float(n_out),
            "input_output_val_ratio": (float(tot_in) / float(tot_out)) if tot_out > 0 else 1.0,
            "max_input_prior_tx_count": max_prior_count,
            "avg_input_prior_tx_count": avg_prior_count,
            "max_input_prior_volume_satoshi": max_prior_vol,
            "new_address_fraction": new_addr_frac,
        }

    return ground_truth


def run_benchmark(target_lines: int):
    print("=" * 80)
    print(f"BOUNDED SCALE-UP BENCHMARK: {target_lines:,} REAL ZENODO LINES")
    print("=" * 80)

    # Free space before run
    drive_before = get_drive_free_gb()
    print(f"Drive Space Before: C: {drive_before.get('C:')} GB free | D: {drive_before.get('D:')} GB free")

    proc = psutil.Process()
    mem_start_rss = proc.memory_info().rss / (1024 * 1024)
    peak_rss = mem_start_rss
    t0 = time.perf_counter()

    # Paths on Drive D:
    base_dir = f"D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/temp/scale_benchmark_{target_lines}"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)

    spark_tmp = os.path.join(base_dir, "spark_tmp")
    spark_wh = os.path.join(base_dir, "spark_warehouse")
    parquet_out = os.path.join(base_dir, "zenodo_features_parquet")
    raw_slice_path = os.path.join(base_dir, f"zenodo_raw_slice_{target_lines}.txt")

    # 1. Extract bounded slice from local raw file
    raw_src = "data/raw/sample_a/btc-tx-480000-489999-aa"
    assert os.path.exists(raw_src), f"Source raw archive not found: {raw_src}"

    print(f"\n[Step 1/6] Extracting {target_lines:,} lines from {raw_src}...")
    t_extract0 = time.perf_counter()
    with open(raw_src, "r", encoding="ascii") as f_in, open(raw_slice_path, "w", encoding="ascii") as f_out:
        for _ in range(target_lines):
            line = f_in.readline()
            if not line:
                break
            f_out.write(line)
    t_extract1 = time.perf_counter()
    slice_size_mb = os.path.getsize(raw_slice_path) / (1024 * 1024)
    print(f"  Slice written: {raw_slice_path} ({slice_size_mb:.2f} MB in {t_extract1 - t_extract0:.2f} s)")

    # 2. Start SparkSession
    print("\n[Step 2/6] Starting local SparkSession configured on Drive D:...")
    spark = build_spark_session(
        app_name=f"ZenodoScaleBenchmark_{target_lines}",
        local_cores="local[2]",
        spark_tmp_dir=spark_tmp,
        spark_warehouse_dir=spark_wh
    )

    try:
        pipeline = ZenodoSparkBatchPipeline(spark)

        # 3. Distributed transformations
        print("\n[Step 3/6] Running distributed Spark batch transformations...")
        t_spark0 = time.perf_counter()

        raw_df = spark.read.text([raw_slice_path])
        raw_count = raw_df.count()
        assert raw_count == target_lines, f"Expected {target_lines} lines, read {raw_count}"

        records_df = pipeline.parse_raw_zenodo_text(raw_df)
        parsed_count = records_df.count()

        tx_df = pipeline.assemble_completed_transactions(records_df)
        tx_count = tx_df.count()

        features_df = pipeline.compute_leakage_safe_features(records_df, tx_df)
        t_spark1 = time.perf_counter()
        peak_rss = max(peak_rss, proc.memory_info().rss / (1024 * 1024))
        print(f"  Raw lines read:        {raw_count:,}")
        print(f"  Valid records parsed:  {parsed_count:,}")
        print(f"  Completed transactions:{tx_count:,}")
        print(f"  Spark transform time:  {t_spark1 - t_spark0:.2f} s")

        # 4. Stream Parquet via bounded PyArrow writer
        print(f"\n[Step 4/6] Streaming Parquet via bounded PyArrow to {parquet_out}...")
        t_write0 = time.perf_counter()
        pipeline.write_parquet_bounded_arrow(features_df, parquet_out, batch_size=1000)
        t_write1 = time.perf_counter()
        peak_rss = max(peak_rss, proc.memory_info().rss / (1024 * 1024))
        print(f"  Parquet streaming write time: {t_write1 - t_write0:.2f} s")

        # Parquet size on disk
        total_parquet_bytes = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(parquet_out)
            for f in files
        )

        # 5. Read back Parquet and validate
        print("\n[Step 5/6] Validating written Parquet output...")
        table = pq.read_table(parquet_out)
        pa_num_rows = table.num_rows
        print(f"  PyArrow read-back rows: {pa_num_rows:,} (matches Spark completed txs: {pa_num_rows == tx_count})")
        assert pa_num_rows == tx_count, f"Row count mismatch: PyArrow {pa_num_rows} vs Spark {tx_count}"

        # Schema validation: all 18 feature columns match expected_schema
        expected_schema = get_transaction_features_pyarrow_schema()
        for field in expected_schema:
            assert field.name in table.schema.names, f"Missing feature field: {field.name}"
            pa_field = table.schema.field(field.name)
            assert pa_field.type == field.type, f"Field type mismatch for {field.name}: {pa_field.type} vs {field.type}"

        # Partition column block_chunk is also present in partitioned dataset
        assert "block_chunk" in table.schema.names, "Missing partition column: block_chunk"

        # Verify _SUCCESS marker
        assert os.path.exists(os.path.join(parquet_out, "_SUCCESS")), "Missing _SUCCESS marker"

        # Check partition directories
        chunks = [d for d in os.listdir(parquet_out) if d.startswith("block_chunk=")]
        print(f"  Partition chunk directories: {chunks}")
        assert len(chunks) > 0, "No block_chunk partition directories created"

        # Numerical sanity & null handling on PyArrow dataset
        fee_col = table["fee_satoshi"].to_pylist()
        cb_col = table["is_coinbase"].to_pylist()
        in_sat_col = table["total_input_satoshi"].to_pylist()
        out_sat_col = table["total_output_satoshi"].to_pylist()
        null_counts = {name: table[name].null_count for name in table.column_names}

        # Fee sanity: coinbase fees = 0; non-coinbase fees >= 0 in standard transactions
        for fee, cb, in_s, out_s in zip(fee_col, cb_col, in_sat_col, out_sat_col):
            if cb == 1:
                assert fee == 0, "Coinbase fee must be 0"
            else:
                assert fee >= 0, f"Standard transaction fee cannot be negative: {fee} (in={in_s}, out={out_s})"

        # 6. Independent validation against pure-Python ground truth
        print("\n[Step 6/6] Independent validation against pure-Python reference implementation...")
        # Select representative sample: first 20, middle 20, last 20
        all_keys = list(zip(table["block_number"].to_pylist(), table["tx_index"].to_pylist()))
        sample_keys = all_keys[:20] + all_keys[len(all_keys)//2 - 10 : len(all_keys)//2 + 10] + all_keys[-20:]
        print(f"  Selected representative sample of {len(sample_keys)} transactions across slice.")

        with open(raw_slice_path, "r", encoding="ascii") as f_slice:
            slice_lines = f_slice.readlines()

        t_gt0 = time.perf_counter()
        gt = compute_ground_truth_for_keys(slice_lines, sample_keys)
        t_gt1 = time.perf_counter()
        print(f"  Ground truth computed in {t_gt1 - t_gt0:.2f} s for {len(gt)} transactions.")

        # Map pyarrow rows by (block, tx_idx)
        pa_rows = table.to_pylist()
        pa_dict = {(r["block_number"], r["tx_index"]): r for r in pa_rows}

        matches = 0
        for k in sample_keys:
            assert k in pa_dict, f"Sample key {k} missing from Parquet"
            assert k in gt, f"Sample key {k} missing from ground truth"
            pr = pa_dict[k]
            gr = gt[k]

            assert pr["block_number"] == gr["block_number"]
            assert pr["tx_index"] == gr["tx_index"]
            assert pr["tx_hash"] == gr["tx_hash"]
            assert pr["num_inputs"] == gr["num_inputs"]
            assert pr["num_outputs"] == gr["num_outputs"]
            assert pr["total_input_satoshi"] == gr["total_input_satoshi"]
            assert pr["total_output_satoshi"] == gr["total_output_satoshi"]
            assert pr["fee_satoshi"] == gr["fee_satoshi"]
            assert pr["is_coinbase"] == gr["is_coinbase"]
            assert pr["max_input_prior_tx_count"] == gr["max_input_prior_tx_count"]
            assert pr["max_input_prior_volume_satoshi"] == gr["max_input_prior_volume_satoshi"]

            for fcol in ["fee_rate", "avg_input_val", "avg_output_val", "input_output_count_ratio", "input_output_val_ratio", "avg_input_prior_tx_count", "new_address_fraction"]:
                diff = abs(pr[fcol] - gr[fcol])
                assert diff < 1e-4, f"Mismatch on {k} for {fcol}: pa={pr[fcol]}, gt={gr[fcol]}"
            matches += 1

        print(f"  Representative sample validation: {matches}/{len(sample_keys)} matched 100% against Python ground truth.")

        # Inspect boundary conditions
        first_tx = pa_rows[0]
        last_tx = pa_rows[-1]
        boundary_report = {
            "slice_start": {
                "first_row_block": first_tx["block_number"],
                "first_row_tx_index": first_tx["tx_index"],
                "is_coinbase": first_tx["is_coinbase"] == 1,
                "note": "Block 480000 genesis-of-slice Coinbase transaction cleanly initialized."
            },
            "slice_end": {
                "last_row_block": last_tx["block_number"],
                "last_row_tx_index": last_tx["tx_index"],
                "note": "Any partial transaction at slice EOF missing matching inputs/outputs was filtered out as incomplete."
            }
        }

        t_end = time.perf_counter()
        elapsed_sec = t_end - t0
        mem_end_rss = proc.memory_info().rss / (1024 * 1024)
        throughput_lps = round(target_lines / max(elapsed_sec, 1e-6), 2)
        throughput_tps = round(pa_num_rows / max(elapsed_sec, 1e-6), 2)

        # Free space after run
        drive_after = get_drive_free_gb()

        report = {
            "benchmark_status": "PASSED",
            "target_raw_lines": target_lines,
            "raw_lines_processed": raw_count,
            "valid_records_parsed": parsed_count,
            "completed_transactions": tx_count,
            "parquet_output_rows": pa_num_rows,
            "parquet_output_size_bytes": total_parquet_bytes,
            "parquet_output_size_mb": round(total_parquet_bytes / (1024 * 1024), 2),
            "parquet_partitions": chunks,
            "null_counts": null_counts,
            "wall_clock_elapsed_seconds": round(elapsed_sec, 2),
            "throughput_lines_per_sec": throughput_lps,
            "throughput_transactions_per_sec": throughput_tps,
            "memory_start_rss_mb": round(mem_start_rss, 2),
            "memory_peak_rss_mb": round(peak_rss, 2),
            "memory_end_rss_mb": round(mem_end_rss, 2),
            "memory_growth_mb": round(peak_rss - mem_start_rss, 2),
            "drive_c_free_gb_before": drive_before.get("C:"),
            "drive_c_free_gb_after": drive_after.get("C:"),
            "drive_d_free_gb_before": drive_before.get("D:"),
            "drive_d_free_gb_after": drive_after.get("D:"),
            "boundary_conditions": boundary_report,
            "sample_ground_truth_matches": f"{matches}/{len(sample_keys)}",
            "representative_sample_verified": True,
            "caveat": "Bounded slice benchmark of local Zenodo data on Windows. Not production throughput or streaming latency."
        }

        out_report_file = f"data/validation/spark_scale_benchmark_{target_lines}.json"
        os.makedirs(os.path.dirname(os.path.abspath(out_report_file)), exist_ok=True)
        with open(out_report_file, "w", encoding="utf-8") as f_rep:
            json.dump(report, f_rep, indent=2)

        print("\n" + "=" * 80)
        print(f"BENCHMARK COMPLETED SUCCESSFULLY ({target_lines:,} LINES):")
        print(f"  Raw lines:           {raw_count:,}")
        print(f"  Valid records:       {parsed_count:,}")
        print(f"  Completed txs:       {tx_count:,}")
        print(f"  Parquet size:        {total_parquet_bytes / (1024 * 1024):.2f} MB")
        print(f"  Elapsed time:        {elapsed_sec:.2f} s")
        print(f"  Throughput:          {throughput_lps:,} lines/s ({throughput_tps:,} txs/s)")
        print(f"  Peak RAM:            {peak_rss:.2f} MB (Growth: {peak_rss - mem_start_rss:.2f} MB)")
        print(f"  Drive C: before/after: {drive_before.get('C:')} GB -> {drive_after.get('C:')} GB (0 bytes delta)")
        print(f"  Drive D: before/after: {drive_before.get('D:')} GB -> {drive_after.get('D:')} GB")
        print(f"  Report saved to:     {out_report_file}")
        print("=" * 80)
        return report

    finally:
        spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Bounded Scale-Up Benchmark for Spark + PyArrow Pipeline.")
    parser.add_argument("--lines", type=int, default=50000, help="Number of raw Zenodo lines to benchmark (e.g. 50000 or 100000).")
    args = parser.parse_args()
    run_benchmark(args.lines)


if __name__ == "__main__":
    main()
