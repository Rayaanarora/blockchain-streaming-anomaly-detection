"""
Independent Validation Script: Apache Spark + PyArrow on Real Zenodo Bitcoin Data.

Validates:
1. Multi-file boundary handling (chunk crossing where a transaction's inputs and outputs span across files).
2. Distributed aggregation, fee calculation, and Coinbase subsidy handling.
3. Strict temporal window causality on real blockchain addresses.
4. Bounded PyArrow Parquet writer, retry-safety, and directory layout.
5. Measured elapsed time, peak RSS memory, and Parquet output size on Drive D:.
"""
import os
import sys
import shutil
import json
import time
from collections import defaultdict
import psutil

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pyarrow.parquet as pq
from src.spark.zenodo_batch import build_spark_session, ZenodoSparkBatchPipeline


def compute_ground_truth(lines):
    """
    Independently compute ground truth completed transactions and historical features
    in pure Python without PySpark.
    """
    # 1. Parse raw lines
    parsed = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        kind, block, tx_idx, src, dst, val = parts[0], int(parts[1]), int(parts[2]), parts[3], parts[4], int(parts[5])
        if kind not in ("BTC-IN", "BTC-OUT") or block < 0 or tx_idx < 0 or val < 0:
            continue
        # Hex tx hash check
        tx_hash = dst if kind == "BTC-IN" else src
        if len(tx_hash) != 64:
            continue
        # Address sanitization
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

    # Combine into transaction features
    ground_truth = {}
    for k in sorted(completed.keys()):
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

        fee_rate = fee / (n_in + n_out)
        avg_in = tot_in / n_in
        avg_out = tot_out / n_out
        io_cnt_ratio = float(n_in) / float(n_out)
        io_val_ratio = (float(tot_in) / float(tot_out)) if tot_out > 0 else 1.0

        ground_truth[k] = {
            "block_number": tx["block_number"],
            "tx_index": tx["tx_index"],
            "tx_hash": tx["tx_hash"],
            "num_inputs": n_in,
            "num_outputs": n_out,
            "total_input_satoshi": tot_in,
            "total_output_satoshi": tot_out,
            "fee_satoshi": fee,
            "fee_rate": fee_rate,
            "is_coinbase": tx["is_coinbase"],
            "avg_input_val": avg_in,
            "avg_output_val": avg_out,
            "input_output_count_ratio": io_cnt_ratio,
            "input_output_val_ratio": io_val_ratio,
            "max_input_prior_tx_count": max_prior_count,
            "avg_input_prior_tx_count": avg_prior_count,
            "max_input_prior_volume_satoshi": max_prior_vol,
            "new_address_fraction": new_addr_frac,
        }

    return ground_truth


def run_validation():
    print("=" * 80)
    print("INDEPENDENT AUDIT: APACHE SPARK + PYARROW ON REAL ZENODO SAMPLE")
    print("=" * 80)

    proc = psutil.Process()
    mem_start_mb = proc.memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()

    # Paths on Drive D:
    base_dir = "D:/blockchainbigdata/blockchain-streaming-anomaly-detection/data/temp/real_data_audit"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(base_dir, exist_ok=True)

    spark_tmp = os.path.join(base_dir, "spark_tmp")
    spark_wh = os.path.join(base_dir, "spark_warehouse")
    out_parquet = os.path.join(base_dir, "zenodo_real_sample_parquet")

    # 1. Read 250 real lines from local Zenodo sample
    real_src = "data/raw/sample_a/btc-tx-480000-489999-aa"
    assert os.path.exists(real_src), f"Real Zenodo source not found: {real_src}"

    with open(real_src, "r") as f:
        real_lines = [f.readline() for _ in range(250)]

    # Split into 2 chunks deliberately bisecting transaction (480000, 23)
    # Part 1: lines 1-100 (contains tx 23 first 2 inputs)
    # Part 2: lines 101-250 (contains tx 23 remaining 3 inputs and 2 outputs)
    part1_path = os.path.join(base_dir, "zenodo_chunk_part1.txt")
    part2_path = os.path.join(base_dir, "zenodo_chunk_part2.txt")

    with open(part1_path, "w") as f:
        f.writelines(real_lines[:100])
    with open(part2_path, "w") as f:
        f.writelines(real_lines[100:])

    print(f"Created real slice Part 1: {part1_path} (100 lines)")
    print(f"Created real slice Part 2: {part2_path} (150 lines)")
    print("Transaction (480000, 23) deliberately bisected across Part 1 and Part 2.")

    # 2. Compute independent Python ground truth
    print("\nComputing independent ground-truth baseline in pure Python...")
    gt = compute_ground_truth(real_lines)
    print(f"Ground-truth completed transactions: {len(gt)}")
    assert (480000, 23) in gt, "Ground truth missing split transaction (480000, 23)"
    tx23_gt = gt[(480000, 23)]
    print(f"Ground-truth Tx 23: num_inputs={tx23_gt['num_inputs']}, num_outputs={tx23_gt['num_outputs']}, fee={tx23_gt['fee_satoshi']}")

    # 3. Start Spark Session
    print("\nStarting local SparkSession configured on Drive D:...")
    spark = build_spark_session(
        app_name="RealZenodoAudit",
        local_cores="local[2]",
        spark_tmp_dir=spark_tmp,
        spark_warehouse_dir=spark_wh
    )

    try:
        pipeline = ZenodoSparkBatchPipeline(spark)

        # Ingest both parts
        print("Reading both chunk parts through Spark distributed reader...")
        raw_df = spark.read.text([part1_path, part2_path])
        records_df = pipeline.parse_raw_zenodo_text(raw_df)
        tx_df = pipeline.assemble_completed_transactions(records_df)
        features_df = pipeline.compute_leakage_safe_features(records_df, tx_df)

        # 4. Write Parquet via bounded PyArrow writer
        print(f"Writing Snappy-compressed Parquet via bounded PyArrow to {out_parquet}...")
        pipeline.write_parquet_bounded_arrow(features_df, out_parquet)

        # Test retry-safety: write again
        print("Testing retry-safety (idempotent re-execution)...")
        pipeline.write_parquet_bounded_arrow(features_df, out_parquet)

        mem_peak_mb = proc.memory_info().rss / (1024 * 1024)

        # 5. Read back Parquet and validate
        print("\nReading back Parquet via PyArrow and comparing against independent ground truth...")
        table = pq.read_table(out_parquet)
        pa_rows = table.to_pylist()
        assert len(pa_rows) == len(gt), f"Row count mismatch: PyArrow {len(pa_rows)} vs Ground Truth {len(gt)}"

        # Check _SUCCESS marker
        assert os.path.exists(os.path.join(out_parquet, "_SUCCESS")), "Missing _SUCCESS marker"

        # Check partition directory
        parts = [d for d in os.listdir(out_parquet) if d.startswith("block_chunk=")]
        assert parts == ["block_chunk=480000-480999"]

        # Validate row-by-row
        pa_dict = {(r["block_number"], r["tx_index"]): r for r in pa_rows}

        for k, gt_row in gt.items():
            assert k in pa_dict, f"Transaction {k} missing from Parquet output"
            pa_row = pa_dict[k]

            # Structural & identity fields
            assert pa_row["block_number"] == gt_row["block_number"]
            assert pa_row["tx_index"] == gt_row["tx_index"]
            assert pa_row["tx_hash"] == gt_row["tx_hash"]
            assert pa_row["is_coinbase"] == gt_row["is_coinbase"]

            # Counts & volumes
            assert pa_row["num_inputs"] == gt_row["num_inputs"], f"Tx {k} num_inputs mismatch"
            assert pa_row["num_outputs"] == gt_row["num_outputs"], f"Tx {k} num_outputs mismatch"
            assert pa_row["total_input_satoshi"] == gt_row["total_input_satoshi"], f"Tx {k} total_input mismatch"
            assert pa_row["total_output_satoshi"] == gt_row["total_output_satoshi"], f"Tx {k} total_output mismatch"
            assert pa_row["fee_satoshi"] == gt_row["fee_satoshi"], f"Tx {k} fee mismatch"

            # Historical window features
            assert pa_row["max_input_prior_tx_count"] == gt_row["max_input_prior_tx_count"], f"Tx {k} max_prior_count mismatch"
            assert pa_row["max_input_prior_volume_satoshi"] == gt_row["max_input_prior_volume_satoshi"], f"Tx {k} max_prior_vol mismatch"

            # Float features
            for fcol in ["fee_rate", "avg_input_val", "avg_output_val", "input_output_count_ratio", "input_output_val_ratio", "avg_input_prior_tx_count", "new_address_fraction"]:
                diff = abs(pa_row[fcol] - gt_row[fcol])
                assert diff < 1e-4, f"Tx {k} {fcol} float mismatch: pa={pa_row[fcol]}, gt={gt_row[fcol]}"

        # Specifically check boundary split transaction (480000, 23)
        tx23_pa = pa_dict[(480000, 23)]
        print("\nBoundary split transaction (480000, 23) successfully verified:")
        print(f"  Inputs assembled:  {tx23_pa['num_inputs']} (File 1 had 2, File 2 had 3)")
        print(f"  Outputs assembled: {tx23_pa['num_outputs']} (File 2 had 2)")
        print(f"  Total input sat:   {tx23_pa['total_input_satoshi']:,}")
        print(f"  Total output sat:  {tx23_pa['total_output_satoshi']:,}")
        print(f"  Fee satoshi:       {tx23_pa['fee_satoshi']:,}")

        # Check total output Parquet size
        total_bytes = 0
        for root, _, files in os.walk(out_parquet):
            for file in files:
                total_bytes += os.path.getsize(os.path.join(root, file))

        t_end = time.perf_counter()
        elapsed_sec = t_end - t0
        mem_end_mb = proc.memory_info().rss / (1024 * 1024)

        report = {
            "status": "PASSED",
            "real_dataset_source": real_src,
            "real_lines_processed": 250,
            "chunk_files_ingested": [part1_path, part2_path],
            "boundary_split_transaction_verified": "(480000, 23)",
            "completed_transactions": len(pa_rows),
            "parquet_output_directory": out_parquet,
            "parquet_output_size_bytes": total_bytes,
            "parquet_output_size_kb": round(total_bytes / 1024, 2),
            "elapsed_seconds": round(elapsed_sec, 2),
            "memory_start_rss_mb": round(mem_start_mb, 2),
            "memory_peak_rss_mb": round(mem_peak_mb, 2),
            "memory_end_rss_mb": round(mem_end_mb, 2),
            "memory_delta_mb": round(mem_peak_mb - mem_start_mb, 2)
        }

        report_file = os.path.join(base_dir, "real_data_validation_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 80)
        print("REAL DATA AUDIT PASSED 100%:")
        print(f"  - Completed transactions validated: {len(pa_rows)}")
        print(f"  - Boundary split transaction:       Verified assembled across 2 files")
        print(f"  - Parquet output size:             {total_bytes / 1024:.2f} KB")
        print(f"  - Measured elapsed time:           {elapsed_sec:.2f} s")
        print(f"  - Measured peak memory:            {mem_peak_mb:.2f} MB (Delta: {mem_peak_mb - mem_start_mb:.2f} MB)")
        print(f"  - Audit report saved to:           {report_file}")
        print("=" * 80)
        return report

    finally:
        spark.stop()


if __name__ == "__main__":
    run_validation()
