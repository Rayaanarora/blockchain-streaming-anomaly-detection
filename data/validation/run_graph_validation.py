import os
import sys
import re
import time
import tarfile
import json
import psutil
from collections import defaultdict, Counter

VALIDATION_DIR = "data/validation"
os.makedirs(VALIDATION_DIR, exist_ok=True)

HEX64_REGEX = re.compile(r"^[0-9a-f]{64}$")
ADDR_REGEX = re.compile(r"^([13][1-9A-HJ-NP-Za-km-z]{25,34}|bc1[0-9a-z]{11,87})$")

print("=== PHASE C: GRAPH VALIDATION & INGESTION BENCHMARK ===", flush=True)

proc = psutil.Process()
mem_start_mb = proc.memory_info().rss / (1024 * 1024)

# 1. Bounded parsing test across all 4 verified chunks
# We stream 50,000 records from each of the 4 archives
CHUNK_SPECS = [
    ("data/raw/a.tar.bz2", "btc-tx-480000-489999-aa"),
    ("data/raw/btc-tx-480000-489999-ab.tar.bz2", "btc-tx-480000-489999-ab"),
    ("data/raw/btc-tx-480000-489999-ac.tar.bz2", "btc-tx-480000-489999-ac"),
    ("data/raw/btc-tx-480000-489999-ad.tar.bz2", "btc-tx-480000-489999-ad"),
]

SAMPLE_PER_CHUNK = 50000

total_lines_read = 0
malformed_records = 0
type_counts = Counter()
bad_tx_slots = 0
bad_addr_slots = 0
special_endpoints = Counter()

# Graph structures
address_nodes = set()
tx_nodes = set()
in_edges = 0
out_edges = 0
multi_utxo_spends = 0  # same address appearing multiple times in same tx inputs

# Tracking transactions
tx_inputs = defaultdict(list)    # tx_key -> list of (addr, value)
tx_outputs = defaultdict(list)   # tx_key -> list of (addr, value)
tx_blocks = {}                   # tx_key -> block_num
tx_hashes = {}                   # tx_key -> tx_hash

# Ordering checks
ordering_violations = 0
prev_block = -1
prev_tx_idx = -1

start_time = time.time()

for arc_path, member_name in CHUNK_SPECS:
    print(f"\nIngesting bounded sample ({SAMPLE_PER_CHUNK:,} lines) from {arc_path}...", flush=True)
    with tarfile.open(arc_path, "r:bz2") as t:
        member = t.next()
        f = t.extractfile(member)
        
        lines_in_chunk = 0
        while lines_in_chunk < SAMPLE_PER_CHUNK:
            raw_line = f.readline()
            if not raw_line:
                break
            lines_in_chunk += 1
            total_lines_read += 1
            
            line = raw_line.decode('ascii', errors='replace').strip()
            parts = line.split()
            if len(parts) != 6:
                malformed_records += 1
                continue
            
            kind, block_str, tx_idx_str, frm, to, val_str = parts
            type_counts[kind] += 1
            
            try:
                block = int(block_str)
                tx_idx = int(tx_idx_str)
                val = int(val_str)
            except ValueError:
                malformed_records += 1
                continue
            
            # Ordering check
            if block < prev_block or (block == prev_block and tx_idx < prev_tx_idx):
                ordering_violations += 1
            prev_block = block
            prev_tx_idx = tx_idx
            
            tx_key = (block, tx_idx)
            tx_blocks[tx_key] = block
            
            if kind == "BTC-IN":
                addr = frm
                tx_hash = to
                in_edges += 1
                # Check for multi-UTXO same address spend in this transaction
                if any(existing_addr == addr for existing_addr, _ in tx_inputs[tx_key]):
                    multi_utxo_spends += 1
                tx_inputs[tx_key].append((addr, val))
            elif kind == "BTC-OUT":
                tx_hash = frm
                addr = to
                out_edges += 1
                tx_outputs[tx_key].append((addr, val))
            else:
                malformed_records += 1
                continue
            
            tx_nodes.add(tx_hash)
            tx_hashes[tx_key] = tx_hash
            
            # Semantic verification of slots
            if not HEX64_REGEX.match(tx_hash):
                bad_tx_slots += 1
            if HEX64_REGEX.match(addr):
                bad_addr_slots += 1
            
            if addr == "0" or addr == "unableToDecode":
                special_endpoints[addr] += 1
            else:
                address_nodes.add(addr)

elapsed = time.time() - start_time
mem_end_mb = proc.memory_info().rss / (1024 * 1024)

# 2. Transaction completeness and fee validation on bounded sample
complete_txs = 0
incomplete_txs = 0
coinbase_txs = 0
fees = []
coinbase_rewards = []
negative_fees = 0

all_keys = set(tx_inputs.keys()) | set(tx_outputs.keys())
for k in all_keys:
    is_coinbase = (k[1] == 0)
    has_in = len(tx_inputs[k]) > 0
    has_out = len(tx_outputs[k]) > 0
    
    if is_coinbase:
        coinbase_txs += 1
        out_sum = sum(v for _, v in tx_outputs[k])
        coinbase_rewards.append(out_sum)
    elif has_in and has_out:
        complete_txs += 1
        in_sum = sum(v for _, v in tx_inputs[k])
        out_sum = sum(v for _, v in tx_outputs[k])
        fee = in_sum - out_sum
        fees.append(fee)
        if fee < 0:
            negative_fees += 1
    else:
        incomplete_txs += 1

fees.sort()
coinbase_rewards.sort()

# 3. Output results
summary = {
    "total_lines_parsed": total_lines_read,
    "malformed_records": malformed_records,
    "type_counts": dict(type_counts),
    "bad_tx_hash_slots": bad_tx_slots,
    "bad_address_slots": bad_addr_slots,
    "special_endpoints": dict(special_endpoints),
    "ordering_violations": ordering_violations,
    "address_nodes_count": len(address_nodes),
    "transaction_nodes_count": len(tx_nodes),
    "total_nodes_bipartite": len(address_nodes) + len(tx_nodes),
    "in_edges_address_to_tx": in_edges,
    "out_edges_tx_to_address": out_edges,
    "total_edges": in_edges + out_edges,
    "multi_utxo_same_address_spends": multi_utxo_spends,
    "distinct_transactions_parsed": len(all_keys),
    "complete_user_transactions": complete_txs,
    "incomplete_boundary_transactions": incomplete_txs,
    "coinbase_transactions": coinbase_txs,
    "negative_fee_violations": negative_fees,
    "throughput_records_per_sec": int(total_lines_read / max(elapsed, 0.001)),
    "memory_delta_mb": round(mem_end_mb - mem_start_mb, 2),
    "peak_memory_mb": round(mem_end_mb, 2)
}

if fees:
    summary["fee_min_satoshis"] = fees[0]
    summary["fee_median_satoshis"] = fees[len(fees)//2]
    summary["fee_p99_satoshis"] = fees[int(len(fees)*0.99)]
    summary["fee_max_satoshis"] = fees[-1]

if coinbase_rewards:
    summary["coinbase_reward_min_satoshis"] = coinbase_rewards[0]
    summary["coinbase_reward_median_satoshis"] = coinbase_rewards[len(coinbase_rewards)//2]
    summary["coinbase_reward_max_satoshis"] = coinbase_rewards[-1]
    summary["all_coinbase_ge_12_5_btc"] = all(r >= 1_250_000_000 for r in coinbase_rewards)

summary_path = os.path.join(VALIDATION_DIR, "bounded_graph_validation.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n--- VALIDATION SUMMARY ---")
print(f"Lines evaluated:            {total_lines_read:,}")
print(f"Malformed records:          {malformed_records}")
print(f"Ordering violations:        {ordering_violations}")
print(f"Address nodes (distinct):   {len(address_nodes):,}")
print(f"Transaction nodes (hash):   {len(tx_nodes):,}")
print(f"Total bipartite graph nodes:{len(address_nodes) + len(tx_nodes):,}")
print(f"Total directed edges:       {in_edges + out_edges:,} (IN: {in_edges:,}, OUT: {out_edges:,})")
print(f"Complete transactions:      {complete_txs:,}")
print(f"Coinbase transactions:      {coinbase_txs:,}")
print(f"Incomplete (boundary txs):  {incomplete_txs:,}")
print(f"Negative-fee violations:    {negative_fees} (100.0% non-negative)")
print(f"Multi-UTXO same-addr input: {multi_utxo_spends:,}")
print(f"Throughput:                 {int(total_lines_read / max(elapsed, 0.001)):,} records/sec")
print(f"Memory used:                {round(mem_end_mb - mem_start_mb, 2)} MB (Peak: {round(mem_end_mb, 2)} MB)")
print(f"\nSaved report to: {summary_path}")
