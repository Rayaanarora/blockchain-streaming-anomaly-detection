"""
Comprehensive unit and integration test suite for the modular streaming pipeline.

Tests:
- Multi-input and multi-output transaction assembly and fee calculation.
- Transactions split across chunk/archive boundaries.
- Coinbase transaction handling, fee isolation, and reward rules.
- Malformed rows, non-hex tx hashes, invalid addresses, and duplicate inputs.
- Historical feature extractor causality and zero future-leakage verification.
- Deterministic event replay ordering and graph topology verification.
"""
import pytest
import numpy as np
from src.data.schema import (
    TransactionRecord,
    RecordType,
    parse_raw_line,
    sanitize_endpoint_address,
    HEX64_REGEX,
    ADDR_REGEX,
)
from src.data.boundary import TransactionBoundaryBuffer, CompletedTransaction
from src.graph.bipartite import BipartiteTransactionGraph
from src.features.extractor import HistoricalFeatureExtractor, FEATURE_NAMES
from src.models.baselines import CostSensitiveAnomalyModel


# ==============================================================================
# 1. Multi-Input and Multi-Output Transactions
# ==============================================================================

def test_multi_input_multi_output_assembly():
    """Verify multi-input and multi-output records correctly assemble into CompletedTransaction."""
    buffer = TransactionBoundaryBuffer(sanitize_endpoints=False)

    records = [
        TransactionRecord(RecordType.BTC_IN, 480100, 5, "1InAddrA", "aa" * 32, 100000),
        TransactionRecord(RecordType.BTC_IN, 480100, 5, "1InAddrB", "aa" * 32, 250000),
        TransactionRecord(RecordType.BTC_OUT, 480100, 5, "aa" * 32, "1OutAddrC", 200000),
        TransactionRecord(RecordType.BTC_OUT, 480100, 5, "aa" * 32, "1OutAddrD", 140000),
    ]

    for rec in records:
        buffer.process_record(rec)

    completed = list(buffer.flush_all())
    assert len(completed) == 1
    tx = completed[0]

    assert tx.block_number == 480100
    assert tx.tx_index == 5
    assert tx.num_inputs == 2
    assert tx.num_outputs == 2
    assert tx.total_input_satoshi == 350000
    assert tx.total_output_satoshi == 340000
    assert tx.fee_satoshi == 10000
    assert not tx.is_coinbase


def test_bipartite_graph_multi_io_no_cartesian_product():
    """Verify bipartite graph strictly creates Address <-> Tx edges without Cartesian product."""
    graph = BipartiteTransactionGraph()

    tx = CompletedTransaction(
        block_number=480100,
        tx_index=5,
        tx_hash="bb" * 32,
        inputs=[("1InA", 50000), ("1InB", 70000), ("1InC", 30000)],
        outputs=[("1OutD", 80000), ("1OutE", 60000)],
        is_coinbase=False
    )
    graph.add_transaction(tx)

    # 3 inputs + 2 outputs = 5 bipartite edges
    assert graph.num_edges == 5
    assert graph.num_addresses == 5
    assert graph.num_transactions == 1

    # Verify each input has out-edge to tx_hash
    for in_addr in ["1InA", "1InB", "1InC"]:
        assert graph.get_address_out_degree(in_addr) == 1
        assert graph.get_address_in_degree(in_addr) == 0

    # Verify tx has 3 in-edges and 2 out-edges
    assert graph.get_transaction_in_degree("bb" * 32) == 3
    assert graph.get_transaction_out_degree("bb" * 32) == 2

    # Verify no direct address-to-address edges exist
    assert len(graph.addr_out_edges["1InA"]) == 1
    target, _ = graph.addr_out_edges["1InA"][0]
    assert target == "bb" * 32


# ==============================================================================
# 2. Boundary-Split Transaction Assembly
# ==============================================================================

def test_boundary_split_transaction_retention():
    """Verify transactions split across chunk boundaries are retained in buffer until complete."""
    buffer = TransactionBoundaryBuffer(sanitize_endpoints=False)

    # Chunk 1 ends with only inputs for tx (480521, 839)
    chunk1_records = [
        TransactionRecord(RecordType.BTC_IN, 480521, 838, "1PrevIn", "11" * 32, 50000),
        TransactionRecord(RecordType.BTC_OUT, 480521, 838, "11" * 32, "1PrevOut", 45000),
        TransactionRecord(RecordType.BTC_IN, 480521, 839, "3Addr1", "22" * 32, 1680000),
        TransactionRecord(RecordType.BTC_IN, 480521, 839, "3Addr2", "22" * 32, 6340000),
    ]

    emitted_chunk1 = []
    for rec in chunk1_records:
        for tx in buffer.process_record(rec):
            emitted_chunk1.append(tx)

    # Tx 838 is ready when advancing to 839
    assert len(emitted_chunk1) == 1
    assert emitted_chunk1[0].tx_index == 838

    # Flush completed at chunk boundary: Tx 839 must NOT be flushed because it has 0 outputs
    boundary_flush = list(buffer.flush_completed())
    assert len(boundary_flush) == 0
    assert buffer.pending_count == 1
    assert (480521, 839) in buffer._pending

    # Chunk 2 supplies the missing outputs for Tx 839 and starts Tx 840
    chunk2_records = [
        TransactionRecord(RecordType.BTC_OUT, 480521, 839, "22" * 32, "1Addr3", 5000000),
        TransactionRecord(RecordType.BTC_OUT, 480521, 839, "22" * 32, "3Addr4", 2891900),
        TransactionRecord(RecordType.BTC_IN, 480521, 840, "1NextIn", "33" * 32, 100000),
        TransactionRecord(RecordType.BTC_OUT, 480521, 840, "33" * 32, "1NextOut", 95000),
    ]

    emitted_chunk2 = []
    for rec in chunk2_records:
        for tx in buffer.process_record(rec):
            emitted_chunk2.append(tx)

    # When advancing to 840, Tx 839 is emitted fully resolved!
    assert len(emitted_chunk2) >= 1
    tx_839 = emitted_chunk2[0]
    assert tx_839.tx_index == 839
    assert tx_839.num_inputs == 2
    assert tx_839.num_outputs == 2
    assert tx_839.total_input_satoshi == 8020000
    assert tx_839.total_output_satoshi == 7891900
    assert tx_839.fee_satoshi == 128100


# ==============================================================================
# 3. Coinbase Transactions and Endpoint Isolation
# ==============================================================================

def test_coinbase_handling_and_fee_isolation():
    """Verify coinbase transactions (tx_index == 0) are handled separately with no fee."""
    buffer = TransactionBoundaryBuffer(sanitize_endpoints=True)

    cb_in = TransactionRecord(
        RecordType.BTC_IN, 480000, 0, "0",
        "5ade05890effc4adce70fbe97908bb803b10dc4a6247615496b21538f132c0c6",
        1397458013
    )
    cb_out = TransactionRecord(
        RecordType.BTC_OUT, 480000, 0,
        "5ade05890effc4adce70fbe97908bb803b10dc4a6247615496b21538f132c0c6",
        "1CK6KHY6MHgYvmRQ4PAafKYDrg1ejbH1cE",
        1397458013
    )
    # Next transaction to trigger emission
    next_in = TransactionRecord(
        RecordType.BTC_IN, 480000, 1, "1UserA",
        "f1a3b08e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef2",
        50000
    )

    emitted = []
    for r in [cb_in, cb_out, next_in]:
        for tx in buffer.process_record(r):
            emitted.append(tx)

    assert len(emitted) == 1
    cb_tx = emitted[0]
    assert cb_tx.is_coinbase
    assert cb_tx.tx_index == 0
    assert cb_tx.fee_satoshi == 0  # Not a standard fee
    assert cb_tx.total_output_satoshi >= 1_250_000_000  # >= 12.5 BTC base reward


def test_special_endpoint_isolation_prevents_supernodes():
    """Verify unableToDecode and 0 endpoints are partitioned to avoid supernodes."""
    tx_hash1 = "11" * 32
    tx_hash2 = "22" * 32

    # Two different transactions with unableToDecode
    addr1 = sanitize_endpoint_address("unableToDecode", tx_hash1, 0)
    addr2 = sanitize_endpoint_address("unableToDecode", tx_hash2, 0)
    assert addr1 != addr2
    assert addr1.startswith("unableToDecode_")
    assert addr2.startswith("unableToDecode_")

    # Coinbase source 0
    cb_src1 = sanitize_endpoint_address("0", tx_hash1)
    cb_src2 = sanitize_endpoint_address("0", tx_hash2)
    assert cb_src1 != cb_src2
    assert cb_src1.startswith("coinbase_source_")


# ==============================================================================
# 4. Malformed and Duplicate Records
# ==============================================================================

def test_malformed_records_rejection():
    """Verify parse_raw_line strictly rejects invalid records."""
    # Invalid length
    with pytest.raises(ValueError, match="Malformed row"):
        parse_raw_line("BTC-IN 480000 1 1addr 500")

    # Invalid record kind
    with pytest.raises(ValueError, match="Invalid record type"):
        parse_raw_line("ETH-IN 480000 1 1addr " + "aa" * 32 + " 500")

    # Invalid block number
    with pytest.raises(ValueError, match="Invalid block number"):
        parse_raw_line("BTC-IN badblock 1 1addr " + "aa" * 32 + " 500")

    # Invalid satoshi value
    with pytest.raises(ValueError, match="Negative satoshi value"):
        parse_raw_line("BTC-IN 480000 1 1addr " + "aa" * 32 + " -500")

    # Non-64 hex tx hash
    with pytest.raises(ValueError, match="not a 64-character lowercase hex"):
        parse_raw_line("BTC-IN 480000 1 1addr not_a_hex_tx 500")

    # Tx hash placed into address slot
    with pytest.raises(ValueError, match="Address slot contains a 64-hex transaction hash"):
        parse_raw_line("BTC-IN 480000 1 " + "aa" * 32 + " " + "bb" * 32 + " 500")


# ==============================================================================
# 5. Feature Extraction Causality and Leakage Prevention
# ==============================================================================

def test_feature_extractor_temporal_causality():
    """
    Verify historical feature extractor computes features strictly from PAST transactions.
    Address history must not incorporate the current transaction until AFTER feature extraction.
    """
    extractor = HistoricalFeatureExtractor()

    # Tx 1: Brand new address 1Alice sends to 1Bob
    tx1 = CompletedTransaction(
        block_number=480000,
        tx_index=1,
        tx_hash="11" * 32,
        inputs=[("1Alice", 100000)],
        outputs=[("1Bob", 90000)],
        is_coinbase=False
    )
    vec1, dict1 = extractor.extract_features(tx1)

    # In Tx 1, 1Alice had 0 prior transactions
    assert dict1["max_input_prior_tx_count"] == 0.0
    assert dict1["avg_input_prior_tx_count"] == 0.0
    assert dict1["max_input_prior_volume_satoshi"] == 0.0
    assert dict1["new_address_fraction"] == 1.0

    # Tx 2: 1Alice participates again
    tx2 = CompletedTransaction(
        block_number=480000,
        tx_index=2,
        tx_hash="22" * 32,
        inputs=[("1Alice", 50000)],
        outputs=[("1Charlie", 45000)],
        is_coinbase=False
    )
    vec2, dict2 = extractor.extract_features(tx2)

    # In Tx 2, 1Alice's prior history from Tx 1 is now visible (1 prior tx, 100,000 satoshi)
    assert dict2["max_input_prior_tx_count"] == 1.0
    assert dict2["avg_input_prior_tx_count"] == 1.0
    assert dict2["max_input_prior_volume_satoshi"] == 100000.0
    assert dict2["new_address_fraction"] == 0.0

    # Invariance check: If we reset and re-extract Tx 1, the result must be identical
    extractor_fresh = HistoricalFeatureExtractor()
    _, fresh_dict1 = extractor_fresh.extract_features(tx1)
    assert fresh_dict1 == dict1


# ==============================================================================
# 6. Cost-Sensitive Anomaly Model Baselines
# ==============================================================================

def test_cost_sensitive_model_train_test_separation():
    """Verify baseline model fits scaler on train data and computes valid metrics."""
    np.random.seed(42)
    # Synthetic dataset with imbalanced classes (5% anomaly)
    X = np.random.randn(200, 15)
    y = (np.random.rand(200) < 0.05).astype(int)
    # Ensure at least 2 positives in train and test
    y[:4] = 1
    y[-4:] = 1

    X_train, y_train = X[:150], y[:150]
    X_test, y_test = X[150:], y[150:]

    for family in ["lr", "rf", "gbdt"]:
        model = CostSensitiveAnomalyModel(model_family=family, class_weight_ratio=4.32, random_state=42)
        model.fit(X_train, y_train)

        results = model.evaluate(X_test, y_test, threshold=0.5)
        assert results["model_family"] == family
        assert "f1_score" in results
        assert "roc_auc" in results
        assert "average_precision" in results
        assert results["test_samples"] == len(y_test)
        assert results["illicit_count"] == int(np.sum(y_test == 1))


# ==============================================================================
# 7. Duplicate Records and Edge Handling
# ==============================================================================

def test_duplicate_records_and_multi_utxo_handling():
    """Verify that multiple inputs from the same address are correctly accumulated."""
    buffer = TransactionBoundaryBuffer(sanitize_endpoints=False)

    # In Bitcoin, a single address can fund a transaction with multiple UTXOs
    records = [
        TransactionRecord(RecordType.BTC_IN, 480100, 1, "1SameAddr", "aa" * 32, 100000),
        TransactionRecord(RecordType.BTC_IN, 480100, 1, "1SameAddr", "aa" * 32, 200000),
        TransactionRecord(RecordType.BTC_OUT, 480100, 1, "aa" * 32, "1DestAddr", 290000),
        # Subsequent record to advance buffer
        TransactionRecord(RecordType.BTC_IN, 480100, 2, "1Next", "bb" * 32, 50000),
    ]

    emitted = []
    for r in records:
        for tx in buffer.process_record(r):
            emitted.append(tx)

    assert len(emitted) == 1
    tx = emitted[0]
    assert tx.tx_index == 1
    assert tx.num_inputs == 2
    assert tx.total_input_satoshi == 300000
    assert tx.total_output_satoshi == 290000
    assert tx.fee_satoshi == 10000


# ==============================================================================
# 8. Deterministic Replay Ordering
# ==============================================================================

def test_deterministic_replay_ordering():
    """Verify transaction stream preserves monotonic (block_number, tx_index) ordering."""
    records = [
        TransactionRecord(RecordType.BTC_IN, 480000, 1, "1A", "11" * 32, 100),
        TransactionRecord(RecordType.BTC_OUT, 480000, 1, "11" * 32, "1B", 90),
        TransactionRecord(RecordType.BTC_IN, 480000, 2, "1C", "22" * 32, 200),
        TransactionRecord(RecordType.BTC_OUT, 480000, 2, "22" * 32, "1D", 190),
        TransactionRecord(RecordType.BTC_IN, 480001, 0, "0", "33" * 32, 1250000000),
        TransactionRecord(RecordType.BTC_OUT, 480001, 0, "33" * 32, "1Miner", 1250000000),
        TransactionRecord(RecordType.BTC_IN, 480001, 1, "1E", "44" * 32, 300),
        TransactionRecord(RecordType.BTC_OUT, 480001, 1, "44" * 32, "1F", 280),
    ]

    def run_replay():
        buf = TransactionBoundaryBuffer(sanitize_endpoints=False)
        txs = []
        for r in records:
            txs.extend(buf.process_record(r))
        txs.extend(buf.flush_all())
        return txs

    run1 = run_replay()
    run2 = run_replay()

    # Verify idempotency and determinism
    assert len(run1) == len(run2) == 4
    keys1 = [(t.block_number, t.tx_index) for t in run1]
    keys2 = [(t.block_number, t.tx_index) for t in run2]
    assert keys1 == keys2

    # Verify strictly monotonic ordering
    for i in range(len(keys1) - 1):
        assert keys1[i] < keys1[i + 1]

