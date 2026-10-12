import re
import pytest
from collections import defaultdict

HEX64 = re.compile(r"^[0-9a-f]{64}$")
ADDR_REGEX = re.compile(r"^([13][1-9A-HJ-NP-Za-km-z]{25,34}|bc1[0-9a-z]{11,87})$")


def parse_zenodo_line(line: str):
    parts = line.strip().split()
    if len(parts) != 6:
        raise ValueError(f"Malformed row: expected 6 fields, got {len(parts)}")
    kind, block_str, tx_idx_str, frm, to, val_str = parts
    if kind not in ("BTC-IN", "BTC-OUT"):
        raise ValueError(f"Unknown transaction kind: {kind}")
    return kind, int(block_str), int(tx_idx_str), frm, to, int(val_str)


def test_zenodo_line_parsing():
    sample_in = "BTC-IN 480000 1 1fLXSjDjK9A3LFjPVKcyzguXjovibCgge f1a3b08e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef2 829539"
    kind, block, tx_idx, frm, to, val = parse_zenodo_line(sample_in)
    assert kind == "BTC-IN"
    assert block == 480000
    assert tx_idx == 1
    assert frm == "1fLXSjDjK9A3LFjPVKcyzguXjovibCgge"
    assert to == "f1a3b08e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef2"
    assert val == 829539

    sample_out = "BTC-OUT 480000 1 f1a3b08e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef2 1w5MSnsACBsxdhDktfnhfmfPhmdAuwriG 805661"
    kind, block, tx_idx, frm, to, val = parse_zenodo_line(sample_out)
    assert kind == "BTC-OUT"
    assert frm == "f1a3b08e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef2"
    assert to == "1w5MSnsACBsxdhDktfnhfmfPhmdAuwriG"
    assert val == 805661


def test_endpoint_semantics_identification():
    tx_hash = "f1a3b08e4cf233e5c9cbd4a2322339be254118ccbfb4deb81dc225f2a3fcbef2"
    addr = "1fLXSjDjK9A3LFjPVKcyzguXjovibCgge"
    assert HEX64.match(tx_hash)
    assert not HEX64.match(addr)
    assert ADDR_REGEX.match(addr)
    assert not ADDR_REGEX.match(tx_hash)


def test_coinbase_handling_and_fee_exclusion():
    # Coinbase transaction (tx_idx == 0)
    cb_in = "BTC-IN 480000 0 0 5ade05890effc4adce70fbe97908bb803b10dc4a6247615496b21538f132c0c6 1397458013"
    cb_out = "BTC-OUT 480000 0 5ade05890effc4adce70fbe97908bb803b10dc4a6247615496b21538f132c0c6 1CK6KHY6MHgYvmRQ4PAafKYDrg1ejbH1cE 1397458013"

    _, b_in, idx_in, src, _, v_in = parse_zenodo_line(cb_in)
    _, b_out, idx_out, _, dst, v_out = parse_zenodo_line(cb_out)

    assert idx_in == 0 and idx_out == 0
    assert src == "0"  # synthetic generation source
    assert v_out >= 1_250_000_000  # at least 12.5 BTC base reward at height 480k

    # Standard user transfer fee must NOT be calculated on coinbase
    is_standard_tx = (idx_in != 0)
    assert not is_standard_tx


def test_standard_tx_fee_calculation():
    # Normal transaction: 829,539 sat in, 805,661 sat out -> fee = 23,878 sat
    in_val = 829539
    out_val = 805661
    fee = in_val - out_val
    assert fee >= 0
    assert fee == 23878


def test_bipartite_graph_direction():
    # Construct bipartite graph: Address -> TxHash (BTC-IN) and TxHash -> Address (BTC-OUT)
    edges = []
    
    in_line = "BTC-IN 480000 1 1addrA 64hexTX000000000000000000000000000000000000000000000000000000000 500"
    out_line = "BTC-OUT 480000 1 64hexTX000000000000000000000000000000000000000000000000000000000 1addrB 450"

    kind_in, _, _, frm_in, to_in, val_in = parse_zenodo_line(in_line)
    edges.append((frm_in, to_in, "BTC-IN", val_in))

    kind_out, _, _, frm_out, to_out, val_out = parse_zenodo_line(out_line)
    edges.append((frm_out, to_out, "BTC-OUT", val_out))

    assert len(edges) == 2
    # Edge 1: Address -> Tx
    assert edges[0][0] == "1addrA" and edges[0][1].startswith("64hexTX")
    # Edge 2: Tx -> Address
    assert edges[1][0].startswith("64hexTX") and edges[1][1] == "1addrB"

    # Verify no Cartesian product was created (direct 1addrA -> 1addrB edge is absent)
    direct_edges = [(u, v) for u, v, _, _ in edges if u.startswith("1addr") and v.startswith("1addr")]
    assert len(direct_edges) == 0


def test_special_endpoint_isolation():
    # 'unableToDecode' and '0' must not be grouped into single user address
    endpoints = ["0", "unableToDecode", "1CK6KHY6MHgYvmRQ4PAafKYDrg1ejbH1cE"]
    special_set = {"0", "unableToDecode"}
    valid_addresses = [e for e in endpoints if e not in special_set]
    assert valid_addresses == ["1CK6KHY6MHgYvmRQ4PAafKYDrg1ejbH1cE"]


def test_boundary_split_transaction_buffer():
    # Simulate transaction cut in half across a chunk boundary:
    # Chunk A provides inputs:
    chunk_a_lines = [
        "BTC-IN 480521 839 3addr1 43d4da0d99137a3486a7bf15567eedc079c58cf137399d937c3a610847a3ce5e 1680000",
        "BTC-IN 480521 839 3addr2 43d4da0d99137a3486a7bf15567eedc079c58cf137399d937c3a610847a3ce5e 6340000"
    ]
    # Chunk B provides outputs:
    chunk_b_lines = [
        "BTC-OUT 480521 839 43d4da0d99137a3486a7bf15567eedc079c58cf137399d937c3a610847a3ce5e 1addr3 5000000",
        "BTC-OUT 480521 839 43d4da0d99137a3486a7bf15567eedc079c58cf137399d937c3a610847a3ce5e 3addr4 2891900"
    ]

    # Resolver buffer
    tx_buffer = defaultdict(lambda: {"inputs": [], "outputs": []})

    for l in chunk_a_lines:
        kind, block, idx, frm, to, val = parse_zenodo_line(l)
        tx_buffer[(block, idx)]["inputs"].append((frm, val))

    # At chunk boundary: tx is incomplete
    key = (480521, 839)
    assert len(tx_buffer[key]["inputs"]) == 2
    assert len(tx_buffer[key]["outputs"]) == 0

    # Stream next chunk:
    for l in chunk_b_lines:
        kind, block, idx, frm, to, val = parse_zenodo_line(l)
        tx_buffer[(block, idx)]["outputs"].append((to, val))

    # After buffer continuation: tx is now complete and fee is non-negative
    tot_in = sum(v for _, v in tx_buffer[key]["inputs"])
    tot_out = sum(v for _, v in tx_buffer[key]["outputs"])
    fee = tot_in - tot_out

    assert tot_in == 8020000
    assert tot_out == 7891900
    assert fee == 128100
    assert fee >= 0


def test_heist_mapping_invariance():
    # BitcoinHeist white addresses must NOT be considered legitimate ground truth
    heist_sample = [
        {"address": "111K8kZAEnJg245r2cM6y9zgJGHZtJPy6", "label": "princetonCerber"},
        {"address": "1123pJv8jzeFQaCV4w644pzQJzVWay2zcA", "label": "white"}
    ]
    positive_addresses = {r["address"] for r in heist_sample if r["label"] != "white"}
    assert "111K8kZAEnJg245r2cM6y9zgJGHZtJPy6" in positive_addresses
    assert "1123pJv8jzeFQaCV4w644pzQJzVWay2zcA" not in positive_addresses
    
    # White address is UNLABELED, not confirmed licit
    label_for_white = "UNLABELED" if heist_sample[1]["label"] == "white" else "ILLICIT"
    assert label_for_white == "UNLABELED"
