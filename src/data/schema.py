"""
Data schemas, record types, and syntactic validations for blockchain transactions.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import re

HEX64_REGEX = re.compile(r"^[0-9a-f]{64}$")
ADDR_REGEX = re.compile(r"^([13][1-9A-HJ-NP-Za-km-z]{25,34}|bc1[0-9a-z]{11,87})$")


class RecordType:
    BTC_IN = "BTC-IN"
    BTC_OUT = "BTC-OUT"
    VALID_TYPES = (BTC_IN, BTC_OUT)


@dataclass(frozen=True)
class TransactionRecord:
    kind: str
    block_number: int
    tx_index: int
    source: str
    destination: str
    value: int  # Satoshis

    @property
    def is_coinbase(self) -> bool:
        return self.tx_index == 0

    @property
    def tx_key(self) -> Tuple[int, int]:
        return (self.block_number, self.tx_index)

    @property
    def address(self) -> str:
        """Extract the user/wallet address endpoint."""
        if self.kind == RecordType.BTC_IN:
            return self.source
        return self.destination

    @property
    def tx_hash(self) -> str:
        """Extract the 64-hex transaction hash endpoint."""
        if self.kind == RecordType.BTC_IN:
            return self.destination
        return self.source


def parse_raw_line(line: str) -> TransactionRecord:
    """
    Parse a single raw space-delimited Zenodo line into a TransactionRecord.
    Raises ValueError on syntax/type violations.
    """
    parts = line.strip().split()
    if len(parts) != 6:
        raise ValueError(f"Malformed row: expected 6 space-separated fields, got {len(parts)}: '{line[:80]}'")

    kind, block_str, tx_idx_str, frm, to, val_str = parts

    if kind not in RecordType.VALID_TYPES:
        raise ValueError(f"Invalid record type '{kind}', expected 'BTC-IN' or 'BTC-OUT'")

    try:
        block_number = int(block_str)
        if block_number < 0:
            raise ValueError(f"Negative block number {block_number}")
    except ValueError as e:
        raise ValueError(f"Invalid block number '{block_str}': {e}")

    try:
        tx_index = int(tx_idx_str)
        if tx_index < 0:
            raise ValueError(f"Negative transaction index {tx_index}")
    except ValueError as e:
        raise ValueError(f"Invalid transaction index '{tx_idx_str}': {e}")

    try:
        value = int(val_str)
        if value < 0:
            raise ValueError(f"Negative satoshi value {value}")
    except ValueError as e:
        raise ValueError(f"Invalid transfer value '{val_str}': {e}")

    # Validate endpoint slots
    if kind == RecordType.BTC_IN:
        tx_hash = to
        addr = frm
    else:
        tx_hash = frm
        addr = to

    if not HEX64_REGEX.match(tx_hash):
        raise ValueError(f"Transaction ID '{tx_hash}' is not a 64-character lowercase hex string")

    # Address slot validation: must not accidentally be a 64-hex transaction ID
    if HEX64_REGEX.match(addr):
        raise ValueError(f"Address slot contains a 64-hex transaction hash '{addr}'")

    return TransactionRecord(
        kind=kind,
        block_number=block_number,
        tx_index=tx_index,
        source=frm,
        destination=to,
        value=value
    )


def sanitize_endpoint_address(addr: str, tx_hash: str, output_idx: Optional[int] = None) -> str:
    """
    Isolate special endpoints so 'unableToDecode' and '0' do not collapse
    into false high-degree supernodes in the transaction graph.
    """
    if addr == "unableToDecode":
        idx_str = f"_{output_idx}" if output_idx is not None else ""
        return f"unableToDecode_{tx_hash[:16]}{idx_str}"
    if addr == "0":
        return f"coinbase_source_{tx_hash[:16]}"
    return addr
