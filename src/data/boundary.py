"""
Boundary transaction accumulator and chunk-boundary state manager.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
from .schema import TransactionRecord, RecordType, sanitize_endpoint_address


@dataclass
class CompletedTransaction:
    block_number: int
    tx_index: int
    tx_hash: str
    inputs: List[Tuple[str, int]] = field(default_factory=list)   # (address, value_satoshi)
    outputs: List[Tuple[str, int]] = field(default_factory=list) # (address, value_satoshi)
    is_coinbase: bool = False

    @property
    def total_input_satoshi(self) -> int:
        return sum(v for _, v in self.inputs)

    @property
    def total_output_satoshi(self) -> int:
        return sum(v for _, v in self.outputs)

    @property
    def fee_satoshi(self) -> Optional[int]:
        """
        Transaction fee = sum(inputs) - sum(outputs).
        Coinbase transactions mint new Bitcoin subsidy, so ordinary transfer fee is None/0.
        """
        if self.is_coinbase:
            return 0
        if not self.inputs or not self.outputs:
            return None
        return self.total_input_satoshi - self.total_output_satoshi

    @property
    def num_inputs(self) -> int:
        return len(self.inputs)

    @property
    def num_outputs(self) -> int:
        return len(self.outputs)

    @property
    def input_addresses(self) -> List[str]:
        return [addr for addr, _ in self.inputs]

    @property
    def output_addresses(self) -> List[str]:
        return [addr for addr, _ in self.outputs]


class TransactionBoundaryBuffer:
    """
    Stateful buffer that accumulates incoming TransactionRecords into CompletedTransactions.
    Retains incomplete transactions across chunk boundaries without premature emission.
    """
    def __init__(self, sanitize_endpoints: bool = True):
        self.sanitize_endpoints = sanitize_endpoints
        # Buffer map: (block, tx_index) -> partial transaction data
        self._pending: Dict[Tuple[int, int], CompletedTransaction] = {}
        self._current_key: Optional[Tuple[int, int]] = None
        self._last_observed_key: Optional[Tuple[int, int]] = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def process_record(self, record: TransactionRecord) -> List[CompletedTransaction]:
        """
        Ingest a single record. If a previous transaction is completed and we have
        moved to a subsequent key in chronological order, return the completed transactions.
        """
        key = record.tx_key
        emitted: List[CompletedTransaction] = []

        # If we advanced to a strictly newer transaction key, check if older pending txs can be emitted
        if self._current_key is not None and key > self._current_key:
            # Yield any prior completed transactions strictly older than current key
            ready_keys = [k for k in sorted(self._pending.keys()) if k < key]
            for k in ready_keys:
                tx = self._pending[k]
                # A standard transaction is complete if it has both inputs and outputs
                # A coinbase transaction is complete if it has outputs
                if (tx.is_coinbase and len(tx.outputs) > 0) or (not tx.is_coinbase and len(tx.inputs) > 0 and len(tx.outputs) > 0):
                    del self._pending[k]
                    emitted.append(tx)

        self._current_key = key
        self._last_observed_key = key

        # Get or initialize the pending transaction container
        if key not in self._pending:
            self._pending[key] = CompletedTransaction(
                block_number=record.block_number,
                tx_index=record.tx_index,
                tx_hash=record.tx_hash,
                is_coinbase=record.is_coinbase
            )

        tx = self._pending[key]

        if record.kind == RecordType.BTC_IN:
            addr = sanitize_endpoint_address(record.source, record.tx_hash) if self.sanitize_endpoints else record.source
            tx.inputs.append((addr, record.value))
        else:
            out_idx = len(tx.outputs)
            addr = sanitize_endpoint_address(record.destination, record.tx_hash, out_idx) if self.sanitize_endpoints else record.destination
            tx.outputs.append((addr, record.value))

        return emitted

    def flush_completed(self) -> List[CompletedTransaction]:
        """
        Flush all transactions currently in the buffer that have sufficient records.
        Retains only truly incomplete boundary transactions.
        """
        ready_keys = []
        for k, tx in sorted(self._pending.items()):
            if (tx.is_coinbase and len(tx.outputs) > 0) or (not tx.is_coinbase and len(tx.inputs) > 0 and len(tx.outputs) > 0):
                ready_keys.append(k)

        return [self._pending.pop(k) for k in ready_keys]

    def flush_all(self) -> List[CompletedTransaction]:
        """Force flush all remaining transactions in buffer (at end of complete stream)."""
        return [self._pending.pop(k) for k in sorted(list(self._pending.keys()))]
