"""
Directed bipartite address-transaction graph representation.
Constructed strictly from BTC-IN and BTC-OUT records without Cartesian expansion.
"""
from typing import Dict, Set, Tuple, List, Optional
from collections import defaultdict, Counter
from src.data.boundary import CompletedTransaction


class BipartiteTransactionGraph:
    """
    Directed bipartite graph:
      - Addresses -> Transactions (funding inputs)
      - Transactions -> Addresses (disbursed outputs)
    """
    def __init__(self, track_edge_weights: bool = True):
        self.track_edge_weights = track_edge_weights
        
        # Node sets
        self.address_nodes: Set[str] = set()
        self.tx_nodes: Set[str] = set()
        
        # Transaction node metadata: tx_hash -> {block, tx_idx, fee, is_coinbase, total_value}
        self.tx_metadata: Dict[str, dict] = {}
        
        # Directed Adjacency: u -> list of (v, weight)
        # Out-edges from Address to TxHash
        self.addr_out_edges: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # In-edges to Address from TxHash
        self.addr_in_edges: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        
        # Out-edges from TxHash to Address
        self.tx_out_edges: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # In-edges to TxHash from Address
        self.tx_in_edges: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        
        # Historical degree and volume counters for O(1) feature lookup
        self.addr_tx_count: Dict[str, int] = Counter()
        self.addr_cumulative_volume: Dict[str, int] = Counter()
        self.total_edge_count: int = 0

    @property
    def num_addresses(self) -> int:
        return len(self.address_nodes)

    @property
    def num_transactions(self) -> int:
        return len(self.tx_nodes)

    @property
    def num_nodes(self) -> int:
        return len(self.address_nodes) + len(self.tx_nodes)

    @property
    def num_edges(self) -> int:
        return self.total_edge_count

    def add_transaction(self, tx: CompletedTransaction) -> None:
        """
        Integrate a CompletedTransaction into the bipartite graph.
        Never produces Cartesian product direct address-to-address edges.
        """
        tx_hash = tx.tx_hash
        self.tx_nodes.add(tx_hash)
        
        self.tx_metadata[tx_hash] = {
            "block_number": tx.block_number,
            "tx_index": tx.tx_index,
            "fee_satoshi": tx.fee_satoshi,
            "is_coinbase": tx.is_coinbase,
            "total_input": tx.total_input_satoshi,
            "total_output": tx.total_output_satoshi
        }

        # 1. Inputs: Address -> TxHash
        for addr, val in tx.inputs:
            self.address_nodes.add(addr)
            self.addr_out_edges[addr].append((tx_hash, val))
            self.tx_in_edges[tx_hash].append((addr, val))
            self.addr_tx_count[addr] += 1
            self.addr_cumulative_volume[addr] += val
            self.total_edge_count += 1

        # 2. Outputs: TxHash -> Address
        for addr, val in tx.outputs:
            self.address_nodes.add(addr)
            self.tx_out_edges[tx_hash].append((addr, val))
            self.addr_in_edges[addr].append((tx_hash, val))
            self.addr_tx_count[addr] += 1
            self.addr_cumulative_volume[addr] += val
            self.total_edge_count += 1

    def get_address_in_degree(self, addr: str) -> int:
        return len(self.addr_in_edges.get(addr, []))

    def get_address_out_degree(self, addr: str) -> int:
        return len(self.addr_out_edges.get(addr, []))

    def get_transaction_in_degree(self, tx_hash: str) -> int:
        return len(self.tx_in_edges.get(tx_hash, []))

    def get_transaction_out_degree(self, tx_hash: str) -> int:
        return len(self.tx_out_edges.get(tx_hash, []))

    def clear(self) -> None:
        self.address_nodes.clear()
        self.tx_nodes.clear()
        self.tx_metadata.clear()
        self.addr_out_edges.clear()
        self.addr_in_edges.clear()
        self.tx_out_edges.clear()
        self.tx_in_edges.clear()
        self.addr_tx_count.clear()
        self.addr_cumulative_volume.clear()
        self.total_edge_count = 0
