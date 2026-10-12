"""
Chronological historical feature extractor for blockchain transactions.
Strictly prevents future-to-past data leakage.
"""
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from src.data.boundary import CompletedTransaction


FEATURE_NAMES = [
    "num_inputs",
    "num_outputs",
    "total_input_satoshi",
    "total_output_satoshi",
    "fee_satoshi",
    "fee_rate",
    "is_coinbase",
    "avg_input_val",
    "avg_output_val",
    "input_output_count_ratio",
    "input_output_val_ratio",
    "max_input_prior_tx_count",
    "avg_input_prior_tx_count",
    "max_input_prior_volume_satoshi",
    "new_address_fraction"
]


class HistoricalFeatureExtractor:
    """
    Extracts transaction-level and historical graph features without future leakage.
    Maintains historical address state: tx count, cumulative satoshi volume, first observed block.
    """
    def __init__(self):
        # Historical address profiles: strictly updated AFTER feature extraction
        self.address_prior_tx_count: Dict[str, int] = defaultdict(int)
        self.address_prior_volume: Dict[str, int] = defaultdict(int)
        self.total_transactions_processed: int = 0

    def extract_features(self, tx: CompletedTransaction) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute features for a transaction strictly using its local attributes and
        prior address history. Updates historical state immediately after computation.
        """
        num_in = tx.num_inputs
        num_out = tx.num_outputs
        tot_in = tx.total_input_satoshi
        tot_out = tx.total_output_satoshi
        is_cb = 1.0 if tx.is_coinbase else 0.0

        fee = float(tx.fee_satoshi) if tx.fee_satoshi is not None else 0.0
        fee_rate = fee / max(num_in + num_out, 1.0)
        avg_in = tot_in / max(num_in, 1.0)
        avg_out = tot_out / max(num_out, 1.0)
        count_ratio = num_in / max(num_out, 1.0)
        val_ratio = tot_in / max(tot_out, 1.0)

        # 2. Historical features computed over input addresses prior to this transaction
        input_addrs = tx.input_addresses
        if input_addrs and not tx.is_coinbase:
            prior_counts = [self.address_prior_tx_count[a] for a in input_addrs]
            prior_vols = [self.address_prior_volume[a] for a in input_addrs]
            max_prior_count = float(max(prior_counts))
            avg_prior_count = float(sum(prior_counts) / len(prior_counts))
            max_prior_vol = float(max(prior_vols))
            new_addrs = sum(1 for c in prior_counts if c == 0)
            new_addr_fraction = float(new_addrs / len(input_addrs))
        else:
            max_prior_count = 0.0
            avg_prior_count = 0.0
            max_prior_vol = 0.0
            new_addr_fraction = 1.0 if not tx.is_coinbase else 0.0

        feature_dict = {
            "num_inputs": float(num_in),
            "num_outputs": float(num_out),
            "total_input_satoshi": float(tot_in),
            "total_output_satoshi": float(tot_out),
            "fee_satoshi": fee,
            "fee_rate": fee_rate,
            "is_coinbase": is_cb,
            "avg_input_val": avg_in,
            "avg_output_val": avg_out,
            "input_output_count_ratio": count_ratio,
            "input_output_val_ratio": val_ratio,
            "max_input_prior_tx_count": max_prior_count,
            "avg_input_prior_tx_count": avg_prior_count,
            "max_input_prior_volume_satoshi": max_prior_vol,
            "new_address_fraction": new_addr_fraction
        }

        feature_vec = np.array([feature_dict[name] for name in FEATURE_NAMES], dtype=np.float32)

        # 3. Post-extraction state update (Strict temporal causality)
        for addr, val in tx.inputs:
            self.address_prior_tx_count[addr] += 1
            self.address_prior_volume[addr] += val

        for addr, val in tx.outputs:
            self.address_prior_tx_count[addr] += 1
            self.address_prior_volume[addr] += val

        self.total_transactions_processed += 1

        return feature_vec, feature_dict

    def reset(self) -> None:
        self.address_prior_tx_count.clear()
        self.address_prior_volume.clear()
        self.total_transactions_processed = 0
