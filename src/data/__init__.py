"""
Data ingestion, parsing, schema validation, and chunk boundary management.
"""
from .schema import TransactionRecord, RecordType, parse_raw_line, sanitize_endpoint_address
from .boundary import CompletedTransaction, TransactionBoundaryBuffer
from .reader import ZenodoStreamReader
from .elliptic import (
    EllipticPreprocessor,
    PreprocessorConfig,
    PreprocessedDataset,
    PreprocessingAuditReport,
    ELLIPTIC_NUM_FEATURES,
    ELLIPTIC_NUM_LOCAL_FEATURES,
    ELLIPTIC_NUM_AGG_FEATURES,
    FEATURE_COL_NAMES,
)

__all__ = [
    "TransactionRecord",
    "RecordType",
    "parse_raw_line",
    "sanitize_endpoint_address",
    "CompletedTransaction",
    "TransactionBoundaryBuffer",
    "ZenodoStreamReader",
    "EllipticPreprocessor",
    "PreprocessorConfig",
    "PreprocessedDataset",
    "PreprocessingAuditReport",
    "ELLIPTIC_NUM_FEATURES",
    "ELLIPTIC_NUM_LOCAL_FEATURES",
    "ELLIPTIC_NUM_AGG_FEATURES",
    "FEATURE_COL_NAMES",
]
