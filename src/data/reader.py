"""
Streaming archive readers for Zenodo transaction datasets (.tar.bz2 and plain text).
"""
import os
import tarfile
from typing import Iterator, List, Optional
from .schema import TransactionRecord, parse_raw_line
from .boundary import TransactionBoundaryBuffer, CompletedTransaction


class ZenodoStreamReader:
    """
    Streams raw records or CompletedTransactions from local plain text or tar.bz2 archives.
    """
    def __init__(self, sanitize_endpoints: bool = True):
        self.sanitize_endpoints = sanitize_endpoints
        self.total_records_read = 0
        self.malformed_records = 0

    def stream_records_from_file(self, file_path: str, max_records: Optional[int] = None) -> Iterator[TransactionRecord]:
        """Stream raw records from an uncompressed plain text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        count = 0
        with open(file_path, "r", encoding="ascii", errors="replace") as f:
            for line in f:
                if max_records is not None and count >= max_records:
                    break
                try:
                    record = parse_raw_line(line)
                    count += 1
                    self.total_records_read += 1
                    yield record
                except ValueError:
                    self.malformed_records += 1
                    continue

    def stream_records_from_archive(self, archive_path: str, max_records: Optional[int] = None) -> Iterator[TransactionRecord]:
        """Stream raw records directly from a compressed .tar.bz2 archive without extracting."""
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        count = 0
        with tarfile.open(archive_path, "r:bz2") as t:
            member = t.next()
            if not member:
                return
            f = t.extractfile(member)
            if not f:
                return

            for raw_line in f:
                if max_records is not None and count >= max_records:
                    break
                line = raw_line.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                try:
                    record = parse_raw_line(line)
                    count += 1
                    self.total_records_read += 1
                    yield record
                except ValueError:
                    self.malformed_records += 1
                    continue

    def stream_transactions_from_paths(
        self,
        paths: List[str],
        max_records_total: Optional[int] = None
    ) -> Iterator[CompletedTransaction]:
        """
        Stream full CompletedTransactions across multiple consecutive archives/files,
        maintaining pending transaction state across chunk boundaries.
        """
        buffer = TransactionBoundaryBuffer(sanitize_endpoints=self.sanitize_endpoints)
        records_yielded = 0

        for p in paths:
            if p.endswith(".tar.bz2") or p.endswith(".bz2"):
                record_iter = self.stream_records_from_archive(p)
            else:
                record_iter = self.stream_records_from_file(p)

            for rec in record_iter:
                if max_records_total is not None and records_yielded >= max_records_total:
                    break
                records_yielded += 1

                for completed_tx in buffer.process_record(rec):
                    yield completed_tx

            if max_records_total is not None and records_yielded >= max_records_total:
                break

        # Flush any remaining completed transactions after processing all files
        for completed_tx in buffer.flush_completed():
            yield completed_tx
