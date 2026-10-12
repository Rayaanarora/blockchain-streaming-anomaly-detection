#!/usr/bin/env python3
"""
Zenodo Dataset Downloader & Checksum Verifier
Designed for: Distributed Streaming Anomaly Detection on Blockchain Transaction Graphs

Repository: D:\blockchainbigdata\blockchain-streaming-anomaly-detection

Key Features:
  - Validates Zenodo REST API Record metadata (defaults to Record 7157854 - Part 2 of 4)
  - Safe by default: runs in dry-run mode unless --download or --verify is explicitly given
  - Robust resumable downloads using HTTP Range headers with HTTP 206/200 status validation
  - Dual-factor archive verification: validates both expected file size and Zenodo MD5 checksum
  - Known alias preservation: protects existing files (a.tar.bz2, b.tar.bz2, sample_a)
  - Comprehensive disk-space auditing for destination drive with extraction and safety buffer margins
  - Strict integrity isolation: downloads to .part files and verifies MD5 before finalization
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

DEFAULT_RECORD_ID = "7157854"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BlockchainAnomalyResearch/1.0"
COMPRESSION_EXPANSION_RATIO = 3.05  # Average observed uncompressed-to-bzip2 ratio
SAFETY_MARGIN_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB safety reserve for host OS/workspace

# Known aliases for existing files in data/raw to avoid accidental re-download or collision
KNOWN_ALIASES = {
    "btc-tx-480000-489999-aa.tar.bz2": ["btc-tx-480000-489999-aa.tar.bz2", "a.tar.bz2"],
    "btc-tx-490000-499999-aa.tar.bz2": ["btc-tx-490000-499999-aa.tar.bz2", "b.tar.bz2"],
}


def fetch_record_metadata(record_id: str) -> Dict:
    """Fetch record file metadata from Zenodo REST API."""
    url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        sys.exit(f"[ERROR] Zenodo API returned HTTP {e.code}: {e.reason} for record {record_id}")
    except Exception as e:
        sys.exit(f"[ERROR] Failed to query Zenodo API for record {record_id}: {e}")


def compute_md5(file_path: str, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Compute streaming MD5 hexadecimal digest of a local file."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def get_drive_free_space(path: str) -> Tuple[str, int, int]:
    """
    Resolve the root mount/drive for a path and return (drive_root, free_bytes, total_bytes).
    Works on both Windows (e.g. D:\\) and Unix-like systems.
    """
    abs_path = os.path.abspath(path)
    drive_prefix = os.path.splitdrive(abs_path)[0]
    drive_root = (drive_prefix + os.path.sep) if drive_prefix else os.path.sep
    
    # If the specific directory doesn't exist yet, check the nearest existing parent or drive
    probe_dir = abs_path
    while probe_dir and not os.path.exists(probe_dir):
        parent = os.path.dirname(probe_dir)
        if parent == probe_dir:
            break
        probe_dir = parent
    
    usage_target = probe_dir if (probe_dir and os.path.exists(probe_dir)) else drive_root
    stat = shutil.disk_usage(usage_target)
    return drive_root, stat.free, stat.total


def format_bytes(b: int) -> str:
    """Format byte count into human-readable representation."""
    val = float(b)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(val) < 1024.0:
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} PB"


def find_local_candidate(key: str, out_dir: str) -> Optional[str]:
    """Find if a key or one of its known aliases exists locally."""
    candidates = KNOWN_ALIASES.get(key, [key])
    for fname in candidates:
        candidate_path = os.path.join(out_dir, fname)
        if os.path.isfile(candidate_path):
            return candidate_path
    return None


def verify_file_integrity(file_path: str, expected_size: int, expected_md5: Optional[str]) -> Tuple[bool, str, Optional[str]]:
    """
    Check if a local file matches both expected size and MD5 checksum.
    Returns (is_valid, description, calculated_md5).
    """
    if not os.path.exists(file_path):
        return False, "file does not exist", None
    
    actual_size = os.path.getsize(file_path)
    if actual_size != expected_size:
        return False, f"size mismatch (local: {actual_size:,} bytes, expected: {expected_size:,} bytes)", None
    
    calc_md5 = compute_md5(file_path)
    if expected_md5:
        if calc_md5.lower() != expected_md5.lower():
            return False, f"MD5 mismatch (local: {calc_md5}, expected: {expected_md5})", calc_md5
    
    return True, "valid", calc_md5


def download_file_resumable(
    url: str,
    target_path: str,
    expected_size: int,
    expected_md5: Optional[str] = None,
    chunk_size: int = 1024 * 1024,
) -> bool:
    """
    Resumable download with HTTP range verification and strict post-download checksum validation.
    Guarantees target_path is never touched unless the download is 100% verified.
    """
    part_path = target_path + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    existing_bytes = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    # If an existing .part file is already at expected size, check if it's valid
    if existing_bytes == expected_size:
        print(f"  [INFO] Temporary file '{os.path.basename(part_path)}' already matches expected size. Verifying...")
        is_valid, msg, calc_md5 = verify_file_integrity(part_path, expected_size, expected_md5)
        if is_valid:
            os.replace(part_path, target_path)
            print(f"  [SUCCESS] Pre-existing part verified and finalized: {os.path.basename(target_path)}")
            return True
        else:
            print(f"  [WARN] Pre-existing part failed verification: {msg}. Restarting from byte 0...")
            os.remove(part_path)
            existing_bytes = 0
    elif existing_bytes > expected_size:
        print(f"  [WARN] Temporary part file exceeds expected size ({existing_bytes} > {expected_size}). Resetting...")
        os.remove(part_path)
        existing_bytes = 0

    headers = {"User-Agent": USER_AGENT}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"  [INFO] Attempting to resume from byte offset {existing_bytes:,} ({format_bytes(existing_bytes)})...")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status_code = getattr(resp, "status", getattr(resp, "code", 200))
            
            # Critical Range verification: ensure server acknowledged partial request
            if existing_bytes > 0:
                if status_code == 206:
                    mode = "ab"
                    downloaded = existing_bytes
                    print(f"  [INFO] HTTP 206 Partial Content confirmed. Appending to part file.")
                elif status_code == 200:
                    mode = "wb"
                    downloaded = 0
                    print(f"  [WARN] Server returned HTTP 200 OK instead of HTTP 206. Restarting download from byte 0.")
                else:
                    print(f"  [ERROR] Unexpected HTTP status code: {status_code}")
                    return False
            else:
                mode = "wb"
                downloaded = 0

            with open(part_path, mode) as f:
                while True:
                    buf = resp.read(chunk_size)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    pct = (downloaded / expected_size) * 100 if expected_size else 0
                    print(
                        f"\r  Progress: {format_bytes(downloaded)} / {format_bytes(expected_size)} ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range Not Satisfiable
            print(f"\n  [WARN] HTTP 416 Range Not Satisfiable. Resetting part file...")
            if os.path.exists(part_path):
                os.remove(part_path)
        else:
            print(f"\n  [ERROR] HTTP error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  [ERROR] Network error during transfer: {e}")
        print(f"  [INFO] Partial file preserved at '{os.path.basename(part_path)}' for future resumption.")
        return False

    # Check size
    final_size = os.path.getsize(part_path)
    if final_size != expected_size:
        print(f"  [ERROR] Incomplete transfer: got {final_size:,} bytes, expected {expected_size:,} bytes.")
        return False

    # Check MD5
    if expected_md5:
        print("  Verifying MD5 checksum...")
        calc_md5 = compute_md5(part_path)
        if calc_md5.lower() != expected_md5.lower():
            print(f"  [ERROR] MD5 verification failed!")
            print(f"    Expected:   {expected_md5}")
            print(f"    Calculated: {calc_md5}")
            corrupted_backup = part_path + ".corrupted"
            if os.path.exists(corrupted_backup):
                os.remove(corrupted_backup)
            os.rename(part_path, corrupted_backup)
            print(f"  [INFO] Corrupted transfer saved to '{os.path.basename(corrupted_backup)}' for inspection.")
            return False
        print(f"  [OK] MD5 checksum verified: {calc_md5}")

    # Atomic rename to final destination
    os.replace(part_path, target_path)
    print(f"  [SUCCESS] Verified and finalized: {os.path.basename(target_path)}")
    return True


def audit_storage(
    files: List[Dict],
    out_dir: str,
    drive_root: str,
    free_bytes: int,
    total_bytes: int,
) -> Tuple[int, int, int, List[Tuple[Dict, str, str]]]:
    """
    Inspect all target files against local storage.
    Returns (needed_download_bytes, verified_count, missing_count, status_list).
    """
    needed_download_bytes = 0
    verified_count = 0
    missing_count = 0
    status_list = []

    for f in files:
        key = f["key"]
        expected_size = f["size"]
        expected_checksum = f.get("checksum", "")
        expected_md5 = expected_checksum.split(":")[-1] if expected_checksum else None

        local_path = find_local_candidate(key, out_dir)
        if local_path:
            is_valid, msg, calc_md5 = verify_file_integrity(local_path, expected_size, expected_md5)
            alias_note = f" (as {os.path.basename(local_path)})" if os.path.basename(local_path) != key else ""
            if is_valid:
                verified_count += 1
                status_list.append((f, "VERIFIED", f"PRESENT & VERIFIED{alias_note}"))
            else:
                missing_count += 1
                needed_download_bytes += expected_size
                status_list.append((f, "CORRUPTED", f"CORRUPTED{alias_note}: {msg}"))
        else:
            missing_count += 1
            needed_download_bytes += expected_size
            status_list.append((f, "MISSING", "NOT DOWNLOADED"))

    return needed_download_bytes, verified_count, missing_count, status_list


def main():
    parser = argparse.ArgumentParser(
        description="Zenodo Blockchain Dataset Downloader & Checksum Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Preview 480k-490k block chunks (safe dry-run):
    python download_zenodo.py --filter 480000 --dry-run

  Verify local archives against Zenodo checksums:
    python download_zenodo.py --filter 480000 --verify

  Download a specific single missing chunk (e.g. part ab):
    python download_zenodo.py --filter btc-tx-480000-489999-ab --download
        """,
    )
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID, help=f"Zenodo record ID (default: {DEFAULT_RECORD_ID})")
    parser.add_argument("--out-dir", default="data/raw", help="Target directory for raw data (default: data/raw)")
    parser.add_argument("--filter", default="", help="Filter files by substring (e.g. 'btc-tx-480000-489999')")
    parser.add_argument("--dry-run", action="store_true", help="Preview chunks, sizes, and disk space without downloading (default mode)")
    parser.add_argument("--verify", action="store_true", help="Verify integrity of existing local files against Zenodo MD5")
    parser.add_argument("--download", action="store_true", help="Explicitly enable downloading of missing/corrupted files")

    args = parser.parse_args()

    # Default to dry-run mode if neither --download nor --verify is explicitly specified
    is_dry_run = args.dry_run or (not args.download and not args.verify)

    print(f"=== ZENODO BLOCKCHAIN DATASET AUDIT & INGESTION ===")
    print(f"Querying Zenodo API for Record {args.record_id}...")
    meta = fetch_record_metadata(args.record_id)
    title = meta.get("metadata", {}).get("title", "Unknown Publication")
    all_files = meta.get("files", [])
    print(f"Publication: {title}")
    print(f"Total files in deposit: {len(all_files)}")

    # Apply filter
    files = [f for f in all_files if args.filter in f["key"]] if args.filter else all_files
    print(f"Filtered chunks matching '{args.filter}': {len(files)}")
    if not files:
        sys.exit("[INFO] No files matched the filter criteria.")

    # Inspect drive storage
    drive_root, free_bytes, total_bytes = get_drive_free_space(args.out_dir)
    needed_download_bytes, verified_count, missing_count, status_list = audit_storage(
        files, args.out_dir, drive_root, free_bytes, total_bytes
    )

    # In verification-only mode
    if args.verify:
        print("\n--- LOCAL FILE CHECKSUM VERIFICATION REPORT ---")
        pass_count = 0
        fail_count = 0
        absent_count = 0
        for f, status, desc in status_list:
            key = f["key"]
            expected_md5 = f.get("checksum", "").split(":")[-1]
            if status == "VERIFIED":
                print(f"  [PASS] {key:<35} {format_bytes(f['size']):>10} | MD5: {expected_md5} | {desc}")
                pass_count += 1
            elif status == "CORRUPTED":
                print(f"  [FAIL] {key:<35} {format_bytes(f['size']):>10} | {desc}")
                fail_count += 1
            else:
                print(f"  [MISS] {key:<35} {format_bytes(f['size']):>10} | {desc}")
                absent_count += 1
        print(f"\nSummary: {pass_count} verified, {fail_count} corrupted, {absent_count} not downloaded.")
        return

    # Storage audit calculations
    est_uncompressed_needed = int(needed_download_bytes * COMPRESSION_EXPANSION_RATIO)
    total_needed_with_extraction = needed_download_bytes + est_uncompressed_needed + SAFETY_MARGIN_BYTES
    free_after_download_only = free_bytes - needed_download_bytes
    free_after_full_extraction = free_bytes - (needed_download_bytes + est_uncompressed_needed)

    print("\n--- DESTINATION STORAGE AUDIT ---")
    print(f"Target directory:               {os.path.abspath(args.out_dir)}")
    print(f"Destination drive root:         {drive_root}")
    print(f"Drive total capacity:           {format_bytes(total_bytes)}")
    print(f"Drive current free space:       {format_bytes(free_bytes)}")
    print(f"Chunks matching filter:         {len(files)} ({verified_count} verified on disk, {missing_count} pending)")
    print(f"Net compressed download needed: {format_bytes(needed_download_bytes)}")
    print(f"Est. uncompressed size (new):   ~{format_bytes(est_uncompressed_needed)}")
    print(f"Safety reserve margin:          {format_bytes(SAFETY_MARGIN_BYTES)}")
    print(f"Free space after download only: {format_bytes(free_after_download_only)}")
    print(f"Free space after extraction:    {format_bytes(free_after_full_extraction)}")

    if total_needed_with_extraction > free_bytes:
        print(f"\n[ALERT] Storage is tight for complete download + extraction with safety margin.")
        if needed_download_bytes + SAFETY_MARGIN_BYTES > free_bytes:
            print(f"[ERROR] Insufficient space even for compressed downloads alone!")
        else:
            print(f"[NOTE] Compressed download is feasible, but decompression must be performed iteratively.")
    else:
        print(f"\n[STATUS] STORAGE FEASIBLE: Drive {drive_root} has ample space for downloads and full extraction.")

    print("\n--- AVAILABLE CHUNKS LISTING ---")
    for idx, (f, status, desc) in enumerate(status_list, 1):
        cs = f.get("checksum", "").split(":")[-1]
        stat_tag = f"[{status}]"
        print(f"  {idx:>2}. {f['key']:<35} {format_bytes(f['size']):>10}  MD5: {cs}  {stat_tag}")

    # Dry-run termination
    if is_dry_run:
        print("\n" + "=" * 60)
        print("[DRY-RUN COMPLETE] No files were downloaded, modified, or deleted.")
        print("Existing archives (a.tar.bz2, b.tar.bz2) and extracted samples remain untouched.")
        print("\nCommands:")
        print("  1. To verify local archives against Zenodo checksums:")
        print("       python download_zenodo.py --filter 480000 --verify")
        print("  2. To dry-run and inspect blocks 490000-499999:")
        print("       python download_zenodo.py --filter 490000 --dry-run")
        print("  3. To download a single missing chunk (when ready):")
        print("       python download_zenodo.py --filter btc-tx-480000-489999-ab --download")
        print("=" * 60)
        return

    # Execution path when --download is explicitly passed
    if needed_download_bytes == 0:
        print("\n[SUCCESS] All requested chunks are already downloaded and verified. Nothing to download.")
        return

    # Strict safety check before initiating download
    if needed_download_bytes + SAFETY_MARGIN_BYTES > free_bytes:
        sys.exit(f"[ABORT] Cannot proceed: Required space ({format_bytes(needed_download_bytes + SAFETY_MARGIN_BYTES)}) exceeds free space ({format_bytes(free_bytes)}).")

    print("\nStarting downloads for pending chunks...")
    downloaded_count = 0
    for idx, (f, status, desc) in enumerate(status_list, 1):
        if status == "VERIFIED":
            print(f"[{idx}/{len(files)}] Already verified: {f['key']} (skipping)")
            continue

        target = os.path.join(args.out_dir, f["key"])
        print(f"\n[{idx}/{len(files)}] Ingesting {f['key']} ({format_bytes(f['size'])})...")
        md5_expected = f.get("checksum", "").split(":")[-1]
        url = f["links"]["self"]

        success = download_file_resumable(url, target, f["size"], md5_expected)
        if not success:
            sys.exit(f"\n[ABORT] Ingestion terminated on chunk '{f['key']}'. Target directory preserved safely.")
        downloaded_count += 1

    print(f"\n[DONE] Successfully downloaded and verified {downloaded_count} chunk(s).")


if __name__ == "__main__":
    main()
