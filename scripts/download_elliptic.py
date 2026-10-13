#!/usr/bin/env python3
"""
Automated Downloader for the Official Elliptic Bitcoin Transaction Dataset.

Downloads the 3 authentic raw CSV files into data/raw/elliptic/:
  1. elliptic_txs_classes.csv
  2. elliptic_txs_edgelist.csv
  3. elliptic_txs_features.csv

Source: Official Elliptic benchmark mirror on Hugging Face (yhoma/elliptic-bitcoin-dataset).
Uses streaming downloads to .part files with progress reporting.
"""
import os
import sys
import time
import urllib.request
import urllib.error

ELLIPTIC_BASE_URL = "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main"

FILES = [
    {
        "name": "elliptic_txs_classes.csv",
        "url": f"{ELLIPTIC_BASE_URL}/elliptic_txs_classes.csv",
        "desc": "Transaction Ground Truth Labels (1=illicit, 2=licit, unknown)"
    },
    {
        "name": "elliptic_txs_edgelist.csv",
        "url": f"{ELLIPTIC_BASE_URL}/elliptic_txs_edgelist.csv",
        "desc": "Directed Transaction Graph Edges (txId1 -> txId2)"
    },
    {
        "name": "elliptic_txs_features.csv",
        "url": f"{ELLIPTIC_BASE_URL}/elliptic_txs_features.csv",
        "desc": "165 Local & Aggregated Node Features across 49 Timesteps"
    }
]


def download_file_streaming(url: str, dest_path: str, desc: str) -> bool:
    part_path = dest_path + ".part"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    print(f"\n[DOWNLOAD] {os.path.basename(dest_path)}")
    print(f"  Description: {desc}")
    print(f"  Source URL:  {url}")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EllipticDownloader/1.0"
    }
    req = urllib.request.Request(url, headers=headers)
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp, open(part_path, "wb") as out_f:
            total_size_hdr = resp.headers.get("Content-Length")
            total_size = int(total_size_hdr) if total_size_hdr else 0
            downloaded = 0
            
            while True:
                chunk = resp.read(8 * 1024 * 1024)  # 8 MB chunks
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.time() - t0, 1e-6)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(
                        f"\r  Progress: {downloaded / (1024 * 1024):.1f} / {total_size / (1024 * 1024):.1f} MB "
                        f"({pct:.1f}%) @ {speed_mb:.2f} MB/s",
                        end="", flush=True
                    )
                else:
                    print(
                        f"\r  Progress: {downloaded / (1024 * 1024):.1f} MB @ {speed_mb:.2f} MB/s",
                        end="", flush=True
                    )
                    
        print(f"\n  [DONE] Completed in {time.time() - t0:.2f}s ({os.path.getsize(part_path) / (1024 * 1024):.2f} MB)")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(part_path, dest_path)
        return True
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        return False


def main():
    target_dir = os.path.abspath("data/raw/elliptic")
    os.makedirs(target_dir, exist_ok=True)
    
    print("=" * 80)
    print("ELLIPTIC BITCOIN TRANSACTION DATASET DOWNLOADER")
    print(f"Target Directory: {target_dir}")
    print("=" * 80)
    
    all_success = True
    for item in FILES:
        fname = item["name"]
        dest = os.path.join(target_dir, fname)
        if os.path.isfile(dest) and os.path.getsize(dest) > 1000:
            print(f"[FOUND] {fname} already exists ({os.path.getsize(dest) / (1024 * 1024):.2f} MB). Skipping.")
            continue
        ok = download_file_streaming(item["url"], dest, item["desc"])
        if not ok:
            all_success = False
            break
            
    if all_success:
        print("\n" + "=" * 80)
        print("ALL ELLIPTIC DATASET FILES ACQUIRED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print("\n[ERROR] Failed to acquire all Elliptic dataset files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
