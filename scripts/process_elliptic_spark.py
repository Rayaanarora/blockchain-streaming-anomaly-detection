"""
Run the Elliptic preprocessing through Apache Spark.

    .venv\\Scripts\\python.exe scripts\\process_elliptic_spark.py
    .venv\\Scripts\\python.exe scripts\\process_elliptic_spark.py --overwrite   # old output moved to *.backup_<ts>

Reads data/raw/elliptic/*.csv (never modified) and writes data/processed/elliptic_spark/.
Then run scripts/validate_elliptic_spark_parity.py.
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.spark.elliptic_spark import run  # noqa: E402
from src.spark.session import get_spark  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "data/raw/elliptic"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data/processed/elliptic_spark"))
    ap.add_argument("--master", default="local[4]")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    spark = get_spark("EllipticSparkPreprocessing", master=args.master)
    try:
        result = run(spark, args.raw_dir, args.out_dir, overwrite=args.overwrite)
    finally:
        spark.stop()
    print(json.dumps(result["manifest"], indent=2))
    print(json.dumps(result["report"]["integrity"], indent=2))
    print(json.dumps(result["report"]["splits"], indent=2))
    if result["backup"]:
        print("previous output preserved at", result["backup"])


if __name__ == "__main__":
    main()
