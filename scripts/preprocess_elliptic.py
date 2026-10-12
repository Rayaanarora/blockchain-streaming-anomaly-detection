"""
Elliptic Dataset Audit and Preprocessing CLI Runner.

Executes schema checks, alignment validation, train-only scaling, and produces
machine-readable JSON and Markdown audit reports.
"""
import os
import sys
import argparse
import json
import pandas as pd

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.elliptic import EllipticPreprocessor, PreprocessorConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Audit and preprocess Elliptic Bitcoin transaction dataset.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw/elliptic",
        help="Directory containing elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/validation",
        help="Directory to save audit report artifacts"
    )
    parser.add_argument(
        "--train-cutoff",
        type=int,
        default=34,
        help="Last time step to include in training split (default: 34)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

    # Fallback to ./data if not in data/raw/elliptic
    if not os.path.exists(features_path) and os.path.exists(os.path.join("data", "elliptic_txs_features.csv")):
        data_dir = "data"
        features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
        classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
        edgelist_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")

    print("=" * 80)
    print("ELLIPTIC BENCHMARK PREPROCESSING & AUDIT RUNNER")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Check file availability
    missing = []
    for name, p in [("Features", features_path), ("Classes", classes_path), ("Edgelist", edgelist_path)]:
        if os.path.exists(p):
            sz_mb = os.path.getsize(p) / (1024 * 1024)
            print(f"  [FOUND] {name}: {p} ({sz_mb:.2f} MB)")
        else:
            print(f"  [MISSING] {name}: {p}")
            missing.append(name)

    if missing:
        print("\n[ERROR] Cannot execute preprocessing on real data: files are missing.")
        print(f"Missing components: {missing}")
        print("Please obtain official Elliptic dataset files before running this script.")
        sys.exit(1)

    print("\nLoading raw CSVs into memory...")
    # Kaggle features file has no header
    features_df = pd.read_csv(features_path, header=None)
    classes_df = pd.read_csv(classes_path)
    edgelist_df = pd.read_csv(edgelist_path)

    print(f"Loaded Features: {features_df.shape}, Classes: {classes_df.shape}, Edgelist: {edgelist_df.shape}")

    config = PreprocessorConfig(train_cutoff_step=args.train_cutoff)
    preprocessor = EllipticPreprocessor(config=config)

    print("\nRunning audit and preprocessing pipeline...")
    dataset, report = preprocessor.audit_and_preprocess(features_df, classes_df, edgelist_df)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "elliptic_audit_report.json")
    md_path = os.path.join(output_dir, "ELLIPTIC_AUDIT_REPORT.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    md_content = preprocessor.generate_markdown_audit_report(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nAudit complete! Reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"\nAligned Nodes: {len(dataset.tx_ids):,}")
    print(f"Train nodes:   {dataset.train_mask.sum():,} (Licit: {report.train_licit_count}, Illicit: {report.train_illicit_count})")
    print(f"Test nodes:    {dataset.test_mask.sum():,} (Licit: {report.test_licit_count}, Illicit: {report.test_illicit_count})")


if __name__ == "__main__":
    main()
