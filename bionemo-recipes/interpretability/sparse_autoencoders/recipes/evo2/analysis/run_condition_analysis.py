"""Driver: per-condition sink analysis on a parquet whose FASTA carries cond= headers.

Usage:
    python run_condition_analysis.py <parquet_dir> <fasta> [--out <json>] [--top-k 200]
"""
import argparse, json
from analysis_lib import analyze_conditions

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet_dir")
    ap.add_argument("fasta")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--subsample", type=int, default=200_000)
    a = ap.parse_args()
    res = analyze_conditions(a.parquet_dir, a.fasta, top_k=a.top_k, subsample=a.subsample)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nsaved -> {a.out}")
