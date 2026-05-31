"""Driver: python run_analysis.py <label> <parquet_dir> [--cross-model <csv>]"""
import argparse
from analysis_lib import analyze_layer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("parquet_dir")
    ap.add_argument("--fasta", default="/data/interp/evo2/scratch/mixed_25M_prokeuk_v2.fasta")
    ap.add_argument("--dp-size", type=int, default=4)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--cross-model", default=None)
    a = ap.parse_args()
    analyze_layer(a.parquet_dir, a.label, a.fasta, a.dp_size, a.n, a.seed,
                  a.top_k, a.cross_model)
