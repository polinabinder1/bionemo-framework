"""Position-0 enrichment with proper variable-length baseline.

Background:
  Pool = 1500 prok @ 8192 bp + 2500 euk @ ~5000 bp.
  Position frequency is NOT uniform — positions 0..~4940 appear in all 4000
  records; positions ~4940..8191 appear only in the 1500 prok records.

For each layer (uses existing /tmp/sink_identity_<layer>.csv):
  - n0_sample : how many of the 1M sampled rows have position==0
  - s0        : how many of the top-1000 sinks have position==0
  - enrichment_0 = (s0 / 1000) / (n0_sample / 1_000_000)
  - same for positions 0..9
  - position histogram + baseline overlay saved as PNG
  - markdown summary

DOES NOT redo SVD or u1. Uses same seed=42 to regenerate the EXACT 1M sample
indices used by sink_identity.py, then maps them to positions for the baseline.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FASTA = "/data/interp/evo2/scratch/mixed_25M_prokeuk_v2.fasta"
DP_SIZE = 4
N_SAMPLE = 1_000_000
TOP_K = 1000
SEED = 42

LAYERS = {
    "L12": "/data/interp/evo2/activations/evo2_1b_base_layer12_parquet_25M_prokeuk_v2",
    "L15": "/data/interp/evo2/activations/evo2_1b_base_layer15_parquet_25M_prokeuk_v2",
    "L19": "/data/interp/evo2/activations/evo2_1b_base_layer19_parquet_25M_prokeuk_v2",
    "L22": "/data/interp/evo2/activations/evo2_1b_base_layer22_parquet_25M_prokeuk_v2",
}


def parse_fasta_lengths(path):
    lens = []
    cur = 0
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur > 0:
                    lens.append(cur)
                cur = 0
            else:
                cur += len(line)
        if cur > 0:
            lens.append(cur)
    return lens


def build_position_per_row(record_lengths, dp_size):
    """Same logic as sink_identity.py build_parquet_row_mapping but returns only positions."""
    records_per_rank = [[] for _ in range(dp_size)]
    for fasta_idx, slen in enumerate(record_lengths):
        records_per_rank[fasta_idx % dp_size].append((fasta_idx, slen))
    pos_chunks = []
    for rank in range(dp_size):
        for fasta_idx, slen in records_per_rank[rank]:
            pos_chunks.append(np.arange(slen, dtype=np.int32))
    return np.concatenate(pos_chunks)


def main():
    print(f"parsing FASTA {FASTA} ...")
    record_lengths = parse_fasta_lengths(FASTA)
    print(f"  {len(record_lengths)} records, total bp = {sum(record_lengths):,}")
    print(f"  length distribution: min={min(record_lengths)}, max={max(record_lengths)}, mean={np.mean(record_lengths):.0f}")

    pos_per_row = build_position_per_row(record_lengths, DP_SIZE)
    total_rows = len(pos_per_row)
    print(f"  total mapped rows: {total_rows:,}")

    # Pool position frequency: count_pool[p] = #sequences with length > p
    max_len = max(record_lengths)
    count_pool = np.zeros(max_len, dtype=np.int64)
    for slen in record_lengths:
        count_pool[:slen] += 1
    # Sanity: total tokens = sum(count_pool) = sum(record_lengths)
    assert count_pool.sum() == sum(record_lengths), f"{count_pool.sum()} vs {sum(record_lengths)}"
    print(f"  pool position freq: pos=0 has {count_pool[0]} occurrences, pos=4999 has {count_pool[4999]}, pos=5000 has {count_pool[5000]}, pos=8191 has {count_pool[8191]}")

    # Reproduce the 1M sample indices (same seed as sink_identity.py)
    print(f"\nregenerating 1M sample indices (seed={SEED}) ...")
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(total_rows, size=N_SAMPLE, replace=False)
    sample_idx.sort()
    sample_positions = pos_per_row[sample_idx]

    # Sample position freq
    n0_sample = int((sample_positions == 0).sum())
    sample_at_pos = np.bincount(sample_positions, minlength=max_len).astype(np.int64)
    sample_at_pos = sample_at_pos[:max_len]
    print(f"  n0_sample (pos==0 in 1M draw): {n0_sample}")
    print(f"  expected analytical: {count_pool[0]} x {N_SAMPLE} / {total_rows} = {count_pool[0] * N_SAMPLE / total_rows:.1f}")

    summary_lines = ["# Position-0 enrichment per layer\n"]
    summary_lines.append(f"Pool: {len(record_lengths)} records, {sum(record_lengths):,} bp; positions 0..~{record_lengths[1500] if len(record_lengths) > 1500 else 4999} appear in all 4000; positions ~5000..8191 appear in only {sum(1 for L in record_lengths if L > 5000)} sequences.\n\n")
    summary_lines.append(f"1M sample regenerated with seed={SEED} (matches sink_identity.py).\n\n")
    summary_lines.append("| layer | s0 (sinks @ pos==0) | n0_sample | enrichment @ pos=0 | s_top10 | enrichment @ pos<10 | verdict |\n")
    summary_lines.append("|---|---:|---:|---:|---:|---:|---|\n")

    for layer_name, parquet in LAYERS.items():
        csv_path = f"/tmp/sink_identity_{layer_name}.csv"
        if not Path(csv_path).exists():
            print(f"\n{layer_name}: no CSV at {csv_path}, skipping (run sink_identity.py first)")
            continue

        print(f"\n{'='*60}\n{layer_name}\n{'='*60}")
        # CSV: global_idx, seq_id, pos_in_seq, u1_abs
        sinks = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        sink_positions = sinks["pos_in_seq"].astype(np.int32)
        n_sinks = len(sink_positions)
        print(f"  loaded {n_sinks} sinks from {csv_path}")

        # Per-position enrichment for p in [0, 10)
        for p in range(10):
            sp = int((sink_positions == p).sum())
            np_sample = int(sample_at_pos[p]) if p < max_len else 0
            if np_sample > 0:
                enrich = (sp / n_sinks) / (np_sample / N_SAMPLE)
            else:
                enrich = float("inf")
            print(f"  pos={p:>2}: sinks={sp:>4}, n_p_sample={np_sample:>5}, enrichment={enrich:>8.2f}x")

        s0 = int((sink_positions == 0).sum())
        s_top10 = int((sink_positions < 10).sum())
        n_top10_sample = int(sample_at_pos[:10].sum())
        enrich0 = (s0 / n_sinks) / (n0_sample / N_SAMPLE) if n0_sample > 0 else float("inf")
        enrich_top10 = (s_top10 / n_sinks) / (n_top10_sample / N_SAMPLE) if n_top10_sample > 0 else float("inf")
        print(f"\n  s0          = {s0}")
        print(f"  n0_sample   = {n0_sample}")
        print(f"  enrichment0 = {enrich0:.2f}x")
        print(f"  s_top10     = {s_top10}")
        print(f"  enrichment_top10 = {enrich_top10:.2f}x")

        # Decide verdict
        if enrich0 > 5 and s0 > n0_sample * 0.005:
            v_str = "STRONG pos-0 enrichment"
        elif enrich0 > 2:
            v_str = "Modest pos-0 enrichment"
        else:
            v_str = "No pos-0 enrichment"
        if s_top10 / n_sinks > 0.5:
            v_str += " + most sinks in first 10 positions"
        elif enrich0 > 2:
            v_str += " + bulk of sinks scattered"
        print(f"  verdict: {v_str}")

        summary_lines.append(f"| {layer_name} | {s0} | {n0_sample} | {enrich0:.2f}x | {s_top10} | {enrich_top10:.2f}x | {v_str} |\n")

        # Histogram + baseline overlay
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: linear-scale histogram, first 100 positions
        bin_edges = np.arange(0, 101, 1)
        h_sinks, _ = np.histogram(sink_positions, bins=bin_edges)
        baseline_first100 = (sample_at_pos[:100] / N_SAMPLE) * n_sinks  # expected count per pos
        axes[0].bar(np.arange(100), h_sinks, width=1.0, alpha=0.6, label=f"sinks (n={n_sinks})", color="C3")
        axes[0].plot(np.arange(100), baseline_first100, "k--", label="baseline (uniform from sample)", linewidth=1.5)
        axes[0].set_xlabel("position in sequence")
        axes[0].set_ylabel("# sinks")
        axes[0].set_title(f"{layer_name}: position 0-99 (linear scale)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Right: full position histogram (log y) showing the full chunk
        bin_size = 64
        bins = np.arange(0, max_len + bin_size, bin_size)
        h_sinks_full, _ = np.histogram(sink_positions, bins=bins)
        h_pool_full, _ = np.histogram(np.arange(max_len), bins=bins, weights=count_pool)
        baseline_full = h_pool_full / total_rows * n_sinks  # expected count per bin
        bin_centers = (bins[:-1] + bins[1:]) / 2
        axes[1].bar(bin_centers, h_sinks_full, width=bin_size, alpha=0.6, label=f"sinks (n={n_sinks})", color="C3")
        axes[1].plot(bin_centers, baseline_full, "k--", label="baseline (variable-length pool)", linewidth=1.5)
        axes[1].set_xlabel("position in sequence")
        axes[1].set_ylabel("# sinks per 64-bp bin")
        axes[1].set_title(f"{layer_name}: full distribution (variable-length pool)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        png_path = f"/tmp/sink_position_hist_{layer_name}.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"  saved -> {png_path}")

    summary_path = "/tmp/sink_position_enrichment.md"
    with open(summary_path, "w") as f:
        f.writelines(summary_lines)
    print(f"\n\nsummary -> {summary_path}")
    for line in summary_lines:
        print(line.rstrip())


if __name__ == "__main__":
    main()
