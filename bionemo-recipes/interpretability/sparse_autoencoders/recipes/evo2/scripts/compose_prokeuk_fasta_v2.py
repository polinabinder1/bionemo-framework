"""Compose a prok+euk FASTA the Goodfire way, adapted to the 1B/8k-context model.

Differences vs compose_prokeuk_fasta.py (v1):
  - Uses *whole* prokaryotic genomes (no 50kb truncation). The "metagenomes"
    file in OpenGenome2 actually contains whole 1-5 Mbp bacterial genome
    assemblies, equivalent to GTDB representative genomes Goodfire used.
  - Samples *random 8k windows* from each genome rather than always taking
    the 5' end -- uniform position coverage matches Goodfire's chunking.
  - Sample with replacement / reservoir for broad taxonomic spread instead
    of the first N records in file order.
  - Each output record is already chunk-sized (no further chunking needed by
    chunk_fasta.py).
  - Defaults to Goodfire's 2,752 prokaryotic genome count.

Output FASTA: each record is one chunk-sized window. With --chunks-per-genome 1
and --n-genomes 2752 plus an equal-bp euk side, the output is ready to feed
straight into extract.py without going through chunk_fasta.py.
"""

import argparse
import gzip
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _iter_records(fh):
    """Yield (header, joined_seq) from a FASTA file handle."""
    header = None
    lines = []
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(lines)
            header = line
            lines = []
        elif line:
            lines.append(line)
    if header is not None:
        yield header, "".join(lines)


def reservoir_sample_records(fh, n, max_scan=None, rng=None):
    """Reservoir-sample n records from a FASTA handle.

    Goes through up to max_scan records; broader scan -> better taxonomic spread
    but slower I/O. Set max_scan=None to scan the whole file.
    """
    rng = rng or np.random.default_rng()
    reservoir = []
    for i, rec in enumerate(_iter_records(fh)):
        if max_scan is not None and i >= max_scan:
            break
        if len(reservoir) < n:
            reservoir.append(rec)
        else:
            j = int(rng.integers(0, i + 1))
            if j < n:
                reservoir[j] = rec
    return reservoir


def random_window(seq: str, window_size: int, rng) -> str:
    """Return a random `window_size`-bp slice of seq; if seq is shorter, return as-is."""
    if len(seq) <= window_size:
        return seq
    start = int(rng.integers(0, len(seq) - window_size + 1))
    return seq[start : start + window_size]


def write_record(out, counter, source, seq):
    """Write one FASTA record with a unique header."""
    out.write(f">seq_{counter} {source}\n")
    # 80-char width lines for portability with downstream tools.
    for i in range(0, len(seq), 80):
        out.write(seq[i : i + 80] + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/data/interp/evo2/OpenGenome2/fasta"),
        help="OpenGenome2 fasta root.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/data/interp/evo2/scratch/mixed_25M_prokeuk_v2.fasta"),
    )
    ap.add_argument("--n-genomes", type=int, default=2752, help="Prokaryotic whole genomes to sample (Goodfire used 2752).")
    ap.add_argument("--chunks-per-genome", type=int, default=1, help="Random 8k windows to draw from each genome.")
    ap.add_argument("--chunk-size", type=int, default=8192, help="Window size in bp; matches 1B trained context.")
    ap.add_argument("--n-euk-records", type=int, default=2752, help="Eukaryotic 5kb-window records to sample.")
    ap.add_argument(
        "--max-scan-prok", type=int, default=50000,
        help="Cap on how many prok records to scan when reservoir sampling (None = full file).",
    )
    ap.add_argument(
        "--max-scan-euk", type=int, default=200000,
        help="Cap on how many euk records to scan when reservoir sampling.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    metagenome_file = args.root / "metagenomes" / "filtered_metagenomes_pt1.fasta.gz"
    euk_parts = sorted((args.root / "eukaryotic_genic_windows").glob("*.fasta.gz.*"))

    if not metagenome_file.exists():
        print(f"ERROR: missing metagenome source at {metagenome_file}", file=sys.stderr)
        sys.exit(1)
    if not euk_parts:
        print(f"ERROR: no eukaryotic_genic_windows parts under {args.root}", file=sys.stderr)
        sys.exit(1)

    counter = 0
    bp_by_source: dict[str, int] = {}

    def _emit(out, source, seq):
        nonlocal counter
        write_record(out, counter, source, seq)
        counter += 1
        bp_by_source[source] = bp_by_source.get(source, 0) + len(seq)

    with open(args.output, "w") as out:
        # 1. Prokaryotic: reservoir-sample whole genomes, then draw random 8k windows.
        print(f"[prok] reservoir-sampling {args.n_genomes} whole genomes (scanning up to {args.max_scan_prok} records)...")
        with _open_text(metagenome_file) as fh:
            prok_records = reservoir_sample_records(fh, args.n_genomes, args.max_scan_prok, rng)
        print(f"[prok] got {len(prok_records)} genomes; drawing {args.chunks_per_genome} random window(s) per genome...")
        for _, seq in prok_records:
            for _ in range(args.chunks_per_genome):
                _emit(out, "prok_genome_random_window", random_window(seq, args.chunk_size, rng))

        # 2. Eukaryotic: reservoir-sample euk genic windows (kept as-is, already ~5kb).
        print(f"[euk] reservoir-sampling {args.n_euk_records} genic windows (scanning up to {args.max_scan_euk} records)...")
        cat_parts = subprocess.Popen(
            ["bash", "-c", f"cat {' '.join(str(p) for p in euk_parts)} | zcat"],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            euk_records = reservoir_sample_records(cat_parts.stdout, args.n_euk_records, args.max_scan_euk, rng)
        finally:
            cat_parts.stdout.close()
            cat_parts.terminate()
        print(f"[euk] got {len(euk_records)} windows; emitting (truncated to {args.chunk_size} bp if longer)...")
        for _, seq in euk_records:
            _emit(out, "euk_genic_window", seq[: args.chunk_size])

    total_bp = sum(bp_by_source.values())
    n_tokens_est = total_bp  # 1 token per bp for DNA
    print()
    print(f"wrote {counter} records, {total_bp:,} bp ({n_tokens_est / 1e6:.1f}M tokens est.)")
    print(f"  output: {args.output}")
    print(f"  bp by source:")
    for src, bp in bp_by_source.items():
        pct = 100 * bp / total_bp if total_bp else 0
        print(f"    {src:<30} {bp:>12,} bp  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
