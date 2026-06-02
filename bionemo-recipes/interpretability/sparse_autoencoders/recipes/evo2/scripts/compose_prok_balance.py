"""Compose a prok-only FASTA to balance an existing prok+euk FASTA to ~50/50.

Hashes every prok record in --exclude-from (the existing FASTA), then
reservoir-samples whole bacterial genomes from OpenGenome2's metagenomes file,
draws random 8k windows, and emits only windows whose 64-bit sequence hash
collides with NEITHER the existing FASTA NOR previously-emitted windows in
this run.

Preserves the GTDB-source header in each new record's FASTA header so future
iterations can also dedup by genome ID:

    >seq_0 prok_genome_random_window|gtdb_id=<original metagenome header>

Defaults sized for the 1B/8k-context Goodfire-style corpus: 44k new records
balances a 30k-existing-prok / 100k-euk FASTA to ~50/50 (605M bp each).
"""
import argparse
import gzip
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _iter_records(fh):
    header, lines = None, []
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
    if len(seq) <= window_size:
        return seq
    start = int(rng.integers(0, len(seq) - window_size + 1))
    return seq[start : start + window_size]


def hash_seq(s: str) -> bytes:
    return hashlib.sha256(s.encode()).digest()[:8]


def collect_existing_hashes(existing_fasta: Path, source_tag: str = "prok_genome_random_window") -> set:
    """One pass over the existing FASTA; return set of 64-bit hashes for records
    whose header contains source_tag."""
    seen = set()
    with open(existing_fasta) as f:
        cur_hdr, cur_seq = None, []
        for line in f:
            if line.startswith(">"):
                if cur_hdr and source_tag in cur_hdr:
                    seen.add(hash_seq("".join(cur_seq)))
                cur_hdr = line.strip()
                cur_seq = []
            else:
                cur_seq.append(line.rstrip("\n"))
        if cur_hdr and source_tag in cur_hdr:
            seen.add(hash_seq("".join(cur_seq)))
    return seen


def write_record(out, counter, source, seq, gtdb_id=None):
    extra = f"|gtdb_id={gtdb_id}" if gtdb_id else ""
    out.write(f">seq_{counter} {source}{extra}\n")
    for i in range(0, len(seq), 80):
        out.write(seq[i : i + 80] + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/data/interp/evo2/OpenGenome2/fasta"))
    ap.add_argument("--exclude-from", type=Path, required=True,
                    help="Existing FASTA whose prok records should be excluded from this sample.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-new-records", type=int, default=44_000,
                    help="Target number of new prok records to emit.")
    ap.add_argument("--oversample", type=int, default=50_000,
                    help="Reservoir size; should be > n-new-records to absorb hash collisions.")
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--max-scan-prok", type=int, default=200_000,
                    help="Cap on prok records scanned by reservoir sampler.")
    ap.add_argument("--seed", type=int, default=43,
                    help="DIFFERENT from the original composer seed (was 42) to start in a different reservoir state.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[exclude] hashing prok records in {args.exclude_from}...")
    exclude = collect_existing_hashes(args.exclude_from)
    print(f"[exclude] {len(exclude):,} unique prok hashes loaded")

    metagenome_file = args.root / "metagenomes" / "filtered_metagenomes_pt1.fasta.gz"
    if not metagenome_file.exists():
        print(f"ERROR: missing metagenome source at {metagenome_file}", file=sys.stderr)
        sys.exit(1)

    print(f"[prok] reservoir-sampling {args.oversample} whole genomes (scanning up to {args.max_scan_prok})...")
    with _open_text(metagenome_file) as fh:
        prok_records = reservoir_sample_records(fh, args.oversample, args.max_scan_prok, rng)
    print(f"[prok] got {len(prok_records)} candidate genomes")

    counter = 0
    new_seen = set()
    n_collisions_existing = 0
    n_collisions_new = 0
    with open(args.output, "w") as out:
        for hdr, seq in prok_records:
            if counter >= args.n_new_records:
                break
            win = random_window(seq, args.chunk_size, rng)
            h = hash_seq(win)
            if h in exclude:
                n_collisions_existing += 1
                continue
            if h in new_seen:
                n_collisions_new += 1
                continue
            new_seen.add(h)
            gtdb_id = hdr.lstrip(">").split()[0] if hdr else None
            write_record(out, counter, "prok_genome_random_window", win, gtdb_id=gtdb_id)
            counter += 1

    total_bp = counter * args.chunk_size  # may overcount slightly if some seqs < 8192
    print(f"\n[done] emitted {counter} records ({total_bp:,} bp ≈ {total_bp/1e6:.1f} M)")
    print(f"[done] collisions vs existing: {n_collisions_existing}, vs in-run dupes: {n_collisions_new}")
    if counter < args.n_new_records:
        print(f"[warn] only got {counter}/{args.n_new_records} -- bump --oversample or --max-scan-prok", file=sys.stderr)
    print(f"[done] output -> {args.output}")


if __name__ == "__main__":
    main()
