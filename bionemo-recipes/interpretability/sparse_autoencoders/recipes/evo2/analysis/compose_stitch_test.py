"""Synthetic sequences with @ stitch tokens at known positions.

Conditions (50 each, all 8192 bp uniform random ACGT base):
  stitch_1k    @ at positions 1024, 2048, ..., 7168   (7 stitch tokens, ~every 1024 bp)
  stitch_2k    @ at positions 2048, 4096, 6144         (3 stitch tokens, ~every 2048 bp)
  stitch_irreg @ at randomly placed positions, 5-10 per sequence (mimics real exon/intron variability)

Stitch positions encoded in header as 'stitch_pos=p1,p2,...' so analysis can map them.
"""
import numpy as np

OUT = "/data/interp/evo2/scratch/stitch_token_test.fasta"
SEQ_LEN = 8192
N_PER = 50
rng = np.random.default_rng(13)
BASES = np.array(list("ACGT"))


def random_dna(n):
    return "".join(BASES[rng.integers(0, 4, n)])


def make_with_stitches(positions):
    """Build an 8192-bp sequence with @ at the given positions (sorted)."""
    chars = list(random_dna(SEQ_LEN))
    for p in positions:
        if 0 <= p < SEQ_LEN:
            chars[p] = "@"
    return "".join(chars)


count = {}
with open(OUT, "w") as f:
    # stitch_1k: @ every 1024 bp
    positions_1k = list(range(1024, SEQ_LEN, 1024))   # 1024, 2048, ..., 7168
    for i in range(N_PER):
        seq = make_with_stitches(positions_1k)
        pos_str = ",".join(str(p) for p in positions_1k)
        f.write(f">cond=stitch_1k|src=synth|src_idx={i}|tag_len=0|total_len={SEQ_LEN}|stitch_pos={pos_str}\n{seq}\n")
    count["stitch_1k"] = N_PER

    # stitch_2k: @ every 2048 bp
    positions_2k = [2048, 4096, 6144]
    for i in range(N_PER):
        seq = make_with_stitches(positions_2k)
        pos_str = ",".join(str(p) for p in positions_2k)
        f.write(f">cond=stitch_2k|src=synth|src_idx={i}|tag_len=0|total_len={SEQ_LEN}|stitch_pos={pos_str}\n{seq}\n")
    count["stitch_2k"] = N_PER

    # stitch_irreg: 5-10 random positions per sequence
    for i in range(N_PER):
        k = rng.integers(5, 11)
        positions = sorted(rng.integers(500, SEQ_LEN - 500, size=k).tolist())
        seq = make_with_stitches(positions)
        pos_str = ",".join(str(p) for p in positions)
        f.write(f">cond=stitch_irreg|src=synth|src_idx={i}|tag_len=0|total_len={SEQ_LEN}|stitch_pos={pos_str}\n{seq}\n")
    count["stitch_irreg"] = N_PER

print(f"wrote {sum(count.values())} records to {OUT}")
for k, v in count.items():
    print(f"  {k}: {v}")
