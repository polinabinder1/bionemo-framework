"""Sanity checks 2 + 3 from the user's spec.

Check 2: Are the top-1000 token sets actually identical or just nearly so?
Check 3: Are the activation values at the same row different across layers?

Check 1 (u1 correlation) would need full SVD which we'd have to re-run; if
checks 2 and 3 give clear answers, we skip it.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

LAYERS = {
    "L12": "/data/interp/evo2/activations/evo2_1b_base_layer12_parquet_25M_prokeuk_v2",
    "L15": "/data/interp/evo2/activations/evo2_1b_base_layer15_parquet_25M_prokeuk_v2",
    "L19": "/data/interp/evo2/activations/evo2_1b_base_layer19_parquet_25M_prokeuk_v2",
}

# ============================================================
# Check 2: Top-1000 token set overlap
# ============================================================
print("=" * 60)
print("CHECK 2: top-1000 global_idx set overlap")
print("=" * 60)
sinks = {}
for name in LAYERS:
    idxs = set()
    with open(f"/tmp/sink_identity_{name}.csv") as f:
        next(f)
        for line in f:
            idxs.add(int(line.split(",")[0]))
    sinks[name] = idxs
    print(f"  {name}: |top-1000| = {len(idxs)}")

print()
for a in ["L12", "L15", "L19"]:
    for b in ["L12", "L15", "L19"]:
        if a < b:
            i = len(sinks[a] & sinks[b])
            sd = len(sinks[a] ^ sinks[b])
            in_a_not_b = len(sinks[a] - sinks[b])
            in_b_not_a = len(sinks[b] - sinks[a])
            verdict = "EXACTLY IDENTICAL (suspicious — possible bug)" if sd == 0 else f"high but not identical ({len(sinks[a])-i} differ each side)"
            print(f"  {a} ∩ {b}: |intersection|={i}, sym_diff={sd}  --> {verdict}")
            print(f"    {a}-only: {in_a_not_b},  {b}-only: {in_b_not_a}")

all3 = sinks["L12"] & sinks["L15"] & sinks["L19"]
print(f"\n  All three intersection: {len(all3)}")

# ============================================================
# Check 3: Same parquet row across layers — should differ
# ============================================================
print()
print("=" * 60)
print("CHECK 3: same parquet row across layers (activation vectors)")
print("=" * 60)
# Pick a row that's a top sink for L12
test_idx = 131074  # L12 #1 sink: seq_id=64, pos=2
shard_idx = test_idx // 100000
row_in_shard = test_idx % 100000
print(f"  global_idx = {test_idx} (shard {shard_idx}, row {row_in_shard})")
print()

vecs = {}
for name, parquet in LAYERS.items():
    shard_path = Path(parquet) / f"shard_{shard_idx:05d}.parquet"
    t = pq.read_table(shard_path)
    vec = np.array([t.column(f"dim_{i}")[row_in_shard].as_py() for i in range(1920)], dtype=np.float32)
    vecs[name] = vec
    print(f"  {name}: norm = {np.linalg.norm(vec):>9.4f}, mean = {vec.mean():>+8.4f}, std = {vec.std():>+8.4f}")
    # Look at the 3 spiky channel values
    print(f"        channel 56  = {vec[56]:+8.4f},  channel 562 = {vec[562]:+8.4f},  channel 1786 = {vec[1786]:+8.4f}")

print()
print("  Strict equality between layers (np.allclose):")
for a in ["L12", "L15", "L19"]:
    for b in ["L12", "L15", "L19"]:
        if a < b:
            same = np.allclose(vecs[a], vecs[b])
            l2_diff = np.linalg.norm(vecs[a] - vecs[b])
            print(f"    {a} == {b}: {same}  (||diff||={l2_diff:.4f})")

print()
print("  Cosine similarities between layers:")
for a in ["L12", "L15", "L19"]:
    for b in ["L12", "L15", "L19"]:
        if a < b:
            c = np.dot(vecs[a], vecs[b]) / (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b]) + 1e-30)
            print(f"    cos({a}, {b}) = {c:+.4f}")
