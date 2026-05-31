"""Check whether evo2 layer-22 activations contain duplicates or near-duplicates,
and whether subsampling to 'independent contexts' raises the PR.

Three checks:
  1. Exact-duplicate row count in one shard (hash rows).
  2. Stride-subsample: take every Nth row, recompute PR. If PR jumps as N increases,
     within-sequence redundancy is inflating the low-rank reading.
  3. Across-shard subsample for comparison.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import hashlib

PARQUET = "/data/interp/evo2/activations/evo2_1b_base_layer22_parquet_25M_prokeuk"
HIDDEN = 1920


def load_shard(idx):
    t = pq.read_table(f"{PARQUET}/shard_{idx:05d}.parquet")
    return np.stack([t.column(f"dim_{i}").to_numpy() for i in range(HIDDEN)], axis=1).astype(np.float32)


def pr(X):
    Xc = X - X.mean(0, keepdims=True)
    S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = S ** 2
    return (lam.sum() ** 2) / (lam ** 2).sum()


print("--- CHECK 1: exact-duplicate rows in shard_00000 ---")
X = load_shard(0)
print(f"shape = {X.shape}")
# Hash by quantizing to 4 decimal places to allow for tiny float noise
Xq = (X * 1e4).astype(np.int64)
hashes = [hashlib.md5(row.tobytes()).hexdigest() for row in Xq[:50000]]   # 50k subset for speed
n_unique = len(set(hashes))
print(f"  exact duplicates (50k rows hashed @ 1e-4 precision): {50000 - n_unique} duplicate rows, {n_unique} unique")
print(f"  duplicate rate: {(50000 - n_unique) / 50000 * 100:.2f}%")

print()
print("--- CHECK 2: stride-subsample within-shard PR ---")
print(f"  full shard PR (n={X.shape[0]:>6}) = {pr(X):.2f}")
for stride in [10, 100, 1000, 8000]:
    sub = X[::stride]
    p = pr(sub)
    print(f"  stride={stride:5d} -> n={sub.shape[0]:>6} -> PR = {p:.2f}")

print()
print("--- CHECK 3: spread across many shards (independent contexts) ---")
# Take 1 token from each of 100 different shards (spread across the parquet),
# then 10 tokens from each of 100 shards, then 100 tokens from each of 100 shards.
shard_ids = list(range(0, 252, 3))[:80]   # 80 spread shards
print(f"  sampling from {len(shard_ids)} different shards: {shard_ids[:5]}...{shard_ids[-5:]}")
for tokens_per_shard in [1, 10, 100, 1000]:
    chunks = []
    for sid in shard_ids:
        Xs = load_shard(sid)
        # take first N tokens (these are all from different sequences across shards)
        chunks.append(Xs[:tokens_per_shard])
        del Xs
    Y = np.concatenate(chunks, axis=0)
    p = pr(Y)
    print(f"  {tokens_per_shard:5d} tok/shard x {len(shard_ids):3d} shards = n={Y.shape[0]:>6} -> PR = {p:.2f}")
