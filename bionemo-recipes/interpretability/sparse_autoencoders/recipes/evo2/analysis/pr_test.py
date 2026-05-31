"""Uniform random 1M-token PR test on a parquet directory.
Usage: python pr_test.py <parquet_dir>
"""
import sys
import json
import time
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict

PARQUET = Path(sys.argv[1])
N_SAMPLE = 1_000_000
SEED = 42

meta = json.load(open(PARQUET / "metadata.json"))
HIDDEN = meta["hidden_dim"]
N_TOKENS = meta["n_samples"]
SHARD_SIZE = meta["shard_size"]
N = min(N_SAMPLE, N_TOKENS)

print(f"=== PR test: {PARQUET.name} ===")
print(f"  ambient = {HIDDEN}, total tokens = {N_TOKENS:,}, sampling = {N:,}")

rng = np.random.default_rng(SEED)
idx = rng.choice(N_TOKENS, size=N, replace=False)
idx.sort()
shard_ids = idx // SHARD_SIZE
row_within = idx % SHARD_SIZE

rows_per_shard = defaultdict(list)
for sid, r in zip(shard_ids, row_within):
    rows_per_shard[int(sid)].append(int(r))
print(f"  loading {len(rows_per_shard)} shards ...")

X = np.empty((N, HIDDEN), dtype=np.float32)
cursor = 0
t0 = time.time()
for sid, rows in sorted(rows_per_shard.items()):
    t = pq.read_table(PARQUET / f"shard_{sid:05d}.parquet")
    full = np.stack([t.column(f"dim_{j}").to_numpy() for j in range(HIDDEN)], axis=1).astype(np.float32)
    rows_arr = np.array(rows)
    rows_arr = rows_arr[rows_arr < full.shape[0]]
    n_taken = len(rows_arr)
    X[cursor:cursor + n_taken] = full[rows_arr]
    cursor += n_taken
    del full
X = X[:cursor]
print(f"  loaded {cursor:,} tokens in {time.time()-t0:.0f}s")

t0 = time.time()
Xc = X - X.mean(0, keepdims=True)
S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
lam = (S ** 2) / (Xc.shape[0] - 1)
PR = (lam.sum() ** 2) / (lam ** 2).sum()
stable_rank = (S ** 2).sum() / (S[0] ** 2)
cum = np.cumsum(lam) / lam.sum()
print(f"  SVD: {time.time()-t0:.0f}s")
print()
print(f"  top-1/2 ratio = {lam[0]/lam[1]:.2f}")
for thr in [0.5, 0.9, 0.99, 0.999]:
    k = int(np.searchsorted(cum, thr) + 1)
    print(f"  {thr*100:5.1f}% var in k = {k:>4d}")
print(f"  *** PR = {PR:.2f}")
print(f"  *** stable_rank = {stable_rank:.2f}")

out = PARQUET.parent / f"pr_test_{PARQUET.name}.npz"
np.savez(out, lam=lam, cum=cum, PR=PR, stable_rank=stable_rank)
print(f"  saved -> {out}")
