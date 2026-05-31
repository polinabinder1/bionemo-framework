"""Uniform-random sampling of 1M tokens from the 25M layer-22 parquet.

Unlike the previous dim analysis (which took 11 contiguous shards), this
spreads the 1M tokens across ALL 3,216 chunks via uniform random indexing.
If PR jumps significantly vs the 2.90 we measured before, within-chunk
redundancy was inflating the low-rank reading.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
import time

PARQUET = Path("/data/interp/evo2/activations/evo2_1b_base_layer22_parquet_25M_prokeuk")
HIDDEN = 1920
N_TOKENS = 25_063_561
SHARD_SIZE = 100_000  # nominal, last shard may be smaller
N_SAMPLE = 1_000_000
SEED = 42

rng = np.random.default_rng(SEED)

# Generate N_SAMPLE distinct random token indices uniformly across the whole parquet.
print(f"generating {N_SAMPLE:,} random indices in [0, {N_TOKENS:,}) ...")
idx = rng.choice(N_TOKENS, size=N_SAMPLE, replace=False)
idx.sort()  # so we can group by shard efficiently
print(f"  done. min={idx[0]}, max={idx[-1]}")

# Map each global index to (shard_id, row_within_shard).
shard_ids = idx // SHARD_SIZE
row_within = idx % SHARD_SIZE

# Group rows by shard so we load each shard once.
rows_per_shard = defaultdict(list)
for sid, r in zip(shard_ids, row_within):
    rows_per_shard[int(sid)].append(int(r))

print(f"\nwill load {len(rows_per_shard)} distinct shards (out of 252)")
print(f"  avg rows/shard sampled = {N_SAMPLE / len(rows_per_shard):.1f}")

# Load each shard once, pull the requested rows, accumulate.
X = np.empty((N_SAMPLE, HIDDEN), dtype=np.float32)
cursor = 0
t0 = time.time()
for i, (sid, rows) in enumerate(sorted(rows_per_shard.items())):
    path = PARQUET / f"shard_{sid:05d}.parquet"
    t = pq.read_table(path)
    # Stack the full shard into a numpy matrix, then index the requested rows.
    full = np.stack([t.column(f"dim_{j}").to_numpy() for j in range(HIDDEN)], axis=1).astype(np.float32)
    rows_arr = np.array(rows)
    rows_arr = rows_arr[rows_arr < full.shape[0]]  # last shard may be smaller than 100k
    n_taken = len(rows_arr)
    X[cursor:cursor + n_taken] = full[rows_arr]
    cursor += n_taken
    if (i + 1) % 25 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1:3d}/{len(rows_per_shard)}] loaded shard {sid:>3d}, accumulated {cursor:>8,} rows ({elapsed:.1f}s)")
    del full

X = X[:cursor]
print(f"\nfinal sample shape = {X.shape}, total load time = {time.time()-t0:.1f}s")
print(f"per-token L2 norm: mean={np.linalg.norm(X, axis=1).mean():.2f}, std={np.linalg.norm(X, axis=1).std():.2f}")

# SVD
print(f"\ncomputing SVD on centered data...")
t0 = time.time()
Xc = X - X.mean(axis=0, keepdims=True)
S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
lam = (S ** 2) / (Xc.shape[0] - 1)
total = lam.sum()
cum = np.cumsum(lam) / total
PR = (lam.sum() ** 2) / (lam ** 2).sum()
stable_rank = (S ** 2).sum() / (S[0] ** 2)
print(f"  SVD time: {time.time()-t0:.1f}s")

print(f"\n=== Uniform-random 1M token sample (covers all 3,216 chunks) ===")
print(f"  top-10 eigenvalues: {lam[:10].round(1)}")
print(f"  top-1 / top-2 ratio: {lam[0]/lam[1]:.3f}")
print()
print(f"  --- cumulative variance ---")
for thr in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]:
    k = int(np.searchsorted(cum, thr) + 1)
    print(f"  {thr*100:>5.2f}% var:  k = {k:>4d}  ({100*k/HIDDEN:>5.2f}% of {HIDDEN} dims)")

print()
print(f"  Participation Ratio    = {PR:.2f}")
print(f"  Stable rank            = {stable_rank:.2f}")
print()
print(f"  --- comparison to prior results ---")
print(f"  Prior (11 contiguous shards, ~140 chunks): PR = 2.90")
print(f"  This  (random across all 3,216 chunks):   PR = {PR:.2f}")
print(f"  Delta:                                     {PR - 2.90:+.2f}")
print()
if PR > 10:
    print(f"  >>> Major jump — within-chunk redundancy was inflating the low-rank reading.")
elif PR > 5:
    print(f"  >>> Modest jump — some within-chunk effect but architecture dominates.")
else:
    print(f"  >>> Essentially unchanged — architecture is the dominant cause of low rank.")

np.savez("/tmp/uniform_random_pr.npz", S=S, lam=lam, cum=cum, PR=PR, stable_rank=stable_rank)
print(f"\nspectrum saved to /tmp/uniform_random_pr.npz")
