"""Diagnose whether the dominant top eigenvalue is DC, sink-token, or massive-activation.

For each parquet:
  - random uniform 1M sample
  - centered SVD
  - inspect top eigenvector (v1) — concentrated on few dims = massive activation
  - inspect top coefficient (u1) — concentrated on few tokens = sink token
  - drop estimated 'first-of-chunk' tokens and re-compute PR

Run on evo2 L12 (OLD + NEW), evo2 L22 (NEW), and codonfm L16.
"""
import sys, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PARQUETS = [
    ("evo2 L12 NEW",  "/data/interp/evo2/activations/evo2_1b_base_layer12_parquet_25M_prokeuk_v2", 1920),
    ("evo2 L22 NEW",  "/data/interp/evo2/activations/evo2_1b_base_layer22_parquet_25M_prokeuk_v2", 1920),
]
N = 1_000_000
SEED = 42


def load_random(parquet_dir, hidden, n):
    meta = json.load(open(Path(parquet_dir) / "metadata.json"))
    total = meta["n_samples"]
    shard_size = meta["shard_size"]
    rng = np.random.default_rng(SEED)
    n_use = min(n, total)
    idx = rng.choice(total, size=n_use, replace=False)
    idx.sort()
    shard_ids = idx // shard_size
    row_within = idx % shard_size
    rows_per_shard = defaultdict(list)
    for sid, r in zip(shard_ids, row_within):
        rows_per_shard[int(sid)].append(int(r))
    X = np.empty((n_use, hidden), dtype=np.float32)
    pos = np.empty(n_use, dtype=np.int64)   # row_within (i.e. position inside its shard)
    cursor = 0
    for sid, rows in sorted(rows_per_shard.items()):
        t = pq.read_table(Path(parquet_dir) / f"shard_{sid:05d}.parquet")
        full = np.stack([t.column(f"dim_{j}").to_numpy() for j in range(hidden)], axis=1).astype(np.float32)
        rows_arr = np.array(rows)
        rows_arr = rows_arr[rows_arr < full.shape[0]]
        n_taken = len(rows_arr)
        X[cursor:cursor + n_taken] = full[rows_arr]
        pos[cursor:cursor + n_taken] = rows_arr
        cursor += n_taken
        del full
    return X[:cursor], pos[:cursor], meta


def pr(eigvals):
    return (eigvals.sum() ** 2) / (eigvals ** 2).sum()


def analyze(name, X, pos):
    print(f"\n=== {name} ===")
    print(f"  shape={X.shape}, per-token L2 mean={np.linalg.norm(X, axis=1).mean():.1f}, std={np.linalg.norm(X, axis=1).std():.1f}")

    mu = X.mean(0)
    print(f"  ||mu|| = {np.linalg.norm(mu):.2f}")

    # Centered SVD — full uv since we need v1 and u1
    Xc = X - mu
    t0 = time.time()
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    print(f"  SVD time: {time.time()-t0:.1f}s")
    lam = S ** 2
    PR_full = pr(lam)
    print(f"  PR (centered, all tokens)              = {PR_full:.2f}")
    print(f"  top-1 / top-2 eigenvalue ratio         = {lam[0]/lam[1]:.2f}")

    # v1 (top right singular vector — direction in feature/channel space)
    v1 = Vt[0]
    v1_abs = np.abs(v1)
    print(f"  v1 (channel direction):")
    print(f"    max|v1|        = {v1_abs.max():.3f}")
    print(f"    std|v1|        = {v1_abs.std():.3f}")
    print(f"    top-5 |v1|     = {sorted(v1_abs, reverse=True)[:5]}")
    print(f"    frac dims >0.1 = {(v1_abs > 0.1).sum() / len(v1)*100:.1f}%")
    # if v1 has only a few dims with high |v1|, the top mode lives on a few channels (massive activation)
    # if v1 is diffuse (most |v1| similar magnitude ~1/sqrt(d)), the top mode is a DC-like direction
    isotropic_floor = 1.0 / np.sqrt(len(v1))
    print(f"    DC reference (1/sqrt(d)) = {isotropic_floor:.3f}")
    if v1_abs.max() > 5 * isotropic_floor:
        print(f"    --> SPIKY: top mode lives on a few channels (massive-activation signature)")
    else:
        print(f"    --> DIFFUSE: top mode is roughly isotropic (DC-like)")

    # u1 (top left singular vector — per-token coefficient on the top mode)
    u1 = U[:, 0]
    u1_abs = np.abs(u1)
    print(f"  u1 (per-token loading on top mode):")
    print(f"    max|u1|        = {u1_abs.max():.4f}")
    print(f"    mean|u1|       = {u1_abs.mean():.4f}")
    print(f"    std|u1|        = {u1_abs.std():.4f}")
    print(f"    99th pct |u1|  = {np.percentile(u1_abs, 99):.4f}")
    print(f"    99.9th pct |u1|= {np.percentile(u1_abs, 99.9):.4f}")
    # if a small number of tokens have huge |u1| and the rest are small, that's a sink-token pattern
    sink_ratio = np.percentile(u1_abs, 99.9) / np.percentile(u1_abs, 50)
    print(f"    99.9th/50th ratio = {sink_ratio:.1f}")
    if sink_ratio > 10:
        print(f"    --> SINK-TOKEN signature (few tokens carry the dominant mode)")
    else:
        print(f"    --> DIFFUSE (top mode spread across many tokens)")

    # PR after dropping "first-of-chunk" tokens.
    # Heuristic: shards contain multiple sequences. Without seq metadata we can
    # only approximate. Drop tokens where row_within_shard < 5 (i.e., near a
    # shard boundary, plus a small buffer). For sink-mode confirmation we use
    # the u1 outliers directly:
    keep = u1_abs < np.percentile(u1_abs, 99)
    Xc_no_sink = X[keep] - X[keep].mean(0, keepdims=True)
    Sb = np.linalg.svd(Xc_no_sink, full_matrices=False, compute_uv=False)
    lam_b = Sb ** 2
    PR_no_sink = pr(lam_b)
    print(f"  PR after dropping top 1% |u1| tokens  = {PR_no_sink:.2f}")

    keep_strict = u1_abs < np.percentile(u1_abs, 95)
    Xc_no_sink_5 = X[keep_strict] - X[keep_strict].mean(0, keepdims=True)
    Sb5 = np.linalg.svd(Xc_no_sink_5, full_matrices=False, compute_uv=False)
    PR_no_sink_5 = pr(Sb5 ** 2)
    print(f"  PR after dropping top 5% |u1| tokens  = {PR_no_sink_5:.2f}")

    # 99% var k after sink removal
    cum = np.cumsum(lam_b) / lam_b.sum()
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    cum5 = np.cumsum(Sb5**2) / (Sb5**2).sum()
    k99_5 = int(np.searchsorted(cum5, 0.99) + 1)
    print(f"  99% var k (drop top 1%) = {k99}")
    print(f"  99% var k (drop top 5%) = {k99_5}")


for name, parquet, hidden in PARQUETS:
    try:
        X, pos, meta = load_random(parquet, hidden, N)
        analyze(name, X, pos)
        del X, pos
    except Exception as e:
        print(f"\n=== {name} FAILED: {e}")
