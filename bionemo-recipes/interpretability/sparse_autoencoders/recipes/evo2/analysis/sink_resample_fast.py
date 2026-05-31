"""Fast sink analysis with N resamples using covariance-matrix eigendecomp.

Replaces full SVD (~200s) with X^T X / eigh on a 1920x1920 matrix (~10-15s).
Same metrics as before: PR, top1/top2, v1 spikiness, u1 sink ratio,
PR after dropping top-X% sinks. Numerically identical up to FP noise.

Usage: python sink_resample_fast.py <parquet_dir> <hidden_dim> [n_resamples]
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

PARQUET = Path(sys.argv[1])
HIDDEN = int(sys.argv[2])
N_RESAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 3
N = 1_000_000
SPIKY_MULT = 5

meta = json.load(open(PARQUET / "metadata.json"))
total = meta["n_samples"]
shard_size = meta["shard_size"]


def load_random(seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=min(N, total), replace=False)
    idx.sort()
    sids = idx // shard_size
    rows = idx % shard_size
    rps = defaultdict(list)
    pos = defaultdict(list)
    for k, (s, r) in enumerate(zip(sids, rows)):
        rps[int(s)].append(int(r))
        pos[int(s)].append(k)
    X = np.empty((len(idx), HIDDEN), dtype=np.float32)
    for s in sorted(rps):
        t = pq.read_table(PARQUET / f"shard_{s:05d}.parquet")
        full = np.stack([t.column(f"dim_{j}").to_numpy() for j in range(HIDDEN)], axis=1).astype(np.float32)
        rarr = np.array(rps[s])
        parr = np.array(pos[s])
        m = rarr < full.shape[0]
        X[parr[m]] = full[rarr[m]]
        del full
    return X, idx


def eigen_top(Xc):
    """Centered X (n, d) -> (eigvals_desc, v1, u1)
    via covariance matrix eigendecomp. Much faster than full SVD for d << n.

    Uses float32 matmul (Xc is fp32; result C is fp32). eigh on 1920x1920 is
    cheap and float32 is plenty for our diagnostics (we care about top-N
    eigenvalues that differ by orders of magnitude).
    """
    n = Xc.shape[0]
    # X^T X is d x d, cheap for d=1920. Stay in float32 to avoid 15GB alloc.
    C = (Xc.T @ Xc).astype(np.float64) / (n - 1)
    eigvals, V = np.linalg.eigh(C)  # ascending; V columns are eigvecs
    eigvals = eigvals[::-1].clip(min=0.0)
    V = V[:, ::-1]
    v1 = V[:, 0].astype(np.float32)
    sigma1 = np.sqrt(eigvals[0] * (n - 1) + 1e-30)
    u1 = (Xc @ v1) / sigma1
    return eigvals, v1, u1


def analyze_sample(seed, X, global_idx):
    print(f"\n--- sample seed={seed} ---", flush=True)
    print(f"  shape={X.shape}, per-token L2 mean={np.linalg.norm(X, axis=1).mean():.2f}, std={np.linalg.norm(X, axis=1).std():.2f}", flush=True)
    mu = X.mean(0)
    mu_norm = np.linalg.norm(mu)
    print(f"  ||mu|| = {mu_norm:.2f}", flush=True)

    t0 = time.time()
    Xc = X - mu
    eigvals, v1, u1 = eigen_top(Xc)
    print(f"  eigendecomp time: {time.time()-t0:.1f}s", flush=True)

    PR = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    top1_2 = eigvals[0] / eigvals[1]
    iso = 1.0 / np.sqrt(HIDDEN)
    v1_abs = np.abs(v1)
    u1_abs = np.abs(u1)
    spiky_chans = np.where(v1_abs > SPIKY_MULT * iso)[0]
    u1_max = u1_abs.max()
    u1_mean = u1_abs.mean()
    u1_999 = np.percentile(u1_abs, 99.9)
    u1_50 = np.percentile(u1_abs, 50)
    print(f"  PR (all)              = {PR:.2f}", flush=True)
    print(f"  top1/top2 eigenvalue  = {top1_2:.2f}", flush=True)
    print(f"  v1 max|v1|            = {v1_abs.max():.3f}  (DC ref = {iso:.3f}, ratio = {v1_abs.max()/iso:.1f}x)", flush=True)
    print(f"  spiky channels (>5x DC ref): {len(spiky_chans)} chans: {spiky_chans.tolist()}", flush=True)
    print(f"  u1 max/mean ratio     = {u1_max/u1_mean:.0f}", flush=True)
    print(f"  u1 99.9th/50th        = {u1_999/u1_50:.1f}", flush=True)

    # Drop top-X% sinks: form Xc_filtered, redo eigh
    drops = {}
    for pct in (99, 95):
        keep = u1_abs < np.percentile(u1_abs, pct)
        Xc2 = X[keep] - X[keep].mean(0, keepdims=True)
        ev2, _, _ = eigen_top(Xc2)
        PR2 = (ev2.sum() ** 2) / (ev2 ** 2).sum()
        cum = np.cumsum(ev2) / ev2.sum()
        k99 = int(np.searchsorted(cum, 0.99) + 1)
        drops[pct] = (PR2, k99)
        print(f"  drop top {100-pct}% |u1|: PR = {PR2:.2f}, 99% var k = {k99}", flush=True)

    thresh = np.percentile(u1_abs, 99)
    top_u1_global = global_idx[u1_abs >= thresh]

    return {
        "seed": seed,
        "mu_norm": float(mu_norm),
        "PR": float(PR),
        "top1_2": float(top1_2),
        "v1_max": float(v1_abs.max()),
        "spiky_chans": spiky_chans.tolist(),
        "u1_max_over_mean": float(u1_max / u1_mean),
        "u1_999_over_50": float(u1_999 / u1_50),
        "PR_drop_1pct": float(drops[99][0]),
        "k99_drop_1pct": int(drops[99][1]),
        "PR_drop_5pct": float(drops[95][0]),
        "k99_drop_5pct": int(drops[95][1]),
        "top_u1_global": top_u1_global,
    }


print(f"\n{'='*70}", flush=True)
print(f"sink analysis (covariance-eigh) x {N_RESAMPLES} resamples: {PARQUET.name}", flush=True)
print(f"  ambient = {HIDDEN}, total tokens = {total:,}, sampling N = {N:,}", flush=True)
print(f"{'='*70}", flush=True)

results = []
for i in range(N_RESAMPLES):
    seed = 42 + i
    print(f"\n[{time.strftime('%H:%M:%S')}] loading sample seed={seed} ...", flush=True)
    t0 = time.time()
    X, gidx = load_random(seed)
    print(f"  load time: {time.time()-t0:.1f}s", flush=True)
    r = analyze_sample(seed, X, gidx)
    results.append(r)
    del X

print(f"\n{'='*70}", flush=True)
print(f"ROBUSTNESS across {N_RESAMPLES} samples:", flush=True)
print(f"{'='*70}", flush=True)
keys = ["mu_norm", "PR", "top1_2", "v1_max", "u1_max_over_mean", "u1_999_over_50",
        "PR_drop_1pct", "k99_drop_1pct", "PR_drop_5pct", "k99_drop_5pct"]
print(f"  {'metric':30s}  " + "  ".join(f"seed={r['seed']:>3d}" for r in results) + "   mean +/- std", flush=True)
for k in keys:
    vals = [r[k] for r in results]
    mean = np.mean(vals)
    std = np.std(vals)
    line = f"  {k:30s}  " + "  ".join(f"{v:>8.3f}" for v in vals) + f"   {mean:>8.3f} +/- {std:.3f}"
    print(line, flush=True)

chan_sets = [set(r["spiky_chans"]) for r in results]
common_chans = set.intersection(*chan_sets) if chan_sets else set()
union_chans = set.union(*chan_sets) if chan_sets else set()
print(f"\n  spiky channels — union: {sorted(union_chans)}", flush=True)
print(f"  spiky channels — intersection (in ALL samples): {sorted(common_chans)}", flush=True)
print(f"  consistency: {len(common_chans)}/{len(union_chans)} channels present in every resample", flush=True)

print(f"\n  top-1% u1 token overlap (pairwise):", flush=True)
for i in range(len(results)):
    for j in range(i+1, len(results)):
        s_i = set(results[i]["top_u1_global"].tolist())
        s_j = set(results[j]["top_u1_global"].tolist())
        ov = s_i & s_j
        expected = len(s_i) * len(s_j) / total
        print(f"    seed {results[i]['seed']} vs seed {results[j]['seed']}: "
              f"|A&B|={len(ov):,}  vs random ~{expected:.0f} ({len(ov)/max(1,expected):.1f}x)", flush=True)
