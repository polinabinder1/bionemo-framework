"""Part 1: Compute L19 sink threshold from a few random shards of the 25M v2 parquet.

Read N random shards in full (each = 100k tokens). For each token compute
max(|x[56]|, |x[562]|, |x[1786]|) on centered activations. 99.9th percentile
of that distribution = threshold. Compare to existing top-1000 |u1| sinks.
"""
import json, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PARQUET = Path("/data/interp/evo2/activations/evo2_1b_base_layer19_parquet_25M_prokeuk_v2")
SINK_CSV = "/tmp/sink_identity_L19.csv"
OUT = "/data/interp/evo2/L19_sink_threshold.txt"
SINK_CHANNELS = [56, 562, 1786]
N_SHARDS = 10  # 10 * 100k = 1M tokens
SEED = 42

meta = json.load(open(PARQUET / "metadata.json"))
hidden = meta["hidden_dim"]; n_shards = meta["n_shards"]; ss = meta["shard_size"]
print(f"parquet: {meta['n_samples']:,} tokens, {n_shards} shards, hidden={hidden}")

rng = np.random.default_rng(SEED)
sel_shards = sorted(rng.choice(n_shards, N_SHARDS, replace=False).tolist())
print(f"reading {N_SHARDS} random shards: {sel_shards}")

t0 = time.time()
Xs, gis = [], []
for s in sel_shards:
    t = pq.read_table(PARQUET / f"shard_{s:05d}.parquet")
    arr = np.stack([t.column(f"dim_{j}").to_numpy() for j in range(hidden)], axis=1).astype(np.float32)
    Xs.append(arr)
    gis.append(np.arange(s * ss, s * ss + arr.shape[0]))
    print(f"  shard {s} loaded {arr.shape[0]} tokens ({time.time()-t0:.1f}s)", flush=True)
X = np.concatenate(Xs, axis=0)
global_idx = np.concatenate(gis, axis=0)
print(f"total: X.shape={X.shape}, load took {time.time()-t0:.1f}s")

# Center per channel
mu = X.mean(0)
X_c = X - mu
sink_mag = np.abs(X_c[:, SINK_CHANNELS]).max(axis=1)

print(f"\nDistribution of max(|x[56]|,|x[562]|,|x[1786]|) on centered activations:")
print(f"  median = {np.median(sink_mag):.4f}")
print(f"  90th   = {np.percentile(sink_mag, 90):.4f}")
print(f"  99th   = {np.percentile(sink_mag, 99):.4f}")
print(f"  99.9th = {np.percentile(sink_mag, 99.9):.4f}")
print(f"  max    = {sink_mag.max():.4f}")

threshold = float(np.percentile(sink_mag, 99.9))
n_flagged = int((sink_mag > threshold).sum())
print(f"\nThreshold (99.9th pct) = {threshold:.4f}")
print(f"Tokens flagged at threshold: {n_flagged}/{len(sink_mag)} ({n_flagged/len(sink_mag)*100:.3f}%)")

# Overlap with existing top-1000 |u1| sinks
sink_csv = np.loadtxt(SINK_CSV, delimiter=",", skiprows=1)
u1_sink_global = set(sink_csv[:, 0].astype(np.int64).tolist())
flagged_global = set(global_idx[sink_mag > threshold].tolist())
sampled_u1 = u1_sink_global & set(global_idx.tolist())
overlap = u1_sink_global & flagged_global
print(f"\nu1 top-1000 sinks: {len(u1_sink_global)}; in our 1M sample: {len(sampled_u1)}")
print(f"magnitude-flagged: {len(flagged_global)}")
print(f"overlap (flagged ∩ u1): {len(overlap)}")
cond_pct = 100 * len(overlap) / max(1, len(sampled_u1))
print(f"CONDITIONAL overlap (of u1-sinks-in-sample): {cond_pct:.1f}%")

with open(OUT, "w") as f:
    f.write(f"{threshold:.6f}\n")
    f.write(f"# L19 sink threshold: 99.9th pct of max(|x[56]|,|x[562]|,|x[1786]|) on centered 1M tokens\n")
    f.write(f"# tokens flagged: {n_flagged}/{len(sink_mag)} ({n_flagged/len(sink_mag)*100:.3f}%)\n")
    f.write(f"# conditional overlap with |u1| top-1000: {cond_pct:.1f}%\n")

print(f"\nsaved -> {OUT}")
