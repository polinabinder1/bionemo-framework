"""Filter a parquet activation store by raw sink-channel magnitude.

For each token: drop if max(|x[c]|) > threshold for c in SINK_CHANNELS.
No centering. The SAE handles its own pre_bias during training.

Writes a new parquet dir with the same shard format + updated metadata.json.
Reports the count and percentage of tokens dropped.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SINK_CHANNELS = [56, 562, 1786]


def verify_threshold(src: Path, hidden: int, n_shards: int,
                     n_sample_shards: int = 10, seed: int = 42):
    """Recompute 99.9th pct of raw max sink-channel magnitude on a sample."""
    rng = np.random.default_rng(seed)
    sel = sorted(rng.choice(n_shards, n_sample_shards, replace=False).tolist())
    print(f"verifying threshold on raw values from shards {sel}...")
    mags = []
    for s in sel:
        t = pq.read_table(src / f"shard_{s:05d}.parquet")
        cols = [t.column(f"dim_{c}").to_numpy() for c in SINK_CHANNELS]
        arr = np.stack(cols, axis=1).astype(np.float32)
        mags.append(np.abs(arr).max(axis=1))
    mag = np.concatenate(mags)
    pct = {p: float(np.percentile(mag, p)) for p in (50, 90, 99, 99.9)}
    print(f"  raw distribution: median={pct[50]:.3f}, 90th={pct[90]:.3f}, "
          f"99th={pct[99]:.3f}, 99.9th={pct[99.9]:.3f}, max={mag.max():.3f}")
    return pct[99.9]


def filter_parquet(src: Path, out: Path, threshold: float):
    meta = json.load(open(src / "metadata.json"))
    hidden, n_shards, ss = meta["hidden_dim"], meta["n_shards"], meta["shard_size"]
    print(f"src: {meta['n_samples']:,} tokens, {n_shards} shards, hidden={hidden}")
    raw_99_9 = verify_threshold(src, hidden, n_shards)
    print(f"  raw 99.9th = {raw_99_9:.3f}, will filter at threshold = {threshold:.3f}\n")

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_in = n_out = n_dropped = 0
    buffer_rows = []
    out_shard_idx = 0

    for s in range(n_shards):
        tbl = pq.read_table(src / f"shard_{s:05d}.parquet")
        arr = np.stack([tbl.column(f"dim_{j}").to_numpy() for j in range(hidden)],
                       axis=1).astype(np.float32)
        n_in += arr.shape[0]
        mag = np.abs(arr[:, SINK_CHANNELS]).max(axis=1)
        keep = mag <= threshold
        n_dropped += int((~keep).sum())
        n_out += int(keep.sum())
        kept = arr[keep]
        if kept.shape[0] == 0:
            continue
        buffer_rows.append(kept)
        total = sum(b.shape[0] for b in buffer_rows)
        while total >= ss:
            X = np.concatenate(buffer_rows, axis=0)
            chunk = X[:ss]
            pq.write_table(pa.table({f"dim_{j}": chunk[:, j] for j in range(hidden)}),
                           out / f"shard_{out_shard_idx:05d}.parquet")
            out_shard_idx += 1
            rem = X[ss:]
            buffer_rows = [rem] if rem.shape[0] > 0 else []
            total = rem.shape[0]
        if (s + 1) % 50 == 0:
            print(f"  src {s+1}/{n_shards}  dst {out_shard_idx}  dropped {n_dropped} ({time.time()-t0:.0f}s)", flush=True)

    if buffer_rows and sum(b.shape[0] for b in buffer_rows) > 0:
        X = np.concatenate(buffer_rows, axis=0)
        pq.write_table(pa.table({f"dim_{j}": X[:, j] for j in range(hidden)}),
                       out / f"shard_{out_shard_idx:05d}.parquet")
        out_shard_idx += 1

    new_meta = {
        "n_samples": n_out, "hidden_dim": hidden,
        "n_shards": out_shard_idx, "shard_size": ss,
        "model_name": meta.get("model_name"), "layer": meta.get("layer"),
        "n_sequences": meta.get("n_sequences"),
        "filtered_from": str(src), "filter_threshold": threshold,
        "filter_channels": SINK_CHANNELS, "filter_type": "raw_max_abs",
        "n_dropped": n_dropped,
    }
    with open(out / "metadata.json", "w") as f:
        json.dump(new_meta, f, indent=2)

    print(f"\n[done] in:{n_in:,}  out:{n_out:,}  dropped:{n_dropped:,} ({n_dropped/n_in*100:.4f}%)")
    print(f"[done] {out_shard_idx} new shards in {out}")
    print(f"[done] elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    a = ap.parse_args()
    filter_parquet(a.src, a.out, a.threshold)
