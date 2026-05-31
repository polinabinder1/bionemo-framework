"""Pre-SAE evo2 activation diagnostics. One file, every layer/model."""
from __future__ import annotations
import json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def load_random_tokens(parquet_dir, n=1_000_000, seed=42):
    """Returns (X[n,hidden], global_idx[n], metadata_dict). Hidden auto-detected."""
    d = Path(parquet_dir)
    meta = json.load(open(d / "metadata.json"))
    total, ss, hidden = meta["n_samples"], meta["shard_size"], meta["hidden_dim"]
    idx = np.sort(np.random.default_rng(seed).choice(total, size=min(n, total), replace=False))
    by_shard = defaultdict(lambda: ([], []))   # shard -> (row_within, sample_pos)
    for k, gi in enumerate(idx):
        s, r = int(gi // ss), int(gi % ss)
        by_shard[s][0].append(r); by_shard[s][1].append(k)
    X = np.empty((len(idx), hidden), dtype=np.float32)
    t0 = time.time()
    for i, s in enumerate(sorted(by_shard)):
        full = np.stack([pq.read_table(d / f"shard_{s:05d}.parquet").column(f"dim_{j}").to_numpy()
                         for j in range(hidden)], axis=1).astype(np.float32)
        rows, pos = map(np.array, by_shard[s])
        keep = rows < full.shape[0]
        X[pos[keep]] = full[rows[keep]]
        if (i + 1) % 50 == 0:
            print(f"  loaded {i+1}/{len(by_shard)} shards ({time.time()-t0:.0f}s)", flush=True)
    return X, idx, meta


def parse_fasta_lengths(path):
    lens, cur = [], 0
    for line in open(path):
        if line.startswith(">"):
            if cur: lens.append(cur)
            cur = 0
        else:
            cur += len(line.rstrip("\n"))
    if cur: lens.append(cur)
    return lens


def row_to_position(lengths, dp_size=4):
    """parquet_row -> (fasta_idx, pos_in_record), assumes DP round-robin then rank-concat."""
    seq_chunks, pos_chunks = [], []
    for rank in range(dp_size):
        for i in range(rank, len(lengths), dp_size):
            seq_chunks.append(np.full(lengths[i], i, dtype=np.int32))
            pos_chunks.append(np.arange(lengths[i], dtype=np.int32))
    return np.concatenate(seq_chunks), np.concatenate(pos_chunks)


def pca_top(X):
    """Centered cov-eigh -> (mu, eigvals_desc, v1, u1)."""
    mu = X.mean(0)
    Xc = X - mu
    n = Xc.shape[0]
    C = (Xc.T @ Xc).astype(np.float64) / (n - 1)
    ev, V = np.linalg.eigh(C)
    ev = ev[::-1].clip(min=0); V = V[:, ::-1]
    v1 = V[:, 0].astype(np.float32)
    u1 = (Xc @ v1) / np.sqrt(ev[0] * (n - 1) + 1e-30)
    return mu, ev, v1, u1


def metrics(ev, v1, u1, hidden, spiky_mult=5):
    """Spectrum + sink diagnostics in one dict."""
    cum = np.cumsum(ev) / ev.sum()
    v1a, u1a = np.abs(v1), np.abs(u1)
    iso = 1.0 / np.sqrt(hidden)
    return {
        "PR": float(ev.sum()**2 / (ev**2).sum()),
        "top1/top2": float(ev[0] / ev[1]),
        "stable_rank": float(ev.sum() / (ev[0] + 1e-30)),
        "k_var": {p: int(np.searchsorted(cum, p/100) + 1) for p in (50, 90, 99, 99.9)},
        "v1_max": float(v1a.max()),
        "v1_max_over_iso": float(v1a.max() / iso),
        "spiky_channels": np.where(v1a > spiky_mult * iso)[0].tolist(),
        "u1_max_over_mean": float(u1a.max() / u1a.mean()),
        "u1_999_over_50": float(np.percentile(u1a, 99.9) / np.percentile(u1a, 50)),
    }


def drop_top_sinks(X, u1, pct):
    """Recompute PR + k99 after dropping the top pct% of |u1| tokens."""
    keep = np.abs(u1) < np.percentile(np.abs(u1), 100 - pct)
    _, ev, _, _ = pca_top(X[keep])
    return {"PR": float(ev.sum()**2 / (ev**2).sum()),
            "k99": int(np.searchsorted(np.cumsum(ev)/ev.sum(), 0.99) + 1)}


def top_sinks(u1, global_idx, seq_per_row, pos_per_row, top_k=1000):
    """Top-K |u1| -> dict of aligned arrays {global_idx, seq_idx, pos_in_seq, u1_abs}."""
    u1a = np.abs(u1)
    sel = np.argsort(u1a)[::-1][:top_k]
    gi = global_idx[sel]; vals = u1a[sel]
    ok = gi < len(seq_per_row)
    gi, vals = gi[ok], vals[ok]
    return {"global_idx": gi, "seq_idx": seq_per_row[gi],
            "pos_in_seq": pos_per_row[gi], "u1_abs": vals}


def save_csv(sinks, path):
    with open(path, "w") as f:
        f.write("global_idx,seq_id,pos_in_seq,u1_abs\n")
        for r in zip(sinks["global_idx"], sinks["seq_idx"], sinks["pos_in_seq"], sinks["u1_abs"]):
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f}\n")


def load_csv(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    return {"global_idx": arr[:, 0].astype(np.int64), "seq_idx": arr[:, 1].astype(np.int32),
            "pos_in_seq": arr[:, 2].astype(np.int32), "u1_abs": arr[:, 3].astype(np.float32)}


def load_single_row(parquet_dir, global_idx):
    """Load one activation row by global index. Returns float32 vector."""
    d = Path(parquet_dir)
    meta = json.load(open(d / "metadata.json"))
    hidden, ss = meta["hidden_dim"], meta["shard_size"]
    t = pq.read_table(d / f"shard_{global_idx//ss:05d}.parquet")
    return np.array([t.column(f"dim_{i}")[global_idx % ss].as_py() for i in range(hidden)],
                    dtype=np.float32)


def position_enrichment(sink_pos, sample_pos, n_top, n_sample, max_pos=10):
    """Per-position enrichment vs sample baseline."""
    base = np.bincount(sample_pos, minlength=max_pos).astype(np.int64)
    return [{"pos": p, "sinks": int((sink_pos == p).sum()),
             "baseline": int(base[p]) if p < len(base) else 0,
             "enrichment": ((sink_pos == p).sum() / n_top) / (base[p] / n_sample)
                            if p < len(base) and base[p] > 0 else float("inf")}
            for p in range(max_pos)]


def analyze_layer(parquet_dir, label, fasta, dp_size=4, n=1_000_000, seed=42,
                  top_k=1000, cross_model_csv=None, csv_out=None, drop_pcts=(1, 5)):
    """One load, full pipeline. Returns results dict + prints summary."""
    csv_out = csv_out or f"/tmp/sink_identity_{label}.csv"
    print(f"\n=== {label}  ({Path(parquet_dir).name}) ===", flush=True)

    X, gi, meta = load_random_tokens(parquet_dir, n, seed)
    print(f"loaded {X.shape}", flush=True)
    mu, ev, v1, u1 = pca_top(X)
    m = metrics(ev, v1, u1, X.shape[1])
    drops = {pct: drop_top_sinks(X, u1, pct) for pct in drop_pcts}

    lens = parse_fasta_lengths(fasta)
    seq_per_row, pos_per_row = row_to_position(lens, dp_size)
    sinks = top_sinks(u1, gi, seq_per_row, pos_per_row, top_k)
    save_csv(sinks, csv_out)
    valid = gi[gi < len(pos_per_row)]
    enrich = position_enrichment(sinks["pos_in_seq"], pos_per_row[valid], top_k, n)

    overlap = None
    if cross_model_csv and Path(cross_model_csv).exists():
        other = set(load_csv(cross_model_csv)["global_idx"].tolist())
        this = set(sinks["global_idx"].tolist())
        ov = this & other
        overlap = {"|this|": len(this), "|other|": len(other), "|ov|": len(ov),
                   "vs_random": len(ov) / max(1, len(this) * len(other) / meta["n_samples"])}

    # Print summary
    print(f"||mu|| = {np.linalg.norm(mu):.2f}")
    print(f"PR = {m['PR']:.2f}  top1/top2 = {m['top1/top2']:.2f}  stable_rank = {m['stable_rank']:.2f}")
    print(f"k_var = {m['k_var']}")
    print(f"v1: max = {m['v1_max']:.3f} ({m['v1_max_over_iso']:.1f}x iso), spiky = {m['spiky_channels']}")
    print(f"u1: max/mean = {m['u1_max_over_mean']:.0f}, 99.9/50 = {m['u1_999_over_50']:.1f}")
    for pct, d in drops.items():
        print(f"drop top {pct}%: PR = {d['PR']:.2f}, k99 = {d['k99']}")
    p = sinks["pos_in_seq"]
    print(f"sinks: prok={int((sinks['seq_idx']<1500).sum())} euk={int((sinks['seq_idx']>=1500).sum())}, "
          f"distinct seqs={len(set(sinks['seq_idx'].tolist()))}, pos==0={int((p==0).sum())}, "
          f"pos<10={int((p<10).sum())}, median pos={int(np.median(p))}")
    print(f"position enrichment first 10:")
    for e in enrich:
        print(f"  pos={e['pos']}: sinks={e['sinks']:>4} baseline={e['baseline']:>4} enrich={e['enrichment']:>6.1f}x")
    if overlap:
        print(f"cross-model overlap with {Path(cross_model_csv).name}: {overlap}")
    print(f"saved -> {csv_out}")

    return dict(meta=meta, metrics=m, drops=drops, sinks=sinks,
                enrichment=enrich, overlap=overlap, csv=csv_out)
