"""Side-by-side dimensionality comparison: evo2 layer-22 vs codonfm layer-16.

For each parquet, samples 10 shards spread across the full range, stacks,
computes SVD on the centered data, and reports the spectrum + effective-dim
scalars side-by-side. Same relative depth (88%) on both models.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import time

PARQUETS = {
    "evo2_1b layer-22 (prok+euk, hidden=1920)": {
        "dir": "/data/interp/evo2/activations/evo2_1b_base_layer22_parquet_25M_prokeuk",
        "n_shards_total": 252,
        "hidden": 1920,
    },
    "codonfm_1b layer-16 (Primates, hidden=2048)": {
        "dir": "/data/jwilber/codonfm/activations/Primates_encodon_1b_layer-2",
        "n_shards_total": 9531,
        "hidden": 2048,
    },
}

N_SAMPLE_SHARDS = 11        # 11 × 100k = 1.1M, then slice down to N_TOKENS_MATCH
N_TOKENS_MATCH  = 1_000_000  # hard cap so both spectra see exactly the same token budget

def load_shards(parquet_dir: str, shard_ids: list[int], hidden: int) -> np.ndarray:
    """Load and stack a list of shards into a single (n_tokens, hidden) array."""
    chunks = []
    for sid in shard_ids:
        path = Path(parquet_dir) / f"shard_{sid:05d}.parquet"
        t = pq.read_table(path)
        chunk = np.stack([t.column(f"dim_{i}").to_numpy() for i in range(hidden)], axis=1)
        chunks.append(chunk.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def analyze(name: str, X: np.ndarray) -> dict:
    """Center, SVD, compute spectrum metrics."""
    print(f"\n=== {name} ===")
    print(f"  shape = {X.shape}, dtype = {X.dtype}")
    print(f"  per-token L2 norm: mean={np.linalg.norm(X, axis=1).mean():.2f}, std={np.linalg.norm(X, axis=1).std():.2f}")
    t0 = time.time()
    Xc = X - X.mean(axis=0, keepdims=True)
    S = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    lam = (S ** 2) / (Xc.shape[0] - 1)
    total = lam.sum()
    cum = np.cumsum(lam) / total
    PR = (lam.sum() ** 2) / (lam ** 2).sum()
    stable_rank = (S ** 2).sum() / (S[0] ** 2)
    print(f"  SVD compute time: {time.time()-t0:.1f}s")
    out = {
        "name": name,
        "ambient": X.shape[1],
        "n_tokens": X.shape[0],
        "lam": lam,
        "cum": cum,
        "PR": PR,
        "stable_rank": stable_rank,
        "top_eigvals": lam[:10],
    }
    return out


def report_table(results: list[dict]):
    print("\n" + "=" * 90)
    print(f"{'metric':<35s} | {'evo2 L22':>22s} | {'codonfm L16':>22s}")
    print("=" * 90)

    e, c = results[0], results[1]
    print(f"{'ambient dim':<35s} | {e['ambient']:>22d} | {c['ambient']:>22d}")
    print(f"{'tokens analyzed':<35s} | {e['n_tokens']:>22,d} | {c['n_tokens']:>22,d}")
    print(f"{'top-1 lambda':<35s} | {e['top_eigvals'][0]:>22.1f} | {c['top_eigvals'][0]:>22.1f}")
    print(f"{'top-1 / top-2 ratio':<35s} | {(e['top_eigvals'][0]/e['top_eigvals'][1]):>22.3f} | {(c['top_eigvals'][0]/c['top_eigvals'][1]):>22.3f}")
    print()

    print(f"{'# components for variance threshold':<35s}")
    for thr in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]:
        ke = int(np.searchsorted(e["cum"], thr) + 1)
        kc = int(np.searchsorted(c["cum"], thr) + 1)
        print(f"  {thr*100:>5.2f}% var{'':<22s} | k={ke:>4d}  ({100*ke/e['ambient']:>4.2f}%) | k={kc:>4d}  ({100*kc/c['ambient']:>4.2f}%)")
    print()

    print(f"{'Participation Ratio':<35s} | {e['PR']:>22.2f} | {c['PR']:>22.2f}")
    print(f"{'Stable rank':<35s} | {e['stable_rank']:>22.2f} | {c['stable_rank']:>22.2f}")
    print()

    print(f"--- relative compression ---")
    print(f"  evo2 effective dim    / ambient = {e['PR']:.1f} / {e['ambient']} = {100*e['PR']/e['ambient']:.3f}%")
    print(f"  codonfm effective dim / ambient = {c['PR']:.1f} / {c['ambient']} = {100*c['PR']/c['ambient']:.3f}%")
    print(f"  ratio: codonfm uses {c['PR']/e['PR']:.1f}x more effective dims than evo2")


def main():
    results = []
    for name, cfg in PARQUETS.items():
        # 10 shards spread evenly across the full parquet
        step = max(1, cfg["n_shards_total"] // N_SAMPLE_SHARDS)
        shard_ids = list(range(0, cfg["n_shards_total"], step))[:N_SAMPLE_SHARDS]
        print(f"\n[{name}] sampling shard IDs: {shard_ids}")
        t0 = time.time()
        X = load_shards(cfg["dir"], shard_ids, cfg["hidden"])
        print(f"  loaded {X.shape[0]:,} tokens in {time.time()-t0:.1f}s; slicing to {N_TOKENS_MATCH:,} for fair comparison")
        X = X[:N_TOKENS_MATCH]
        assert X.shape[0] == N_TOKENS_MATCH, f"only {X.shape[0]} tokens available — increase N_SAMPLE_SHARDS"
        results.append(analyze(name, X))
        del X  # free RAM before loading the next parquet

    report_table(results)

    # Save for plotting
    np.savez(
        "/tmp/dim_compare.npz",
        evo2_lam=results[0]["lam"],
        evo2_cum=results[0]["cum"],
        codonfm_lam=results[1]["lam"],
        codonfm_cum=results[1]["cum"],
    )
    print("\nSpectra saved to /tmp/dim_compare.npz")


if __name__ == "__main__":
    main()
