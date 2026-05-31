"""Cross-layer sanity:
 (A) top-K |u1| token-set overlap across layers (uses saved sink CSVs)
 (B) same parquet row activation across layers — sink-row cos~1; non-sink rows
     show the normal layer-divergence pattern.
"""
import numpy as np
from itertools import combinations
from analysis_lib import load_csv, load_single_row

LAYERS = {
    "L12": "/data/interp/evo2/activations/evo2_1b_base_layer12_parquet_25M_prokeuk_v2",
    "L15": "/data/interp/evo2/activations/evo2_1b_base_layer15_parquet_25M_prokeuk_v2",
    "L19": "/data/interp/evo2/activations/evo2_1b_base_layer19_parquet_25M_prokeuk_v2",
}
TOTAL_ROWS = 24_637_383


print("=== (A) top-K token-set overlap across layers ===")
sinks = {n: set(load_csv(f"/tmp/sink_identity_{n}.csv")["global_idx"].tolist()) for n in LAYERS}
for a, b in combinations(sinks, 2):
    ov, sd = sinks[a] & sinks[b], sinks[a] ^ sinks[b]
    print(f"  {a} ∩ {b}: |∩|={len(ov)}, sym_diff={len(sd)}")
all_sinks = set().union(*sinks.values())
print(f"  union across layers: {len(all_sinks)}")


def row_stats(label, gi):
    vecs = {n: load_single_row(p, gi) for n, p in LAYERS.items()}
    print(f"\n  {label} (global_idx={gi}):")
    for n, v in vecs.items():
        print(f"    {n}: ||v||={np.linalg.norm(v):>9.2f}, mean={v.mean():+8.4f}")
    for a, b in combinations(vecs, 2):
        c = np.dot(vecs[a], vecs[b]) / (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b]) + 1e-30)
        print(f"    cos({a}, {b}) = {c:+.4f}")


print("\n=== (B) same-row activation across layers ===")
row_stats("sink row", 131074)
rng = np.random.default_rng(7)
for _ in range(3):
    while True:
        c = int(rng.integers(0, TOTAL_ROWS))
        if c not in all_sinks:
            break
    row_stats("non-sink row", c)
