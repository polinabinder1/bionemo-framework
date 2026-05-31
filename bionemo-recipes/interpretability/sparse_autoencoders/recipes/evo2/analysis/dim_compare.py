"""Cross-parquet PCA spectrum comparison (evo2 vs codonfm, or any two parquets)."""
from analysis_lib import load_random_tokens, pca_top, metrics

PARQUETS = {
    "evo2 L22":    "/data/interp/evo2/activations/evo2_1b_base_layer22_parquet_25M_prokeuk",
    "codonfm L16": "/data/jwilber/codonfm/activations/Primates_encodon_1b_layer-2",
}
N = 1_000_000

for name, path in PARQUETS.items():
    print(f"\n=== {name} ===")
    X, _, _ = load_random_tokens(path, N)
    _, ev, v1, u1 = pca_top(X)
    m = metrics(ev, v1, u1, X.shape[1])
    print(f"  PR = {m['PR']:.2f}, top1/top2 = {m['top1/top2']:.2f}, stable_rank = {m['stable_rank']:.2f}")
    print(f"  k_var = {m['k_var']}")
    print(f"  spiky channels = {m['spiky_channels']}")
    del X
