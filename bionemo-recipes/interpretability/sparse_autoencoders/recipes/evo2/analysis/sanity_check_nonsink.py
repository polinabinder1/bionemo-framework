"""Check 3 robustness: pull NON-sink rows across layers, expect lower cosine."""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

LAYERS = {
    "L12": "/data/interp/evo2/activations/evo2_1b_base_layer12_parquet_25M_prokeuk_v2",
    "L15": "/data/interp/evo2/activations/evo2_1b_base_layer15_parquet_25M_prokeuk_v2",
    "L19": "/data/interp/evo2/activations/evo2_1b_base_layer19_parquet_25M_prokeuk_v2",
}

# Build set of ALL sink global_idx across all 3 layers
all_sinks = set()
for name in LAYERS:
    with open(f"/tmp/sink_identity_{name}.csv") as f:
        next(f)
        for line in f:
            all_sinks.add(int(line.split(",")[0]))
print(f"total distinct sink global_idx across L12/L15/L19: {len(all_sinks)}")

# Pick 5 random NON-sink rows
rng = np.random.default_rng(7)
total_rows = 24_637_383
shard_size = 100_000
test_idxs = []
while len(test_idxs) < 5:
    cand = int(rng.integers(0, total_rows))
    if cand not in all_sinks:
        test_idxs.append(cand)

print(f"\nselected non-sink rows: {test_idxs}\n")

def get_row(parquet_dir, global_idx):
    sh = global_idx // shard_size
    r = global_idx % shard_size
    t = pq.read_table(Path(parquet_dir) / f"shard_{sh:05d}.parquet")
    n_rows = t.num_rows
    if r >= n_rows:
        return None
    return np.array([t.column(f"dim_{i}")[r].as_py() for i in range(1920)], dtype=np.float32)

for test_idx in test_idxs:
    vecs = {}
    print(f"=== row {test_idx} (non-sink) ===")
    for name, parquet in LAYERS.items():
        v = get_row(parquet, test_idx)
        if v is None:
            print(f"  {name}: ROW NOT IN SHARD (last shard short)"); break
        vecs[name] = v
        print(f"  {name}: norm = {np.linalg.norm(v):>8.4f}, mean = {v.mean():+7.4f}, std = {v.std():+7.4f}, ch56={v[56]:+7.3f} ch562={v[562]:+7.3f} ch1786={v[1786]:+7.3f}")
    if len(vecs) < 3:
        continue
    print("  cosine + L2 diff:")
    for a in ["L12", "L15", "L19"]:
        for b in ["L12", "L15", "L19"]:
            if a < b:
                c = np.dot(vecs[a], vecs[b]) / (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b]) + 1e-30)
                d = np.linalg.norm(vecs[a] - vecs[b])
                print(f"    {a} vs {b}: cos = {c:+.4f},  ||diff|| = {d:.4f}")
    print()
