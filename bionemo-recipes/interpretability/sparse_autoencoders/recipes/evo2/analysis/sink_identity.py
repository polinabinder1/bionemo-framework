"""Sink-token identity analysis for L12, L15, L19, L22 NEW.

For each layer:
  1. Load 1M random tokens (seed=42, same indices across layers for direct
     cross-layer comparison)
  2. Compute centered SVD via covariance method -> u1
  3. Top-1000 |u1| indices -> global parquet row indices
  4. Map (parquet_row -> sequence_id, position_in_sequence) via FASTA replay
     (assumes DP=4 round-robin, then rank-concatenated)
  5. Position histogram (does pos 0 dominate?), sequence diversity count
  6. Save (global_idx, seq_id, pos, u1_value, layer) as CSV

Then cross-layer:
  - Pairwise top-1000 token overlap (set intersection)
  - All-four intersection
  - Pairwise spiky-channel overlap (v1 indices > 5*1/sqrt(d))

Output: /tmp/sink_identity_<layer>.csv + /tmp/sink_identity_summary.md

NOTE: u1 isn't saved to disk from prior runs so we have to recompute. Each
SVD on 1M x 1920 via covariance method is ~5-10s; the bottleneck is the
parquet load (~700s per layer cold cache).
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pyarrow.parquet as pq

FASTA = "/data/interp/evo2/scratch/mixed_25M_prokeuk_v2.fasta"
DP_SIZE = 4
N = 1_000_000
SEED = 42
TOP_K = 1000
SPIKY_MULT = 5

LAYERS = {
    "L12": ("/data/interp/evo2/activations/evo2_1b_base_layer12_parquet_25M_prokeuk_v2", 1920),
    "L15": ("/data/interp/evo2/activations/evo2_1b_base_layer15_parquet_25M_prokeuk_v2", 1920),
    "L19": ("/data/interp/evo2/activations/evo2_1b_base_layer19_parquet_25M_prokeuk_v2", 1920),
    "L22": ("/data/interp/evo2/activations/evo2_1b_base_layer22_parquet_25M_prokeuk_v2", 1920),
}


def parse_fasta(path):
    """Return list of (header_id, sequence_string) in file order."""
    recs = []
    cur_id = None
    cur_seq = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_id is not None:
                    recs.append((cur_id, "".join(cur_seq)))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            recs.append((cur_id, "".join(cur_seq)))
    return recs


def build_parquet_row_mapping(records, dp_size):
    """Reconstruct parquet_row -> (fasta_record_index, pos_in_record).

    Predict_evo2 round-robins records across DP ranks: rank r processes
    fasta indices [r, r+dp_size, r+2*dp_size, ...]. Within each rank,
    tokens are concatenated in record order. The merged parquet
    concatenates rank-0 tokens, then rank-1, etc.
    """
    records_per_rank = [[] for _ in range(dp_size)]
    for fasta_idx, (_, seq) in enumerate(records):
        records_per_rank[fasta_idx % dp_size].append((fasta_idx, len(seq)))
    seq_chunks = []
    pos_chunks = []
    for rank in range(dp_size):
        for fasta_idx, slen in records_per_rank[rank]:
            seq_chunks.append(np.full(slen, fasta_idx, dtype=np.int32))
            pos_chunks.append(np.arange(slen, dtype=np.int32))
    return np.concatenate(seq_chunks), np.concatenate(pos_chunks)


def load_random(parquet_dir, hidden, n, seed):
    meta = json.load(open(Path(parquet_dir) / "metadata.json"))
    total = meta["n_samples"]
    shard_size = meta["shard_size"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=min(n, total), replace=False)
    idx.sort()
    sids = idx // shard_size
    rws = idx % shard_size
    rps = defaultdict(list)
    pos_in_sample = defaultdict(list)
    for k, (s, r) in enumerate(zip(sids, rws)):
        rps[int(s)].append(int(r))
        pos_in_sample[int(s)].append(k)
    X = np.empty((len(idx), hidden), dtype=np.float32)
    for s in sorted(rps):
        t = pq.read_table(Path(parquet_dir) / f"shard_{s:05d}.parquet")
        full = np.stack([t.column(f"dim_{j}").to_numpy() for j in range(hidden)], axis=1).astype(np.float32)
        rarr = np.array(rps[s])
        parr = np.array(pos_in_sample[s])
        m = rarr < full.shape[0]
        X[parr[m]] = full[rarr[m]]
        del full
    return X, idx


def eigen_top(Xc):
    """Centered X -> (eigvals desc, v1, u1) via cov-matrix eigh."""
    n = Xc.shape[0]
    C = (Xc.T @ Xc).astype(np.float64) / (n - 1)
    eigvals, V = np.linalg.eigh(C)
    eigvals = eigvals[::-1].clip(min=0.0)
    V = V[:, ::-1]
    v1 = V[:, 0].astype(np.float32)
    sigma1 = np.sqrt(eigvals[0] * (n - 1) + 1e-30)
    u1 = (Xc @ v1) / sigma1
    return eigvals, v1, u1


def main():
    print(f"parsing FASTA {FASTA} ...", flush=True)
    records = parse_fasta(FASTA)
    print(f"  {len(records)} records, total bp = {sum(len(s) for _, s in records):,}", flush=True)
    n_prok = sum(1 for h, _ in records[:1500])  # first 1500 are prok by convention from compose_prokeuk_fasta_v2
    n_euk = len(records) - n_prok
    print(f"  inferred prok (first 1500): {n_prok}, euk (rest): {n_euk}", flush=True)

    print(f"building parquet_row -> (seq_id, pos) mapping (DP={DP_SIZE}) ...", flush=True)
    seq_idx_per_row, pos_per_row = build_parquet_row_mapping(records, DP_SIZE)
    print(f"  total mapped rows: {len(seq_idx_per_row):,}", flush=True)

    per_layer_results = {}

    for name, (parquet, hidden) in LAYERS.items():
        print(f"\n{'='*70}\n{name}\n{'='*70}", flush=True)
        t0 = time.time()
        X, global_idx = load_random(parquet, hidden, N, SEED)
        print(f"  load time: {time.time()-t0:.1f}s, X.shape={X.shape}", flush=True)

        mu = X.mean(0)
        Xc = X - mu
        t0 = time.time()
        eigvals, v1, u1 = eigen_top(Xc)
        print(f"  eigh time: {time.time()-t0:.1f}s, ||mu||={np.linalg.norm(mu):.2f}", flush=True)

        u1_abs = np.abs(u1)
        v1_abs = np.abs(v1)
        iso = 1.0 / np.sqrt(hidden)

        # Top-K sink tokens
        top_local = np.argsort(u1_abs)[::-1][:TOP_K]
        top_global = global_idx[top_local]
        top_values = u1_abs[top_local]

        # Filter out global indices that fell off the FASTA replay (shouldn't normally happen)
        valid = top_global < len(seq_idx_per_row)
        top_global = top_global[valid]
        top_values = top_values[valid]
        top_seq = seq_idx_per_row[top_global]
        top_pos = pos_per_row[top_global]

        # spiky channels
        spiky_chans = np.where(v1_abs > SPIKY_MULT * iso)[0]

        # Position histogram bins
        pos_at_0 = int((top_pos == 0).sum())
        pos_0_3 = int((top_pos < 4).sum())
        pos_0_15 = int((top_pos < 16).sum())
        pos_0_127 = int((top_pos < 128).sum())
        pos_0_1023 = int((top_pos < 1024).sum())
        pos_gt_1024 = int((top_pos >= 1024).sum())

        # Sequence diversity
        seqs_in_top = set(top_seq.tolist())
        n_distinct_seqs = len(seqs_in_top)
        seq_counts = Counter(top_seq.tolist())
        most_common = seq_counts.most_common(5)

        # Domain split (prok vs euk)
        n_prok_sinks = int((top_seq < 1500).sum())
        n_euk_sinks = int((top_seq >= 1500).sum())

        print(f"  top-{TOP_K} |u1| range: [{top_values.min():.4f}, {top_values.max():.4f}]", flush=True)
        print(f"  spiky channels (>{SPIKY_MULT}/sqrt(d)): {spiky_chans.tolist()}", flush=True)
        print(f"\n  --- POSITION HISTOGRAM ---", flush=True)
        print(f"  position == 0          : {pos_at_0:>5} ({100*pos_at_0/len(top_pos):>5.1f}%)", flush=True)
        print(f"  position < 4           : {pos_0_3:>5} ({100*pos_0_3/len(top_pos):>5.1f}%)", flush=True)
        print(f"  position < 16          : {pos_0_15:>5} ({100*pos_0_15/len(top_pos):>5.1f}%)", flush=True)
        print(f"  position < 128         : {pos_0_127:>5} ({100*pos_0_127/len(top_pos):>5.1f}%)", flush=True)
        print(f"  position < 1024        : {pos_0_1023:>5} ({100*pos_0_1023/len(top_pos):>5.1f}%)", flush=True)
        print(f"  position >= 1024       : {pos_gt_1024:>5} ({100*pos_gt_1024/len(top_pos):>5.1f}%)", flush=True)
        print(f"  position min / med / max: {top_pos.min()} / {int(np.median(top_pos))} / {top_pos.max()}", flush=True)

        # Periodicity
        for k in (2, 3, 4, 8, 64, 1024, 8192):
            h = np.bincount(top_pos % k, minlength=k)
            max_v = h.max()
            expected = len(top_pos) / k
            if max_v > 1.5 * expected:
                top_bin = int(h.argmax())
                print(f"  periodicity mod {k}: bin {top_bin} = {max_v} ({max_v/expected:.2f}x expected)", flush=True)

        print(f"\n  --- SEQUENCE DIVERSITY ---", flush=True)
        print(f"  distinct sequences in top-{TOP_K}: {n_distinct_seqs}", flush=True)
        print(f"  top-5 most common (seq_id, count): {most_common}", flush=True)
        print(f"  prok sinks: {n_prok_sinks}, euk sinks: {n_euk_sinks}", flush=True)
        if n_distinct_seqs < 50:
            print(f"  --> few sequences are pathological", flush=True)
        elif n_distinct_seqs > 500:
            print(f"  --> general phenomenon across many sequences", flush=True)

        # Save per-layer CSV
        csv_path = f"/tmp/sink_identity_{name}.csv"
        with open(csv_path, "w") as f:
            f.write("global_idx,seq_id,pos_in_seq,u1_abs\n")
            for gi, si, ps, vv in zip(top_global, top_seq, top_pos, top_values):
                f.write(f"{gi},{si},{ps},{vv:.6f}\n")
        print(f"  saved -> {csv_path}", flush=True)

        # Verdict
        if pos_0_15 >= 0.9 * len(top_pos):
            verdict = "POSITIONAL (>90% in first 16 positions)"
        elif n_distinct_seqs < 50:
            verdict = "CONTENT-DEFINED (small set of source sequences)"
        elif len(spiky_chans) <= 5 and len(spiky_chans) > 0:
            verdict = "ARCHITECTURAL (few specific channels dominate)"
        else:
            verdict = "MIXED"

        per_layer_results[name] = {
            "top_global": set(top_global.tolist()),
            "spiky_chans": set(spiky_chans.tolist()),
            "n_distinct_seqs": n_distinct_seqs,
            "pos_at_0": pos_at_0,
            "pos_0_15": pos_0_15,
            "pos_gt_1024": pos_gt_1024,
            "verdict": verdict,
            "spiky_list": spiky_chans.tolist(),
            "most_common_seqs": most_common,
            "n_prok_sinks": n_prok_sinks,
            "n_euk_sinks": n_euk_sinks,
        }
        print(f"\n  VERDICT: {verdict}", flush=True)

    # ===== CROSS-LAYER =====
    print(f"\n\n{'='*70}\nCROSS-LAYER OVERLAP\n{'='*70}", flush=True)
    layer_names = list(LAYERS.keys())

    print(f"\n--- top-{TOP_K} |u1| TOKEN overlap (pairwise) ---", flush=True)
    for i, a in enumerate(layer_names):
        for b in layer_names[i+1:]:
            ov = per_layer_results[a]["top_global"] & per_layer_results[b]["top_global"]
            expected = TOP_K * TOP_K / N
            print(f"  {a} ∩ {b}: {len(ov):>4d}  (vs random ~{expected:.1f}, {len(ov)/max(1,expected):.1f}x)", flush=True)
    all4 = (per_layer_results[layer_names[0]]["top_global"]
            & per_layer_results[layer_names[1]]["top_global"]
            & per_layer_results[layer_names[2]]["top_global"]
            & per_layer_results[layer_names[3]]["top_global"])
    print(f"  ALL FOUR intersection: {len(all4)} tokens", flush=True)

    print(f"\n--- SPIKY CHANNEL overlap ---", flush=True)
    for i, a in enumerate(layer_names):
        for b in layer_names[i+1:]:
            ov = per_layer_results[a]["spiky_chans"] & per_layer_results[b]["spiky_chans"]
            print(f"  {a} {sorted(per_layer_results[a]['spiky_chans'])} ∩ {b} {sorted(per_layer_results[b]['spiky_chans'])}: {sorted(ov)}", flush=True)
    all4_chans = (per_layer_results[layer_names[0]]["spiky_chans"]
                  & per_layer_results[layer_names[1]]["spiky_chans"]
                  & per_layer_results[layer_names[2]]["spiky_chans"]
                  & per_layer_results[layer_names[3]]["spiky_chans"])
    print(f"  ALL FOUR spiky channels: {sorted(all4_chans)}", flush=True)

    # Save markdown summary
    md_path = "/tmp/sink_identity_summary.md"
    with open(md_path, "w") as f:
        f.write("# Sink Identity Summary — L12/L15/L19/L22 NEW (25M v2 prokeuk)\n\n")
        f.write(f"Top-{TOP_K} |u1| tokens analyzed per layer (seed={SEED}, n_sample={N}).\n\n")
        f.write("| layer | spiky channels | distinct seqs in top-1000 | pos==0 | pos<16 | pos>=1024 | prok/euk | verdict |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for name in layer_names:
            r = per_layer_results[name]
            f.write(f"| {name} | {sorted(r['spiky_chans'])} | {r['n_distinct_seqs']} | {r['pos_at_0']} | {r['pos_0_15']} | {r['pos_gt_1024']} | {r['n_prok_sinks']}/{r['n_euk_sinks']} | {r['verdict']} |\n")
        f.write("\n## Cross-layer token overlap (top-1000)\n\n")
        for i, a in enumerate(layer_names):
            for b in layer_names[i+1:]:
                ov = per_layer_results[a]["top_global"] & per_layer_results[b]["top_global"]
                expected = TOP_K * TOP_K / N
                f.write(f"- **{a} ∩ {b}**: {len(ov)} tokens (vs random ~{expected:.1f}, {len(ov)/max(1,expected):.1f}x)\n")
        f.write(f"- **All four**: {len(all4)} tokens\n\n")
        f.write("## Cross-layer spiky channel overlap\n\n")
        for name in layer_names:
            f.write(f"- {name}: {sorted(per_layer_results[name]['spiky_chans'])}\n")
        f.write(f"- All four: {sorted(all4_chans)}\n")
    print(f"\nmarkdown summary saved -> {md_path}", flush=True)


if __name__ == "__main__":
    main()
