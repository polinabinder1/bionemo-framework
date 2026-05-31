# Evo2 pre-SAE activation analysis

Self-contained diagnostics that read a parquet of activations (from
`scripts/extract.py`) and report:

- PCA spectrum (PR, top1/top2, k_var, stable rank)
- Sink-channel + sink-token signatures (`v1`, `u1`)
- PR after dropping the top X% of `|u1|` tokens (sink-removal robustness)
- Top-K sink token identity: which (sequence, position) carries the dominant mode
- Position-0 / first-N-positions enrichment vs the variable-length pool baseline
- Cross-layer / cross-model token-set overlap

None of these scripts train SAEs or modify parquets.

## Quick start

For any (model, layer) extraction:

```bash
python run_analysis.py L26_7B \
    /data/interp/evo2/activations/evo2_7b_layer26_parquet_25M_prokeuk_v2 \
    --fasta /data/interp/evo2/scratch/mixed_25M_prokeuk_v2.fasta \
    --cross-model /tmp/sink_identity_L19.csv
```

One load, full analysis, prints all sections to stdout, saves a sink CSV.
Works for any hidden_dim (auto-detected from the parquet's metadata.json).

## Layout

```
analysis_lib.py       single shared module — load, FASTA replay, eigh, sink metrics,
                      identity, enrichment, cross-model overlap. 400 LOC.
run_analysis.py       thin driver wrapping analysis_lib.analyze_layer.  40 LOC.

sanity_checks.py             top-K set overlap + same-row across layers
sanity_check_nonsink.py      companion: non-sink rows confirm cross-layer divergence
dup_check.py                 within/across-shard duplicate sanity (one-off)
dim_compare.py               two-parquet cross-model comparison (evo2 vs codonfm)
dim_analysis.py              single-shard PCA (one-off; subsumed by run_analysis)
extract_layer.sh             generic extract.py wrapper (LAYER/FASTA/OUT_DIR env)

archive/                     earlier one-off prototypes preserved for reference
                             (pr_test, uniform_random_pr, sink_analysis,
                              sink_resample_fast, sink_identity, position_enrichment)
                             — superseded by analysis_lib but kept for repro
```

## What `run_analysis.py` does, in order

For one parquet:

1. Loads N (default 1M) uniform-random rows from the parquet's shards.
2. Centers by per-channel mean, runs covariance-eigh (`X^T X`, then `np.linalg.eigh`).
3. Reports: PR, top1/top2 ratio, stable rank, cumulative-variance k for
   50/80/90/95/99/99.9%.
4. Reports v1 (channel direction) spikiness: max |v1| vs the isotropic floor
   1/√d; lists spiky channel indices.
5. Reports u1 (per-token loading on top mode) sink-shape: max/mean, 99.9th/50th
   percentile ratio.
6. Re-runs centered eigh after dropping the top 1% and top 5% of |u1| tokens,
   reports PR and 99% var k — tells you the "true rank" once sinks are removed.
7. Identifies the top-K |u1| tokens (default 1000), maps each to
   (fasta_record, position_in_record) via the DP=4 round-robin reconstruction,
   saves the CSV.
8. Position histogram diagnostics: pos==0 count, position <10 count,
   median position, mod-k periodicity check.
9. Position enrichment factor at positions 0-9 against the empirical sample
   baseline (correctly handles variable-length records).
10. Cross-model token-set overlap if you pass `--cross-model <other.csv>`.

## Re-using analysis_lib programmatically

```python
from analysis_lib import analyze_layer

r = analyze_layer(
    parquet_dir="/data/interp/evo2/activations/evo2_7b_layer26_parquet_25M_prokeuk_v2",
    label="L26_7B",
    fasta="/data/interp/evo2/scratch/mixed_25M_prokeuk_v2.fasta",
    cross_model_csv="/tmp/sink_identity_L19.csv",
)

# r is a dict with the spectrum, sink, drops, top_sinks, enrichment,
# cross_overlap, and meta fields. Use them however.
```

Lower-level building blocks if you want to deviate from the full pipeline:

```python
from analysis_lib import (
    load_random_tokens,                # parquet -> (X, global_idx, meta)
    parse_fasta_lengths,               # FASTA -> [int]
    build_row_to_position_mapping,     # lengths, dp_size -> (seq_per_row, pos_per_row)
    centered_eigen_top,                # X -> (mu, eigvals, v1, u1)
    spectrum_metrics,                  # eigvals -> {PR, top1/2, stable_rank, k_var}
    sink_diagnostics,                  # (v1, u1, hidden) -> {v1/u1 stats, spiky chans}
    drop_top_sinks_metrics,            # (X, u1, pct) -> {PR, k99}
    identify_top_sinks,                # (u1, idx, seq_per_row, pos_per_row, k) -> TopSinks
    save_sink_csv / load_sink_csv,
    position_enrichment,
)
```

## Path conventions

```
parquets    /data/interp/evo2/activations/<model>_layer<N>_parquet_<scale>_<tag>/
FASTA       /data/interp/evo2/scratch/mixed_<scale>_prokeuk_v2.fasta
sink CSVs   /tmp/sink_identity_<label>.csv
```

Adapt as needed.

## Findings to reproduce

With seed=42 on the 25M v2 prok+euk parquets (1B model):

- Same 3 spiky v1 channels at L12/L15/L19: `[56, 562, 1786]`
- Top-1000 token overlap 988-998/1000 between any two of L12/L15/L19
- L22 has no spiky channels, diffuse top eigenvector, true low-rank
- After dropping top 5% sinks at L12/L15/L19: 99% var k ≈ 1816-1833
  (out of 1920) — i.e. nearly full rank
- L22 is genuinely 99% var in k=8 even with sinks removed

## Caveats

- The DP=4 round-robin assumption matches predict_evo2's distributed sampler
  for our extractions. If you change `--nproc_per_node`, update `dp_size`.
- The FASTA replay assumes record order in the parquet matches the order
  the FASTA was processed in. Verified empirically for our pipeline.
- Float32 throughout; covariance eigh has plenty of precision for top-N
  eigenvalues separated by orders of magnitude.
