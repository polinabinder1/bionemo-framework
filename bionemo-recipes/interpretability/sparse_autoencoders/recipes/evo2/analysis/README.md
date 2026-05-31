# Evo2 pre-SAE activation analysis scripts

Self-contained diagnostic scripts written to characterize evo2 1B residual-stream
activations before training SAEs. All scripts are stand-alone — they read parquets
written by `scripts/extract.py`, do an analysis, and print/save results. None of
them write back to the parquets; none train SAEs.

These are the exploratory tools that produced the findings:

- evo2 1B residual stream has a near-rank-1 PCA spectrum at layers 12-19
- The dominance is a **sink-token / massive-activation** signature
  (a few hundred specific input tokens carry huge values on 3 specific channels
  `[56, 562, 1786]`)
- Underlying rank after sink removal is high (~1800 of 1920 dims for 99% variance)
- Layer 22 is in a totally different regime (no sink channels, diffuse top eigenvector)

## Path conventions

All scripts hardcode these paths from the original experiment runs:

```
parquets:  /data/interp/evo2/activations/<model>_layer<N>_parquet_<size>_<tag>/
FASTA:     /data/interp/evo2/scratch/mixed_25M_prokeuk_v2.fasta
```

Adapt these when re-running elsewhere. Each script's top-level constants
make the substitution straightforward.

## Script catalog

Listed roughly in the order they were developed/used. Each script is a single
file; run with `python <name>.py`.

### Dimensionality / rank diagnostics

- **`dim_analysis.py`** — single-shard SVD on one layer's parquet. Computes PCA
  spectrum, participation ratio, stable rank, cumulative variance thresholds.
  First-pass tool. Output prints + `.npz`.

- **`dim_compare.py`** — two-parquet comparison (evo2 vs codonfm). Loads 11
  shards from each, slices to 1M tokens, computes spectra side-by-side.
  Produced the headline "codonfm uses 82× more effective dims than evo2"
  result.

- **`uniform_random_pr.py`** — same metric as dim_compare but on a *uniform-random*
  1M sample from one parquet (vs contiguous shard slices). Sanity check that
  contiguous sampling wasn't biasing the result.

### Sink / massive-activation diagnostics

- **`sink_analysis.py`** — first full-SVD sink analysis (with U+V matrices).
  Reports `‖μ‖`, top1/top2 eigenvalue ratio, v1 spikiness, u1 sink-token
  signature, and PR after dropping the top X% of `|u1|`. Slow (~200s per SVD).
  Superseded by `sink_resample_fast.py`.

- **`sink_resample_fast.py`** — same diagnostics as `sink_analysis.py` but via
  covariance-eigh (X^T X, eigh on 1920×1920) instead of full SVD. ~5-10×
  faster. Also does N resamples with different seeds for robustness — reports
  per-metric mean ± std across seeds plus pairwise top-1% token overlap.

- **`dup_check.py`** — checks whether the rank measurement is inflated by
  duplicate or near-duplicate tokens. Exact-row duplicate count + stride-subsample
  PR + cross-shard PR with 1, 10, 100, 1000 tokens per shard.

- **`sink_identity.py`** — per-token sink identification. For each layer:
  1. Load 1M random tokens (seed=42 for cross-layer comparability)
  2. Compute centered SVD via covariance method → u1, v1
  3. Take top-1000 `|u1|` tokens
  4. Map each parquet row to `(sequence_id, position_in_sequence)` via
     FASTA-replay (assumes DP=4 round-robin)
  5. Report: position histogram, sequence diversity, spiky channels (v1
     indices), prok/euk split, cross-layer overlap, periodicity checks
  6. Emit per-layer CSV `(global_idx, seq_id, pos_in_seq, u1_abs)`

- **`position_enrichment.py`** — properly-baselined position enrichment.
  Variable-length sequence pool means uniform `[0, 8191]` isn't the right
  baseline. This script regenerates the same 1M-row sample, computes the
  empirical position-frequency baseline, then loads each layer's sink CSV
  and reports enrichment factor at position 0 and positions 0..9. Produces
  a histogram-with-baseline-overlay PNG per layer.

### Sanity checks

- **`sanity_checks.py`** — checks 2 and 3 from the original "are these results
  real" review:
  - Check 2: token-set overlap across layers (set intersection of top-1000)
  - Check 3: same parquet row across layers (cosine similarity of activation
    vectors)
  Confirms that overlapping top-1000 isn't a data-reading bug — the sets
  are 988-998/1000 (high but not exactly identical).

- **`sanity_check_nonsink.py`** — companion to check 3: pulls 5 random *non-sink*
  rows and compares activations across layers. Shows that non-sink rows have
  much lower cosine similarity (0.7-0.95) than sink rows (>0.9999), confirming
  the sink-row cosine-1.0 is a sink-channel-dominance artifact and not a
  data bug.

- **`sink_loss_checks.py`** — numerical verification of the SAE loss formula
  on the trained 500M layer-22 checkpoint. Confirms `fvu + var_explained = 1`
  exactly, inspects `pre_bias` drift, and traces the aux-loss residual
  formulation. Used to find the aux-loss bug fixed in the parent branch
  (`topk.py`: residual = x - recon, not x - recon + pre_bias).

### Helpers

- **`extract_layer.sh`** — generic single-extraction shell wrapper around
  `scripts/extract.py`. Takes `LAYER`, `FASTA`, `OUT_DIR`, `MICRO_BATCH` from
  env. Used to drive layer-specific extractions outside the main `1b.sh`
  pipeline.

- **`pr_test.py`** — minimal uniform-random PR test for one parquet (used as
  a chained step in some pipeline shell scripts during development).

## Typical workflow

1. Train an SAE recipe extraction → parquet at
   `evo2_1b_base_layer<N>_parquet_<scale>_<tag>/`.
2. Run `python uniform_random_pr.py` on it → check spectrum.
3. If PR looks suspiciously low: `python sink_resample_fast.py <parquet>` →
   confirm sink-token signature with 3 resamples.
4. If sinks confirmed: `python sink_identity.py` → identify which tokens.
5. `python position_enrichment.py` → check if sinks are positional.
6. `python sanity_checks.py` and `python sanity_check_nonsink.py` → confirm
   results aren't artifacts.

## Reproducing the key results

Run with the same seeds (`SEED=42` throughout) and the same 25M v2 prok+euk
FASTA, against the four layer-12/15/19/22 parquets, to reproduce:

- Same 3 spiky channels at L12/L15/L19: `[56, 562, 1786]`
- 988-998/1000 top-token overlap between any two of L12/L15/L19
- L22 has no spiky channels, diffuse top eigenvector
- pos-0 enrichment 34.48× at L12/L15/L19, 0× at L22
- Underlying rank after dropping top 5% sinks: ~1816-1833 dims (out of 1920)
  for L12/L15/L19; only ~8 dims for L22

## Caveats

- All scripts assume DP=4 round-robin record ordering for the FASTA-replay
  position mapping (`build_parquet_row_mapping` in `sink_identity.py`).
  This matches how `predict_evo2`'s distributed sampler arranged sequences
  for our extractions. Different DP size → adjust `DP_SIZE` constant.
- The v1/v2 FASTA composition matters: position-based stats use the
  knowledge that the v2 FASTA has 1500 prok records of 8192bp each + 2500
  euk records of ~5000bp each. Different mixes need recomposed baselines.
- `sink_loss_checks.py` was specific to debugging the aux-loss residual.
  It loads a particular trained SAE checkpoint and probably won't run
  elsewhere without path edits.
