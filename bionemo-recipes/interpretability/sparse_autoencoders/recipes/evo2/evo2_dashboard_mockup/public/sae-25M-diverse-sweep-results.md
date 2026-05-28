# Evo2 SAE — 25M diverse sweep (A/B/C) results

**Date**: 2026-05-27
**Pod**: `sae-polina-777f9787c4-x5pc8` (Lepton, 4× H100 80GB)
**Wandb project**: [`clara-discovery/evo2-sae-25M-diverse`](https://wandb.ai/clara-discovery/evo2-sae-25M-diverse)
**Goal**: Compare three TopK SAE auxk variants on a 25M-token diverse-euk subset, pick a winner before scaling to 100M / 500M.

---

## What it was run on

**Base model**: `arcinstitute/savanna_evo2_1b_base` (Evo2 1B, Hyena, 25 layers, hidden dim 1920)
**Layer**: 20 (out of 25 — proportional to 7B's sweet spot around layer 26–27)
**Context**: 8192 bp (model's trained context for the 1B)

**Source FASTA** (`/data/interp/evo2/scratch/mixed_25M.fasta`, 174 MB):

| Source | Records | Notes |
|---|---|---|
| OpenGenome2 `organelles` | 250 | mito/plastid (euk) |
| OpenGenome2 `promoters` | all (~173,000) | euk transcription start sites, individually short |
| OpenGenome2 `mrna_splice` | 2,000 | euk splice junctions |
| OpenGenome2 `eukaryotic_genic_windows` | 1,500 | 5kb euk genic regions |
| **Total source** | **176,991 sequences** | ~174 Mbp |

**All-eukaryotic.** No prokaryotic data — the "diverse" label was relative to the *original* organelle-only 25M run.

After chunking to 8192 bp: **184,929 chunks** → `predict_evo2` produced 46,233 `.pt` activation files (all 4 DP ranks × ~11,558 batches each).

`pt_to_parquet.py` then **shuffled the .pt list with seed=42** and capped writes at 25M tokens → final parquet with:

- `n_samples`: **25,005,916 tokens** (within 0.02% of cap)
- `n_sequences`: 26,085 (14% random sample of the 184,929 chunks)
- `hidden_dim`: 1920
- 252 parquet shards, 71 GB total

All three configs trained on the **same shuffled 25M-token parquet**.

---

## Configs trained

Common: TopK SAE, top_k=32, lr=3e-4, batch_size=4096, dp_size=4, init_pre_bias=True, n_epochs=5, dead_tokens_threshold=500,000 (codonfm-style strict).

| Config | expansion_factor | auxk | auxk_coef | What varies |
|---|---|---|---|---|
| **A** (baseline) | 8 | 512 | 0.03125 | codonfm defaults |
| **B** | 8 | **2048** | 0.03125 | 4× more revival latents per step |
| **C** | 8 | 512 | **0.125** | 4× stronger revival loss |
| **D** | **16** | 512 | 0.03125 | 2× larger dictionary (skipped — watchdog killed sweep after C) |

---

## Final results (step 7500, 5 epochs)

| Config | Wall time | fvu | dead_latents (%) | var_exp | mse | wandb run |
|---|---|---|---|---|---|---|
| A | 24m 25s | **0.012** | 70.3% | 0.988 | 0.012 | `config_A_baseline_auxk512_coef0.03125_exp8` |
| **B** | 23m 5s | 0.014 | **66.7%** ← best | 0.986 | 0.014 | `config_B_auxk2048_coef0.03125_exp8` |
| C | 23m 1s | 0.030 | 78.6% ← worst | 0.967 | 0.042 | `config_C_auxk512_coef0.125_exp8` |

Wandb URLs: prefix with `https://wandb.ai/clara-discovery/evo2-sae-25M-diverse/runs/`
- A: `nyucqd95`
- B: (see project view)
- C: `s7ccgrlx`

Checkpoints saved at `/data/interp/evo2/sae/evo2_1b_base_layer20_25M_diverse_{A,B,C}/checkpoints/checkpoint_final.pt`.

---

## Findings

1. **None of the three configs achieved codonfm-style low dead latents** (codonfm baseline is <5%). The all-eukaryotic input mix is the most likely culprit — Evo2 1B is OpenGenome2-trained, which is prok-dominated, so euk-only activations probably under-cover the feature dictionary.

2. **B (auxk=2048) is the marginal winner.** 4× more revival latents per step gave a ~3.6 pp absolute improvement over A (70.3% → 66.7% dead) with essentially identical fvu. Mild win; the auxk-count lever is not transformative on its own.

3. **C (auxk_coef=0.125) is a counter-result.** The early step-600 reading of 14.8% dead suggested it would win, but training destabilized — fvu rose to 0.030 (3× worse than A/B), gradient norm climbed (1.4 vs 0.2), and dead latents finished at 78.6%. The 4× higher revival loss weight overpowered reconstruction. **Lesson**: early dead-latent readings are misleading; always wait for convergence.

4. **D (capacity lever) was the right next thing to try** — expansion=16 doubles latent count, so each latent has more headroom before going dead. We deferred it to focus on the **prok+euk data axis** instead (the layer-20 organelle 25M run finished at 79% dead; adding diversity dropped that to 67% with config B, suggesting *data* is a bigger lever than *capacity* here).

5. **Pipeline lessons** (documented separately in commits + memory):
   - `predict_evo2` writes fp32 `.pt` even though the activation tensors could be bf16; halving on-disk volume would have saved ~30 min on this run.
   - `pt_to_parquet` was the wall-clock bottleneck (~25 min for the 25M cap, vs ~17 min for predict on the full 184M bp). Switched ThreadPool → ProcessPool mid-run for a ~2× speedup, but the real fix is **streaming directly to ActivationStore inside the inference loop** (codonfm pattern). That's the next-run upgrade.
   - The `predict_evo2` empty-batch crash on DP shard boundaries needed a 2-line patch ([commit `863122a7`](https://github.com/polinabinder1/bionemo-framework/commit/863122a7)).

---

## What's next

- **prok+euk 25M run** (launched same day): same hyperparams as winning config B, but FASTA is 40% prok (metagenomes) + 60% euk (eukaryotic_genic_windows). Wandb project: `evo2-sae-25M-prokeuk`. Uses the new streaming extractor (`scripts/extract.py`), skipping the `.pt` + pt_to_parquet detour. Hypothesis: prokaryotic activations will activate features that euk-only training leaves dead → much lower dead-latent fraction.
- **Then scale**: 100M and 500M token runs with winning hyperparams + prok+euk mix.
- **Long-term**: 7B model + 16384 context for Goodfire-comparable results — requires multi-node compute.
