#!/bin/bash
# Generic extractor — takes LAYER, FASTA, OUT_DIR, MICRO_BATCH from env.
set -euo pipefail
LAYER=${LAYER:?LAYER required}
FASTA=${FASTA:?FASTA required}
OUT_DIR=${OUT_DIR:?OUT_DIR required}
MICRO_BATCH=${MICRO_BATCH:-16}
EVO2_MEGATRON_DIR=/workspace/bionemo-framework/bionemo-recipes/recipes/evo2_megatron
RECIPE_DIR=/workspace/bionemo-framework/bionemo-recipes/interpretability/sparse_autoencoders/recipes/evo2
CKPT_DIR=/data/interp/evo2/checkpoints/evo2_1b_base_mbridge
MODEL=arcinstitute/savanna_evo2_1b_base

source "${EVO2_MEGATRON_DIR}/.venv/bin/activate"

echo "[$(date +%H:%M:%S)] extracting layer=${LAYER} micro_batch=${MICRO_BATCH}"
echo "  fasta:  $FASTA"
echo "  out:    $OUT_DIR"

if [[ -f "${OUT_DIR}/metadata.json" ]]; then
    echo "  output already exists, skipping"
    exit 0
fi

torchrun --nproc_per_node 4 --master-port "${MASTER_PORT:-29500}" "${RECIPE_DIR}/scripts/extract.py" \
    --activation-store-dir "$OUT_DIR" \
    --max-tokens 0 \
    --model-name "$MODEL" \
    --fasta "$FASTA" \
    --ckpt-dir "$CKPT_DIR" \
    --embedding-layer "$LAYER" \
    --micro-batch-size "$MICRO_BATCH"

echo "[$(date +%H:%M:%S)] layer-${LAYER} done -> $OUT_DIR"
cat "${OUT_DIR}/metadata.json"
