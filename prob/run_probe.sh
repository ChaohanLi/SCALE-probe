#!/usr/bin/env bash
# =============================================================================
# SCALE Probe — run script
# Edit the dataset block below, then execute:
#   bash run_probe.sh
# or to run in background:
#   nohup bash run_probe.sh > run_probe.log 2>&1 &
# =============================================================================
set -euo pipefail

# ─── Dataset configuration ──────────────────────────────────────────────────
#  Pick one dataset block and comment out the rest.

# Dataset: 5w_GSE196830_atac (top-12k stratified, 29 classes)
DATASET_ID="5w_GSE196830_atac"

# Dataset: 10w_GSE196830_atac (noncoding33, 29 classes)
# DATASET_ID="10w_GSE196830_atac"

# Dataset: 20w_GSE196830_atac (noncoding33, 29 classes)
# DATASET_ID="20w_GSE196830_atac"

# Dataset: 40w_GSE196830_atac (noncoding33, 29 classes)
# DATASET_ID="40w_GSE196830_atac"

# Dataset: 80w_GSE196830_atac (noncoding33, 29 classes)
# DATASET_ID="80w_GSE196830_atac"

# Dataset: 120w_GSE196830_atac (noncoding33, 29 classes)
# DATASET_ID="120w_GSE196830_atac"

# Dataset: GSE96583_atac (noncoding33, 8 classes)
# DATASET_ID="GSE96583_atac"

# ─── Run configuration ──────────────────────────────────────────────────────
RUN_NAME="probe"
WANDB_PROJECT="scale-probe"
N_LATENT=20            # matches PeakVI for a fair LinearSVC comparison
ENCODE_DIMS="1024,128" # SCALE paper default
DECODE_DIMS=""         # SCALE paper default: linear decoder
N_CENTROIDS=""         # blank → defaults to dataset n_class from registry
MAX_EPOCHS=200         # epoch budget; early stopping on quick-probe F1 exits sooner
BATCH_SIZE_TRAIN=1024  # matches peakvi_prob; VAE/Adam is robust at this batch
LR=2e-3                # SCALE upstream default
WEIGHT_DECAY=5e-4
BETA=1.0
WARMUP_N=200
GRAD_CLIP=10.0
EARLY_STOPPING="--early_stopping"   # use "" to disable
EARLY_STOPPING_PATIENCE=24
N_JOBS=16
MAX_ITER=2000
SAVE_EMBEDDINGS="--save_embeddings"

PYTHON="/lichaohan/miniconda3/envs/scvi/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs_probe"

# ─── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

EXTRA_ARGS=()
if [[ -n "${N_CENTROIDS}" ]]; then
    EXTRA_ARGS+=(--n_centroids "${N_CENTROIDS}")
fi

$PYTHON probe.py \
    --dataset_id        "${DATASET_ID}" \
    --output_dir        "${OUTPUT_DIR}" \
    --run_name          "${RUN_NAME}" \
    --wandb_project     "${WANDB_PROJECT}" \
    --n_latent          "${N_LATENT}" \
    --encode_dims       "${ENCODE_DIMS}" \
    --decode_dims       "${DECODE_DIMS}" \
    --batch_size_train  "${BATCH_SIZE_TRAIN}" \
    --max_epochs        "${MAX_EPOCHS}" \
    --lr                "${LR}" \
    --weight_decay      "${WEIGHT_DECAY}" \
    --beta              "${BETA}" \
    --warmup_n          "${WARMUP_N}" \
    --grad_clip         "${GRAD_CLIP}" \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
    --n_jobs            "${N_JOBS}" \
    --max_iter          "${MAX_ITER}" \
    "${EXTRA_ARGS[@]}" \
    ${EARLY_STOPPING} \
    ${SAVE_EMBEDDINGS}
