#!/usr/bin/env bash
# =============================================================================
# PeakVI Probe — visualize embeddings
# Requires probe to have been run with SAVE_EMBEDDINGS="--save_embeddings".
# Usage:
#   bash run_visualize.sh
# or in background:
#   nohup bash run_visualize.sh > run_visualize.log 2>&1 &
# =============================================================================
set -euo pipefail

PYTHON="/lichaohan/miniconda3/envs/scvi/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Configuration ──────────────────────────────────────────────────────────
# Point RUN_DIR to a single run directory (e.g. outputs_probe/probe_10w_GSE196830_atac)
# or to the outputs_probe/ root to visualize all runs at once.
RUN_DIR="${SCRIPT_DIR}/outputs_probe"

METHOD="both"      # umap | tsne | both
MAX_CELLS=20000    # subsample to this many cells before UMAP/t-SNE
SEED=42
N_JOBS=4           # runs processed in parallel

# Comma-separated list of splits to visualize: val, train, all, joint
SPLITS="joint"

# Filter run directories by name substrings (comma-separated, ALL must match).
# Leave empty ("") to process every run under RUN_DIR.
# Example: RUN_NAME_FILTER="20260519"
RUN_NAME_FILTER=""

# ─── Run ────────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}"

FILTER_ARG=()
if [[ -n "${RUN_NAME_FILTER}" ]]; then
    FILTER_ARG=(--run_name_filter "${RUN_NAME_FILTER}")
fi

$PYTHON visualize.py \
    --run_dir   "${RUN_DIR}" \
    --method    "${METHOD}" \
    --max_cells "${MAX_CELLS}" \
    --seed      "${SEED}" \
    --n_jobs    "${N_JOBS}" \
    --splits    "${SPLITS}" \
    "${FILTER_ARG[@]}"
