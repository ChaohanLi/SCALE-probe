"""
Cell-type evaluation for SCALE embeddings.

Protocol (identical to scVI / PeakVI / scFoundation / scGPT probe baselines):
  - Same stratified 80/20 train/val split (seed=42)
  - Train SCALE **unsupervised** on train cells only (no cell-type labels)
  - Extract latent z (encodeBatch, out='z') from val cells
  - 5-fold StratifiedKFold on val embeddings:
      each fold: StandardScaler -> optional PCA -> LinearSVC(dual=False)
  - Report mean CV train/test accuracy, macro-F1, balanced-accuracy

Input data:
  Same cell × peak parquet format used by peakvi_prob — parquet (cells × sites)
  + site_to_gene TSV + cell-type label CSV.  SCALE expects binarized peak
  accessibility, which we apply once on the loaded AnnData.

SCALE reference: Xiong et al. 2019 (Nature Communications).
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
import scanpy as sc
import torch
import wandb
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from scale.model import SCALE
from scale.dataset import SingleCellDataset
from scale.layer import DeterministicWarmup

_PROB_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Dataset registry — must mirror perturbfm/peakvi_prob exactly so the same
# dataset_id resolves to the same parquet across baselines.
# ---------------------------------------------------------------------------
DATASET_REGISTRY = {
    "5w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/5w_PBMC_GSE196830/counts_top12k_stratified_allchr.parquet",
        "site_tsv":  "/lichaohan/readData/site_to_gene_index_stratified_top12k_bl.tsv",
        "label_csv": "/lichaohan/readData/5w_PBMC_GSE196830/filtered_5w_all_cells.csv",
        "n_class":   29,
    },
    "10w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/10w_PBMC_GSE196830/stratified_noncoding33_counts.parquet",
        "site_tsv":  "/lichaohan/readData/10w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_10w_nochr5.tsv",
        "label_csv": "/lichaohan/readData/10w_PBMC_GSE196830/filtered_10w_all_celltype.csv",
        "n_class":   29,
    },
    "20w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/20w_PBMC_GSE196830/stratified_noncoding33_counts.parquet",
        "site_tsv":  "/lichaohan/readData/20w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_20w_nochr5_18.tsv",
        "label_csv": "/lichaohan/readData/20w_PBMC_GSE196830/filtered_20w_all_celltype.csv",
        "n_class":   29,
    },
    "40w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/40w_PBMC_GSE196830/stratified_noncoding33_40w_counts.parquet",
        "site_tsv":  "/lichaohan/readData/40w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_40w.tsv",
        "label_csv": "/lichaohan/readData/40w_PBMC_GSE196830/filtered_40w_all_celltype.csv",
        "n_class":   29,
    },
    "80w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/80w_PBMC_GSE196830/stratified_noncoding33_80w_counts.parquet",
        "site_tsv":  "/lichaohan/readData/80w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_80w.tsv",
        "label_csv": "/lichaohan/readData/80w_PBMC_GSE196830/filtered_80w_all_celltype.csv",
        "n_class":   29,
    },
    "120w_GSE196830_atac": {
        "parquet":   "/lichaohan/readData/120w_PBMC_GSE196830/stratified_noncoding33_120w_counts.parquet",
        "site_tsv":  "/lichaohan/readData/120w_PBMC_GSE196830/site_to_gene_index_stratified_noncoding33_120w.tsv",
        "label_csv": "/lichaohan/readData/120w_PBMC_GSE196830/filtered_120w_all_celltype.csv",
        "n_class":   29,
    },
    "GSE96583_atac": {
        "parquet":   "/lichaohan/readData/GSE96583_PBMC/stratified_noncoding33_counts.parquet",
        "site_tsv":  "/lichaohan/readData/GSE96583_PBMC/site_map.tsv",
        "label_csv": "/lichaohan/readData/GSE96583_PBMC/filtered_stratified_noncoding33_celltype.csv",
        "n_class":   8,
    },
}


# ---------------------------------------------------------------------------
# Build AnnData from parquet (cell × raw-peak counts) — copy of peakvi_prob
# ---------------------------------------------------------------------------
def parquet_to_split_adatas(
    parquet_path: str,
    site_tsv_path: str,
    label_csv_path: str,
    train_size: float = 0.8,
    random_state: int = 42,
    min_cells_per_peak: int = 0,
):
    """
    Memory-efficient replacement for parquet_to_adata() + train/val split.

    Two-pass strategy to avoid ever building the full combined CSR matrix:
      Pass 1 — barcode column only → compute train/val split (tiny memory).
      Pass 2 — full data → route each row to train or val on-the-fly,
               accumulating raw uint8 arrays (4× smaller than float32).
               At build time, concatenate and cast to float32 once per split.

    Peak memory per split (train ~80%, val ~20%):
      accumulation  : 0.8×NNZ×5 B  (uint8 data + int32 indices) for train
                    + 0.2×NNZ×5 B  for val  ≈ NNZ×5 B total
      build (train) : 0.8×NNZ×9 B  (uint8→float32 copy + indices in-place)
      Total peak    : ~NNZ×(5 + 0.8×4) ≈ NNZ×8.2 B
    For 120w (NNZ≈72B): ~590 GB — fits within 800 GB available.

    Also applies min_cells_per_peak filter using train column sums only
    (consistent with the split-before-filter protocol), then restricts val
    to the same peak set.

    Returns
    -------
    adata_train, adata_val : AnnData
    y_train, y_val         : int64 label arrays (aligned to adata row order)
    labels                 : full label array (parquet barcode order)
    train_idx, val_idx     : int indices into the original parquet barcode order
    classes                : sorted class name list
    type2idx               : class-name → int mapping
    """
    import pandas as pd
    import scipy.sparse as sp
    import anndata as ad
    import pyarrow.parquet as pq

    # ── Peak metadata ─────────────────────────────────────────────────
    print(f"  Loading site TSV: {site_tsv_path}", flush=True)
    site_df = pd.read_csv(site_tsv_path, sep="\t")
    n_sites = len(site_df)
    if "chrom" in site_df.columns and "col_idx_0based" in site_df.columns:
        peak_names = (
            site_df["chrom"].astype(str) + ":" + site_df["col_idx_0based"].astype(str)
        ).tolist()
    else:
        peak_names = [str(i) for i in range(n_sites)]
    data_cols = [str(i + 1) for i in range(n_sites)]
    print(f"  {n_sites} peaks", flush=True)

    # ── Labels dict ───────────────────────────────────────────────────
    print(f"  Loading cell-type labels: {label_csv_path}", flush=True)
    lbl_df = pd.read_csv(label_csv_path)
    ct_col = "cell_type" if "cell_type" in lbl_df.columns else lbl_df.columns[1]
    bc_to_ct = dict(zip(lbl_df["cell_barcode"].values, lbl_df[ct_col].values))
    del lbl_df

    # ── Pass 1: scan barcode column only to get ordering + split ─────
    print(f"  Pass 1: reading barcodes from parquet ...", flush=True)
    pf = pq.ParquetFile(parquet_path)
    bc_parts = []
    for batch in pf.iter_batches(batch_size=50_000, columns=["cell_barcode"]):
        bc_parts.append(batch.column("cell_barcode").to_pylist())
    all_barcodes = np.array([bc for part in bc_parts for bc in part]); del bc_parts
    print(f"  {len(all_barcodes):,} cells found in parquet", flush=True)

    cell_types_all = np.array([bc_to_ct[bc] for bc in all_barcodes])
    classes  = sorted(set(cell_types_all))
    type2idx = {c: i for i, c in enumerate(classes)}
    labels   = np.array([type2idx[ct] for ct in cell_types_all], dtype=np.int64)
    del cell_types_all

    train_bc_arr, val_bc_arr = _stratified_train_val_split(
        all_barcodes, labels, train_size=train_size, random_state=random_state,
    )
    train_bc_set = set(train_bc_arr.tolist())

    bc2pos    = {bc: i for i, bc in enumerate(all_barcodes)}
    train_idx = np.array([bc2pos[bc] for bc in train_bc_arr], dtype=np.int64)
    val_idx   = np.array([bc2pos[bc] for bc in val_bc_arr],   dtype=np.int64)
    y_train   = labels[train_idx]
    y_val     = labels[val_idx]
    del bc2pos, all_barcodes
    print(f"  Split: {len(train_idx):,} train / {len(val_idx):,} val", flush=True)

    # ── Pass 2: stream data, route to train / val ────────────────────
    print(f"  Pass 2: loading peak data (streaming, uint8) ...", flush=True)
    pf = pq.ParquetFile(parquet_path)
    missing = [c for c in data_cols[:5] if c not in set(pf.schema_arrow.names)]
    if missing:
        raise ValueError(f"Parquet missing expected columns {missing}.")

    # Accumulate raw arrays to avoid vstack peak
    tr_data, tr_indices, tr_nnz_cum = [], [], 0
    vl_data, vl_indices, vl_nnz_cum = [], [], 0
    tr_iptr_parts = [np.array([0], dtype=np.int64)]
    vl_iptr_parts = [np.array([0], dtype=np.int64)]
    tr_barcodes, vl_barcodes = [], []
    n_batches = 0

    for batch in pf.iter_batches(batch_size=2000, columns=["cell_barcode"] + data_cols):
        df_b  = batch.to_pandas()
        bcs   = df_b["cell_barcode"].values
        # Binarize immediately → uint8 (4× smaller than float32, lossless for ATAC)
        vals  = (df_b[data_cols].values > 0).astype(np.uint8)
        del df_b

        tr_mask = np.array([bc in train_bc_set for bc in bcs], dtype=bool)
        vl_mask = ~tr_mask

        for mask, d_list, i_list, iptr_parts, bc_list, nnz_ref in [
            (tr_mask, tr_data, tr_indices, tr_iptr_parts, tr_barcodes, [tr_nnz_cum]),
            (vl_mask, vl_data, vl_indices, vl_iptr_parts, vl_barcodes, [vl_nnz_cum]),
        ]:
            if not mask.any():
                continue
            sub = sp.csr_matrix(vals[mask])   # uint8 CSR
            bc_list.extend(bcs[mask].tolist())
            d_list.append(sub.data.copy())
            i_list.append(sub.indices.astype(np.int64))
            iptr_parts.append(sub.indptr[1:].astype(np.int64) + nnz_ref[0])
            nnz_ref[0] += sub.nnz
            del sub

        # Write back mutable cumulative NNZ
        tr_nnz_cum = tr_iptr_parts[-1][-1] if len(tr_iptr_parts) > 1 else 0
        vl_nnz_cum = vl_iptr_parts[-1][-1] if len(vl_iptr_parts) > 1 else 0

        del vals
        n_batches += 1
        if n_batches % 100 == 0:
            print(f"    … {n_batches * 2000:,} rows processed", flush=True)

    del train_bc_set

    # ── Build CSR matrices (uint8 → float32, sequential to limit peak) ─
    def _build_csr(d_parts, i_parts, iptr_parts, n_rows, n_cols, tag):
        print(f"  Building {tag} CSR ...", flush=True)
        data_u8  = np.concatenate(d_parts);  d_parts.clear()
        data_f32 = data_u8.astype(np.float32); del data_u8
        indices  = np.concatenate(i_parts).astype(np.int64);   i_parts.clear()
        indptr   = np.concatenate(iptr_parts).astype(np.int64)
        mat = sp.csr_matrix((data_f32, indices, indptr), shape=(n_rows, n_cols))
        print(f"    {tag}: {mat.shape}, nnz={mat.nnz:,}", flush=True)
        return mat

    X_tr = _build_csr(tr_data, tr_indices, tr_iptr_parts, len(tr_barcodes), n_sites, "train")
    X_vl = _build_csr(vl_data, vl_indices, vl_iptr_parts, len(vl_barcodes), n_sites, "val")

    # ── Optional peak filter (fit on train, apply to both) ───────────
    if min_cells_per_peak > 0:
        col_accessible = np.array((X_tr > 0).sum(axis=0)).ravel()
        keep = col_accessible >= min_cells_per_peak
        n_kept = int(keep.sum())
        if n_kept < n_sites:
            print(f"  filter_peaks(min_cells={min_cells_per_peak}): "
                  f"{n_sites} → {n_kept} peaks", flush=True)
            X_tr = X_tr[:, keep]
            X_vl = X_vl[:, keep]
            peak_names = [peak_names[i] for i in np.where(keep)[0]]

    var_df = pd.DataFrame(index=pd.Index(peak_names, name=""))

    adata_train = ad.AnnData(
        X=X_tr,
        obs=pd.DataFrame(
            {"cell_type": [classes[y] for y in y_train]},
            index=pd.Index(np.array(tr_barcodes), name=""),
        ),
        var=var_df,
    )
    adata_val = ad.AnnData(
        X=X_vl,
        obs=pd.DataFrame(
            {"cell_type": [classes[y] for y in y_val]},
            index=pd.Index(np.array(vl_barcodes), name=""),
        ),
        var=var_df,
    )
    print(f"  Done: train={adata_train.shape}, val={adata_val.shape}", flush=True)
    return adata_train, adata_val, y_train, y_val, labels, train_idx, val_idx, classes, type2idx


def parquet_to_adata(parquet_path: str, site_tsv_path: str, label_csv_path: str):
    import pandas as pd
    import scipy.sparse as sp
    import anndata as ad

    print(f"  Loading site TSV: {site_tsv_path}")
    site_df = pd.read_csv(site_tsv_path, sep="\t")
    n_sites = len(site_df)
    if "chrom" in site_df.columns and "col_idx_0based" in site_df.columns:
        peak_names = (
            site_df["chrom"].astype(str) + ":" + site_df["col_idx_0based"].astype(str)
        ).tolist()
    else:
        peak_names = [str(i) for i in range(n_sites)]
    print(f"  {n_sites} peaks")

    print(f"  Loading parquet: {parquet_path}")
    import pyarrow.parquet as pq
    data_cols = [str(i + 1) for i in range(n_sites)]
    cols_to_read = ["cell_barcode"] + data_cols

    pf = pq.ParquetFile(parquet_path)
    parquet_col_set = set(pf.schema_arrow.names)
    missing = [c for c in data_cols[:5] if c not in parquet_col_set]
    if missing:
        raise ValueError(
            f"Parquet does not contain expected site columns (e.g. {missing}). "
            f"Check that site_tsv and parquet correspond to the same dataset."
        )

    barcodes_list: list = []
    chunks: list = []
    n_batches = 0
    for batch in pf.iter_batches(batch_size=2000, columns=cols_to_read):
        df_b = batch.to_pandas()
        barcodes_list.append(df_b["cell_barcode"].values)
        chunks.append(sp.csr_matrix(df_b[data_cols].values, dtype=np.float32))
        del df_b
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"    … {n_batches * 2000:,} rows loaded", flush=True)

    barcodes = np.concatenate(barcodes_list)
    X = sp.vstack(chunks, format="csr")
    del chunks, barcodes_list
    print(f"  Parquet loaded: {X.shape[0]:,} cells × {X.shape[1]:,} peaks  "
          f"(nnz={X.nnz:,}, density={X.nnz/X.shape[0]/X.shape[1]:.4f})", flush=True)

    print(f"  Loading cell-type labels: {label_csv_path}")
    lbl_df = pd.read_csv(label_csv_path)
    ct_col = "cell_type" if "cell_type" in lbl_df.columns else lbl_df.columns[1]
    bc_to_ct = dict(zip(lbl_df["cell_barcode"].values, lbl_df[ct_col].values))
    cell_types = [bc_to_ct[bc] for bc in barcodes]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cell_type": cell_types},
            index=pd.Index(barcodes, name=""),
        ),
        var=pd.DataFrame(index=pd.Index(peak_names, name="")),
    )
    print(f"  AnnData built: {adata.shape}, "
          f"{adata.obs['cell_type'].nunique()} cell types", flush=True)
    return adata


# ---------------------------------------------------------------------------
# Stratified 80/20 split — bit-identical to peakvi_prob / scvi_prob
# ---------------------------------------------------------------------------
def _stratified_train_val_split(barcodes, labels, train_size=0.8, random_state=42):
    barcodes = np.asarray(barcodes)
    labels   = np.asarray(labels)
    rng = np.random.default_rng(int(random_state))
    train_parts, val_parts = [], []
    for lab in np.unique(labels):
        idx  = np.flatnonzero(labels == lab)
        perm = rng.permutation(idx)
        n_tr = int(np.floor(len(idx) * train_size))
        if len(idx) >= 2:
            n_tr = min(max(n_tr, 1), len(idx) - 1)
        else:
            n_tr = len(idx)
        train_parts.append(perm[:n_tr])
        val_parts.append(perm[n_tr:])
    train_idx = np.concatenate(train_parts)
    val_idx   = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return barcodes[train_idx], barcodes[val_idx]


# ---------------------------------------------------------------------------
# Binarize cell × peak counts in place — SCALE assumes binary accessibility
# ---------------------------------------------------------------------------
def _binarize_adata(adata):
    import scipy.sparse as sp
    if sp.issparse(adata.X):
        X = adata.X.copy()
        X.data = (X.data > 0).astype(np.float32)
        X.eliminate_zeros()
        adata.X = X
    else:
        adata.X = (adata.X > 0).astype(np.float32)
    return adata


def _ensure_batch_column(adata):
    """SCALE's SingleCellDataset hard-references obs['batch'].cat.codes; inject
    a single-batch categorical so the DataLoader doesn't crash."""
    import pandas as pd
    if "batch" not in adata.obs.columns:
        adata.obs["batch"] = pd.Categorical(["0"] * adata.shape[0])
    elif not isinstance(adata.obs["batch"].dtype, pd.CategoricalDtype):
        adata.obs["batch"] = pd.Categorical(adata.obs["batch"].astype(str))
    return adata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train SCALE and evaluate a LinearSVC probe on val embeddings"
    )
    p.add_argument("--dataset_id", type=str, default="10w_GSE196830_atac",
                   help="Registry key (e.g. 10w_GSE196830_atac, GSE96583_atac). "
                        "Overrides --parquet / --site_tsv / --label_csv.")
    p.add_argument("--parquet",   type=str, default=None)
    p.add_argument("--site_tsv",  type=str, default=None)
    p.add_argument("--label_csv", type=str, default=None)
    # Split / seed
    p.add_argument("--train_size", type=float, default=0.8)
    p.add_argument("--seed",       type=int,   default=42)
    # SCALE architecture
    # SCALE official preprocessing (mirrors scale/dataset.py: preprocessing_atac)
    # Note: filter_cells is intentionally omitted to preserve all cells for
    # downstream task alignment.
    p.add_argument("--min_cells_per_peak", type=int, default=3,
                   help="Filter peaks accessible in fewer cells (SCALE default: 3). "
                        "Set 0 to skip.")
    p.add_argument("--n_top_peaks", type=int, default=30000,
                   help="Keep top N highly variable peaks via seurat_v3 HVG selection. "
                        "SCALE default is 30000. Set -1 to keep all peaks.")
    p.add_argument("--n_latent", type=int, default=20,
                   help="SCALE latent dimension (default 20 to match PeakVI for "
                        "fair LinearSVC comparison; SCALE paper default is 10)")
    p.add_argument("--encode_dims", type=str, default="1024,128",
                   help="Comma-separated encoder hidden dims (default '1024,128' "
                        "matches SCALE paper)")
    p.add_argument("--decode_dims", type=str, default="",
                   help="Comma-separated decoder hidden dims (default '' — direct "
                        "linear decoder, matches SCALE paper)")
    p.add_argument("--n_centroids", type=int, default=None,
                   help="GMM-prior centroid count; defaults to dataset n_class")
    # SCALE training
    p.add_argument("--max_epochs", type=int, default=200,
                   help="Maximum training epochs. SCALE upstream uses max_iter, "
                        "but epoch budget is more interpretable across dataset sizes.")
    p.add_argument("--batch_size_train", type=int, default=1024,
                   help="Mini-batch size (default 1024 matches peakvi_prob)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="SCALE default lr (Adam, weight_decay=5e-4)")
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--beta",        type=float, default=1.0,
                   help="DeterministicWarmup ceiling for KL (default 1.0)")
    p.add_argument("--warmup_epochs", type=int, default=20,
                   help="Epochs over which KL beta is annealed. Converted to "
                        "iterations automatically (= warmup_epochs * iter_per_epoch), "
                        "so the schedule is dataset-size-independent.")
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--loss_reduction", type=str, default="per_element",
                   choices=["per_element", "per_cell"],
                   help="'per_element' divides loss by (batch * input_dim) — required "
                        "for high-dimensional ATAC peaks (≥100k) to keep Adam stable. "
                        "'per_cell' matches SCALE upstream (divide by batch only).")
    p.add_argument("--init_gmm", default=False,
                   action=argparse.BooleanOptionalAction,
                   help="Initialise GMM centroids by fitting GaussianMixture on the "
                        "untrained encoder output (SCALE upstream behaviour). On "
                        "high-dim inputs this collapses var_c → 0 and produces "
                        "KL=inf, so it is OFF by default for this probe. The "
                        "in-model defaults (mu_c=0, var_c=1) are used instead.")
    p.add_argument("--early_stopping",   default=True,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--early_stopping_patience", type=int, default=24,
                   help="Epochs with no quick-probe-F1 improvement before stop")
    # LinearSVC probe
    p.add_argument("--cv_folds",    type=int, default=5)
    p.add_argument("--max_samples", type=int, default=0,
                   help="If >0, subsample each 5-fold training partition to this "
                        "many cells before fitting LinearSVC. Default 0 = use all "
                        "val cells per fold (matches scVI/PeakVI probe protocol).")
    p.add_argument("--pca_dim",     type=int, default=None,
                   help="PCA before SVC. Default None (latent already compact).")
    p.add_argument("--max_iter",    type=int, default=2000)
    p.add_argument("--n_jobs",      type=int, default=16)
    # Output
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(_PROB_DIR, "outputs_probe"))
    p.add_argument("--run_name",   type=str, default=None)
    p.add_argument("--save_embeddings", action="store_true")
    # Weights & Biases
    p.add_argument("--wandb_project", type=str, default="scale-probe")
    p.add_argument("--no_wandb",      action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# LinearSVC probe (identical to other baselines)
# ---------------------------------------------------------------------------
def build_probe(train_embeddings, args):
    steps = [("scaler", StandardScaler())]
    if args.pca_dim is not None:
        pca_dim = min(
            int(args.pca_dim),
            train_embeddings.shape[0],
            train_embeddings.shape[1],
        )
        if pca_dim >= 1 and pca_dim < train_embeddings.shape[1]:
            steps.append(("pca", PCA(n_components=pca_dim, random_state=args.seed)))
    steps.append(("svc", LinearSVC(
        random_state=args.seed,
        dual=False,
        max_iter=args.max_iter,
    )))
    return Pipeline(steps)


def compute_metrics(labels, preds):
    return {
        "accuracy":          float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1":          float(f1_score(labels, preds, average="macro",
                                            zero_division=0)),
        "weighted_f1":       float(f1_score(labels, preds, average="weighted",
                                            zero_division=0)),
        "n_samples":         int(len(labels)),
    }


def run_svc_cv(embeddings, labels, args):
    unique, counts = np.unique(labels, return_counts=True)
    keep_classes   = unique[counts >= args.cv_folds]
    if len(keep_classes) < 2:
        raise ValueError(
            f"Need at least 2 classes with >= {args.cv_folds} samples; "
            f"got {len(keep_classes)}."
        )
    dropped_info = [
        {"class_id": int(c), "count": int(n)}
        for c, n in zip(unique, counts) if n < args.cv_folds
    ]
    if len(keep_classes) != len(unique):
        mask       = np.isin(labels, keep_classes)
        embeddings = embeddings[mask]
        labels     = labels[mask]
        labels     = np.searchsorted(keep_classes, labels)

    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                               random_state=args.seed)
    n_fold_jobs = min(args.cv_folds, args.n_jobs)
    n_jobs_ovr  = max(1, args.n_jobs // n_fold_jobs)
    print(f"Parallelism: {n_fold_jobs} fold workers × {n_jobs_ovr} OvR cores "
          f"= {n_fold_jobs * n_jobs_ovr} cores used (of {args.n_jobs})",
          flush=True)
    splits = list(splitter.split(embeddings, labels))

    def _run_fold(fold_idx, train_idx, test_idx):
        x_train = embeddings[train_idx]
        y_train = labels[train_idx]
        x_test  = embeddings[test_idx]
        y_test  = labels[test_idx]
        if args.max_samples and len(x_train) > args.max_samples:
            sampled_idx = np.random.choice(len(x_train), args.max_samples, replace=False)
            x_train_fit = x_train[sampled_idx]
            y_train_fit = y_train[sampled_idx]
        else:
            x_train_fit = x_train
            y_train_fit = y_train
        probe = build_probe(x_train_fit, args)
        print(f"  Fold {fold_idx}/{args.cv_folds}: fitting SVC on "
              f"{len(x_train_fit)} samples...", flush=True)
        probe.fit(x_train_fit, y_train_fit)
        train_preds = probe.predict(x_train)
        test_preds  = probe.predict(x_test)
        print(f"  Fold {fold_idx}/{args.cv_folds}: done.", flush=True)
        n_kept = len(keep_classes)
        return {
            "fold":           fold_idx,
            "train_size":     int(len(x_train)),
            "train_fit_size": int(len(x_train_fit)),
            "test_size":      int(len(x_test)),
            "train":          compute_metrics(y_train, train_preds),
            "test":           compute_metrics(y_test,  test_preds),
            "probe_steps":    list(probe.named_steps.keys()),
            "test_per_class": {
                "f1":        f1_score(y_test, test_preds, average=None,
                                      labels=np.arange(n_kept), zero_division=0).tolist(),
                "precision": precision_score(y_test, test_preds, average=None,
                                             labels=np.arange(n_kept), zero_division=0).tolist(),
                "recall":    recall_score(y_test, test_preds, average=None,
                                          labels=np.arange(n_kept), zero_division=0).tolist(),
                "support":   [int((y_test == c).sum()) for c in range(n_kept)],
            },
        }

    fold_metrics = Parallel(n_jobs=n_fold_jobs, backend="loky")(
        delayed(_run_fold)(fold_idx, train_idx, test_idx)
        for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1)
    )

    mean_metrics = {}
    for split in ["train", "test"]:
        mean_metrics[split] = {
            key: float(np.mean([f[split][key] for f in fold_metrics]))
            for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
        }

    n_kept = len(keep_classes)
    per_class_cv = []
    for c in range(n_kept):
        fold_f1s   = [fm["test_per_class"]["f1"][c]        for fm in fold_metrics]
        fold_precs = [fm["test_per_class"]["precision"][c] for fm in fold_metrics]
        fold_recs  = [fm["test_per_class"]["recall"][c]    for fm in fold_metrics]
        fold_sups  = [fm["test_per_class"]["support"][c]   for fm in fold_metrics]
        per_class_cv.append({
            "class_idx":      int(c),
            "mean_f1":        float(np.mean(fold_f1s)),
            "std_f1":         float(np.std(fold_f1s)),
            "mean_precision": float(np.mean(fold_precs)),
            "std_precision":  float(np.std(fold_precs)),
            "mean_recall":    float(np.mean(fold_recs)),
            "std_recall":     float(np.std(fold_recs)),
            "mean_support":   float(np.mean(fold_sups)),
        })

    return {
        "fold_metrics":           fold_metrics,
        "mean_metrics":           mean_metrics,
        "per_class_cv":           per_class_cv,
        "kept_classes":           [int(x) for x in keep_classes.tolist()],
        "dropped_classes":        dropped_info,
        "n_samples_after_filter": int(len(labels)),
    }


def _quick_probe_f1(z_val: np.ndarray, y_val: np.ndarray, args) -> float:
    """Stratified 70/30 single-split macro-F1 — used for per-epoch ckpt selection."""
    from sklearn.model_selection import StratifiedShuffleSplit
    unique, counts = np.unique(y_val, return_counts=True)
    keep = unique[counts >= 2]
    if len(keep) < 2:
        return 0.0
    if len(keep) < len(unique):
        mask = np.isin(y_val, keep)
        z_val = z_val[mask]
        y_val = y_val[mask]
        y_val = np.searchsorted(keep, y_val)
    if len(z_val) > 6000:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(z_val), 6000, replace=False)
        z_val = z_val[idx]
        y_val = y_val[idx]
        # Re-filter classes that lost samples during downsampling; otherwise
        # StratifiedShuffleSplit below can fail silently → quick_f1 = 0.
        unique2, counts2 = np.unique(y_val, return_counts=True)
        keep2 = unique2[counts2 >= 2]
        if len(keep2) < 2:
            return 0.0
        if len(keep2) < len(unique2):
            mask = np.isin(y_val, keep2)
            z_val = z_val[mask]
            y_val = y_val[mask]
            y_val = np.searchsorted(keep2, y_val)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(z_val, y_val))
    except ValueError:
        return 0.0
    z_tr, z_te = z_val[tr_idx], z_val[te_idx]
    y_tr, y_te = y_val[tr_idx], y_val[te_idx]
    if args.max_samples and len(z_tr) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(z_tr), args.max_samples, replace=False)
        z_tr = z_tr[idx]
        y_tr = y_tr[idx]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc",    LinearSVC(dual=False, max_iter=args.max_iter,
                             random_state=args.seed)),
    ])
    try:
        pipe.fit(z_tr, y_tr)
        # sklearn signature: f1_score(y_true, y_pred, ...). Reversing args
        # gives a systematically biased macro-F1 when class supports differ.
        return float(f1_score(y_te, pipe.predict(z_te),
                              average="macro", zero_division=0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Custom SCALE training loop — mirrors scale.model.VAE.fit() but adds:
#   * per-epoch quick-probe F1 logging
#   * best-checkpoint saving by downstream F1
#   * early stopping on quick-probe F1 (not ELBO)
# ---------------------------------------------------------------------------
def train_scale(
    model:        SCALE,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    y_val:        np.ndarray,
    args,
    device:       torch.device,
    best_ckpt_path: str,
    final_ckpt_path: str,
    use_wandb:    bool,
    input_dim:    int,
):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    n_iter_per_epoch_warmup = max(1, len(train_loader))
    warmup_n = args.warmup_epochs * n_iter_per_epoch_warmup
    print(f"  KL warmup: {args.warmup_epochs} epochs × {n_iter_per_epoch_warmup} iter = {warmup_n} iterations",
          flush=True)
    beta_schedule = DeterministicWarmup(n=warmup_n, t_max=args.beta)
    loss_denom_extra = float(input_dim) if args.loss_reduction == "per_element" else 1.0

    best_f1     = -1.0
    best_epoch  = -1
    bad_epochs  = 0
    elbo_history: list = []   # (epoch, recon_loss, kl_loss)
    f1_history:   list = []   # (epoch, quick_f1)

    n_iter_per_epoch = max(1, len(train_loader))
    print(f"  Iterations per epoch: {n_iter_per_epoch}")

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_recon, epoch_kl, n_cells_seen = 0.0, 0.0, 0
        for x in train_loader:
            x = x.float().to(device)
            optimizer.zero_grad()
            recon_loss, kl_loss = model.loss_function(x)
            beta_t = next(beta_schedule)
            denom = len(x) * loss_denom_extra
            loss = (recon_loss + beta_t * kl_loss) / denom
            if not torch.isfinite(loss):
                print(f"    [WARN] non-finite loss "
                      f"(recon={float(recon_loss):.4g}, kl={float(kl_loss):.4g}); "
                      f"skipping step.", flush=True)
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_recon += float(recon_loss.detach().cpu())
            epoch_kl    += float(kl_loss.detach().cpu())
            n_cells_seen += len(x)

        if n_cells_seen == 0:
            print(f"  [ERROR] Epoch {epoch}: all {n_iter_per_epoch} batches had "
                  f"non-finite loss; model was NOT updated this epoch. "
                  f"Likely cause: encoder logvar overflowing exp() — check "
                  f"GaussianSample clamp range in scale/layer.py.", flush=True)
        mean_recon = epoch_recon / max(1, n_cells_seen)
        mean_kl    = epoch_kl    / max(1, n_cells_seen)
        elbo_history.append((epoch, mean_recon, mean_kl))

        # ── per-epoch quick probe ──
        model.eval()
        with torch.no_grad():
            z_val = model.encodeBatch(val_loader, device=device, out='z')
        if np.isnan(z_val).any():
            n_nan = int(np.isnan(z_val).sum())
            print(f"    [WARN] z_val has {n_nan} NaN entries this epoch; "
                  f"sanitizing for quick-probe.", flush=True)
            z_val = np.nan_to_num(z_val, nan=0.0, posinf=0.0, neginf=0.0)
        f1 = _quick_probe_f1(z_val, y_val, args)
        f1_history.append((epoch, f1))

        if use_wandb:
            wandb.log({
                "epoch":                epoch,
                "train/recon_loss":     mean_recon,
                "train/kl_loss":        mean_kl,
                "train/total_loss":     mean_recon + mean_kl,
                "epoch_probe/quick_f1": f1,
            })
        print(
            f"  [Epoch {epoch:3d}/{args.max_epochs}] "
            f"recon={mean_recon:.4f}  kl={mean_kl:.4f}  quick_f1={f1:.4f}",
            flush=True,
        )

        if f1 > best_f1:
            best_f1    = f1
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    → New best downstream F1={f1:.4f} at epoch {epoch} "
                  f"— checkpoint saved.", flush=True)
        else:
            bad_epochs += 1
            if args.early_stopping and bad_epochs >= args.early_stopping_patience:
                print(f"  Early stopping: no quick-F1 improvement for "
                      f"{bad_epochs} epochs (best F1={best_f1:.4f} "
                      f"at epoch {best_epoch}).")
                break

    torch.save(model.state_dict(), final_ckpt_path)
    trained_epochs = epoch
    print(f"  Final model saved → {final_ckpt_path}")
    print(f"  Best downstream F1={best_f1:.4f} at epoch {best_epoch} "
          f"→ {best_ckpt_path}")

    return {
        "best_f1":         best_f1,
        "best_epoch":      best_epoch,
        "trained_epochs":  trained_epochs,
        "elbo_history":    elbo_history,
        "f1_history":      f1_history,
    }


def save_fold_metrics(path, cv_result):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "train_size", "train_fit_size", "test_size",
            "train_accuracy", "test_accuracy",
            "train_macro_f1", "test_macro_f1",
        ])
        for fold in cv_result["fold_metrics"]:
            writer.writerow([
                fold["fold"], fold["train_size"], fold["train_fit_size"],
                fold["test_size"],
                fold["train"]["accuracy"], fold["test"]["accuracy"],
                fold["train"]["macro_f1"], fold["test"]["macro_f1"],
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.dataset_id in DATASET_REGISTRY:
        cfg = DATASET_REGISTRY[args.dataset_id]
        args.parquet   = cfg["parquet"]
        args.site_tsv  = cfg["site_tsv"]
        args.label_csv = cfg["label_csv"]
        if args.n_centroids is None:
            args.n_centroids = cfg["n_class"]

    if args.parquet is None or args.site_tsv is None or args.label_csv is None:
        raise ValueError("Provide --dataset_id or all of --parquet/--site_tsv/--label_csv.")
    if args.n_centroids is None:
        raise ValueError("--n_centroids required when --dataset_id is not from registry.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or time.strftime("probe_%Y%m%d_%H%M%S")
    run_name = f"{run_name}_{args.dataset_id}"
    out_dir  = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device:           {device}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)

    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # ── Load data + split (streaming, memory-efficient) ──────────────
    # parquet_to_split_adatas never builds the full combined matrix:
    # it streams the parquet once and routes rows directly to train/val,
    # using uint8 accumulation. Peak memory is ~NNZ×8 B instead of ~NNZ×16 B.
    # Binarization and min_cells_per_peak filtering are handled inside.
    print(f"\nLoading ATAC data (streaming split) ...", flush=True)
    (adata_train, adata_val,
     y_train, y_val,
     labels, train_idx, val_idx,
     classes, type2idx) = parquet_to_split_adatas(
        parquet_path   = args.parquet,
        site_tsv_path  = args.site_tsv,
        label_csv_path = args.label_csv,
        train_size     = args.train_size,
        random_state   = args.seed,
        min_cells_per_peak = args.min_cells_per_peak,
    )
    n_class = len(classes)
    print(f"  Classes: {n_class}  "
          f"train={adata_train.shape}  val={adata_val.shape}", flush=True)

    # Data is already binarized (uint8→float32) inside parquet_to_split_adatas;
    # call _binarize_adata as a no-op safety pass (X.data is already 0/1 float32).
    _binarize_adata(adata_train)
    _binarize_adata(adata_val)

    # HVG selection (n_top_peaks=-1 means disabled for 80w/120w runs).
    # If enabled, fit on train and restrict val to the same peak set.
    if args.n_top_peaks != -1:
        n_top = min(args.n_top_peaks, adata_train.n_vars)
        if n_top < adata_train.n_vars:
            sc.pp.highly_variable_genes(
                adata_train, n_top_genes=n_top,
                flavor="seurat_v3", subset=False,
            )
            hvg_mask = adata_train.var["highly_variable"].values
            adata_train = adata_train[:, hvg_mask].copy()
            adata_val   = adata_val[:, hvg_mask].copy()
            print(f"  After HVP selection: train={adata_train.shape} "
                  f"val={adata_val.shape}", flush=True)
        else:
            print(f"  HVP selection skipped "
                  f"(n_vars={adata_train.n_vars} <= n_top_peaks={n_top})", flush=True)
    print(f"  Final shape: train={adata_train.shape}  val={adata_val.shape}", flush=True)

    # SCALE's SingleCellDataset reads obs['batch'].cat.codes — set it.
    _ensure_batch_column(adata_train)
    _ensure_batch_column(adata_val)

    # Pre-densify X: eliminates per-sample CSR→dense conversion in __getitem__.
    # Skip if the combined dense footprint exceeds _DENSIFY_LIMIT_GB to avoid OOM.
    _DENSIFY_LIMIT_GB = 200
    import scipy.sparse as sp
    _n_train, _n_feat = adata_train.shape
    _n_val            = adata_val.shape[0]
    _dense_gb = (_n_train + _n_val) * _n_feat * 4 / 1e9
    if sp.issparse(adata_train.X) and _dense_gb <= _DENSIFY_LIMIT_GB:
        print(f"Pre-densifying train X (sparse → dense float32, "
              f"~{_dense_gb:.0f} GB total)...", flush=True)
        adata_train.X = adata_train.X.toarray()
        adata_val.X   = adata_val.X.toarray()
    elif sp.issparse(adata_train.X):
        print(f"Skipping pre-densify: dense footprint ~{_dense_gb:.0f} GB "
              f"> {_DENSIFY_LIMIT_GB} GB limit; keeping sparse "
              f"(num_workers will parallelise toarray()).", flush=True)

    train_ds = SingleCellDataset(adata_train)
    val_ds   = SingleCellDataset(adata_val)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size_train, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False,
    )

    # ── Build SCALE model ─────────────────────────────────────────────
    encode_dims = [int(d) for d in args.encode_dims.split(",") if d.strip()]
    decode_dims = [int(d) for d in args.decode_dims.split(",") if d.strip()]
    input_dim   = adata_train.shape[1]
    dims = [input_dim, args.n_latent, encode_dims, decode_dims]
    model = SCALE(dims=dims, n_centroids=args.n_centroids)
    print(f"\nSCALE model: input_dim={input_dim}, n_latent={args.n_latent}, "
          f"encode_dims={encode_dims}, decode_dims={decode_dims}, "
          f"n_centroids={args.n_centroids}", flush=True)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    # ── GMM init for centroids (SCALE upstream protocol) ──────────────
    model.to(device)
    if args.init_gmm:
        print("Initialising GMM centroids from val embeddings (untrained encoder)...",
              flush=True)
        model.init_gmm_params(val_loader, device=device)
    else:
        print("Skipping init_gmm_params (using built-in defaults mu_c=0, var_c=1)",
              flush=True)

    # ── Train ─────────────────────────────────────────────────────────
    best_ckpt_path  = os.path.join(out_dir, "scale_best_f1.pt")
    final_ckpt_path = os.path.join(out_dir, "scale_final.pt")
    print(f"\nTraining SCALE (unsupervised) — max_epochs={args.max_epochs}, "
          f"batch_size={args.batch_size_train}, lr={args.lr}, "
          f"loss_reduction={args.loss_reduction}, init_gmm={args.init_gmm}, "
          f"early_stopping={args.early_stopping} (patience="
          f"{args.early_stopping_patience}) ...", flush=True)
    train_info = train_scale(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        y_val           = y_val,
        args            = args,
        device          = device,
        best_ckpt_path  = best_ckpt_path,
        final_ckpt_path = final_ckpt_path,
        use_wandb       = not args.no_wandb,
        input_dim       = input_dim,
    )

    # ── Extract embeddings from best checkpoint ────────────────────────
    print(f"\nLoading best-downstream checkpoint (epoch {train_info['best_epoch']})...")
    best_model = SCALE(dims=dims, n_centroids=args.n_centroids).to(device)
    best_model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    best_model.eval()

    print("Extracting validation embeddings (best checkpoint)...", flush=True)
    with torch.no_grad():
        z_val = best_model.encodeBatch(val_loader, device=device, out='z')
    print(f"  {z_val.shape[1]} dims  |  {len(y_val)} val samples", flush=True)
    n_bad_val = int((~np.isfinite(z_val)).sum())
    if n_bad_val > 0:
        print(f"  [WARN] z_val contains {n_bad_val} non-finite entries "
              f"({n_bad_val / z_val.size:.2%}); replacing with 0.", flush=True)
    z_val = np.nan_to_num(z_val, nan=0.0, posinf=0.0, neginf=0.0)
    print("Extracting training embeddings (best checkpoint)...")
    with torch.no_grad():
        z_train = best_model.encodeBatch(train_loader, device=device, out='z')
    print(f"  {len(y_train)} train samples")

    # Note: train_loader was shuffle=True, so its iteration order does NOT match
    # adata_train order. Rebuild an in-order loader for z_train to align with y_train.
    train_loader_ordered = DataLoader(
        train_ds, batch_size=args.batch_size_train, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False,
    )
    with torch.no_grad():
        z_train = best_model.encodeBatch(train_loader_ordered, device=device, out='z')
    n_bad_tr = int((~np.isfinite(z_train)).sum())
    if n_bad_tr > 0:
        print(f"  [WARN] z_train contains {n_bad_tr} non-finite entries; "
              f"replacing with 0.", flush=True)
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0)

    _z_concat   = np.vstack([z_train, z_val])
    _y_concat   = np.concatenate([y_train, y_val])
    _orig_order = np.argsort(np.concatenate([train_idx, val_idx]))
    z_all = _z_concat[_orig_order]
    y_all = _y_concat[_orig_order]
    del _z_concat, _y_concat, _orig_order

    # ── Full 5-fold SVC on best checkpoint (primary) ─────────────────
    print("\nRunning full 5-fold SVC probe on best-downstream checkpoint...")
    cv_result = run_svc_cv(z_val, y_val, args)

    # ── Full 5-fold SVC on final checkpoint (comparison) ─────────────
    print("\nRunning full 5-fold SVC probe on final checkpoint...")
    final_model = SCALE(dims=dims, n_centroids=args.n_centroids).to(device)
    final_model.load_state_dict(torch.load(final_ckpt_path, map_location=device))
    final_model.eval()
    with torch.no_grad():
        z_val_final = final_model.encodeBatch(val_loader, device=device, out='z')
    z_val_final = np.nan_to_num(z_val_final, nan=0.0, posinf=0.0, neginf=0.0)
    cv_result_final = run_svc_cv(z_val_final, y_val, args)

    # ── Decide primary checkpoint ─────────────────────────────────────
    # If quick_f1 never improved past its initial 0 during training (e.g. due to
    # rare-class StratifiedShuffleSplit failures), best_ckpt is whatever was saved
    # at epoch 1 (degenerate). In that case, treat final_ckpt as the primary
    # result and record the fallback so downstream analysis can detect it.
    if train_info["best_f1"] > 0:
        primary_cv      = cv_result
        secondary_cv    = cv_result_final
        primary_ckpt    = "best_downstream"
    else:
        primary_cv      = cv_result_final
        secondary_cv    = cv_result
        primary_ckpt    = "final_fallback"
        print(f"\n[INFO] best_f1={train_info['best_f1']:.4f} ≤ 0 across all epochs; "
              f"using FINAL checkpoint as primary result "
              f"(primary_checkpoint='final_fallback').", flush=True)

    # ── Save results ───────────────────────────────────────────────────
    result = {
        "metrics":                primary_cv["mean_metrics"],
        "fold_metrics":           primary_cv["fold_metrics"],
        "per_class_cv":           primary_cv["per_class_cv"],
        "embedding_dim":          int(z_val.shape[1]),
        "class_names":            classes,
        "type2idx":               type2idx,
        "kept_classes":           primary_cv["kept_classes"],
        "dropped_classes":        primary_cv["dropped_classes"],
        "n_samples_after_filter": primary_cv["n_samples_after_filter"],
        "final_metrics":          secondary_cv["mean_metrics"],
        "primary_checkpoint":     primary_ckpt,
        "best_downstream_epoch":  train_info["best_epoch"],
        "best_downstream_f1":     train_info["best_f1"],
        "scale_trained_epochs":   train_info["trained_epochs"],
        "elbo_history":           train_info["elbo_history"],
        "f1_history":             train_info["f1_history"],
        "args":                   vars(args),
        "protocol":               "scale_train_val_embeddings_5fold_svc_cv_best_ckpt",
    }
    with open(os.path.join(out_dir, "probe_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "class_names.json"), "w") as f:
        json.dump(classes, f, indent=2)
    save_fold_metrics(os.path.join(out_dir, "probe_fold_metrics.csv"), primary_cv)

    if args.save_embeddings:
        np.save(os.path.join(out_dir, "embeddings_val.npy"),   z_val)
        np.save(os.path.join(out_dir, "labels_val.npy"),       y_val)
        np.save(os.path.join(out_dir, "embeddings_train.npy"), z_train)
        np.save(os.path.join(out_dir, "labels_train.npy"),     y_train)
        np.save(os.path.join(out_dir, "embeddings_all.npy"),   z_all)
        np.save(os.path.join(out_dir, "labels_all.npy"),       y_all)

    # ── Print summary ──────────────────────────────────────────────────
    best_tag  = "  ← PRIMARY" if primary_ckpt == "best_downstream" else ""
    final_tag = "  ← PRIMARY (fallback)" if primary_ckpt == "final_fallback" else ""
    print(f"\n── Best-downstream checkpoint ──{best_tag}")
    for split in ["train", "test"]:
        m = cv_result["mean_metrics"][split]
        print(f"cv {split:>5}: acc={m['accuracy']:.4f} "
              f"bal_acc={m['balanced_accuracy']:.4f} "
              f"macro_f1={m['macro_f1']:.4f} "
              f"weighted_f1={m['weighted_f1']:.4f}")
    print(f"  (epoch {train_info['best_epoch']} of "
          f"{train_info['trained_epochs']} total)")
    print(f"\n── Final (early-stopped) checkpoint ──{final_tag}")
    for split in ["train", "test"]:
        m = cv_result_final["mean_metrics"][split]
        print(f"cv {split:>5}: acc={m['accuracy']:.4f} "
              f"bal_acc={m['balanced_accuracy']:.4f} "
              f"macro_f1={m['macro_f1']:.4f} "
              f"weighted_f1={m['weighted_f1']:.4f}")
    if primary_cv["dropped_classes"]:
        print(f"Dropped classes with < {args.cv_folds} samples: "
              f"{primary_cv['dropped_classes']}")
    print(f"\nSaved to: {out_dir}")

    # ── Weights & Biases logging ───────────────────────────────────────
    if not args.no_wandb:
        # cv_test/* = PRIMARY (used by sweep metric); secondary/* = the other ckpt
        mean     = primary_cv["mean_metrics"]
        mean_sec = secondary_cv["mean_metrics"]
        wandb.log({
            "cv_train/accuracy":          mean["train"]["accuracy"],
            "cv_train/balanced_accuracy": mean["train"]["balanced_accuracy"],
            "cv_train/macro_f1":          mean["train"]["macro_f1"],
            "cv_train/weighted_f1":       mean["train"]["weighted_f1"],
            "cv_test/accuracy":           mean["test"]["accuracy"],
            "cv_test/balanced_accuracy":  mean["test"]["balanced_accuracy"],
            "cv_test/macro_f1":           mean["test"]["macro_f1"],
            "cv_test/weighted_f1":        mean["test"]["weighted_f1"],
            "secondary/cv_test/accuracy":           mean_sec["test"]["accuracy"],
            "secondary/cv_test/balanced_accuracy":  mean_sec["test"]["balanced_accuracy"],
            "secondary/cv_test/macro_f1":           mean_sec["test"]["macro_f1"],
            "secondary/cv_test/weighted_f1":        mean_sec["test"]["weighted_f1"],
            "primary_checkpoint":         primary_ckpt,
            "embedding_dim":              int(z_val.shape[1]),
            "n_val_samples":              int(len(y_val)),
            "n_classes_used":             len(primary_cv["kept_classes"]),
            "n_classes_dropped":          len(primary_cv["dropped_classes"]),
            "best_downstream_epoch":      train_info["best_epoch"],
            "best_downstream_f1":         train_info["best_f1"],
            "scale_trained_epochs":       train_info["trained_epochs"],
        })
        fold_table = wandb.Table(
            columns=["fold", "train_size", "test_size",
                     "train_acc", "test_acc", "train_macro_f1", "test_macro_f1"]
        )
        for fold in primary_cv["fold_metrics"]:
            fold_table.add_data(
                fold["fold"], fold["train_size"], fold["test_size"],
                fold["train"]["accuracy"], fold["test"]["accuracy"],
                fold["train"]["macro_f1"], fold["test"]["macro_f1"],
            )
        wandb.log({"fold_metrics": fold_table})

        per_class_table = wandb.Table(
            columns=["class_name", "mean_f1", "std_f1",
                     "mean_recall", "mean_precision", "mean_support"]
        )
        kept = primary_cv["kept_classes"]
        for entry in primary_cv["per_class_cv"]:
            orig = kept[entry["class_idx"]]
            name = classes[orig] if orig < len(classes) else str(orig)
            per_class_table.add_data(
                name,
                round(entry["mean_f1"],        4),
                round(entry["std_f1"],         4),
                round(entry["mean_recall"],    4),
                round(entry["mean_precision"], 4),
                round(entry["mean_support"],   1),
            )
        wandb.log({"per_class_metrics": per_class_table})

        elbo_table = wandb.Table(columns=["epoch", "recon_loss", "kl_loss"])
        for ep, rl, kl in train_info["elbo_history"]:
            elbo_table.add_data(ep, round(rl, 4), round(kl, 4))
        wandb.log({"elbo_history": elbo_table})

        f1_table = wandb.Table(columns=["epoch", "quick_f1"])
        for ep, f1 in train_info["f1_history"]:
            f1_table.add_data(ep, round(f1, 4))
        wandb.log({"epoch_f1_history": f1_table})

        wandb.finish()


if __name__ == "__main__":
    main()
