# SCALE Probe

Unsupervised **SCALE** (VAE + GMM-prior, Xiong et al. 2019 Nat Commun) trained
on train cells, then a **LinearSVC** probe evaluated via 5-fold cross-validation
on val embeddings.  Matches the protocol of the scVI / PeakVI / scFoundation /
scGPT probe baselines so results are directly comparable.

---

## Supported Datasets

Same registry as `/lichaohan/perturbfm/peakvi_prob/probe.py`:

| `--dataset_id`         | Cells | Classes | Peak set            |
|------------------------|-------|---------|---------------------|
| `5w_GSE196830_atac`    |  50 k |   29    | top-12k stratified  |
| `10w_GSE196830_atac`   | 100 k |   29    | noncoding33         |
| `20w_GSE196830_atac`   | 200 k |   29    | noncoding33         |
| `40w_GSE196830_atac`   | 400 k |   29    | noncoding33         |
| `80w_GSE196830_atac`   | 800 k |   29    | noncoding33         |
| `120w_GSE196830_atac`  | 1.2 M |   29    | noncoding33         |
| `GSE96583_atac`        |  41 k |    8    | noncoding33         |

Input parquets are integer peak counts; the probe **binarizes them once on
load** (SCALE's modeling assumption — `binary=True` in the VAE decoder Sigmoid).

---

## Quick Start

### Interactive (foreground)
```bash
bash run_probe.sh
```
Edit `run_probe.sh` to select the dataset (uncomment the block you want).

### Background (nohup)
```bash
nohup bash run_probe.sh > run_probe.log 2>&1 &
tail -f run_probe.log
```

### Manual CLI
```bash
python probe.py \
    --dataset_id GSE96583_atac \
    --run_name my_run \
    --wandb_project scale-probe
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_id` | `10w_GSE196830_atac` | Registry key (overrides --parquet/--site_tsv/--label_csv) |
| `--n_latent` | `20` | SCALE latent dim. Set to 20 to match PeakVI; SCALE paper used 10. |
| `--encode_dims` | `1024,128` | Encoder MLP hidden sizes (SCALE paper default) |
| `--decode_dims` | `` (empty) | Decoder MLP hidden sizes (SCALE paper default: linear decoder) |
| `--n_centroids` | (registry n_class) | GMM-prior component count |
| `--max_epochs` | `200` | Training-epoch budget (early stopping ends earlier) |
| `--batch_size_train` | `1024` | Mini-batch size |
| `--lr` | `2e-3` | Adam learning rate (SCALE upstream default) |
| `--weight_decay` | `5e-4` | Adam weight decay |
| `--beta` | `1.0` | DeterministicWarmup ceiling for KL term |
| `--warmup_n` | `200` | Iterations over which KL is annealed from 0 → beta |
| `--grad_clip` | `10.0` | Global grad-norm clip |
| `--early_stopping` / `--no_early_stopping` | on | Early-stop on quick-probe F1 (not ELBO) |
| `--early_stopping_patience` | `24` | Epochs of no F1 improvement before stop |
| `--cv_folds` | `5` | StratifiedKFold splits |
| `--max_samples` | `5000` | Per-fold cap on SVC training samples |
| `--pca_dim` | `None` | PCA before SVC; default off (latent already compact) |
| `--max_iter` | `2000` | LinearSVC max_iter |
| `--n_jobs` | `16` | Parallel cores for fold evaluation |
| `--save_embeddings` | off | Save val/train/all `.npy` embeddings (needed by `visualize.py`) |
| `--wandb_project` | `scale-probe` | wandb project name |
| `--no_wandb` | off | Disable wandb logging |

---

## Protocol

```
SCALE (unsupervised, no labels used during training)
  └─ trained on 80% train cells with the same seed=42 stratified split
       as every other probe baseline (scVI / PeakVI / scFoundation / scGPT)
  └─ after each epoch: encodeBatch(val) → quick LinearSVC F1 → save best ckpt
  └─ early stop on quick-probe F1 plateau (patience epochs)

Final evaluation on the val embeddings from BOTH:
  • best-downstream checkpoint (primary metric, "cv_test/*")
  • final  early-stopped     checkpoint ("final/cv_test/*")

5-fold StratifiedKFold (shuffle, seed=42)
  └─ StandardScaler → optional PCA → LinearSVC(dual=False, max_iter=2000)
     per-fold training samples capped at --max_samples (default 5000)
     folds parallelized via joblib (--n_jobs)
```

The 80/20 split, label encoding, fold splitter, and LinearSVC pipeline are
**bit-identical** across all probe baselines.  Differences vs PeakVI:

- SCALE binarizes its input by construction; PeakVI uses raw counts.
- SCALE has no Lightning callback hook, so the per-epoch quick-probe +
  checkpoint logic is implemented inline in `train_scale()` (see `probe.py`).
- **Loss normalization (`--loss_reduction per_element`, default ON)** divides
  the elbo by `batch * input_dim` rather than just `batch`.  This is required
  for our 240k-peak inputs — upstream SCALE expects ~10–30k peaks, and at our
  scale a `/len(x)` normalization leaves the per-cell loss at ~2×10⁶, which
  blows up Adam at the default lr.  Use `--loss_reduction per_cell` only when
  you have first reduced the input dimensionality.
- **`--init_gmm` is OFF by default.** Upstream SCALE seeds the GMM centroids
  by fitting `GaussianMixture` on the untrained encoder output.  On 240k-peak
  inputs the random encoder produces extreme outputs, which makes the GMM fit
  collapse `var_c` to near-zero and turns the KL term to `inf`.  The in-model
  defaults (`mu_c=0`, `var_c=1`) are stable.
- Non-finite training steps are skipped (logged as `[WARN] non-finite loss`)
  to harden against the residual numerical drift in the GMM-prior KL term
  when the encoder's `logvar` head produces large outputs early in training.
  Typically a small handful of batches are skipped in the first epoch.

---

## Output

`outputs_probe/<run_name>_<dataset_id>/`:

| File | Description |
|---|---|
| `probe_metrics.json` | All metrics, per-fold + per-class CV, args, training history |
| `probe_fold_metrics.csv` | Per-fold train/test accuracy & F1 |
| `class_names.json` | Ordered class-label strings |
| `scale_best_f1.pt` | Best-downstream-F1 SCALE state-dict |
| `scale_final.pt` | Final (early-stopped) SCALE state-dict |
| `embeddings_{val,train,all}.npy` | Latent embeddings (with `--save_embeddings`) |
| `labels_{val,train,all}.npy`     | Integer labels (with `--save_embeddings`) |

### wandb

**Scalar metrics** (`cv_train/*`, `cv_test/*`, `final/cv_test/*`):
`accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`.

**Per-epoch curves**:
`elbo_history` (recon_loss / kl_loss),
`epoch_f1_history` (quick_f1).

**Tables**:
`fold_metrics`, `per_class_metrics`.

---

## Visualization

`run_visualize.sh` (UMAP + t-SNE) is identical to the other prob/ visualizers
and points at `outputs_probe/` by default.  Requires `--save_embeddings` on the
probe run.

```bash
bash run_visualize.sh
# or filter by run name substring:
RUN_NAME_FILTER="20260522" bash run_visualize.sh
```

---

## Sweep

`run_sweep.sh` launches a wandb grid sweep over all dataset_ids, with all
non-dataset hparams fixed (see `sweep.yaml`).  Logs to project `scale-probe`.
