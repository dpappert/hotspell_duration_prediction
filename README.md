# [Paper title]

> Code accompanying "[paper title]" ([link when available]).

Five model families (Random Forest, GAM, Logistic Regression, tabular MLP,
CNN) predicting [target] from [predictors], evaluated across three regions.
Four models use tabular predictors; the CNN uses gridded 2D input.

## Setup

```bash
git clone <this repo>
cd <repo>
conda env create -f environment.yml
conda activate hwml
pip install -e .
```

Then populate `data/` with the project's data -- see `data/README.md` for
exactly what goes where, and run:

```bash
python scripts/check_data.py
```

to confirm it's laid out correctly before running anything else.

`data/cnn_grid/` (the CNN model's gridded input, ~6GB) is not shipped in
the repo -- see `data/cnn_grid/README.md`. Every other model runs fine
without it.

## Reproducing results

Each model has a **validate** entry point (hyperparameter sweep, 7-fold
member-based CV) and an **eval** entry point (retrain on all non-test data
with the best/fixed hyperparameters, report held-out test performance).

```bash
# Random Forest
python scripts/validate_rf.py --region w --n-estimators 100 --max-depth 5 \
    --min-samples-split 10 --min-samples-leaf 5 --max-features log2 --criterion gini
python scripts/eval_rf.py --region w

# GAM
python scripts/validate_gam.py --region w --lam 1 --n-splines 10
python scripts/eval_gam.py --region w

# Logistic Regression (fixed baseline -- no validation sweep needed for eval)
python scripts/eval_lr.py --region w

# Tabular MLP (4 architectures: a, b, c, lin)
python scripts/validate_mlp.py --arch a --region sw
python scripts/eval_mlp.py --arch a --region sw

# CNN (7 architectures: a, b, c, d, lin, lr, mlp)
python scripts/validate_cnn.py --arch a --region w
python scripts/eval_cnn.py --arch a --region w
```

To run a full hyperparameter grid rather than one combo at a time, use the
matching driver in `scripts/sweeps/` (e.g. `scripts/sweeps/sweep_rf.sh`
runs every RF combo across all 3 regions). `scripts/sweeps/sweep_eval.sh`
runs every eval script across all archs/regions once validation results
exist.

Region codes: `sw`, `w`, `n` (matching the original filename suffixes).

## Layout

```
src/hwml/          shared library: data loading, splits, metrics, losses,
                   model architectures, hyperparameter-selection logic
scripts/           one validate_*.py / eval_*.py per model family, plus
                   check_data.py and scripts/sweeps/ (hyperparameter grid drivers)
config/            .env.example documenting path overrides (see below)
data/              targets/, tab_preds/ (shipped); cnn_grid/ (not shipped)
results/           script outputs land here by default (gitignored)
tests/             pytest suite for the pure-Python shared logic
legacy/            original one-off scripts, kept until Step 8 verification
                   (diffing new-script output against these) is complete;
                   will be removed afterward
```

### Path configuration

Nothing needs configuring for a fresh clone -- `data/` and `results/`
default to the repo-relative folders above. To point at data living
elsewhere (e.g. a cluster scratch directory), set environment variables
before running anything; see `config/.env.example` for the full list:

```bash
export HWML_DATA_DIR=/scratch3/.../CESM2/ML
```

## Testing

```bash
pytest tests/
```

Covers the pure-Python shared logic (metrics, splits, hyperparameter-tag
parsing) that doesn't require the real dataset or torch. This is not a
substitute for diffing real script output against `legacy/` (in progress,
see below).

## Status: verification in progress

This repo was reorganized from a flat collection of ~40 near-duplicate
scripts (see `legacy/`) into the structure above. The refactor is
believed behavior-preserving except for the fixes listed below, but has
not yet been fully verified against the original scripts' outputs on real
data (Step 8 of the reorganization) -- `legacy/` stays in the repo until
that's done.

### Confirmed fixes vs. the original scripts

These change computed results, deliberately, with confirmation at the
time each was found:

- **RF hyperparameter selection bug** (`eval_RF.py`): the `criterion`
  field was parsed from result filenames using a hardcoded character
  offset that silently mis-parsed 96 of 192 possible hyperparameter
  combos (e.g. `'gini'` truncated to `'ini'`), which would have crashed
  `RandomForestClassifier`. Fixed with proper parsing in
  `hwml.model_selection.parse_rf_tag`; regression-tested against the full
  grid in `tests/`.
- **CNN/MLP eval reproducibility bug** (`eval_all2D.py`, `eval_MLP.py`):
  the validation fold used for early-stopping during final retraining was
  selected via an unseeded shuffle, making it different (and the reported
  test metrics with it) on every run. Fixed to use the same seeded split
  logic as every validate script.
- **LR normalization** (`lr_trial7_valid.py`): originally used a different
  feature-column offset, a different (pre-computed, file-based) split
  scheme, and a different target-folder date suffix than every other
  model. All three confirmed as unintentional and normalized to match
  RF/GAM/MLP/CNN. LR's validation metric aggregation also changed from a
  per-fold breakdown to a pooled statistic, matching every other model.
- **LR eval** (`eval_LR.py`): confirmed as the study's fixed baseline (not
  hyperparameter-tuned), so `eval_lr.py` has no validation-dependent
  hyperparameter selection, unlike every other model. Its class-weighting
  scheme was changed from unweighted to RF's manual `sample_weight`
  formula (confirmed); note this now differs in scale from
  `validate_lr.py`'s `class_weight="balanced"`, since validate's role for
  LR is exploratory only.
- **NumPy 2.0 compatibility**: `np.trapz` (used in the AUPRGC metric) was
  removed in NumPy >=2.0. `hwml.metrics` falls back to `np.trapezoid`
  automatically -- same computation, just avoids a hard crash on a modern
  install.

### Known non-bugs, preserved as-is (flagging for awareness)

- `eval_GAM.py` used `max_iter=2000` for the final fit vs. `validate_gam.py`'s
  `max_iter=5000` -- asymmetry preserved, not confirmed as a bug.
- CNN/MLP eval scripts deliberately exclude Focal-loss combos when
  selecting "best" hyperparameters -- preserved (see
  `hwml.model_selection.select_best_combo`'s `exclude_pattern`).
- `cnn_t7d_valid_sw.py` alone wrote results to a `nh/CNNd/valid/` path
  instead of the `CNNd/valid/` used by every other arch/region
  combination -- the new `validate_cnn.py` always uses the consistent
  path.
