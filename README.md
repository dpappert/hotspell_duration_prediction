# Hotspell duration prediction

This repository contains code relating to the study in [journal, once
known], titled '[paper title]'. Additional information can be found in the
Supplement.

It contains the Python code to perform the following:

- validate (hyperparameter sweep, 7-fold member-based cross-validation) and
  evaluate (held-out test set performance) five model families -- Random
  Forest, GAM, Logistic Regression, a tabular MLP, and a CNN -- predicting
  hot spell duration from precursor predictors
- four of the five models (RF, GAM, LR, MLP) use tabular predictors; the
  CNN uses gridded 2D input
- all five are run across three regions (region codes `sw`, `w`, `n`)

## Referencing

If you use this code in your publication, please cite the corresponding
article:

- [Author list]: [Paper title], [Journal], [volume], [pages],
  [https://doi.org/xxx], [year].

Please report any issues on the GitHub portal.

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

## Supporting information about the scripts

Each model has a **validate** entry point (hyperparameter sweep, 7-fold
member-based cross-validation) and an **eval** entry point (retrain on all
non-test data with the best/fixed hyperparameters, report held-out test
performance). To run a full hyperparameter grid rather than one combo at a
time, use the matching driver in `scripts/sweeps/`.

### `scripts/check_data.py`

**Input:** nothing (reads `data/` as populated).
**Output:** none written -- prints an OK/MISSING report per region for
`data/targets/`, `data/tab_preds/`, and `data/cnn_grid/` to the console.
Run this first, before anything else.

### `scripts/validate_rf.py --region {sw,w,n} --n-estimators ... --max-depth ... [...]`

**Input:** `data/targets/`, `data/tab_preds/` for the given region; RF
hyperparameters as CLI flags (see `--help`).
**Output:** `results/RF/scores/{region}/trial6/valid/metrics_{tag}.csv` --
one row of pooled 7-fold train/val ROC-AUC and PRG-AUC for that combo.

### `scripts/eval_rf.py --region {sw,w,n}`

**Input:** all `metrics_*.csv` files under the matching `.../trial6/valid/`
directory (i.e. run `validate_rf.py` -- ideally via `scripts/sweeps/sweep_rf.sh`
-- first).
**Output:** `results/RF/scores/{region}/trial6/eval/{tag}.csv` -- per-row
`y_true`/`y_proba` on the held-out test set, using the best hyperparameter
combo auto-selected from the validation results.

### `scripts/validate_gam.py --region {sw,w,n} --lam ... --n-splines ...`

**Input/Output:** as RF above, under `results/GAM/scores/{region}/trial4/valid/`.

### `scripts/eval_gam.py --region {sw,w,n}`

**Input/Output:** as `eval_rf.py`, under `results/GAM/scores/{region}/trial4/eval/`.

### `scripts/validate_lr.py --region {sw,w,n} --reg ... --penalty ... --solver ...`

**Input/Output:** as RF above, under `results/LR/scores/{region}/trial7/valid/`.
Exploratory only -- does not gate `eval_lr.py`'s hyperparameters (see below).

### `scripts/eval_lr.py --region {sw,w,n}`

**Input:** `data/targets/`, `data/tab_preds/` only -- no validation results
needed. LR is this study's fixed baseline (C=1, L2, liblinear), not tuned.
**Output:** `results/LR/scores/{region}/eval/{tag}.csv`, same format as
`eval_rf.py`.

### `scripts/validate_mlp.py --arch {a,b,c,lin} --region {sw,w,n}`

**Input:** as RF above. Note: in the original study, `MLPa` was only ever
run on region `sw`, and `MLPb`/`MLPc`/`MLPlin` only on region `n`.
**Output:** `results/MLP/scores/{region}/MLP{arch}/valid/HPS_*.xlsx` -- one
file per hyperparameter combo (`metrics`, `losses`, `epochs` sheets).

### `scripts/eval_mlp.py --arch {a,b,c,lin} --region {sw,w,n}`

**Input:** all `.xlsx` files under the matching `.../MLP{arch}/valid/` directory.
**Output:** `results/MLP/scores/{region}/MLP{arch}/eval/{tag}.csv`,
`y_true`/`y_proba` on the test set.

### `scripts/validate_cnn.py --arch {a,b,c,d,lin,lr,mlp} --region {sw,w,n}`

**Input:** `data/targets/`, `data/tab_preds/`, AND `data/cnn_grid/` for the
given region (see `data/cnn_grid/README.md` -- not shipped in the repo).
**Output:** `results/CNN/scores/{region}/{subdir}/valid/HPS_*.xlsx`, same
format as the MLP validate output.

### `scripts/eval_cnn.py --arch {a,b,c,d,lin,lr,mlp} --region {sw,w,n}`

**Input:** all `.xlsx` files under the matching `.../valid/` directory,
plus the same gridded data as `validate_cnn.py`.
**Output:** `results/CNN/scores/{region}/{subdir}/eval/{tag}.csv`.

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
