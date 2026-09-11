#!/usr/bin/env python3
"""
Validation (single hyperparameter combo) entry point for the Logistic
Regression model. Replaces `lr_trial7_valid.py`.

Four things were normalized here vs. the original LR script, per explicit
confirmation (all four originals differed from RF/GAM/CNN/MLP):
  1. Feature columns: now `iloc[:, 13:]` (was `12:`).
  2. Splits: now `generate_member_splits` (7 on-the-fly folds), same as
     RF/GAM/CNN/MLP -- not the old pre-computed `splits_9f_v5/` file (5 folds).
  3. Target folder naming: now uses `REGION_FOLDERS` (`...18512010...`),
     same as everywhere else -- not the `...18512000...` variant.
  4. Validation metrics: now POOLED across all folds into one rocauc/prgauc
     (matching RF/GAM/CNN/MLP), not a per-fold breakdown table. This is a
     confirmed, deliberate change -- the original reported a different
     statistic (mean-of-per-fold vs. pooled), so re-running this after the
     fix will very likely shift LR's previously-reported validation numbers.
     Flag this specifically when comparing against anything already written
     up using the old LR results.

One thing kept, not a bug: saving the fitted pipeline (`model-f1_*.pkl`)
for inspection was LR-specific extra behaviour (e.g. for coefficient
analysis), not present in RF/GAM. Kept here, now saved on the first of the
7 folds (fold index 0) rather than the old scheme's fold 1 of 5.

Usage
-----
    python scripts/validate_lr.py --region w --reg 1.0 --penalty l2 --solver lbfgs
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hwml import data, splits as splits_mod, metrics as metrics_mod
from hwml.paths import RESULTS_DIR

SEED = 0
MAX_ITER = 2000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    p.add_argument("--reg", type=float, required=True, help="Inverse regularization strength (sklearn's C).")
    p.add_argument("--penalty", type=str, required=True)
    p.add_argument("--solver", type=str, required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    region_index, region_cl = data.REGION_SUFFIX_TO_FC[args.region]

    region_folder = data.REGION_FOLDERS[region_index]
    outfolder = RESULTS_DIR / "LR" / "scores" / region_folder / "trial7" / "valid"
    outfolder.mkdir(parents=True, exist_ok=True)

    tag = f"F{region_index}_C{region_cl}_reg{args.reg:.3g}_pen{args.penalty}_solv{args.solver}"
    metrics_path = outfolder / f"metrics_{tag}.csv"

    table_full, member_ids = data.load_table(region_index)  # drop_na=True
    test_membs, val_membs = splits_mod.generate_member_splits(member_ids, seed=SEED)

    SERIES_y_val_true, SERIES_y_val_proba = [], []
    SERIES_y_train_true, SERIES_y_train_proba = [], []

    for k in range(7):
        train_mask = ~table_full["member_id"].isin(np.concatenate([val_membs[k], test_membs]))
        X_train = table_full[train_mask].iloc[:, 13:]
        y_train = table_full[train_mask].iloc[:, 3]

        val_mask = table_full["member_id"].isin(val_membs[k])
        X_val = table_full[val_mask].iloc[:, 13:]
        y_val = table_full[val_mask].iloc[:, 3]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=args.reg,
                penalty=args.penalty,
                solver=args.solver,
                class_weight="balanced",
                max_iter=MAX_ITER,
                random_state=SEED,
                n_jobs=args.n_jobs,
            )),
        ])
        pipeline.fit(X_train, y_train)

        if np.any(pipeline.named_steps["model"].n_iter_ == MAX_ITER):
            print(f"Warning: hit max_iter on fold {k}", flush=True)

        if k == 0:
            model_path = outfolder / f"model-f1_{tag}.pkl"
            with open(model_path, "wb") as f_out:
                pickle.dump(pipeline, f_out)
            print(f"Saved model for fold 1 to {model_path}", flush=True)

        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]

        SERIES_y_val_true.append(y_val.values)
        SERIES_y_val_proba.append(y_val_proba)
        SERIES_y_train_true.append(y_train.values)
        SERIES_y_train_proba.append(y_train_proba)

    val_m = metrics_mod.compute_metrics(np.concatenate(SERIES_y_val_true), np.concatenate(SERIES_y_val_proba))
    train_m = metrics_mod.compute_metrics(np.concatenate(SERIES_y_train_true), np.concatenate(SERIES_y_train_proba))

    metrics_df = pd.DataFrame({
        "seed": [SEED],
        "val_rocauc": [val_m["rocauc"]],
        "train_rocauc": [train_m["rocauc"]],
        "val_prgauc": [val_m["prgauc"]],
        "train_prgauc": [train_m["prgauc"]],
    })
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
