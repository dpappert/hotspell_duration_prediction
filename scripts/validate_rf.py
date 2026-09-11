#!/usr/bin/env python3
"""
Validation (single hyperparameter combo) entry point for the Random Forest
model. Replaces `rf_trial6_valid.py`; driven by CLI args instead of env
vars (the sweep over combos moves to scripts/sweeps/sweep_rf.sh).

Usage
-----
    python scripts/validate_rf.py --region w --n-estimators 100 --max-depth 5 \\
        --min-samples-split 10 --min-samples-leaf 5 --max-features log2 --criterion gini
"""
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from hwml import data, splits as splits_mod, metrics as metrics_mod
from hwml.paths import RESULTS_DIR

SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    p.add_argument("--n-estimators", type=int, required=True)
    p.add_argument("--max-depth", type=int, required=True)
    p.add_argument("--min-samples-split", type=int, required=True)
    p.add_argument("--min-samples-leaf", type=int, required=True)
    p.add_argument("--max-features", type=str, required=True, help="'log2', 'sqrt', or a float like '0.33'")
    p.add_argument("--criterion", type=str, required=True, choices=["gini", "log_loss"])
    p.add_argument("--n-jobs", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    region_index, region_cl = data.REGION_SUFFIX_TO_FC[args.region]

    max_features = args.max_features
    if max_features == "0.33":
        max_features = float(max_features)

    region_folder = data.REGION_FOLDERS[region_index]
    outfolder = RESULTS_DIR / "RF" / "scores" / region_folder / "trial6" / "valid"
    outfolder.mkdir(parents=True, exist_ok=True)

    tag = (f"F{region_index}_C{region_cl}_"
           f"nest{args.n_estimators}_depth{args.max_depth}_"
           f"msplit{args.min_samples_split}_mleaf{args.min_samples_leaf}_"
           f"mfeat{max_features}_crit{args.criterion}")
    metrics_path = outfolder / f"metrics_{tag}.csv"

    table_full, member_ids = data.load_table(region_index)  # drop_na=True, matches original
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

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        pos_weight = n_neg / n_pos
        sample_weight = np.where(y_train == 1, pos_weight, 1.0)

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            criterion=args.criterion,
            random_state=SEED,
            n_jobs=args.n_jobs,
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_train_proba = model.predict_proba(X_train)[:, 1]

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
