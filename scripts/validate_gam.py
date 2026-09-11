#!/usr/bin/env python3
"""
Validation (single hyperparameter combo) entry point for the GAM model.
Replaces `gam_trial4_valid.py`; driven by CLI args instead of env vars.

Usage
-----
    python scripts/validate_gam.py --region w --lam 1 --n-splines 10
"""
import argparse
from functools import reduce
from operator import add

import numpy as np
import pandas as pd
from pygam import LogisticGAM, s
from sklearn.preprocessing import StandardScaler

from hwml import data, splits as splits_mod, metrics as metrics_mod
from hwml.paths import RESULTS_DIR

SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    p.add_argument("--lam", type=int, required=True)
    p.add_argument("--n-splines", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    region_index, region_cl = data.REGION_SUFFIX_TO_FC[args.region]

    region_folder = data.REGION_FOLDERS[region_index]
    outfolder = RESULTS_DIR / "GAM" / "scores" / region_folder / "trial4" / "valid"
    outfolder.mkdir(parents=True, exist_ok=True)

    tag = f"F{region_index}_C{region_cl}_lam{args.lam}_nspl{args.n_splines}"
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

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        raw_pos_weight = n_neg / n_pos
        pos_weight = np.sqrt(raw_pos_weight)
        sample_weight = np.where(y_train == 1, pos_weight, 1.0)

        n_features = X_train.shape[1]
        terms = reduce(add, [s(i, n_splines=args.n_splines) for i in range(n_features)])

        model = LogisticGAM(terms, lam=args.lam, fit_intercept=True, max_iter=5000)
        model.fit(X_train, y_train, weights=sample_weight)

        y_val_proba = model.predict_mu(X_val)
        y_train_proba = model.predict_mu(X_train)

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
