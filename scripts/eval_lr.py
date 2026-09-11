#!/usr/bin/env python3
"""
Test-set evaluation for LR, the study's fixed baseline model. Replaces
`eval_LR.py`.

Unlike RF/GAM/MLP/CNN, LR does NOT auto-select hyperparameters from
validation results -- confirmed deliberate: LR's hyperparameters are fixed
by design (C=1, L2, liblinear) since it serves as the baseline, not a
tuned model. `validate_lr.py`'s sweep is exploratory only and doesn't gate
this script (consequently, the `val_membs` computation present in the
original `eval_LR.py` was genuinely dead code -- not reproduced here).

WEIGHTING: confirmed deliberate -- uses RF's manual sample_weight formula
(pos_weight = n_neg/n_pos via sample_weight=, not sklearn's
class_weight="balanced"). These differ in absolute scale, which matters
for a regularized model like LogisticRegression (equivalent to scaling C).
This means eval_lr.py now uses a DIFFERENT weighting scale than
validate_lr.py (which still uses class_weight="balanced", unchanged) --
confirmed as acceptable since validate's role for LR is exploratory only.

Usage
-----
    python scripts/eval_lr.py --region w
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hwml import data, splits as splits_mod
from hwml.paths import RESULTS_DIR

SEED = 0
C = 1
PENALTY = "l2"
SOLVER = "liblinear"
MAX_ITER = 2000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    region_index, region_cl = data.REGION_SUFFIX_TO_FC[args.region]
    region_folder = data.REGION_FOLDERS[region_index]

    eval_dir = RESULTS_DIR / "LR" / "scores" / region_folder / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    tag = f"F{region_index}_C{region_cl}_reg{C}_pen{PENALTY}_solv{SOLVER}"

    table_full, member_ids = data.load_table(region_index)
    test_membs, _ = splits_mod.generate_member_splits(member_ids, seed=SEED)

    test_mask = table_full["member_id"].isin(test_membs)
    X_train_full = table_full.loc[~test_mask].iloc[:, 13:]
    y_train_full = table_full.loc[~test_mask, "DURATION_bin"]
    X_test = table_full.loc[test_mask].iloc[:, 13:]
    y_test = table_full.loc[test_mask, "DURATION_bin"]

    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    n_pos = (y_train_full == 1).sum()
    n_neg = (y_train_full == 0).sum()
    pos_weight = n_neg / n_pos
    sample_weight = np.where(y_train_full == 1, pos_weight, 1.0)

    model = LogisticRegression(
        C=C,
        penalty=PENALTY,
        solver=SOLVER,
        max_iter=MAX_ITER,
        random_state=SEED,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train_full_scaled, y_train_full, sample_weight=sample_weight)

    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

    df_out = pd.DataFrame({"y_true": y_test, "y_proba": y_test_proba})
    out_path = eval_dir / f"{tag}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
