#!/usr/bin/env python3
"""
Test-set evaluation for GAM: auto-selects the best hyperparameter combo
from validate_gam.py's results, retrains on all non-test data, evaluates
once on the held-out test set. Replaces `eval_GAM.py`.

Usage
-----
    python scripts/eval_gam.py --region w

NOTE: eval_GAM.py used `max_iter=2000` for the final fit, vs. `max_iter=5000`
in validate_gam.py's sweep -- preserved as-is (not obviously a bug, but
flagging the asymmetry in case it wasn't deliberate).
"""
import argparse
from functools import reduce
from operator import add

import numpy as np
import pandas as pd
from pygam import LogisticGAM, s
from sklearn.preprocessing import StandardScaler

from hwml import data, splits as splits_mod, model_selection
from hwml.paths import RESULTS_DIR

SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    return p.parse_args()


def main():
    args = parse_args()
    region_index, region_cl = data.REGION_SUFFIX_TO_FC[args.region]
    region_folder = data.REGION_FOLDERS[region_index]

    valid_dir = RESULTS_DIR / "GAM" / "scores" / region_folder / "trial4" / "valid"
    eval_dir = RESULTS_DIR / "GAM" / "scores" / region_folder / "trial4" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- SELECT BEST HYPERPARAMETERS --------------------
    csv_files = sorted(valid_dir.glob("metrics_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No validation results found in {valid_dir}; run validate_gam.py first.")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source"] = f.stem
        dfs.append(df)
    combined = pd.concat(dfs).set_index("source")

    best_tag, scored = model_selection.select_best_combo(combined)
    print(scored, flush=True)
    print(f"Best combo: {best_tag}", flush=True)

    hp_tag = best_tag.removeprefix("metrics_")
    hp = model_selection.parse_gam_tag(hp_tag)

    # -------------------- LOAD DATA --------------------
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
    raw_pos_weight = n_neg / n_pos
    pos_weight = np.sqrt(raw_pos_weight)
    sample_weight = np.where(y_train_full == 1, pos_weight, 1.0)

    n_features = X_train_full_scaled.shape[1]
    terms = reduce(add, [s(i, n_splines=hp["n_splines"]) for i in range(n_features)])

    model = LogisticGAM(terms, lam=hp["lam"], fit_intercept=True, max_iter=2000)
    model.fit(X_train_full_scaled, y_train_full, weights=sample_weight)

    y_test_proba = model.predict_mu(X_test_scaled)

    df_out = pd.DataFrame({"y_true": y_test, "y_proba": y_test_proba})
    out_path = eval_dir / f"{hp_tag}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
