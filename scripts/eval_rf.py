#!/usr/bin/env python3
"""
Test-set evaluation for RF: auto-selects the best hyperparameter combo from
validate_rf.py's results, retrains on all non-test data, evaluates once on
the held-out test set. Replaces `eval_RF.py`.

BUG FIX vs. the original: hyperparameter-tag parsing now uses
hwml.model_selection.parse_rf_tag (regex-based) instead of the original's
hardcoded character-offset slice for `criterion`, which silently mis-parsed
roughly half of the possible combos (see model_selection.py docstring).

Usage
-----
    python scripts/eval_rf.py --region w
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from hwml import data, splits as splits_mod, model_selection
from hwml.paths import RESULTS_DIR

SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    p.add_argument("--n-jobs", type=int, default=1)  # original eval_RF.py used n_jobs=1 (validate used 3)
    return p.parse_args()


def main():
    args = parse_args()
    region_index, region_cl = data.REGION_SUFFIX_TO_FC[args.region]
    region_folder = data.REGION_FOLDERS[region_index]

    valid_dir = RESULTS_DIR / "RF" / "scores" / region_folder / "trial6" / "valid"
    eval_dir = RESULTS_DIR / "RF" / "scores" / region_folder / "trial6" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- SELECT BEST HYPERPARAMETERS --------------------
    csv_files = sorted(valid_dir.glob("metrics_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No validation results found in {valid_dir}; run validate_rf.py first.")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source"] = f.stem  # filename without extension, e.g. "metrics_F0_C1_nest50_..."
        dfs.append(df)
    combined = pd.concat(dfs).set_index("source")

    best_tag, scored = model_selection.select_best_combo(combined)
    print(scored, flush=True)
    print(f"Best combo: {best_tag}", flush=True)

    hp_tag = best_tag.removeprefix("metrics_")
    hp = model_selection.parse_rf_tag(hp_tag)

    # -------------------- LOAD DATA --------------------
    table_full, member_ids = data.load_table(region_index)
    test_membs, _ = splits_mod.generate_member_splits(member_ids, seed=SEED)

    test_mask = table_full["member_id"].isin(test_membs)
    X_train_full = table_full.loc[~test_mask].iloc[:, 13:]
    y_train_full = table_full.loc[~test_mask, "DURATION_bin"]
    X_test = table_full.loc[test_mask].iloc[:, 13:]
    y_test = table_full.loc[test_mask, "DURATION_bin"]

    n_pos = (y_train_full == 1).sum()
    n_neg = (y_train_full == 0).sum()
    pos_weight = n_neg / n_pos
    sample_weight = np.where(y_train_full == 1, pos_weight, 1.0)

    # -------------------- TRAIN + EVALUATE --------------------
    model = RandomForestClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        min_samples_split=hp["min_samples_split"],
        min_samples_leaf=hp["min_samples_leaf"],
        max_features=hp["max_features"],
        criterion=hp["criterion"],
        random_state=SEED,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train_full, y_train_full, sample_weight=sample_weight)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    df_out = pd.DataFrame({"y_true": y_test, "y_proba": y_test_proba})
    out_path = eval_dir / f"{hp_tag}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
