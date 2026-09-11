"""
Smoke tests for the shared hwml modules. Run with `pytest` from the repo
root (after `pip install -e .`). Intentionally light -- these cover the
pure-Python logic (metrics, splits, tag parsing) that doesn't need real
data or torch, since that's what's actually testable without your dataset
in place. Not a substitute for Step 8 (diffing real outputs against the
legacy scripts).
"""
import numpy as np
import pandas as pd
import pytest

from hwml import data, metrics, model_selection, splits


def test_compute_auprgc_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = metrics.compute_metrics(y_true, y_score)
    assert m["rocauc"] == 1.0
    assert 0.0 <= m["prgauc"] <= 1.0


def test_compute_auprgc_handles_degenerate_input():
    # single-point / near-degenerate precision-recall curves shouldn't crash
    # (this is what GAM's original redundant `if len(...) < 2: return 0.0`
    # guard was defending against -- confirmed unnecessary, see metrics.py)
    y_true = np.array([0, 1])
    y_score = np.array([0.5, 0.5])
    result = metrics.compute_auprgc(y_true, y_score)
    assert np.isfinite(result)


def test_generate_member_splits_is_deterministic():
    member_ids = np.arange(50)
    test1, val1 = splits.generate_member_splits(member_ids, seed=0)
    test2, val2 = splits.generate_member_splits(member_ids, seed=0)
    assert np.array_equal(test1, test2)
    assert all(np.array_equal(a, b) for a, b in zip(val1, val2))


def test_generate_member_splits_train_test_disjoint():
    member_ids = np.arange(50)
    test_membs, val_membs = splits.generate_member_splits(member_ids, seed=0)
    all_val = np.concatenate(val_membs)
    assert len(set(test_membs) & set(all_val)) == 0
    assert len(val_membs) == 7


def test_region_mappings_are_consistent():
    # every region_index used in REGION_SUFFIX_TO_FC must exist in REGION_FOLDERS
    for suffix, (region_index, _cl) in data.REGION_SUFFIX_TO_FC.items():
        assert region_index in data.REGION_FOLDERS


def test_parse_rf_tag_regression_full_grid():
    """
    Regression test for the bug fixed in Step 6: the original eval_RF.py
    parsed `criterion` with a hardcoded character offset that mis-parsed
    96/192 combos in the actual hyperparameter grid. This checks every
    combo in that grid round-trips correctly through parse_rf_tag.
    """
    for n in [50, 100, 250, 500]:
        for d in [3, 5, 8]:
            for ms in [10, 20]:
                for ml in [5, 10]:
                    for mf in ["log2", "0.33"]:
                        for crit in ["gini", "log_loss"]:
                            tag = f"F0_C1_nest{n}_depth{d}_msplit{ms}_mleaf{ml}_mfeat{mf}_crit{crit}"
                            parsed = model_selection.parse_rf_tag(tag)
                            assert parsed["criterion"] == crit
                            assert parsed["n_estimators"] == n
                            assert parsed["max_depth"] == d
                            assert parsed["min_samples_split"] == ms
                            assert parsed["min_samples_leaf"] == ml
                            expected_mf = float(mf) if mf == "0.33" else mf
                            assert parsed["max_features"] == expected_mf


def test_select_best_combo_excludes_focal():
    df = pd.DataFrame({
        "val_rocauc": [0.70, 0.75, 0.72],
        "train_rocauc": [0.90, 0.78, 0.95],
        "val_prgauc": [0.30, 0.40, 0.32],
        "train_prgauc": [0.60, 0.42, 0.70],
    }, index=[
        "HPS_mpe500-pat50_bs32_lr0.01_wd0.001_critBCE_optAdam",
        "HPS_mpe500-pat50_bs64_lr0.001_wd0.0003_critFocal-g2_optAdam",
        "HPS_mpe500-pat50_bs128_lr0.0001_wd0.001_critBCE_optAdam",
    ])
    best_tag, _ = model_selection.select_best_combo(df, exclude_pattern="Focal")
    assert "Focal" not in best_tag


def test_parse_nn_tag_roundtrip():
    tag = "HPS_mpe500-pat50_bs64_lr0.01_wd0.0003_critBCE_optAdam"
    hp = model_selection.parse_nn_tag(tag)
    assert hp == {
        "max_epochs": 500,
        "patience": 50,
        "batch_size": 64,
        "lr": 0.01,
        "weight_decay": 0.0003,
        "criterion": "BCE",
        "optimizer": "Adam",
    }
