#!/usr/bin/env python3
"""
Sanity-checks that `data/` is populated the way the pipeline expects,
before you spend time launching an actual training/validation run. Run
this after dropping your targets/tab_preds/(optionally) cnn_grid files in.

Usage
-----
    python scripts/check_data.py
"""
from hwml.data import REGION_FOLDERS
from hwml.paths import TARGETS_DIR, TAB_PREDS_DIR, CNN_GRID_DIR

REGION_CL = {0: 1, 1: 2, 2: 4}  # region_index -> cl value, matches REGION_SUFFIX_TO_FC


def check_targets():
    print("== data/targets/ ==")
    ok = True
    for idx, folder in REGION_FOLDERS.items():
        path = TARGETS_DIR / f"{folder}.txt"
        status = "OK" if path.exists() else "MISSING"
        if not path.exists():
            ok = False
        print(f"  [{status}] {path}")
    return ok


def check_tab_preds():
    print("== data/tab_preds/ ==")
    ok = True
    for idx, folder in REGION_FOLDERS.items():
        subdir = TAB_PREDS_DIR / folder
        n_files = len(list(subdir.glob("*.txt"))) if subdir.is_dir() else 0
        status = "OK" if n_files > 0 else "MISSING / EMPTY"
        if n_files == 0:
            ok = False
        print(f"  [{status}] {subdir}  ({n_files} .txt files)")
    return ok


def check_cnn_grid():
    print("== data/cnn_grid/ (optional -- only needed for the CNN model) ==")
    any_present = False
    for idx, cl in REGION_CL.items():
        subdir = CNN_GRID_DIR / f"cl{cl}"
        n_files = len(list(subdir.glob("r*.nc"))) if subdir.is_dir() else 0
        if n_files > 0:
            any_present = True
        status = "OK" if n_files > 0 else "not present (placeholder only)"
        print(f"  [{status}] {subdir}  ({n_files} .nc files)")
    if not any_present:
        print("  -> CNN scripts will raise FileNotFoundError until this is populated.")
        print("     Every other model (RF, GAM, LR, MLP) works without it.")
    return any_present


def main():
    targets_ok = check_targets()
    print()
    tab_preds_ok = check_tab_preds()
    print()
    cnn_ok = check_cnn_grid()
    print()

    if targets_ok and tab_preds_ok:
        print("Tabular data OK -- RF/GAM/LR/MLP scripts are ready to run.")
    else:
        print("Tabular data incomplete -- see MISSING entries above.")

    if not cnn_ok:
        print("CNN gridded data not present -- see data/cnn_grid/README.md.")


if __name__ == "__main__":
    main()
