"""
Loading and assembling the tabular data (targets + tabular predictors) that
feeds the RF / GAM / LR / MLP models, plus the alignment check used when
pairing tabular rows with the CNN's gridded input.

This logic was duplicated near-verbatim across every validation/eval
script. It's reproduced faithfully here, with one flagged discrepancy: see
``drop_na`` below.
"""
import glob

import numpy as np
import pandas as pd

from .paths import TARGETS_DIR, TAB_PREDS_DIR

# index -> folder name, matches the `f` env var / hardcoded value used
# throughout the original scripts (0, 1, 2 => regions cl1, cl2, cl4).
# This is the folder naming used by CNN, MLP, RF and GAM.
REGION_FOLDERS = {
    0: "targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1",
    1: "targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1",
    2: "targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1",
}

# lr_trial7_valid.py originally used a *different* date suffix (18512000,
# not 18512010) for all three regions. CONFIRMED (2024) to be a bug, not
# intentional -- LR should use REGION_FOLDERS like every other model.
# Kept here, unused, only as a record of what the original script did;
# Step 5 normalizes LR to REGION_FOLDERS + drop_na=True + iloc[:, 13:].
REGION_FOLDERS_LR = {
    0: "targets_cl1_HWs_m50_18512000_fmstat_1.5sd_r1",
    1: "targets_cl2_HWs_m50_18512000_fmstat_1.5sd_r1",
    2: "targets_cl4_HWs_m50_18512000_fmstat_1.5sd_r1",
}


def load_table(region_index, region_folders=None, drop_na=True, glob_pattern="*.txt"):
    """
    Load + assemble TABLE_FULL for a given region.

    Reproduces, in order:
      1. read the target file for the region
      2. read + horizontally concat all tabular predictor files for that region
      3. drop DURATION == 7 rows, insert a binary DURATION_bin label at column 3
      4. drop rows containing +/-inf in any numeric column
      5. optionally drop rows containing NaN (see `drop_na`)

    Parameters
    ----------
    region_index : int
        0, 1, or 2 -- indexes into `region_folders`.
    region_folders : dict, optional
        Defaults to `REGION_FOLDERS`. Pass `REGION_FOLDERS_LR` to match the
        LR script's folder naming (see module docstring).
    drop_na : bool
        RF, GAM, MLP, LR (after normalization) and every `eval_*.py` script
        call `.dropna()` after the inf-removal step. Only the CNN
        validation scripts do not. Defaults to True -- pass False when
        reproducing CNN exactly.
    glob_pattern : str
        All scripts glob tab_preds with "*.txt" (LR originally used a bare
        "*", normalized in Step 5). Defaults to "*.txt".

    Returns
    -------
    table_full : pd.DataFrame
    member_ids : np.ndarray
        Unique member IDs from the target file (pre-filtering).
    """
    if region_folders is None:
        region_folders = REGION_FOLDERS

    folder = region_folders[region_index]
    table_init = pd.read_csv(TARGETS_DIR / f"{folder}.txt", sep="\t")
    member_ids = np.unique(table_init["member_id"])

    file_list = sorted(glob.glob(str(TAB_PREDS_DIR / folder / glob_pattern)))
    preds = [pd.read_csv(f, sep="\t") for f in file_list]
    table_full = pd.concat([table_init, pd.concat(preds, axis=1)], axis=1)

    table_full = table_full[table_full["DURATION"] != 7]
    table_full.insert(3, "DURATION_bin", (table_full["DURATION"] > 7).astype(int))

    numeric_cols = table_full.select_dtypes(include=[np.number])
    table_full = table_full[~np.isinf(numeric_cols).any(axis=1)]
    if drop_na:
        table_full = table_full.dropna()

    return table_full, member_ids


def load_cnn_grid(region_cl, grid_dir=None):
    """
    Load the gridded (2D) CNN input for a region and normalize per-channel,
    matching `cnn_t7*_valid_*.py`. Requires the ~2GB/region `.nc` files in
    `data/cnn_grid/cl{region_cl}/` -- these are NOT shipped in the repo (see
    `data/cnn_grid/README.md`), so this will raise if they're missing.

    Parameters
    ----------
    region_cl : int
        1, 2, or 4 -- matches the `c` value / `cl{c}` folder naming.
    """
    import xarray as xr

    from .paths import CNN_GRID_DIR

    if grid_dir is None:
        grid_dir = CNN_GRID_DIR

    file_list = sorted(glob.glob(str(grid_dir / f"cl{region_cl}" / "r*.nc")))
    if not file_list:
        raise FileNotFoundError(
            f"No .nc files found in {grid_dir / f'cl{region_cl}'}. "
            f"This data isn't shipped in the repo -- see data/cnn_grid/README.md."
        )

    datasets = [xr.open_dataarray(f) for f in file_list]
    x = xr.concat(datasets, dim="event")
    x = x.where(x != -np.inf, np.nan)

    means, stds = [], []
    for j in range(x.shape[1]):
        vals = x[:, j, :, :]
        means.append(vals.mean(skipna=True).item())
        stds.append(vals.std(skipna=True).item())
    for j in range(x.shape[1]):
        x[:, j, :, :] = (x[:, j, :, :] - means[j]) / stds[j]

    return x


def check_row_alignment(table_full, grid):
    """
    Reproduces the CNN scripts' date-based alignment check between the
    tabular TABLE_FULL rows and the gridded array's `event` dimension.
    Returns True/False; the original scripts only printed this, they never
    actually acted on a False result -- worth deciding in Step 3 whether
    that should become a hard assertion.
    """
    x_dates = pd.to_datetime(grid.start_date.values)
    table_dates = pd.to_datetime(table_full.start_date.values)
    return bool(np.array_equal(x_dates, table_dates))
