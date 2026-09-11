"""
Central place for resolving data / results locations.

Defaults assume the repo layout (``data/...``, ``results/...``), so a fresh
``git clone`` works with zero configuration using the small data shipped in
the repo. On a cluster or local machine where the full data lives elsewhere
(e.g. the original ``/scratch3/dpappert/CESM2/ML/`` layout), override via
environment variables -- no code changes needed:

    export HWML_DATA_DIR=/scratch3/dpappert/CESM2/ML
    export HWML_RESULTS_DIR=/scratch3/dpappert/CESM2/ML/results
"""
import os
from pathlib import Path

# repo root = two levels up from this file (src/hwml/paths.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("HWML_DATA_DIR", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("HWML_RESULTS_DIR", REPO_ROOT / "results"))

TARGETS_DIR = DATA_DIR / "targets"
TAB_PREDS_DIR = DATA_DIR / "tab_preds"
CNN_GRID_DIR = DATA_DIR / "cnn_grid"

# Only used by the LR validation sweep, which reads pre-computed member
# splits from disk instead of generating them on the fly (see splits.py).
# `splits_9f_v5/` wasn't part of the uploaded scripts/data -- confirm
# whether it's still needed and, if so, where it should live before Step 5.
SPLITS_DIR = Path(os.environ.get("HWML_SPLITS_DIR", DATA_DIR / "precomputed_splits"))
