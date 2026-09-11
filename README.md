# [Paper title]

> Code accompanying "[paper title]" ([link when available]).

**Status: work in progress reorganization.** This README is a placeholder —
full setup and reproduction instructions land in Step 9 of the repo cleanup.

## Layout (preview)

- `src/hwml/` — shared library code (data loading, splits, metrics, models)
- `scripts/` — entry points to run validation / evaluation
- `config/` — paths and default hyperparameters
- `data/` — small data shipped with the repo (targets, tabular predictors);
  see `data/README.md` for exactly what goes where, and run
  `python scripts/check_data.py` to verify it's populated correctly.
  `data/cnn_grid/` is a placeholder (not shipped), see its own README
- `legacy/` — original one-off scripts, kept temporarily during refactor,
  will be removed once the new entry points are verified equivalent
