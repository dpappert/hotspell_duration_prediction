# Data layout

Run `python scripts/check_data.py` any time to verify this folder is
populated correctly (it checks filenames/paths without needing you to
launch an actual training run).

## `targets/` (ship in repo, ~2.4MB total)

Three files, one per region:

```
targets/
├── targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1.txt
├── targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1.txt
└── targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1.txt
```

## `tab_preds/` (ship in repo, ~21MB total)

One subfolder per region, matching the target filenames above (minus
`.txt`), each containing the tabular predictor files for that region:

```
tab_preds/
├── targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1/*.txt
├── targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1/*.txt
└── targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1/*.txt
```

## `cnn_grid/` (NOT shipped, ~6GB total -- see its own README)

Only needed for the CNN model; every other model (RF, GAM, LR, MLP) runs
fine without it.
