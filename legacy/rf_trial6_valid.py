#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

SEED = 0

path = '/scratch3/dpappert/CESM2/ML/'
midpath = 'trial6/valid/'

f = int(os.environ["F"])
c = int(os.environ["C"])

n_estimators = int(os.environ["N_EST"])
max_depth = int(os.environ["DEPTH"])
min_samples_split = int(os.environ["minSPLIT"])
min_samples_leaf = int(os.environ["minLEAF"])
max_features = os.environ["maxFEAT"]
if max_features == '0.33':
    max_features = float(max_features)
criterion = os.environ["CRIT"]

folders = [
    'targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1'
]

outfolder = (
    f"{path}"
    f"RF/scores/{folders[f]}/"
    f"{midpath}"
)
os.makedirs(outfolder, exist_ok=True)

tag = (
    f"F{f}_C{c}_"
    f"nest{n_estimators}_depth{max_depth}_"
    f"msplit{min_samples_split}_mleaf{min_samples_leaf}_"
    f"mfeat{max_features}_crit{criterion}"
)

# -------------------------------------------------------------------
# Load initial data
# -------------------------------------------------------------------

TABLE_init = pd.read_csv(path + f'targets/{folders[f]}.txt', sep='\t')
member_ids = np.unique(TABLE_init['member_id'])

file_list = sorted(glob.glob(f"{path}/tab_preds/{folders[f]}/*.txt"))
preds_ = [pd.read_csv(file, sep='\t') for file in file_list]

TABLE_FULL = pd.concat([TABLE_init, pd.concat(preds_, axis=1)], axis=1)

# Remove duration==7 and add binary label
TABLE_FULL = TABLE_FULL[TABLE_FULL['DURATION'] != 7]
TABLE_FULL.insert(3, 'DURATION_bin', (TABLE_FULL['DURATION'] > 7).astype(int))

# Remove rows that contain np.nan or inf values
TABLE_FULL = TABLE_FULL[~np.isinf(TABLE_FULL.select_dtypes(include=[np.number])).any(axis=1)]
TABLE_FULL = TABLE_FULL.dropna()

# -------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------

def compute_auprgc(y_true, y_score):
    pi = y_true.mean()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    precG = np.zeros_like(precision); recG = np.zeros_like(recall)
    mask_prec = precision > 0; mask_rec = recall > 0
    precG[mask_prec] = (precision[mask_prec] - pi) / ((1 - pi) * precision[mask_prec])
    recG[mask_rec] = (recall[mask_rec] - pi) / ((1 - pi) * recall[mask_rec])
    mask = (precG >= 0) & (recG >= 0)
    precG_plot = precG[mask]
    recG_plot = recG[mask]
    sorted_indices = np.argsort(recG_plot)
    recG_sorted = recG_plot[sorted_indices]
    precG_sorted = precG_plot[sorted_indices]
    auc = np.trapz(precG_sorted, recG_sorted)
    return auc


# -------------------------------------------------------------------
# CV
# -------------------------------------------------------------------

### SPLIT BY MEMBERS

rng = np.random.default_rng(seed=SEED)
test_set_prop = 0.3
test_size = int(test_set_prop * 50)
split_sets = pd.DataFrame(
    [rng.choice(member_ids, size=test_size, replace=False)
     for _ in range(100)],
    columns=[f"member_{i+1}" for i in range(test_size)]
)

test_membs = split_sets.iloc[0, :].values
#print(test_membs, flush=True)
train_membs = np.setdiff1d(member_ids, test_membs)
#print(train_membs, flush=True)

rng.shuffle(train_membs)
val_membs = np.array_split(train_membs, 7)

#for i, part in enumerate(val_membs, 1):
#    print(f"Group {i}: {part}", flush=True)


SERIES_y_val_true = []
SERIES_y_val_proba = []
SERIES_y_train_true = []
SERIES_y_train_proba = []

for k in range(7):
    train_mask = ~TABLE_FULL['member_id'].isin(np.concatenate([val_membs[k], test_membs]))

    X_train = TABLE_FULL[train_mask].iloc[:, 13:]
    y_train = TABLE_FULL[train_mask].iloc[:, 3]

    val_mask = TABLE_FULL['member_id'].isin(val_membs[k])
    X_val = TABLE_FULL[val_mask].iloc[:, 13:]
    y_val = TABLE_FULL[val_mask].iloc[:, 3]

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = n_neg / n_pos
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    # RF model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=SEED,
        n_jobs=3
    )

    model.fit(X_train, y_train, sample_weight=sample_weight)

    # predictions
    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_train_proba = model.predict_proba(X_train)[:, 1]

    SERIES_y_val_true.append(y_val.values)
    SERIES_y_val_proba.append(y_val_proba)
    SERIES_y_train_true.append(y_train.values)
    SERIES_y_train_proba.append(y_train_proba)

val_rocauc = roc_auc_score(
    np.concatenate(SERIES_y_val_true),
    np.concatenate(SERIES_y_val_proba)
)
val_prgauc = compute_auprgc(
    np.concatenate(SERIES_y_val_true),
    np.concatenate(SERIES_y_val_proba)
)

train_rocauc = roc_auc_score(
    np.concatenate(SERIES_y_train_true),
    np.concatenate(SERIES_y_train_proba)
)
train_prgauc = compute_auprgc(
    np.concatenate(SERIES_y_train_true),
    np.concatenate(SERIES_y_train_proba)
)

metrics_df = pd.DataFrame({
    'seed': [SEED],
    'val_rocauc': [val_rocauc],
    'train_rocauc': [train_rocauc],
    'val_prgauc': [val_prgauc],
    'train_prgauc': [train_prgauc]
})
metrics_path = f"{outfolder}/metrics_{tag}.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved metrics to {metrics_path}", flush=True)