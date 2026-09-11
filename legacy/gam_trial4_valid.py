#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import random
from pygam import LogisticGAM, s
from functools import reduce
from operator import add
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

SEED = 0

path = '/scratch3/dpappert/CESM2/ML/'
midpath = 'trial4/valid/'

f = int(os.environ["F"])
c = int(os.environ["C"])

lam = int(os.environ["LAM"])
n_splines = int(os.environ["SPLINES"])

folders = [
    'targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1'
]

outfolder = (
    f"{path}"
    f"GAM/scores/{folders[f]}/"
    f"{midpath}"
)
os.makedirs(outfolder, exist_ok=True)

tag = f"F{f}_C{c}_lam{lam}_nspl{n_splines}"

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
    if len(recG_sorted) < 2:
        return 0.0
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
print(test_membs, flush=True)
train_membs = np.setdiff1d(member_ids, test_membs)
print(train_membs, flush=True)

rng.shuffle(train_membs)
val_membs = np.array_split(train_membs, 7)

for i, part in enumerate(val_membs, 1):
    print(f"Group {i}: {part}", flush=True)


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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    raw_pos_weight = n_neg / n_pos
    pos_weight = np.sqrt(raw_pos_weight)
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)

    n_features = X_train.shape[1]
    terms = reduce(add, [s(i, n_splines=n_splines) for i in range(n_features)])
    
    # GAM model
    model = LogisticGAM(
        terms,
        lam=lam,
        fit_intercept=True,
        max_iter=5000
    )
    
    model.fit(X_train, y_train, weights=sample_weight)

    y_val_proba = model.predict_mu(X_val)
    y_train_proba = model.predict_mu(X_train)

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