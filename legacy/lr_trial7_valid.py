#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import time
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
import pickle

SEED = 0
np.random.seed(SEED)
random.seed(SEED)

path = '/scratch3/dpappert/CESM2/ML/'
midpath = 'trial7/valid/'

f = int(os.environ["F"])
c = int(os.environ["C"])
s = int(os.environ["FOLD_SPLIT"])

C = float(os.environ["REG"])
penalty = os.environ["PEN"]
solver = os.environ["SOLV"]

folders = [
    'targets_cl1_HWs_m50_18512000_fmstat_1.5sd_r1',
    'targets_cl2_HWs_m50_18512000_fmstat_1.5sd_r1',
    'targets_cl4_HWs_m50_18512000_fmstat_1.5sd_r1'
]

outfolder = (
    f"{path}"
    f"LR/scores/{folders[f]}/"
    f"{midpath}"
    f"s{s}"
)
os.makedirs(outfolder, exist_ok=True)
print("Output folder:", outfolder, flush=True)
tag = f"F{f}_C{c}_S{s}_reg{C:.3g}_pen{penalty}_solv{solver}"

# -------------------------------------------------------------------
# Load initial data
# -------------------------------------------------------------------

TABLE_init = pd.read_csv(path + f'targets/{folders[f]}.txt', sep='\t')
member_ids = np.unique(TABLE_init['member_id'])

file_list = sorted(glob.glob(f"{path}/tab_preds/{folders[f]}/*"))
preds_ = [pd.read_csv(file, sep='\t') for file in file_list]

TABLE_FULL = pd.concat([TABLE_init, pd.concat(preds_, axis=1)], axis=1)

# Remove duration==7 and add binary label
TABLE_FULL = TABLE_FULL[TABLE_FULL['DURATION'] != 7]
TABLE_FULL.insert(3, 'DURATION_bin', (TABLE_FULL['DURATION'] > 7).astype(int))

# Remove rows that contain np.nan or inf values
TABLE_FULL = TABLE_FULL[~np.isinf(TABLE_FULL.select_dtypes(include=[np.number])).any(axis=1)]

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
# Search best hyperparameters across validation sets
# -------------------------------------------------------------------

start_time = time.time()
splits = pd.read_csv(path + f"splits_9f_v5/cl{c}/k{s}_split_members.txt", sep="\t")
test_membs = splits.iloc[:, 0]

fold_rocaucs = []
fold_rocaucs_train = []
fold_prgaucs = []
fold_prgaucs_train = []

max_iter = 2000

# 5-fold CV
for fold in range(1, 6):
    val_membs = splits.iloc[:, fold]
    train_mask = ~TABLE_FULL['member_id'].isin(pd.concat([val_membs, test_membs]))

    X_train = TABLE_FULL[train_mask].iloc[:, 12:]
    y_train = TABLE_FULL[train_mask].iloc[:, 3]

    X_val = TABLE_FULL[TABLE_FULL['member_id'].isin(val_membs)].iloc[:, 12:]
    y_val = TABLE_FULL[TABLE_FULL['member_id'].isin(val_membs)].iloc[:, 3]

    # LR model as pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=SEED,
            n_jobs=1
        ))
    ])

    # fit model
    pipeline.fit(X_train, y_train)

    if np.any(pipeline.named_steps["model"].n_iter_ == max_iter):
        print(f"Warning: hit max_iter on fold {fold}", flush=True)

    if fold == 1:
        model_path = f"{outfolder}/model-f1_{tag}.pkl"
        with open(model_path, "wb") as f_out:
            pickle.dump(pipeline, f_out)
        print(f"Saved model for fold 1 to {model_path}", flush=True)

    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    fold_rocaucs.append(roc_auc_score(y_val, y_val_proba))
    fold_prgaucs.append(compute_auprgc(y_val, y_val_proba))

    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    fold_rocaucs_train.append(roc_auc_score(y_train, y_train_proba))
    fold_prgaucs_train.append(compute_auprgc(y_train, y_train_proba))

metrics_df = pd.DataFrame({
    'fold': range(1, len(fold_rocaucs)+1),
    'train_rocauc': fold_rocaucs_train,
    'val_rocauc': fold_rocaucs,
    'train_prgauc': fold_prgaucs_train,
    'val_prgauc': fold_prgaucs
})

metrics_path = f"{outfolder}/metrics_{tag}.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved metrics to {metrics_path}", flush=True)
