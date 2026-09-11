#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import time
import random
from functools import reduce
from operator import add
from pygam import LogisticGAM, s
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

SEED = 0
np.random.seed(SEED)

model = 'GAM'
path = '/scratch3/dpappert/CESM2/ML/'
midpath = 'trial4/eval/'

f = int(os.environ["F"])
c = int(os.environ["C"])

folders = [
    'targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1'
]

outfolder = (
    f"{path}"
    f"{model}"
    f"/scores/{folders[f]}/"
    f"{midpath}"
)
os.makedirs(outfolder, exist_ok=True)


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
# Split data
# -------------------------------------------------------------------

members_ids = np.unique(TABLE_FULL.member_id.values)
rng = np.random.default_rng(seed=SEED)
test_set_prop = 0.3
test_size = int(test_set_prop * 50)
split_sets = pd.DataFrame(
    [rng.choice(member_ids, size=test_size, replace=False)
     for _ in range(100)],
    columns=[f"member_{i+1}" for i in range(test_size)]
)

test_membs = split_sets.iloc[0, :].values
print(test_membs)
train_membs = np.setdiff1d(members_ids, test_membs)
print(train_membs)

np.random.shuffle(train_membs)
val_membs = np.array_split(train_membs, 7)


# -------------------------------------------------------------------
# Select best hyperparameter set
# -------------------------------------------------------------------

csv_files = sorted(glob.glob(f"{outfolder[:-6]}/valid/*.csv"))

dfs = []

for f in csv_files:
    df = pd.read_csv(f)
    df["source"] = os.path.basename(f)
    dfs.append(df)

combined = pd.concat(dfs)
combined = combined.set_index("source")
combined.index = combined.index.str.replace(".csv", "", regex=False)

print(combined, flush=True)

combined["score_roc"] = 2 * combined["val_rocauc"] - combined["train_rocauc"]
combined["score_prg"] = 2 * combined["val_prgauc"] - combined["train_prgauc"]
combined["score_both"] = (combined["score_roc"] + combined["score_prg"]) / 2
best_roc = combined["score_roc"].idxmax()
best_prg = combined["score_prg"].idxmax()
best_both = combined["score_both"].idxmax()
print(best_roc, flush=True)
print(best_prg, flush=True)
print(best_both, flush=True)

tag = best_both[8:]
lam = int(best_both.split("_")[3][3:])
n_splines = int(best_both.split("_")[4][4:])


# -------------------------------------------------------------------
# Evaluate on test set
# -------------------------------------------------------------------

test_mask = TABLE_FULL['member_id'].isin(test_membs)
feature_cols = TABLE_FULL.columns[13:]
target_col = 'DURATION_bin'

X_train_full = TABLE_FULL.loc[~test_mask, feature_cols]
y_train_full = TABLE_FULL.loc[~test_mask, target_col]

X_test = TABLE_FULL.loc[test_mask, feature_cols]
y_test = TABLE_FULL.loc[test_mask, target_col]

scaler = StandardScaler()
scaler.fit(X_train_full)
X_train_full_scaled = scaler.transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

n_pos = (y_train_full == 1).sum()
n_neg = (y_train_full == 0).sum()
raw_pos_weight = n_neg / n_pos
pos_weight = np.sqrt(raw_pos_weight)
sample_weight = np.where(y_train_full == 1, pos_weight, 1.0)

n_features = X_train_full_scaled.shape[1]
terms = reduce(add, [s(i, n_splines=n_splines) for i in range(n_features)])

# GAM model
model = LogisticGAM(
    terms,
    lam=lam,
    fit_intercept=True,
    max_iter=2000
)

model.fit(X_train_full_scaled, y_train_full, weights=sample_weight)

y_test_proba = model.predict_mu(X_test_scaled)


# -------------------------------------------------------------------
# Save predictions
# -------------------------------------------------------------------

df_out = pd.DataFrame({
    "y_true": y_test,
    "y_proba": y_test_proba
})

df_out.to_csv(f"{outfolder}/{tag}.csv", index=False)

print("Done.", flush=True)
