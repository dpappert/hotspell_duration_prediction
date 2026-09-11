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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

SEED = 0
np.random.seed(SEED)

path = '/scratch3/dpappert/CESM2/ML/'
midpath = 'eval/'

f = int(os.environ["F"])
c = int(os.environ["C"])

C = 1
penalty = 'l2'
solver = 'liblinear'

folders = [
    'targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1'
]

outfolder = (
    f"{path}"
    f"LR/scores/{folders[f]}/"
    f"{midpath}"
)
os.makedirs(outfolder, exist_ok=True)

tag = f"F{f}_C{c}_reg{C}_pen{penalty}_solv{solver}"


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
# Evaluate on test (for LR params are fixed either way..)
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

#w_pos = len(y_train_full) / (2 * y_train_full.sum())
#w_neg = len(y_train_full) / (2 * (len(y_train_full) - y_train_full.sum()))

model = LogisticRegression(
    C=C,
    penalty=penalty,
    solver=solver,
    #class_weight={0: 1.0, 1: w_pos},
    max_iter=2000,
    random_state=SEED,
    n_jobs=1
)
model.fit(X_train_full_scaled, y_train_full)

y_test_proba = model.predict_proba(X_test_scaled)[:, 1]


# -------------------------------------------------------------------
# Save predictions
# -------------------------------------------------------------------

df_out = pd.DataFrame({
    "y_true": y_test,
    "y_proba": y_test_proba
})

df_out.to_csv(f"{outfolder}/{tag}.csv", index=False)

print("Done.", flush=True)
