#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import time
import random
#from functools import reduce
#from operator import add
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

def build_model(model_name, input_dim):

    if model_name == "MLPlin":
        return nn.Sequential(
            nn.Linear(input_dim, 1)
        )

    elif model_name == "MLPa":
        return nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    elif model_name == "MLPb":
        return nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    elif model_name == "MLPc":
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    raise ValueError(f"Unknown model '{model_name}'")


SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(3)
torch.set_num_interop_threads(3)
torch.use_deterministic_algorithms(True)

models = ["MLPlin", "MLPa", "MLPb", "MLPc"]
m = int(os.environ["M"])
model_name = models[m]
model = 'MLP'
path = '/scratch3/dpappert/CESM2/ML/'
midpath = f'{model_name}/eval/'

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

csv_files = sorted(glob.glob(f"{outfolder[:-6]}/valid/*.xlsx"))

dfs = []

for f in csv_files:
    df = pd.read_excel(f)
    df["source"] = os.path.basename(f)
    dfs.append(df)

combined = pd.concat(dfs)
combined = combined.set_index("source")
combined.index = combined.index.str.replace(".xlsx", "", regex=False)
combined = combined[~combined.index.str.contains("Focal")]

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

tag = best_both[4:]

parts = best_both.split("_")
params = {}
for p in parts:
    if p.startswith("mpe"):
        params["max_epochs"] = int(p.replace("mpe", "").split("-")[0])
        params["patience"] = int(p.split("-pat")[1])
    elif p.startswith("bs"):
        params["batch_size"] = int(p.replace("bs", ""))
    elif p.startswith("lr"):
        params["lr"] = float(p.replace("lr", ""))
    elif p.startswith("wd"):
        params["weight_decay"] = float(p.replace("wd", ""))
    elif p.startswith("crit"):
        params["criterion"] = p.replace("crit", "")
    elif p.startswith("opt"):
        params["optimizer"] = p.replace("opt", "")


# -------------------------------------------------------------------
# Evaluate on test set
# -------------------------------------------------------------------

# deterministic DataLoader
gen = torch.Generator()
gen.manual_seed(SEED)

val_mask = TABLE_FULL['member_id'].isin(val_membs[3])
test_mask = TABLE_FULL['member_id'].isin(test_membs)
train_mask = ~TABLE_FULL['member_id'].isin(np.concatenate([val_membs[3], test_membs]))

X_train = TABLE_FULL[train_mask].iloc[:, 13:]
y_train = TABLE_FULL[train_mask].iloc[:, 3]
X_val = TABLE_FULL[val_mask].iloc[:, 13:]
y_val = TABLE_FULL[val_mask].iloc[:, 3]
X_test = TABLE_FULL[test_mask].iloc[:, 13:]
y_test = TABLE_FULL[test_mask].iloc[:, 3]

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
test_ds  = TensorDataset(X_test_t, y_test_t)
test_loader  = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=False, generator=gen)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                         batch_size=params["batch_size"], shuffle=True, generator=gen)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                       batch_size=params["batch_size"], shuffle=False, generator=gen)

model = build_model(model_name, X_train.shape[1])

pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

if params["criterion"] == 'BCE':
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
else:
    criterion = FocalLoss(alpha=1-(y_train == 1).mean(), gamma=2)

optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

patience = 50
epochs_since_improvement = 0
best_val_loss = float('inf')
best_state = None
best_epoch = 0

train_losses = []
val_losses = []

for epoch in range(params["max_epochs"]):

    model.train()
    total_train_loss = 0

    for Xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            total_val_loss += criterion(model(Xb), yb).item()

    val_loss = total_val_loss / len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}", flush=True)

    # --- early stopping ---
    if epochs_since_improvement >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
        break

# --- restore best model ---
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Best model restored from epoch {best_epoch}, val loss={best_val_loss:.4f}", flush=True)

model.eval()
y_pred_list = []

with torch.no_grad():
    for Xb, _ in test_loader:
        logits = model(Xb)
        probs = torch.sigmoid(logits)
        y_pred_list.append(probs.cpu().numpy())

y_pred = np.vstack(y_pred_list).ravel()


# -------------------------------------------------------------------
# Save predictions
# -------------------------------------------------------------------

df_out = pd.DataFrame({
    "y_true": y_test,
    "y_proba": y_pred
})

df_out.to_csv(f"{outfolder}/{tag}.csv", index=False)

print("Done.", flush=True)
