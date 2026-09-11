#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import time
import random
import xarray as xr
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Definitions
# -------------------------------------------------------------------

class LogReg2D(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()
        self.linear = nn.Linear(in_channels * height * width, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

class MLP2D(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels * height * width, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

class CNNlin(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class CNNa(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class CNNb(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=True),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 16, 3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        # Nonlinear MLP head (same size as MLP2D)
        self.classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class CNNc(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.AdaptiveAvgPool2d(1)
        )
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)

class CNNd(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Depthwise separable conv block
        def dw_sep(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        self.features = nn.Sequential(
            dw_sep(in_channels, 8),
            nn.MaxPool2d(2),

            dw_sep(8, 16),
            nn.MaxPool2d(2),

            dw_sep(16, 32),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

MODEL_FACTORY = {
    "LogReg2D": lambda X: LogReg2D(
        in_channels=X.shape[1],
        height=X.shape[2],
        width=X.shape[3],
    ),
    "MLP2D": lambda X: MLP2D(
        in_channels=X.shape[1],
        height=X.shape[2],
        width=X.shape[3],
    ),
    "CNNlin": lambda X: CNNlin(
        in_channels=X.shape[1],
    ),
    "CNNa": lambda X: CNNa(
        in_channels=X.shape[1],
    ),
    "CNNb": lambda X: CNNb(
        in_channels=X.shape[1],
    ),
    "CNNc": lambda X: CNNc(
        in_channels=X.shape[1],
    ),
    "CNNd": lambda X: CNNd(
        in_channels=X.shape[1],
    ),
}


SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(3)
torch.set_num_interop_threads(3)
torch.use_deterministic_algorithms(True)

model_name = os.environ["M"]

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
    f"/CNN/scores/{folders[f]}/"
    f"{midpath}"
)
os.makedirs(outfolder, exist_ok=True)


# -------------------------------------------------------------------
# Load initial data
# -------------------------------------------------------------------

TABLE_init = pd.read_csv(path + f"/targets/{folders[f]}.txt", sep='\t')
member_ids = np.unique(TABLE_init['member_id'])
file_list = sorted(glob.glob(f"{path}/tab_preds/{folders[f]}/*.txt"))
preds_ = [pd.read_csv(file, sep='\t') for file in file_list]
TABLE_FULL = pd.concat([TABLE_init, pd.concat(preds_, axis=1)], axis=1)
# Remove duration==7 and add binary label
ind7 = TABLE_FULL.index[TABLE_FULL['DURATION'] == 7].tolist()
TABLE_FULL = TABLE_FULL[TABLE_FULL['DURATION'] != 7]
TABLE_FULL.insert(3, 'DURATION_bin', (TABLE_FULL['DURATION'] > 7).astype(int))
# Mask rows without inf
mask = ~np.isinf(TABLE_FULL.select_dtypes(include=[np.number])).any(axis=1)
TABLE_FULL = TABLE_FULL[mask]

# Gridded cnn data
file_list = sorted(glob.glob(f"/scratch3/dpappert/CESM2/ML/CNN/input_per_member/3x3/cl{c}/r*.nc"))
datasets = [xr.open_dataarray(f) for f in file_list]
X = xr.concat(datasets, dim="event")
X = X.where(X != -np.inf, np.nan)

# Compute per-channel mean and std, ignoring nan
means = []
stds = []
for j in range(X.shape[1]):
    vals = X[:, j, :, :]
    means.append(vals.mean(skipna=True).item())
    stds.append(vals.std(skipna=True).item())
# Normalise per channel
for j in range(X.shape[1]):
    X[:, j, :, :] = (X[:, j, :, :] - means[j]) / stds[j]

X = X[~np.isin(np.arange(X.shape[0]), ind7), :, :, :]
X = X[mask, :, :, :]

# Convert both to the same type (e.g., pandas datetime)
x_dates = pd.to_datetime(X.start_date.values)
table_dates = pd.to_datetime(TABLE_FULL.start_date.values)
# Check if all values match
alignment_check = np.array_equal(x_dates, table_dates)
print(f"Row alignment check: {alignment_check}", flush=True)


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

# --- fold data prep
train_mask = ~TABLE_FULL['member_id'].isin(np.concatenate([val_membs[3], test_membs]))
val_mask = TABLE_FULL['member_id'].isin(val_membs[3])
test_mask = TABLE_FULL['member_id'].isin(test_membs)
y_train = TABLE_FULL[train_mask].iloc[:, 3].values
y_val = TABLE_FULL[val_mask].iloc[:, 3].values
y_test = TABLE_FULL[test_mask].iloc[:, 3].values

# grid data (must be N, C, H, W)
X_grid_train_t = torch.tensor(X[train_mask.values, :, :, :].values, dtype=torch.float32)
X_grid_val_t   = torch.tensor(X[val_mask.values, :, :, :].values, dtype=torch.float32)
X_grid_test_t   = torch.tensor(X[test_mask.values, :, :, :].values, dtype=torch.float32)

# labels
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test_t   = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# class weights
classes, counts = torch.unique(y_train_t, return_counts=True)
weights = 1.0 / counts.float()
weights = weights / weights.sum() * len(classes)
pos_weight = weights[1]

# criterion
if params["criterion"] == 'BCE':
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
else:
    criterion = FocalLoss(alpha=1-(y_train == 1).mean(), gamma=2)

# model
if model_name not in MODEL_FACTORY:
    raise ValueError(
        f"Unknown model '{model_name}'. "
        f"Choose from {list(MODEL_FACTORY.keys())}"
    )

model = MODEL_FACTORY[model_name](X)

print(f"Using model: {model_name}", flush=True)
print(model, flush=True)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

train_dataset = TensorDataset(X_grid_train_t, y_train_t)
val_dataset   = TensorDataset(X_grid_val_t, y_val_t)
test_dataset   = TensorDataset(X_grid_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, generator=gen)
val_loader   = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False, generator=gen)
test_loader   = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, generator=gen)

patience = 50
epochs_since_improvement = 0
best_val_loss = float('inf')
best_model_state = None
best_epoch = 0

train_losses = []
val_losses = []

# --- training loop ---
for epoch in range(params["max_epochs"]):

    # --- TRAIN ---
    model.train()
    total_train_loss = 0

    for Xb, yb in train_loader:
        optimizer.zero_grad()

        logits_train = model(Xb)
        loss_train = criterion(logits_train, yb)

        loss_train.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_train_loss += loss_train.item()

    train_loss_epoch = total_train_loss / len(train_loader)
    train_losses.append(train_loss_epoch)

    # --- DEBUG (use last batch just for stats, or track separately) ---
    if epoch % 10 == 0:
        print(f"[epoch {epoch}] last-batch logits mean={logits_train.mean().item():.4f}, std={logits_train.std().item():.4f}", flush=True)

    # --- VALIDATION ---
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for Xb, yb in val_loader:
            logits_val = model(Xb)
            loss_val = criterion(logits_val, yb)
            total_val_loss += loss_val.item()

    val_loss_epoch = total_val_loss / len(val_loader)
    val_losses.append(val_loss_epoch)

    # --- check improvement ---
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_since_improvement = 0
    else:
        epochs_since_improvement += 1

    print(f"Epoch {epoch}: train={train_loss_epoch:.4f}, val={val_loss_epoch:.4f}", flush=True)

    # --- early stopping ---
    if epochs_since_improvement >= patience:
        print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
        break

# --- restore best model ---
if best_model_state is not None:
    model.load_state_dict(best_model_state)

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
