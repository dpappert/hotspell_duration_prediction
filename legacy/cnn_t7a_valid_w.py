#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import time
import random
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# -------------------- DETERMINISM --------------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(3)
torch.set_num_interop_threads(3)
torch.use_deterministic_algorithms(True)

# -------------------- PATHS --------------------
path = '/scratch3/dpappert/CESM2/ML/'
midpath = 'CNNa/valid/'

f = 1#int(os.environ["F"])
c = 2#int(os.environ["C"])

folders = [
    'targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1'
]

# -------------------- LOAD DATA --------------------
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

# -------------------- FUNCTIONS --------------------
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

def predict_proba(model, loader):
    model.eval()
    probs = []

    with torch.no_grad():
        for Xb, _ in loader:
            logits = model(Xb)
            p = torch.sigmoid(logits).squeeze().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs)

def compute_roc_auc(model, loader):
    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            probs = torch.sigmoid(logits).view(-1).numpy()
            y_prob_all.extend(probs)
            y_true_all.extend(y.numpy())

    return roc_auc_score(y_true_all, y_prob_all)

def compute_prg_auc(model, loader):
    y_true_all = []
    y_prob_all = []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            probs = torch.sigmoid(logits).view(-1).numpy()
            y_prob_all.extend(probs)
            y_true_all.extend(y.numpy())

    return compute_auprgc(np.concatenate(y_true_all), np.array(y_prob_all))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # ensure correct shape
        targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()

        # BCE with logits (no reduction!)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # compute pt = p_t
        pt = torch.exp(-bce_loss)

        # focal loss
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        # reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# -------------------- SPLITS --------------------
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

# -------------------- HYPERPARAMS --------------------
param_grid = {
    'max_epochs': [500],
    'batch_size': [32, 64, 128],
    'lr': [0.1, 0.01, 0.001, 0.0001], 
    'weight_decay': [1e-3, 3e-4],
    'criterion': ['BCE', 'Focal-g2'],
    'optimizer': ['Adam']
}

# -------------------- SINGLE RUN --------------------
results_dir = path + 'CNN/scores/' + folders[f] + '/' + midpath
os.makedirs(results_dir, exist_ok=True)
print(f'Made dir at {results_dir}', flush=True)

# -------------------- TRAINING --------------------
for params in itertools.product(*param_grid.values()):
    max_epochs, batch_size, lr, wd, crit_name, opt_name = params
    param_id = f"HPS_mpe{max_epochs}-pat50_bs{batch_size}_lr{lr}_wd{wd}_crit{crit_name}_opt{opt_name}"
    file_path = os.path.join(results_dir, f"{param_id}.xlsx")

    # Skip if already done
    if os.path.exists(file_path):
        print(f"Skipping {param_id} (already exists)", flush=True)
        continue

    print(f"Running {param_id}", flush=True)

    SERIES_y_val_true = []
    SERIES_y_val_proba = []
    SERIES_y_train_true = []
    SERIES_y_train_proba = []

    all_train_losses = []
    all_val_losses = []

    for k in range(7):

        # deterministic DataLoader
        gen = torch.Generator()
        gen.manual_seed(SEED)

        # --- fold data prep
        train_mask = ~TABLE_FULL['member_id'].isin(np.concatenate([val_membs[k], test_membs]))
        val_mask = TABLE_FULL['member_id'].isin(val_membs[k])
        y_train = TABLE_FULL[train_mask].iloc[:, 3].values
        y_val = TABLE_FULL[val_mask].iloc[:, 3].values

        # grid data (must be N, C, H, W)
        X_grid_train_t = torch.tensor(X[train_mask.values, :, :, :].values, dtype=torch.float32)
        X_grid_val_t   = torch.tensor(X[val_mask.values, :, :, :].values, dtype=torch.float32)
        
        # labels
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_t   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        # class weights
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

        # criterion
        if crit_name == 'BCE':
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif crit_name == 'Focal-g2':
            criterion = FocalLoss(alpha=1-(y_train == 1).mean(), gamma=2)

        model = CNNa(in_channels=X.shape[1])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        train_dataset = TensorDataset(X_grid_train_t, y_train_t)
        val_dataset   = TensorDataset(X_grid_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=gen)

        # --- training setup ---
        patience = 50
        epochs_since_improvement = 0
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        train_losses = []
        val_losses = []

        # --- training loop ---
        for epoch in range(max_epochs):
        
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
        
            train_loss = total_train_loss / len(train_loader)
            train_losses.append(train_loss)
        
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
        
            val_loss = total_val_loss / len(val_loader)
            val_losses.append(val_loss)
        
            # --- check improvement ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
        
            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}", flush=True)
        
            # --- early stopping ---
            if epochs_since_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
                break
        
        model.load_state_dict(best_state)
        
        y_train_proba = predict_proba(model, train_loader)
        y_val_proba = predict_proba(model, val_loader)

        SERIES_y_train_true.append(y_train)
        SERIES_y_train_proba.append(y_train_proba)
        SERIES_y_val_true.append(y_val)
        SERIES_y_val_proba.append(y_val_proba)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

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

    # -------- SAVE EXCEL --------
    metrics_df = pd.DataFrame({
        'seed': [SEED],
        'val_rocauc': [val_rocauc],
        'train_rocauc': [train_rocauc],
        'val_prgauc': [val_prgauc],
        'train_prgauc': [train_prgauc]
    })

    losses_df = pd.DataFrame({
        'train_losses': all_train_losses,
        'val_losses': all_val_losses
    })

    with pd.ExcelWriter(file_path) as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        losses_df.to_excel(writer, sheet_name="losses", index=False)

    print(f"Saved results.", flush=True)
