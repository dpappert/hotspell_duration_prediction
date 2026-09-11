#!/usr/bin/env mamba-new
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
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
midpath = 'MLPb/valid/'

f = 2#int(os.environ["F"])
c = 4#int(os.environ["C"])

folders = [
    'targets_cl1_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl2_HWs_m50_18512010_fmstat_1.5sd_r1',
    'targets_cl4_HWs_m50_18512010_fmstat_1.5sd_r1'
]

# -------------------- LOAD DATA --------------------
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
    idx = np.argsort(recG_plot)
    return np.trapz(precG_plot[idx], recG_plot[idx])

def predict_proba(model, loader):
    model.eval()
    probs = []

    with torch.no_grad():
        for Xb, _ in loader:
            logits = model(Xb)
            p = torch.sigmoid(logits).squeeze().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()

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
results_dir = path + 'MLP/scores/' + folders[f] + '/' + midpath
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

    train_losses = []
    val_losses = []
    best_epochs = []

    for k in range(7):

        # deterministic DataLoader
        gen = torch.Generator()
        gen.manual_seed(SEED)

        train_mask = ~TABLE_FULL['member_id'].isin(np.concatenate([val_membs[k], test_membs]))
        X_train = TABLE_FULL[train_mask].iloc[:, 13:]
        y_train = TABLE_FULL[train_mask].iloc[:, 3]

        val_mask = TABLE_FULL['member_id'].isin(val_membs[k])
        X_val = TABLE_FULL[val_mask].iloc[:, 13:]
        y_val = TABLE_FULL[val_mask].iloc[:, 3]

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

        drop_last_train = (len(X_train_t) % batch_size == 1)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                                 batch_size=batch_size, shuffle=True, generator=gen,
                                 drop_last=drop_last_train)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                               batch_size=batch_size, shuffle=False)

        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        if crit_name == 'BCE':
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else:
            criterion = FocalLoss(alpha=1-(y_train == 1).mean(), gamma=2)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        patience = 50
        epochs_since_improvement = 0
        best_val_loss = float('inf')
        best_state = None
        best_epoch = 0

        all_train_losses = []
        all_val_losses = []

        for epoch in range(max_epochs):

            model.train()
            total_train_loss = 0

            for Xb, yb in train_loader:
                optimizer.zero_grad()
                logits_train = model(Xb)
                loss = criterion(logits_train, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train_loss += loss.item()

            train_loss = total_train_loss / len(train_loader)

            if epoch % 10 == 0:
                print(f"[epoch {epoch}] last-batch logits mean={logits_train.mean().item():.4f}, std={logits_train.std().item():.4f}", flush=True)

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    total_val_loss += criterion(model(Xb), yb).item()

            val_loss = total_val_loss / len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # --- Early stopping logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}", flush=True)

            if epochs_since_improvement >= patience:
                print(f"Stopping early at epoch {epoch} (no improvement for {patience} epochs)")
                break

        model.load_state_dict(best_state)

        y_train_proba = predict_proba(model, train_loader)
        y_val_proba = predict_proba(model, val_loader)

        y_train_true_used = []
        for _, yb in train_loader:
            y_train_true_used.append(yb.numpy())
        y_train_true_used = np.concatenate(y_train_true_used)

        SERIES_y_train_true.append(y_train_true_used)
        SERIES_y_train_proba.append(y_train_proba)
        SERIES_y_val_true.append(y_val.values)
        SERIES_y_val_proba.append(y_val_proba)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        best_epochs.append(best_epoch)

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

    epochs_df = pd.DataFrame({
        'best_epoch': best_epochs
    })

    with pd.ExcelWriter(file_path) as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        losses_df.to_excel(writer, sheet_name="losses", index=False)
        epochs_df.to_excel(writer, sheet_name="epochs", index=False)

    print(f"Saved results.", flush=True)
