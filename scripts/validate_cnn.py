#!/usr/bin/env python3
"""
Validation (hyperparameter sweep) entry point for the CNN / 2D-input
models. Replaces the 21 scripts `cnn_t7{a,b,c,d,lin,lr,mlp}_valid_{n,w,sw}.py`.

Usage
-----
    python scripts/validate_cnn.py --arch a --region w
    python scripts/validate_cnn.py --arch lr --region sw

Each (arch, region) combination reproduces exactly one of the original 21
scripts. The training loop, hyperparameter grid, split logic, and output
format are all unchanged from the originals -- only the architecture
selection and (f, c) region lookup are now parameterized instead of copy-
pasted per file.

One deliberate fix vs. the originals: `cnn_t7d_valid_sw.py` alone wrote
results to `nh/CNNd/valid/` instead of `CNNd/valid/` (every other
arch/region combination used the consistent path). This script always uses
the consistent path. If that `nh/` prefix was meaningful (e.g. marking a
distinct run), flag it before relying on this for the `d`/`sw` combination.
"""
import argparse
import copy
import itertools
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hwml import data, splits as splits_mod, metrics as metrics_mod
from hwml.losses import FocalLoss
from hwml.models.cnn import CNN_REGISTRY, build_model
from hwml.paths import RESULTS_DIR

warnings.filterwarnings("ignore")

# Identical across all 21 original scripts.
PARAM_GRID = {
    "max_epochs": [500],
    "batch_size": [32, 64, 128],
    "lr": [0.1, 0.01, 0.001, 0.0001],
    "weight_decay": [1e-3, 3e-4],
    "criterion": ["BCE", "Focal-g2"],
    "optimizer": ["Adam"],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=list(CNN_REGISTRY), help="Which CNN architecture to validate.")
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC),
                   help="Region suffix from the original filenames (sw/w/n).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-threads", type=int, default=3, help="torch.set_num_threads / interop threads (original default: 3).")
    return p.parse_args()


def main():
    args = parse_args()
    SEED = args.seed

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.n_threads)
    torch.set_num_interop_threads(args.n_threads)
    torch.use_deterministic_algorithms(True)

    region_index, region_cl = data.REGION_SUFFIX_TO_FC[args.region]
    _, _, results_subdir = CNN_REGISTRY[args.arch]

    # -------------------- LOAD DATA --------------------
    table_full, member_ids, keep_mask = data.load_table_for_cnn(region_index)
    X = data.load_cnn_grid(region_cl)
    X = X[keep_mask, :, :, :]

    aligned = data.check_row_alignment(table_full, X)
    print(f"Row alignment check: {aligned}", flush=True)

    in_channels = X.shape[1]
    height, width = X.shape[2], X.shape[3]

    # -------------------- SPLITS --------------------
    test_membs, val_membs = splits_mod.generate_member_splits(member_ids, seed=SEED)

    # -------------------- OUTPUT DIR --------------------
    region_folder = data.REGION_FOLDERS[region_index]
    results_dir = RESULTS_DIR / "CNN" / "scores" / region_folder / results_subdir / "valid"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Made dir at {results_dir}", flush=True)

    # -------------------- TRAINING --------------------
    for params in itertools.product(*PARAM_GRID.values()):
        max_epochs, batch_size, lr, wd, crit_name, opt_name = params
        param_id = f"HPS_mpe{max_epochs}-pat50_bs{batch_size}_lr{lr}_wd{wd}_crit{crit_name}_opt{opt_name}"
        file_path = results_dir / f"{param_id}.xlsx"

        if file_path.exists():
            print(f"Skipping {param_id} (already exists)", flush=True)
            continue

        print(f"Running {param_id}", flush=True)

        SERIES_y_val_true, SERIES_y_val_proba = [], []
        SERIES_y_train_true, SERIES_y_train_proba = [], []
        all_train_losses, all_val_losses = [], []

        for k in range(7):
            gen = torch.Generator()
            gen.manual_seed(SEED)

            train_mask = ~table_full["member_id"].isin(np.concatenate([val_membs[k], test_membs]))
            val_mask = table_full["member_id"].isin(val_membs[k])
            y_train = table_full[train_mask].iloc[:, 3].values
            y_val = table_full[val_mask].iloc[:, 3].values

            X_grid_train_t = torch.tensor(X[train_mask.values, :, :, :].values, dtype=torch.float32)
            X_grid_val_t = torch.tensor(X[val_mask.values, :, :, :].values, dtype=torch.float32)

            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)

            if crit_name == "BCE":
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            elif crit_name == "Focal-g2":
                criterion = FocalLoss(alpha=1 - (y_train == 1).mean(), gamma=2)

            model = build_model(args.arch, in_channels=in_channels, height=height, width=width)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            train_dataset = TensorDataset(X_grid_train_t, y_train_t)
            val_dataset = TensorDataset(X_grid_val_t, y_val_t)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=gen)

            patience = 50
            epochs_since_improvement = 0
            best_val_loss = float("inf")
            train_losses, val_losses = [], []

            for epoch in range(max_epochs):
                model.train()
                total_train_loss = 0
                for Xb, yb in train_loader:
                    optimizer.zero_grad()
                    logits_train = model(Xb)
                    loss_train = criterion(logits_train, yb)
                    loss_train.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_train_loss += loss_train.item()

                train_loss = total_train_loss / len(train_loader)
                train_losses.append(train_loss)

                if epoch % 10 == 0:
                    print(f"[epoch {epoch}] last-batch logits mean={logits_train.mean().item():.4f}, "
                          f"std={logits_train.std().item():.4f}", flush=True)

                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        loss_val = criterion(model(Xb), yb)
                        total_val_loss += loss_val.item()
                val_loss = total_val_loss / len(val_loader)
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}", flush=True)

                if epochs_since_improvement >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
                    break

            model.load_state_dict(best_state)

            def predict_proba(m, loader):
                m.eval()
                probs = []
                with torch.no_grad():
                    for Xb, _ in loader:
                        p = torch.sigmoid(m(Xb)).squeeze().cpu().numpy()
                        probs.append(p)
                return np.concatenate(probs)

            SERIES_y_train_true.append(y_train)
            SERIES_y_train_proba.append(predict_proba(model, train_loader))
            SERIES_y_val_true.append(y_val)
            SERIES_y_val_proba.append(predict_proba(model, val_loader))
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)

        val_m = metrics_mod.compute_metrics(np.concatenate(SERIES_y_val_true), np.concatenate(SERIES_y_val_proba))
        train_m = metrics_mod.compute_metrics(np.concatenate(SERIES_y_train_true), np.concatenate(SERIES_y_train_proba))

        metrics_df = pd.DataFrame({
            "seed": [SEED],
            "val_rocauc": [val_m["rocauc"]],
            "train_rocauc": [train_m["rocauc"]],
            "val_prgauc": [val_m["prgauc"]],
            "train_prgauc": [train_m["prgauc"]],
        })
        losses_df = pd.DataFrame({"train_losses": all_train_losses, "val_losses": all_val_losses})

        with pd.ExcelWriter(file_path) as writer:
            metrics_df.to_excel(writer, sheet_name="metrics", index=False)
            losses_df.to_excel(writer, sheet_name="losses", index=False)

        print("Saved results.", flush=True)


if __name__ == "__main__":
    main()
