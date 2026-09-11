#!/usr/bin/env python3
"""
Validation (hyperparameter sweep) entry point for the tabular MLP models.
Replaces `mlp_t7{a,b,c,lin}_valid.py`.

Usage
-----
    python scripts/validate_mlp.py --arch a --region sw
    python scripts/validate_mlp.py --arch b --region n

Note: in the originals, MLPa was only ever run on region "sw" and
MLPb/MLPc/MLPlin only ever on region "n" -- those are the combinations with
existing results to diff against for Step 8 verification. This CLI accepts
any arch x region combination for future use.
"""
import argparse
import copy
import itertools
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hwml import data, splits as splits_mod, metrics as metrics_mod
from hwml.losses import FocalLoss
from hwml.models.mlp import MLP_REGISTRY, build_model
from hwml.paths import RESULTS_DIR

warnings.filterwarnings("ignore")

# Identical across all 4 original scripts.
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
    p.add_argument("--arch", required=True, choices=list(MLP_REGISTRY))
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-threads", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    SEED = args.seed

    np.random.seed(SEED)
    import random
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.n_threads)
    torch.set_num_interop_threads(args.n_threads)
    torch.use_deterministic_algorithms(True)

    region_index, _ = data.REGION_SUFFIX_TO_FC[args.region]
    _, results_subdir, original_fc = MLP_REGISTRY[args.arch]
    if (region_index, data.REGION_SUFFIX_TO_FC[args.region][1]) != original_fc:
        print(f"NOTE: original mlp_t7{args.arch}_valid.py was only ever run with "
              f"(f,c)={original_fc}; you're running region='{args.region}' -> "
              f"(f,c)=({region_index}, {data.REGION_SUFFIX_TO_FC[args.region][1]}). "
              f"No original results exist for this combo to compare against.", flush=True)

    # -------------------- LOAD DATA --------------------
    table_full, member_ids = data.load_table(region_index)  # drop_na=True default, matches MLP originals

    # -------------------- SPLITS --------------------
    test_membs, val_membs = splits_mod.generate_member_splits(member_ids, seed=SEED)

    # -------------------- OUTPUT DIR --------------------
    region_folder = data.REGION_FOLDERS[region_index]
    results_dir = RESULTS_DIR / "MLP" / "scores" / region_folder / results_subdir / "valid"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Made dir at {results_dir}", flush=True)

    def predict_proba(model, loader):
        model.eval()
        probs = []
        with torch.no_grad():
            for Xb, _ in loader:
                probs.append(torch.sigmoid(model(Xb)).squeeze().cpu().numpy())
        return np.concatenate(probs)

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
        train_losses_all_folds, val_losses_all_folds = [], []
        best_epochs = []

        for k in range(7):
            gen = torch.Generator()
            gen.manual_seed(SEED)

            train_mask = ~table_full["member_id"].isin(np.concatenate([val_membs[k], test_membs]))
            X_train = table_full[train_mask].iloc[:, 13:]
            y_train = table_full[train_mask].iloc[:, 3]

            val_mask = table_full["member_id"].isin(val_membs[k])
            X_val = table_full[val_mask].iloc[:, 13:]
            y_val = table_full[val_mask].iloc[:, 3]

            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

            X_train_t = torch.tensor(X_train, dtype=torch.float32)
            y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

            # avoids a stray singleton final batch (e.g. BatchNorm1d in MLPc chokes on batch size 1)
            drop_last_train = (len(X_train_t) % batch_size == 1)

            train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size,
                                       shuffle=True, generator=gen, drop_last=drop_last_train)
            val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

            model = build_model(args.arch, in_features=X_train.shape[1])

            pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
            if crit_name == "BCE":
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
            else:
                criterion = FocalLoss(alpha=1 - (y_train == 1).mean(), gamma=2)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            patience = 50
            epochs_since_improvement = 0
            best_val_loss = float("inf")
            best_state = None
            best_epoch = 0
            fold_train_losses, fold_val_losses = [], []

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
                    print(f"[epoch {epoch}] last-batch logits mean={logits_train.mean().item():.4f}, "
                          f"std={logits_train.std().item():.4f}", flush=True)

                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for Xb, yb in val_loader:
                        total_val_loss += criterion(model(Xb), yb).item()
                val_loss = total_val_loss / len(val_loader)

                fold_train_losses.append(train_loss)
                fold_val_losses.append(val_loss)

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

            # NOTE: true labels for train are re-extracted from train_loader
            # (not the original y_train array) to stay aligned with
            # drop_last_train possibly dropping a remainder sample -- this
            # matches the original scripts exactly, not a simplification.
            y_train_true_used = []
            for _, yb in train_loader:
                y_train_true_used.append(yb.numpy())
            y_train_true_used = np.concatenate(y_train_true_used)

            SERIES_y_train_true.append(y_train_true_used)
            SERIES_y_train_proba.append(y_train_proba)
            SERIES_y_val_true.append(y_val.values)
            SERIES_y_val_proba.append(y_val_proba)

            train_losses_all_folds.append(fold_train_losses)
            val_losses_all_folds.append(fold_val_losses)
            best_epochs.append(best_epoch)

        val_m = metrics_mod.compute_metrics(np.concatenate(SERIES_y_val_true), np.concatenate(SERIES_y_val_proba))
        train_m = metrics_mod.compute_metrics(np.concatenate(SERIES_y_train_true), np.concatenate(SERIES_y_train_proba))

        metrics_df = pd.DataFrame({
            "seed": [SEED],
            "val_rocauc": [val_m["rocauc"]],
            "train_rocauc": [train_m["rocauc"]],
            "val_prgauc": [val_m["prgauc"]],
            "train_prgauc": [train_m["prgauc"]],
        })
        losses_df = pd.DataFrame({"train_losses": train_losses_all_folds, "val_losses": val_losses_all_folds})
        epochs_df = pd.DataFrame({"best_epoch": best_epochs})

        with pd.ExcelWriter(file_path) as writer:
            metrics_df.to_excel(writer, sheet_name="metrics", index=False)
            losses_df.to_excel(writer, sheet_name="losses", index=False)
            epochs_df.to_excel(writer, sheet_name="epochs", index=False)

        print("Saved results.", flush=True)


if __name__ == "__main__":
    main()
