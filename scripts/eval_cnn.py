#!/usr/bin/env python3
"""
Test-set evaluation for the CNN / 2D-input models: auto-selects the best
hyperparameter combo from validate_cnn.py's results (excluding Focal-loss
combos, same deliberate exclusion as MLP eval), retrains with an
early-stopping validation split, evaluates once on the held-out test set.
Replaces `eval_all2D.py`.

BUG FIX vs. the original (same as eval_mlp.py): the early-stopping
validation fold (`val_membs[3]`) now comes from the same seeded
`generate_member_splits` used everywhere else, instead of an unseeded
`np.random.shuffle` that made this fold -- and the final reported test
metrics -- different and non-reproducible on every run.

Usage
-----
    python scripts/eval_cnn.py --arch a --region sw
"""
import argparse
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hwml import data, splits as splits_mod, model_selection
from hwml.losses import FocalLoss
from hwml.models.cnn import CNN_REGISTRY, build_model
from hwml.paths import RESULTS_DIR

SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=list(CNN_REGISTRY))
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    p.add_argument("--n-threads", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.n_threads)
    torch.set_num_interop_threads(args.n_threads)
    torch.use_deterministic_algorithms(True)

    region_index, region_cl = data.REGION_SUFFIX_TO_FC[args.region]
    _, _, results_subdir = CNN_REGISTRY[args.arch]
    region_folder = data.REGION_FOLDERS[region_index]

    valid_dir = RESULTS_DIR / "CNN" / "scores" / region_folder / results_subdir / "valid"
    eval_dir = RESULTS_DIR / "CNN" / "scores" / region_folder / results_subdir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- SELECT BEST HYPERPARAMETERS --------------------
    xlsx_files = sorted(valid_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No validation results found in {valid_dir}; run validate_cnn.py first.")

    dfs = []
    for f in xlsx_files:
        df = pd.read_excel(f, sheet_name="metrics")
        df["source"] = f.stem
        dfs.append(df)
    combined = pd.concat(dfs).set_index("source")

    best_tag, scored = model_selection.select_best_combo(combined, exclude_pattern="Focal")
    print(scored, flush=True)
    print(f"Best combo: {best_tag}", flush=True)
    hp = model_selection.parse_nn_tag(best_tag)

    # -------------------- LOAD DATA --------------------
    table_full, member_ids, keep_mask = data.load_table_for_cnn(region_index)
    X = data.load_cnn_grid(region_cl)
    X = X[keep_mask, :, :, :]

    aligned = data.check_row_alignment(table_full, X)
    print(f"Row alignment check: {aligned}", flush=True)

    in_channels, height, width = X.shape[1], X.shape[2], X.shape[3]

    test_membs, val_membs = splits_mod.generate_member_splits(member_ids, seed=SEED)

    gen = torch.Generator()
    gen.manual_seed(SEED)

    train_mask = ~table_full["member_id"].isin(np.concatenate([val_membs[3], test_membs]))
    val_mask = table_full["member_id"].isin(val_membs[3])
    test_mask = table_full["member_id"].isin(test_membs)

    y_train = table_full[train_mask].iloc[:, 3].values
    y_val = table_full[val_mask].iloc[:, 3].values
    y_test = table_full[test_mask].iloc[:, 3].values

    X_grid_train_t = torch.tensor(X[train_mask.values, :, :, :].values, dtype=torch.float32)
    X_grid_val_t = torch.tensor(X[val_mask.values, :, :, :].values, dtype=torch.float32)
    X_grid_test_t = torch.tensor(X[test_mask.values, :, :, :].values, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    classes, counts = torch.unique(y_train_t, return_counts=True)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(classes)
    pos_weight = weights[1]

    if hp["criterion"] == "BCE":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        criterion = FocalLoss(alpha=1 - (y_train == 1).mean(), gamma=2)

    model = build_model(args.arch, in_channels=in_channels, height=height, width=width)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    train_loader = DataLoader(TensorDataset(X_grid_train_t, y_train_t), batch_size=hp["batch_size"],
                               shuffle=True, generator=gen)
    val_loader = DataLoader(TensorDataset(X_grid_val_t, y_val_t), batch_size=hp["batch_size"],
                             shuffle=False, generator=gen)
    test_loader = DataLoader(TensorDataset(X_grid_test_t, y_test_t), batch_size=hp["batch_size"],
                              shuffle=False, generator=gen)

    patience = 50
    epochs_since_improvement = 0
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    for epoch in range(hp["max_epochs"]):
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
        train_loss_epoch = total_train_loss / len(train_loader)

        if epoch % 10 == 0:
            print(f"[epoch {epoch}] last-batch logits mean={logits_train.mean().item():.4f}, "
                  f"std={logits_train.std().item():.4f}", flush=True)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                loss_val = criterion(model(Xb), yb)
                total_val_loss += loss_val.item()
        val_loss_epoch = total_val_loss / len(val_loader)

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print(f"Epoch {epoch}: train={train_loss_epoch:.4f}, val={val_loss_epoch:.4f}", flush=True)

        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            probs = torch.sigmoid(model(Xb))
            y_pred_list.append(probs.cpu().numpy())
    y_pred = np.vstack(y_pred_list).ravel()

    df_out = pd.DataFrame({"y_true": y_test, "y_proba": y_pred})
    out_path = eval_dir / f"{best_tag}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
