#!/usr/bin/env python3
"""
Test-set evaluation for the tabular MLP models: auto-selects the best
hyperparameter combo from validate_mlp.py's results (excluding Focal-loss
combos -- a deliberate original design choice, kept), retrains with an
early-stopping validation split, evaluates once on the held-out test set.
Replaces `eval_MLP.py`.

BUG FIX vs. the original: the early-stopping validation fold (`val_membs[3]`)
is now drawn from the same seeded `generate_member_splits` used everywhere
else. The original used an unseeded `np.random.shuffle`, making this fold
-- and therefore the final reported test metrics -- different and
non-reproducible on every run. Confirmed fix, not silently applied.

Usage
-----
    python scripts/eval_mlp.py --arch a --region sw
"""
import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hwml import data, splits as splits_mod, model_selection
from hwml.losses import FocalLoss
from hwml.models.mlp import MLP_REGISTRY, build_model
from hwml.paths import RESULTS_DIR

SEED = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=list(MLP_REGISTRY))
    p.add_argument("--region", required=True, choices=list(data.REGION_SUFFIX_TO_FC))
    p.add_argument("--n-threads", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    import random
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.n_threads)
    torch.set_num_interop_threads(args.n_threads)
    torch.use_deterministic_algorithms(True)

    region_index, _ = data.REGION_SUFFIX_TO_FC[args.region]
    _, results_subdir, _ = MLP_REGISTRY[args.arch]
    region_folder = data.REGION_FOLDERS[region_index]

    valid_dir = RESULTS_DIR / "MLP" / "scores" / region_folder / results_subdir / "valid"
    eval_dir = RESULTS_DIR / "MLP" / "scores" / region_folder / results_subdir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- SELECT BEST HYPERPARAMETERS --------------------
    xlsx_files = sorted(valid_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No validation results found in {valid_dir}; run validate_mlp.py first.")

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
    table_full, member_ids = data.load_table(region_index)
    test_membs, val_membs = splits_mod.generate_member_splits(member_ids, seed=SEED)

    val_mask = table_full["member_id"].isin(val_membs[3])
    test_mask = table_full["member_id"].isin(test_membs)
    train_mask = ~table_full["member_id"].isin(np.concatenate([val_membs[3], test_membs]))

    X_train = table_full[train_mask].iloc[:, 13:]
    y_train = table_full[train_mask].iloc[:, 3]
    X_val = table_full[val_mask].iloc[:, 13:]
    y_val = table_full[val_mask].iloc[:, 3]
    X_test = table_full[test_mask].iloc[:, 13:]
    y_test = table_full[test_mask].iloc[:, 3]

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

    gen = torch.Generator()
    gen.manual_seed(SEED)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=hp["batch_size"],
                               shuffle=True, generator=gen)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=hp["batch_size"],
                             shuffle=False, generator=gen)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=hp["batch_size"],
                              shuffle=False, generator=gen)

    model = build_model(args.arch, in_features=X_train.shape[1])

    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    if hp["criterion"] == "BCE":
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    else:
        criterion = FocalLoss(alpha=1 - (y_train == 1).mean(), gamma=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    patience = 50
    epochs_since_improvement = 0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(hp["max_epochs"]):
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}", flush=True)

        if epochs_since_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best model restored from epoch {best_epoch}, val loss={best_val_loss:.4f}", flush=True)

    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            probs = torch.sigmoid(model(Xb))
            y_pred_list.append(probs.cpu().numpy())
    y_pred = np.vstack(y_pred_list).ravel()

    df_out = pd.DataFrame({"y_true": y_test.values, "y_proba": y_pred})
    out_path = eval_dir / f"{best_tag}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
