"""
Tabular MLP model architectures.

Collapses mlp_t7{a,b,c,lin}_valid.py -- 4 scripts -- into 4 build functions
+ a registry, driven by `scripts/validate_mlp.py --arch ...`.

Unlike the CNN scripts, these 4 were NOT a clean architecture x region
grid: in the originals, MLPa was only ever run on region "sw" (f=0, c=1)
and MLPb/MLPc/MLPlin were only ever run on region "n" (f=2, c=4). The CLI
here supports any arch x region combination, but if you're verifying
against the original outputs (Step 8), those are the specific combos that
have existing results to diff against.

Architectures are plain `nn.Sequential` blocks in the originals (not custom
classes like the CNN ones), reproduced the same way via factory functions
so `in_features` (dependent on the number of tabular predictor columns)
can be plugged in at runtime.
"""
import torch.nn as nn


def build_mlp_a(in_features):
    return nn.Sequential(
        nn.Linear(in_features, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def build_mlp_b(in_features):
    return nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
    )


def build_mlp_c(in_features):
    return nn.Sequential(
        nn.Linear(in_features, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 1),
    )


def build_mlp_lin(in_features):
    """Plain logistic regression (no hidden layer), was `mlp_t7lin_valid.py`."""
    return nn.Sequential(nn.Linear(in_features, 1))


# arch key (matches original filename suffix) -> (builder fn, results subdir,
# the (region_index, region_cl) combo actually run in the original script)
MLP_REGISTRY = {
    "a": (build_mlp_a, "MLPa", (0, 1)),      # region "sw"
    "b": (build_mlp_b, "MLPb", (2, 4)),      # region "n"
    "c": (build_mlp_c, "MLPc", (2, 4)),      # region "n"
    "lin": (build_mlp_lin, "MLPlin", (2, 4)),  # region "n"
}


def build_model(arch, in_features):
    if arch not in MLP_REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'; choose from {list(MLP_REGISTRY)}")
    builder, _, _ = MLP_REGISTRY[arch]
    return builder(in_features)
