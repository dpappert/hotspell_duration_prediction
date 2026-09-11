"""
Shared "pick the best hyperparameter combo from validation results" logic,
used by every eval_*.py script except eval_lr.py (which uses fixed
hyperparameters -- LR is the study's baseline model, not tuned).
"""
import re


def select_best_combo(records_df, exclude_pattern=None):
    """
    Reproduces the generalization-penalized selection formula used
    identically across eval_RF.py, eval_GAM.py, eval_MLP.py, eval_all2D.py:

        score_roc  = 2*val_rocauc - train_rocauc
        score_prg  = 2*val_prgauc - train_prgauc
        score_both = mean(score_roc, score_prg)

    `records_df` must be indexed by combo tag, with columns val_rocauc,
    train_rocauc, val_prgauc, train_prgauc. `exclude_pattern`, if given, is
    a substring to filter out of the index before scoring (MLP/CNN eval
    scripts exclude Focal-loss combos: exclude_pattern="Focal" -- a
    deliberate design choice in the originals, not a bug).

    Returns (best_tag, scored_df).
    """
    df = records_df.copy()
    if exclude_pattern:
        df = df[~df.index.str.contains(exclude_pattern)]
    df["score_roc"] = 2 * df["val_rocauc"] - df["train_rocauc"]
    df["score_prg"] = 2 * df["val_prgauc"] - df["train_prgauc"]
    df["score_both"] = (df["score_roc"] + df["score_prg"]) / 2
    best_tag = df["score_both"].idxmax()
    return best_tag, df


_RF_TAG_RE = re.compile(
    r"F(?P<f>\d+)_C(?P<c>\d+)_nest(?P<nest>\d+)_depth(?P<depth>\d+)_"
    r"msplit(?P<msplit>\d+)_mleaf(?P<mleaf>\d+)_mfeat(?P<mfeat>[^_]+)_crit(?P<crit>.+)$"
)


def parse_rf_tag(tag):
    """
    Parses an RF combo tag, e.g. 'F0_C1_nest50_depth3_msplit10_mleaf5_mfeat0.33_critgini'.

    BUG FIX vs. the original eval_RF.py: it extracted `criterion` with a
    hardcoded character offset (`best_both[59:]`), which only happens to be
    correct when every preceding field has exactly the digit-width the
    offset was calibrated for. Checked against the actual hyperparameter
    grid: 96 of 192 possible combos were mis-parsed this way (e.g. 'gini'
    truncated to 'ini'), which would crash RandomForestClassifier with an
    invalid criterion value. This parses by named field instead, via regex,
    so it's correct regardless of digit widths -- and handles 'log_loss'
    correctly despite its embedded underscore, same as the original
    intended (n_estimators/depth/etc used split()-based parsing already,
    which was fine; only `criterion` used the broken offset).
    """
    m = _RF_TAG_RE.match(tag)
    if not m:
        raise ValueError(f"Could not parse RF tag: {tag!r}")
    d = m.groupdict()
    max_features = d["mfeat"]
    if max_features == "0.33":
        max_features = float(max_features)
    return {
        "n_estimators": int(d["nest"]),
        "max_depth": int(d["depth"]),
        "min_samples_split": int(d["msplit"]),
        "min_samples_leaf": int(d["mleaf"]),
        "max_features": max_features,
        "criterion": d["crit"],
    }


def parse_gam_tag(tag):
    """Parses a GAM combo tag, e.g. 'F0_C1_lam1_nspl5'. Original parsing was
    already robust (split-based, no fixed offsets) -- reproduced as-is."""
    parts = tag.split("_")
    return {
        "lam": int(parts[2][3:]),
        "n_splines": int(parts[3][4:]),
    }


def parse_nn_tag(tag):
    """
    Parses a CNN/MLP combo tag, e.g.
    'HPS_mpe500-pat50_bs64_lr0.01_wd0.0003_critBCE_optAdam' (the `param_id`
    format from validate_cnn.py / validate_mlp.py). Original parsing was
    already robust (startswith-based prefix matching, no fixed offsets) --
    reproduced as-is.
    """
    parts = tag.split("_")
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
    return params
