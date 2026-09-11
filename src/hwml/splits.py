"""
Member-based train/validation/test splitting.

CNN, MLP, RF and GAM validation scripts all generate this split identically,
on the fly, from `member_ids`. LR is the outlier: it reads a pre-computed
split-membership table from disk instead. Both are provided here; see
`load_precomputed_splits` for the LR-specific caveat.
"""
import numpy as np
import pandas as pd


def generate_member_splits(member_ids, seed=0, test_prop=0.3, n_members_total=50, n_val_folds=7):
    """
    Reproduces the on-the-fly split used by CNN / MLP / RF / GAM validation
    scripts exactly (same RNG calls, same order of operations -- do not
    reorder these lines, the exact sequence matters for reproducibility
    since it consumes the seeded RNG step by step):

      1. draw 100 candidate test-member subsets of size `test_prop * n_members_total`
      2. take the *first* candidate subset as the actual test set
      3. the remaining members are shuffled, then split into `n_val_folds` groups

    Returns
    -------
    test_membs : np.ndarray
    val_membs : list[np.ndarray]
        `n_val_folds` arrays partitioning the non-test members.
    """
    rng = np.random.default_rng(seed=seed)
    test_size = int(test_prop * n_members_total)

    split_sets = pd.DataFrame(
        [rng.choice(member_ids, size=test_size, replace=False) for _ in range(100)],
        columns=[f"member_{i + 1}" for i in range(test_size)],
    )

    test_membs = split_sets.iloc[0, :].values
    train_membs = np.setdiff1d(member_ids, test_membs)

    rng.shuffle(train_membs)
    val_membs = np.array_split(train_membs, n_val_folds)

    return test_membs, val_membs


def load_precomputed_splits(path):
    """
    The original `lr_trial7_valid.py` did NOT use `generate_member_splits`
    above -- it read a pre-made split-membership table from disk instead.
    CONFIRMED (2024) this was a leftover from an earlier pipeline version,
    not intentional: LR should use `generate_member_splits`, same as
    RF/GAM/MLP. Kept here, unused, only as a record of the original
    behaviour in case it's ever needed again.
    """
    return pd.read_csv(path, sep="\t")
