import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def get_splits(n, groups: list, seed=2025):
    """
    Grouped splits by guide_sequence: 60% train, 20% val, 20% test.
    Returns (train_idx, val_idx, test_idx) as numpy arrays of row indices.
    """

    gss1 = GroupShuffleSplit(test_size=0.20, n_splits=1, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(np.arange(n), groups=groups))

    gss2 = GroupShuffleSplit(test_size=0.25, n_splits=1, random_state=seed + 1)
    rel_train_idx, rel_val_idx = next(gss2.split(train_val_idx, groups=groups.iloc[train_val_idx]))

    train_idx = train_val_idx[rel_train_idx]
    val_idx = train_val_idx[rel_val_idx]

    return train_idx, val_idx, test_idx
