import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import torch


from sklearn.preprocessing import StandardScaler


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


def fit_num_scaler(x_num):
    """
    Standardize numerical values (Start bp, End bp)
    """

    sc = StandardScaler(with_mean=True, with_std=True).fit(x_num)
    return sc


def get_collate_fn(num_scaler):
    def _collate_fn(batch):
        x_seq, x_num, x_chr, x_strand, x_cas9, x_source, y = zip(*batch)

        x_num = torch.stack(x_num)
        mean = torch.tensor(num_scaler.mean_, dtype=x_num.dtype, device=x_num.device)
        std = torch.tensor(num_scaler.scale_, dtype=x_num.dtype, device=x_num.device)
        x_num = (x_num - mean) / std

        return (
            torch.stack(x_seq),
            x_num,
            torch.stack(x_chr),
            torch.stack(x_strand),
            torch.stack(x_cas9),
            torch.stack(x_source),
            torch.stack(y),
        )

    return _collate_fn
