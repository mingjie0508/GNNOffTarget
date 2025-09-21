import torch
from sklearn.preprocessing import StandardScaler


def fit_num_scaler(x_num):
    """
    Standardize numerical values (Start bp, End bp)
    """

    sc = StandardScaler(with_mean=True, with_std=True).fit(x_num)
    return sc


def get_collate_fn(num_scaler):
    def _collate_fn(batch):
        row_id, x_seq, x_num, x_chr, x_strand, x_cas9, x_source, y = zip(*batch)

        # Standardize numerical values
        x_num = torch.stack(x_num)
        mean = torch.tensor(num_scaler.mean_, dtype=x_num.dtype, device=x_num.device)
        std = torch.tensor(num_scaler.scale_, dtype=x_num.dtype, device=x_num.device)
        x_num = (x_num - mean) / std

        return (
            row_id,
            torch.stack(x_seq),
            x_num,
            torch.stack(x_chr),
            torch.stack(x_strand),
            torch.stack(x_cas9),
            torch.stack(x_source),
            torch.stack(y),
        )

    return _collate_fn
