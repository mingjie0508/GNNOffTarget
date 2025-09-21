import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from .cfg import RunConfig

from sklearn.preprocessing import StandardScaler


def _cat_to_index(series: pd.Series, vocab: list = None):
    """
    Map a categorical series to integer indices with an <UNK> bucket.
    If `vocab` is provided, reuse it; unseen/NaN -> <UNK>.
    Returns (idx[int64], vocab[list[str]]).
    """

    if vocab[-1] != "<UNK>":
        raise ValueError(
            f"Expected last vocab entry to be '<UNK>', got '{vocab[-1]}' instead. "
            "Make sure you always append '<UNK>' to vocab when building it from train."
        )

    if vocab is None:
        cats = pd.Categorical(series)
        vocab = [str(c) for c in cats.categories.tolist()] + ["<UNK>"]
    else:
        cats = pd.Categorical(series, categories=vocab[:-1])

    codes = cats.codes.astype(np.int64)
    unk_idx = len(vocab) - 1
    codes = np.where(codes == -1, unk_idx, codes)

    return codes, vocab


class CRISPRoffTDataset(Dataset):
    """
    Wraps guide/target embeddings + selected metadata into a PyTorch Dataset.

    Expects metadata columns:
      guide_sequence, chr, start, end, strand, cas9_type, source, score

    __getitem__ returns:
      (x_seq[1536], x_num[2], chr_i, strand_i, cas9_i, source_i, y)

    Attributes after init:
      .vocab  : dict[str]->list[str] for {'chr','strand','cas9_type','source'} (with '<UNK>')
      .groups : pd.Series[str] (guide_sequence) for grouped splits
      .N      : number of rows
    """

    CAT_KEYS = ["chr", "strand", "cas9_type", "source"]

    def __init__(self, cfg: RunConfig, idxs: list[int] | None = None, vocabs: dict = {}):
        super().__init__()

        guide_path = cfg.data.guide_seq_path
        target_path = cfg.data.target_seq_path
        metadata_path = cfg.data.metadata_path

        # Load data
        xg = np.load(guide_path).astype(np.float32)  # [N, 768]
        xt = np.load(target_path).astype(np.float32)  # [N, 768]
        metadata = pd.read_csv(metadata_path)

        # Data split groups (by: Guide Sequence)
        self.groups = xg[idxs]

        # Load indices for filtering
        N = xg.shape[0]
        if idxs is not None:
            idxs = np.arange(N)
        else:
            idxs = np.array(idxs)

        self.idxs = idxs
        self.N = len(idxs)

        # Guide, Target Sequence
        self.X_seq = np.concatenate([xg, xt], axis=1).astype(np.float32)  # [N, 1536]
        self.X_seq = self.X_seq[idxs]

        # Targets
        self.y = metadata["score"].astype(np.float32).to_numpy()
        self.y = self.y[idxs]

        # Numerical: start, end
        start_bp = metadata["start"].astype(np.float32).to_numpy()
        end_bp = metadata["end"].astype(np.float32).to_numpy()
        self.X_num = np.stack([start_bp, end_bp], axis=1).astype(np.float32)  # [N, 2]

        # Categoricals: chr, strand, cas9_type, source
        self.X_cat = {}
        self.vocabs = {}

        for key in self.CAT_KEYS:
            code, vocab = _cat_to_index(metadata[key][idxs], vocab=vocabs.get(key))
            self.X_cat[key] = code
            self.vocabs[key] = vocab

        self.N = self.X_seq.shape[0]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int):
        idx = self.idxs[i]
        x_seq = torch.from_numpy(self.X_seq[idx]).to(torch.float32)  # [1536]
        x_num = torch.from_numpy(self.X_num[idx]).to(torch.float32)  # [2]
        x_chr = torch.tensor(self.X_cat["chr"][idx], dtype=torch.long)
        x_strand = torch.tensor(self.X_cat["strand"][idx], dtype=torch.long)
        x_cas9 = torch.tensor(self.X_cat["cas9_type"][idx], dtype=torch.long)
        x_source = torch.tensor(self.X_cat["source"][idx], dtype=torch.long)
        y = torch.tensor(self.y[i], dtype=torch.float32)

        return (x_seq, x_num, x_chr, x_strand, x_cas9, x_source, y)
