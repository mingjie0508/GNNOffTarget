# model_fcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetadataEncoder(nn.Module):
    """
    (X_num, X_cat) -> h_metadata (32-d)
    X_cat fields expected: chr, strand, cas9, source (passed as integer indices)
    """

    def __init__(
        self,
        vocabs: dict,
        num_in_dim: int = 2,
        cat_in_dims: dict | None = None,
        metadata_out_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        cat_in_dims = cat_in_dims or {"chr": 8, "strand": 2, "cas9_type": 8, "source": 8}

        self.emb_chr = nn.Embedding(len(vocabs["chr"]), cat_in_dims["chr"])
        self.emb_strand = nn.Embedding(len(vocabs["strand"]), cat_in_dims["strand"])
        self.emb_cas9 = nn.Embedding(len(vocabs["cas9_type"]), cat_in_dims["cas9_type"])
        self.emb_source = nn.Embedding(len(vocabs["source"]), cat_in_dims["source"])

        cat_in_dim = sum(cat_in_dims.values())
        in_dim = num_in_dim + cat_in_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, metadata_out_dim),
        )

        # Manual Initialization
        for emb in [self.emb_chr, self.emb_strand, self.emb_cas9, self.emb_source]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_num, x_chr, x_strand, x_cas9, x_source):
        x_cat = torch.cat(
            [
                self.emb_chr(x_chr),
                self.emb_strand(x_strand),
                self.emb_cas9(x_cas9),
                self.emb_source(x_source),
            ],
            dim=-1,
        )

        z = torch.cat([x_num, x_cat], dim=-1)
        out = self.mlp(z)  # [B, 32]
        return out


class FCNNRegressor(nn.Module):
    """
    X_seq (1536) + h_metadata (32) -> regression score
    """

    def __init__(
        self,
        vocabs: dict,
        seq_in_dim: int = 1536,
        num_in_dim: int = 2,
        cat_in_dims: dict | None = None,
        metadata_out_dim: int = 32,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.metadata_encoder = MetadataEncoder(
            vocabs, num_in_dim=num_in_dim, cat_in_dims=cat_in_dims, metadata_out_dim=metadata_out_dim, dropout=0.1
        )

        in_dim = seq_in_dim + metadata_out_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_seq, x_num, chr_i, strand_i, cas9_i, source_i):
        h_metadata = self.metadata_encoder(x_num, chr_i, strand_i, cas9_i, source_i)  # [B, 32]
        x = torch.cat([x_seq, h_metadata], dim=-1)  # [B, 1568]
        out = self.mlp(x).squeeze(-1)
        
        return out
