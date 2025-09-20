import torch
import torch.nn as nn
from gnn import GATLinkPredictor
from transformers import AutoTokenizer, AutoModel

class CRISPROffTClassifier(nn.Module):
    def __init__(self, embed_model, in_channels, hidden_channels, out_channels, 
                 in_features, hidden_features,
                 heads=4, dropout=0.0, local_files_only=False):
        super(CRISPROffTClassifier, self).__init__()
        # Embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(
            embed_model, local_files_only=local_files_only, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            embed_model, local_files_only=local_files_only, trust_remote_code=True
        )
        # MLP for additional structured features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features)
        )
        # GNN
        self.gnn = GATLinkPredictor(
            in_channels+hidden_features, hidden_channels, out_channels, heads=heads, dropout=dropout
        )

    def forward(self, x_seq, x_feat, edge_index, edge_pairs):
        """
        x: [num_nodes, in_channels] node features
        edge_index: [2, num_edges] graph structure (for GNN)
        edge_pairs: [2, num_edges_to_predict] node pairs we want predictions for
        """
        # Embed sequence
        with torch.no_grad():
            embedding = []
            for seq in x_seq:
                inputs = self.tokenizer(seq, return_tensors = 'pt')["input_ids"]
                inputs = inputs.to(edge_index.device)
                hidden_states = self.model(inputs)[0] # [1, sequence_length, 768]
                # embedding with mean pooling
                embedding_mean = torch.mean(hidden_states[0], dim=0)
                embedding.append(embedding_mean)
        
        embedding = torch.stack(embedding).detach()
        # Embed additional structured features
        features = self.mlp(x_feat)
        # Concatenate
        embedding = torch.concat((embedding, features), dim=-1)
        # Pass through GNN
        out = self.gnn(embedding, edge_index, edge_pairs)
        return out



if __name__ == "__main__":
    # Example configuration
    in_channels = 768
    hidden_channels = 16
    out_channels = 16
    in_features = 4
    hidden_features = 16
    embed_model = 'zhihan1996/DNABERT-S'

    # Example inputs
    x_seq = [
        'CGGCGCTGGTGCCCAGGACGAGGATGGAGATT', 
        'CGGCGCTGGTGCCCAGGACGAGGATGGAGATT', 
        'GGCGCTGTGTGTCCTGGACGAGGAACTGGACT', 
        'AGGAACTGGAGCAAAGGACAAGGAGATGGTTT'
    ]
    edge_index = torch.tensor([[0, 1, 0, 2],
                               [1, 0, 2, 0]], dtype=torch.long)
    edge_label_index = torch.tensor([[0, 1, 0, 2, 0, 3],
                                     [1, 0, 2, 0, 3, 0]], dtype=torch.long)
    x_feat = torch.randn(len(x_seq), in_features)

    # Model
    model = CRISPROffTClassifier(
        embed_model, in_channels, hidden_channels, out_channels, 
        in_features, hidden_features,
        local_files_only=True
    )
    model.eval()

    # Forward
    with torch.no_grad():
        pred = model(x_seq, x_feat, edge_index, edge_label_index)
        print("Predicted edge logits:", pred)