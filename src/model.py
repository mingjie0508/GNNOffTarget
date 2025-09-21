import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TransformerClassifier(nn.Module):
    def __init__(self, embed_model: str,
                 embedding_dim: int,
                 dim_feedforward: int, 
                 dropout: float = 0.0, 
                 local_files_only: bool = False):
        super(TransformerClassifier, self).__init__()
        # Embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(
            embed_model, local_files_only=local_files_only, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            embed_model, local_files_only=local_files_only, trust_remote_code=True
        )
        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x_seq):
        """
        x: [num_nodes, in_channels] node features
        edge_index: [2, num_edges] graph structure (for GNN)
        edge_pairs: [2, num_edges_to_predict] node pairs we want predictions for
        """
        # Embed sequence
        embedding = []
        for seq in x_seq:
            inputs = self.tokenizer(seq, return_tensors = 'pt')["input_ids"]
            inputs = inputs.to(self.model.device)
            hidden_states = self.model(inputs)[0] # [1, sequence_length, 768]
            # embedding with mean pooling
            embedding_mean = torch.mean(hidden_states[0], dim=0)
            embedding.append(embedding_mean)
        
        embedding = torch.stack(embedding)
        # Pass through classification head
        out = self.classifier(embedding).squeeze(-1)
        return out



if __name__ == "__main__":
    # Example configuration
    embed_model = 'zhihan1996/DNABERT-S'
    embedding_dim = 768
    dim_feedforward = 64

    # Example inputs
    x_seq = [
        'CGGCGCTGGTGCCCAGGACGAGGATGGAGATT[SEP]CGGCGCTGGTGCCCAGGACGAGGATGGAGATT', 
        'CGGCGCTGGTGCCCAGGACGAGGATGGAGATT[SEP]GGCGCTGTGTGTCCTGGACGAGGAACTGGACT', 
        'CGGCGCTGGTGCCCAGGACGAGGATGGAGATT[SEP]AGGAACTGGAGCAAAGGACAAGGAGATGGTTT'
    ]

    # Model
    model = TransformerClassifier(
        embed_model=embed_model,
        embedding_dim=embedding_dim,
        dim_feedforward=dim_feedforward,
        local_files_only=True
    )
    model.eval()

    # Forward
    with torch.no_grad():
        pred = model(x_seq)
        print("Predicted off-target effect logits:", pred)
