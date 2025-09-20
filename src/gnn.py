import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCNLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        super(GCNLinkPredictor, self).__init__()
        # Node encoder GNN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # Edge predictor (MLP on concatenated embeddings of two nodes)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  # Binary edge prediction
        )

    def forward(self, x, edge_index, edge_pairs):
        """
        x: [num_nodes, in_channels] node features
        edge_index: [2, num_edges] graph structure (for GNN)
        edge_pairs: [2, num_edges_to_predict] node pairs we want predictions for
        """
        # Get node embeddings
        h = self.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.conv2(h, edge_index)

        # Get embeddings for source & target nodes
        src, dst = edge_pairs
        h_src = h[src]
        h_dst = h[dst]

        # Concatenate and classify
        edge_features = torch.cat([h_src, h_dst], dim=-1)
        out = self.edge_mlp(edge_features).squeeze(-1)
        return out  # logits of edge existing


class GATLinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.0):
        super(GATLinkPredictor, self).__init__()
        # First GAT layer (multi-head attention)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        # Second GAT layer (average heads into final output)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        # Edge predictor (MLP on concatenated embeddings of two nodes)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  # Binary edge prediction
        )

    def forward(self, x, edge_index, edge_pairs):
        """
        x: [num_nodes, in_channels] node features
        edge_index: [2, num_edges] graph structure (for GNN)
        edge_pairs: [2, num_edges_to_predict] node pairs we want predictions for
        """
        # First GAT layer
        h = self.conv1(x, edge_index)
        h = self.elu(h)  # ELU often used in GAT

        # Dropout for regularization
        h = self.dropout(h)

        # Second GAT layer
        h = self.conv2(h, edge_index)

        # Get embeddings for source & target nodes
        src, dst = edge_pairs
        h_src = h[src]
        h_dst = h[dst]

        # Concatenate and classify
        edge_features = torch.cat([h_src, h_dst], dim=-1)
        out = self.edge_mlp(edge_features).squeeze(-1)
        return out  # logits of edge existing


if __name__ == "__main__":
    # Example inputs
    num_nodes = 6
    in_channels = 8
    hidden_channels = 16
    out_channels = 16

    # Random node features
    x = torch.randn((num_nodes, in_channels))

    # Graph edges for message passing
    edge_index = torch.tensor([[0, 1, 2, 3],
                            [1, 2, 3, 4]], dtype=torch.long)

    # Candidate edges to predict (e.g., link prediction task)
    edge_pairs = torch.tensor([[0, 2, 4],
                            [1, 3, 5]], dtype=torch.long)

    # Model
    model = GATLinkPredictor(in_channels, hidden_channels, out_channels)

    # Forward
    pred = model(x, edge_index, edge_pairs)
    print("Predicted edge logits:", pred)
