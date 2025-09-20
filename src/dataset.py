import os
import torch
from torch_geometric.data import Dataset, Data

class CRISPRoffTDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # Expect these raw files to exist under `root/raw/`
        return ['raw_graphs.pt']

    @property
    def processed_file_names(self):
        # Create one file per graph in `root/processed/`
        return [f'data_{i}.pt' for i in range(3)]  # 3 toy graphs

    def process(self):
        # Example: make 3 toy graphs
        for i in range(3):
            # x = torch.randn((4, 2))  # 4 nodes, 2 features
            x = [
                'CGGCGCTGGTGCCCAGGACGAGGATGGAGATT', 
                'CGGCGCTGGTGCCCAGGACGAGGATGGAGATT', 
                'GGCGCTGTGTGTCCTGGACGAGGAACTGGACT', 
                'AGGAACTGGAGCAAAGGACAAGGAGATGGTTT'
            ]
            num_nodes = len(x)
            edge_index = torch.tensor([[0, 1, 0, 2],
                                       [1, 0, 2, 0]], dtype=torch.long)
            edge_label_index = torch.tensor([[0, 1, 0, 2, 0, 3],
                                           [1, 0, 2, 0, 3, 0]], dtype=torch.long)
            y = torch.tensor([1, 1, 1, 1, 0, 0])  # binary label
            data = Data(x=x, edge_index=edge_index, edge_label_index=edge_label_index, y=y, num_nodes=num_nodes)
            # Apply pre_transform if given
            if self.pre_transform:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load a single graph by index
        data_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        data = torch.load(data_path, weights_only=False)
        return data


if __name__ == "__main__":
    # Example inputs
    dataset = CRISPRoffTDataset(root='../data')

    print("Number of graphs:", len(dataset))
    print("First graph:", dataset[0])
    print("Node features:", dataset[0].x)
    print("Edge index:", dataset[0].edge_index.shape)
    print("Edge pairs (positive and negative):", dataset[0].edge_pairs.shape)
    print("Label:", dataset[0].y)
