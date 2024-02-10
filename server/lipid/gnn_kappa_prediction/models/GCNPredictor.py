import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNPredictor(torch.nn.Module):
    def __init__(self, input_features, hidden_features):
        super(GCNPredictor, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.fc = torch.nn.Linear(hidden_features, 1)  # Output one value

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)  # Pooling
        x = self.fc(x)
        return x.squeeze()