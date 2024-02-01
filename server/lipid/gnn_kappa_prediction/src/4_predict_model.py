import re

import torch
import ast

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Function to safely convert string representations to Python objects
def safe_ast_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s

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
        return x.squeeze()  # Ensure output is 1D

# Load the trained model, feature map, and node map
model = torch.load('../models/gcn_complete_model.pth')
feature_map = torch.load('../models/feature_map.pth')
node_map = torch.load('../models/node_map.pth')
model.eval()

standard_feature_size = 1  # Ensure this matches the size used in training

# Function to process node features for prediction
def process_node_features_for_prediction(features):
    numeric_features = []
    for feature in features[1:]:
        numeric_feature = feature_map.get(feature, len(feature_map))
        numeric_features.append(numeric_feature)

    # Pad or truncate to standard feature size
    return numeric_features[:standard_feature_size] + [0] * (standard_feature_size - len(numeric_features))

# Prediction function
def predict_kappa_q_4(model, node_features_str, edge_list_str, graph_features_str):
    node_features = safe_ast_literal_eval(node_features_str)
    edge_list = safe_ast_literal_eval(edge_list_str)
    graph_features = safe_ast_literal_eval(graph_features_str)

    num_nodes = len(node_map)
    node_features_tensor = torch.zeros((num_nodes, standard_feature_size), dtype=torch.float)

    for features in node_features:
        node_idx = node_map.get(features[0], None)
        if node_idx is not None:
            processed_features = process_node_features_for_prediction(features)
            node_features_tensor[node_idx] = torch.tensor(processed_features, dtype=torch.float)

    edge_index_tensor = torch.tensor([[node_map[node] for node in edge] for edge in edge_list], dtype=torch.long).t().contiguous()
    graph_features_tensor = torch.tensor(graph_features, dtype=torch.float)

    data = Data(x=node_features_tensor, edge_index=edge_index_tensor, graph_features=graph_features_tensor)

    model.eval()
    with torch.no_grad():
        prediction = model(data)
        predicted_kappa_q_4 = prediction.item()
    return predicted_kappa_q_4


def extract_percentages(composition_str):
    pattern = r"(\d+(?:\.\d+)?)%"
    percentages = re.findall(pattern, composition_str)
    return [float(p)/100 for p in percentages]

# Example input data
composition = "100% DYPC"
n_lipids_layer = 2916
n_water = 205428
temperature_k = 310
avg_membrane_thickness = 3.23
node_features_str = "[('D2A', 'C3'), ('D2B', 'C3'), ('GL2', 'Na'), ('NC3', 'Q0'), ('PO4', 'Qa'), ('C3A', 'C1'), ('C1B', 'C1'), ('C3B', 'C1'), ('GL1', 'Na'), ('C1A', 'C1')]"
edge_list_str = "[('C1A', 'D2A'), ('GL1', 'PO4'), ('C3B', 'D2B'), ('C3A', 'D2A'), ('GL1', 'GL2'), ('C1B', 'D2B'), ('NC3', 'PO4'), ('C1B', 'GL2'), ('C1A', 'GL1')]"
graph_features_str = extract_percentages(composition)


# Predicting Kappa q^-4
predicted_kappa_q_4 = predict_kappa_q_4(model, node_features_str, edge_list_str, graph_features_str)
print("Predicted Kappa q^-4:", predicted_kappa_q_4)


composition = "85% POPC; 15% POPE"
n_lipids_layer = 2915
n_water = 195599
temperature_k = 310
avg_membrane_thickness = 3.91
node_features_str = "[('D2A', 'C3'), ('GL2', 'Na'), ('NC3', 'Q0'), ('PO4', 'Qa'), ('C3A', 'C1'), ('C1B', 'C1'), ('C4A', 'C1'), ('C2B', 'C1'), ('NH3', 'Qd'), ('C4B', 'C1'), ('C3B', 'C1'), ('GL1', 'Na'), ('C1A', 'C1')]"
edge_list_str = "[('C1A', 'D2A'), ('NH3', 'PO4'), ('GL1', 'PO4'), ('C3A', 'D2A'), ('C3B', 'C4B'), ('GL1', 'GL2'), ('C1B', 'C2B'), ('C2B', 'C3B'), ('NC3', 'PO4'), ('C1B', 'GL2'), ('C1A', 'GL1'), ('C3A', 'C4A')]"
graph_features_str = extract_percentages(composition)


# Predicting Kappa q^-4
predicted_kappa_q_4 = predict_kappa_q_4(model, node_features_str, edge_list_str, graph_features_str)
print("Predicted Kappa q^-4:", predicted_kappa_q_4)