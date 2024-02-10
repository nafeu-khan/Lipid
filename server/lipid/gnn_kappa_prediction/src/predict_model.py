import json
import os
import random
import re

import pandas as pd
import torch
import ast

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

def process_nodes(lipid_name):
    current_directory = os.path.dirname(__file__)
    node_feature_dir = os.path.join(current_directory, '../data/TextFiles/Node_Features/')
    adjacency_matrix_dir = os.path.join(current_directory, '../data/TextFiles/Adjacency_Matrix/')

    try:
        # Define the path to your node feature matrix data file
        node_feature_file_path = os.path.join(node_feature_dir, f"{lipid_name}.txt")

        # Read the node feature matrix data into a DataFrame
        node_feature_df = pd.read_csv(node_feature_file_path, sep='\t', index_col=0)

        # Define the path to your adjacency matrix data file
        adjacency_matrix_file_path = os.path.join(adjacency_matrix_dir, f"{lipid_name} .txt")
    except Exception as e:
        return ([], [])
    # Read the adjacency matrix data into a DataFrame
    adjacency_df = pd.read_csv(adjacency_matrix_file_path, sep='\t', index_col=0)
    node_features = []
    edge_features = []
    # Process the node feature DataFrame
    for index, row in node_feature_df.iterrows():
        features = [column for column in node_feature_df.columns if row[column] > 0.0]
        if features:
            for i in features:
                node_features.append((row.name, i))
    for source_node, row in adjacency_df.iterrows():
        for target_node, value in row.items():
            if value > 0.0:
                edge_features.append((source_node, target_node))
    return (node_features, edge_features)

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


def load_model():
    current_directory = os.path.dirname(__file__)
    model_path = os.path.join(current_directory, '../models/gcn_complete_model.pth')
    model = GCNPredictor(input_features=1, hidden_features=32)
    model.load_state_dict(torch.load(model_path))# Set the model to inference mode
    return model

  # Ensure output is 1D
def predict_model(request):
    model=load_model()
    current_directory = os.path.dirname(__file__)
    # Declare the paths for the model, feature map, and node map
    feature_map_path = os.path.join(current_directory, '../models/feature_map.pth')
    node_map_path = os.path.join(current_directory, '../models/node_map.pth')

    # Load the models and maps using the declared paths
    feature_map = torch.load(feature_map_path)
    node_map = torch.load(node_map_path)

    model.eval()

    adjacency_file = request.FILES.get('adjacencyFile')
    node_feature_file = request.FILES.get('nodeFeatureFile')
    # Accessing text and other data
    adjacency_text = request.POST.get('adjacencyText')
    node_feature_text = request.POST.get('nodeFeatureText')
    type = request.POST.get('type')
    compositions = request.POST.get('compositions')
    data = request.POST.get('data')
    compositions = json.loads(compositions)
    data = json.loads(data)
    comp_name = compositions["comp1"]["name"]
    comp_name_format = f'{compositions["comp1"]["percentage"]}% {compositions["comp1"]["name"]}'

    # Specify the desired path to save the adjacency_file
    current_directory = os.path.dirname(__file__)
    save_path = os.path.join(current_directory,
                             f'../data/TextFiles/Adjacency_Matrix/{comp_name} .txt')
    # Save the adjacency_file to the specified path
    with open(save_path, 'wb') as destination:
        for chunk in adjacency_file.chunks():
            destination.write(chunk)
    save_path = os.path.join(current_directory, f'../data/TextFiles/Node_Features/{comp_name}.txt')

    # Save the adjacency_file to the specified path
    with open(save_path, 'wb') as destination:
        for chunk in node_feature_file.chunks():
            destination.write(chunk)

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
            node_idx = node_map.get(features[0], num_nodes)
            if node_idx is not None:
                processed_features = process_node_features_for_prediction(features)
                node_features_tensor[node_idx] = torch.tensor(processed_features, dtype=torch.float)

        edge_index_tensor = torch.tensor([[node_map[node] if node in node_map else random.randint(0, num_nodes-1) for node in edge] for edge in edge_list], dtype=torch.long).t().contiguous()
        graph_features_tensor = torch.tensor(graph_features, dtype=torch.float)

        data = Data(x=node_features_tensor, edge_index=edge_index_tensor, graph_features=graph_features_tensor)

        with torch.no_grad():
            prediction = model(data)
            predicted_kappa_q_4 = prediction.item()
        return predicted_kappa_q_4


    def extract_percentages(composition_str):
        pattern = r"(\d+(?:\.\d+)?)%"
        percentages = re.findall(pattern, composition_str)
        return [float(p)/100 for p in percentages]

    # # Example input data
    # composition = "100% DYPC"
    # n_lipids_layer = 2916
    # n_water = 205428
    # temperature_k = 310
    # avg_membrane_thickness = 3.23
    # node_features_str = "[('D2A', 'C3'), ('D2B', 'C3'), ('GL2', 'Na'), ('NC3', 'Q0'), ('PO4', 'Qa'), ('C3A', 'C1'), ('C1B', 'C1'), ('C3B', 'C1'), ('GL1', 'Na'), ('C1A', 'C1')]"
    # edge_list_str = "[('C1A', 'D2A'), ('GL1', 'PO4'), ('C3B', 'D2B'), ('C3A', 'D2A'), ('GL1', 'GL2'), ('C1B', 'D2B'), ('NC3', 'PO4'), ('C1B', 'GL2'), ('C1A', 'GL1')]"
    # graph_features_str = extract_percentages(composition)
    #
    #
    # # Predicting Kappa q^-4
    # print("Predicted Kappa q^-4:", predicted_kappa_q_4)


    composition = comp_name_format
    n_lipids_layer = data["Number of Lipid Per Layer"]
    n_water = data["Number of Water Molecules"]
    temperature_k = data["Temperature"]
    avg_membrane_thickness = data["Membrane Thickness"]
    graph_features_str = extract_percentages(composition)
    kappa_BW_DCF=data["Kappa BW-DCF(Bandwidth Dependent Die-electric Constant Fluctuation)"]
    kappa_RSF=data["Kappa-RSF"]

    graph_features_str.extend([n_lipids_layer,n_water,temperature_k,avg_membrane_thickness,kappa_BW_DCF,kappa_RSF])

    # print(graph_features_str)

    # from server.lipid.gnn_kappa_prediction.src.extract_dataset import process_nodes
    node_features_str,edge_list_str =process_nodes(comp_name)

    predicted_kappa_q_4 = predict_kappa_q_4(model, node_features_str, edge_list_str, graph_features_str)


    return {'pred':predicted_kappa_q_4}

