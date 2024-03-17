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
from io import StringIO


def process_nodes_from_content(node_feature_content, adjacency_matrix_content):
    try:
        # Convert the node feature content string into a file-like object
        node_feature_file_like = StringIO(node_feature_content)

        # Read the node feature matrix data into a DataFrame
        node_feature_df = pd.read_csv(node_feature_file_like, sep='\t', index_col=0)

        # Convert the adjacency matrix content string into a file-like object
        adjacency_matrix_file_like = StringIO(adjacency_matrix_content)

        # Read the adjacency matrix data into a DataFrame
        adjacency_df = pd.read_csv(adjacency_matrix_file_like, sep='\t', index_col=0)
    except Exception as e:
        return ([], [])

    node_features = []
    edge_features = []
    # Process the node feature DataFrame
    for index, row in node_feature_df.iterrows():
        features = [column for column in node_feature_df.columns if row[column] > 0.0]
        if features:
            for i in features:
                node_features.append((row.name, i))
    # Process the adjacency DataFrame
    for source_node, row in adjacency_df.iterrows():
        for target_node, value in row.items():
            if value > 0.0:
                edge_features.append((source_node, target_node))
    return (node_features, edge_features)


def file_to_string(file):
    if file:
        return file.read().decode('utf-8')
    return None


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
    model.load_state_dict(torch.load(model_path))  # Set the model to inference mode
    return model


# Ensure output is 1D
def predict_model(request):
    model = load_model()
    current_directory = os.path.dirname(__file__)
    # Declare the paths for the model, feature map, and node map
    feature_map_path = os.path.join(current_directory, '../models/feature_map.pth')
    node_map_path = os.path.join(current_directory, '../models/node_map.pth')

    # Load the models and maps using the declared paths
    feature_map = torch.load(feature_map_path)
    node_map = torch.load(node_map_path)

    model.eval()

    adjacency_file1 = request.FILES.get('adjacencyFile1')
    node_feature_file1 = request.FILES.get('nodeFeatureFile1')

    # Accessing text and other data
    adjacency_text1 = request.POST.get('adjacencyText1')
    node_feature_text1 = request.POST.get('nodeFeatureText1')

    type = request.POST.get('type')
    print(type)
    compositions = request.POST.get('compositions')
    data = request.POST.get('data')
    compositions = json.loads(compositions)
    print(compositions)
    data = json.loads(data)

    node_features_list = []
    adj_edge_list = []

    comp_name_format = f'{compositions["comp1"]["percentage"]}% {compositions["comp1"]["name"]}'

    print(node_feature_file1)

    node_features_0, edge_list_1 = process_nodes_from_content(file_to_string(node_feature_file1),
                                                              file_to_string(adjacency_file1))
    node_features_list.extend(node_features_0)
    adj_edge_list.extend(edge_list_1)

    if type != 'single':
        adjacency_file2 = request.FILES.get('adjacencyFile2')
        node_feature_file2 = request.FILES.get('nodeFeatureFile2')

        adjacency_text2 = request.POST.get('adjacencyText2')
        node_feature_text2 = request.POST.get('nodeFeatureText2')

        comp_name_format = f'{compositions["comp1"]["percentage"]}% {compositions["comp1"]["name"]}; {compositions["comp2"]["percentage"]}% {compositions["comp2"]["name"]}'
        node_features_0, edge_list_1 = process_nodes_from_content(file_to_string(node_feature_file2),
                                                                  file_to_string(adjacency_file2))

        node_features_list.extend(node_features_0)
        adj_edge_list.extend(edge_list_1)

    standard_feature_size = 1  # Ensure this matches the size used in training

    def process_node_features_for_prediction(features):
        numeric_features = []
        for feature in features[1:]:
            numeric_feature = feature_map.get(feature, len(feature_map))
            numeric_features.append(numeric_feature)

        # Pad or truncate to standard feature size
        return numeric_features[:standard_feature_size] + [0] * (standard_feature_size - len(numeric_features))

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

        edge_index_tensor = torch.tensor(
            [[node_map[node] if node in node_map else random.randint(0, num_nodes - 1) for node in edge] for edge in
             edge_list], dtype=torch.long).t().contiguous()
        graph_features_tensor = torch.tensor(graph_features, dtype=torch.float)

        data = Data(x=node_features_tensor, edge_index=edge_index_tensor, graph_features=graph_features_tensor)

        with torch.no_grad():
            prediction = model(data)
            predicted_kappa_q_4 = prediction.item()
        return predicted_kappa_q_4

    def extract_percentages(composition_str):
        pattern = r"(\d+(?:\.\d+)?)%"
        percentages = re.findall(pattern, composition_str)
        return [float(p) / 100 for p in percentages]

    composition = comp_name_format
    n_lipids_layer = data["Number of Lipid Per Layer"]
    n_water = data["Number of Water Molecules"]
    temperature_k = data["Temperature"]
    avg_membrane_thickness = data["Membrane Thickness"]
    graph_features_str = extract_percentages(composition)
    kappa_BW_DCF = data["Kappa BW-DCF(Bandwidth Dependent Die-electric Constant Fluctuation)"]
    kappa_RSF = data["Kappa-RSF"]

    node_features_list = list(set(node_features_list))
    edge_list_list = sort_tuple(adj_edge_list)


    graph_features_str.extend([n_lipids_layer, n_water, temperature_k, avg_membrane_thickness, kappa_BW_DCF, kappa_RSF])


    predicted_kappa_q_4 = predict_kappa_q_4(model, list(set(node_features_list)), sort_tuple(edge_list_list), graph_features_str)

    return {'pred': predicted_kappa_q_4}


def sort_tuple(my_list):
    my_list = [(min(i[0], i[1]), max(i[0], i[1])) for i in my_list]
    my_list.sort()
    return list(set(my_list))