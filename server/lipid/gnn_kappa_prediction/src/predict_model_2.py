import json
import os
import re
from io import StringIO
import pandas as pd
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GraphPredictor(torch.nn.Module):
    def __init__(self, input_features, hidden_features):
        super(GraphPredictor, self).__init__()
        self.conv1 = SAGEConv(input_features, hidden_features, aggr='mean')  # First GraphSAGE layer
        self.conv2 = SAGEConv(hidden_features, hidden_features, aggr='mean')  # Second GraphSAGE layer
        self.fc = torch.nn.Linear(hidden_features, 1)  # Linear layer for output

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))  # Apply first GraphSAGE layer and ReLU activation
        x = F.relu(self.conv2(x, edge_index))  # Apply second GraphSAGE layer and ReLU activation
        x = global_mean_pool(x, batch)  # Global mean pooling
        x = self.fc(x)  # Apply final linear layer
        return x.squeeze()  # Ensure output is 1D

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

def sort_tuple(my_list):
    my_list = [(min(i[0], i[1]), max(i[0], i[1])) for i in my_list]
    my_list.sort()
    return list(set(my_list))
# Ensure output is 1D
def predict_model(request):
    # Instantiate the model again with the same parameters
    model = GraphPredictor(input_features=9, hidden_features=32)

    current_directory = os.path.dirname(__file__)
    feature_map_path = os.path.join(current_directory, '../models/feature_map_2.pth')
    node_map_path = os.path.join(current_directory, '../models/node_map_2.pth')
    unique_nodes_path = os.path.join(current_directory, '../models/unique_nodes_2.pth')
    model_path = os.path.join(current_directory, '../models/graph_sage_model.pth')

    # Load the models and maps using the declared paths
    feature_map = torch.load(feature_map_path)
    node_map = torch.load(node_map_path)
    unique_nodes = torch.load(unique_nodes_path)
    loaded_model_state_dict = torch.load(model_path)

    model.load_state_dict(loaded_model_state_dict)

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



    def extract_percentages(composition_str):
        pattern = r"(\d+(?:\.\d+)?)%"
        percentages = re.findall(pattern, composition_str)
        return [float(p) / 100 for p in percentages]

    composition = comp_name_format
    n_lipids_layer = data["Number of Lipid Per Layer"]
    n_water = data["Number of Water Molecules"]
    temperature_k = data["Temperature"]
    avg_membrane_thickness = data["Membrane Thickness"]
    graph_features = extract_percentages(composition)
    kappa_BW_DCF = data["Kappa BW-DCF(Bandwidth Dependent Die-electric Constant Fluctuation)"]
    kappa_RSF = data["Kappa-RSF"]

    node_features_list = list(set(node_features_list))
    edge_list = sort_tuple(adj_edge_list)
    graph_features.extend([n_lipids_layer, n_water, temperature_k, avg_membrane_thickness, kappa_BW_DCF, kappa_RSF])

    sample_feature_size = len(node_features_list[0]) - 1 + len(
        graph_features)  # Adjust based on your actual data structure

    max_feature_size = 9

    if sample_feature_size > max_feature_size:
        max_feature_size = sample_feature_size

    # Create a mapping for all unique nodes in both node features and edge list
    for edge in edge_list:
        unique_nodes.update(edge)
    for features in node_features_list:
        unique_nodes.add(features[0])  # Assuming first element of each feature list is the node identifier

    node_map = {node_id: i for i, node_id in enumerate(unique_nodes)}

    standard_feature_size = 1  # Set this based on your data analysis

    def process_node_features(features):
        numeric_features = []
        for feature in features[1:]:  # Skip the first element which is node identifier
            if isinstance(feature, str):
                numeric_feature = feature_map.get(feature, len(feature_map))
                feature_map[feature] = numeric_feature
            else:
                numeric_feature = feature
            numeric_features.append(numeric_feature)

        # Pad the feature vector if it's shorter than the standard size
        if len(numeric_features) < standard_feature_size:
            numeric_features += [0] * (standard_feature_size - len(numeric_features))

        return numeric_features[:standard_feature_size]  # Ensure the feature vector is of standard size

    # Prepare node features tensor
    # Initialize a tensor filled with zeros for each node
    num_nodes = len(unique_nodes)
    num_features = len(next(iter(node_features_list), []))  # Number of features per node
    node_features_tensor = torch.zeros((num_nodes, standard_feature_size), dtype=torch.float)

    for features in node_features_list:
        node_id = features[0]
        if node_id in node_map:
            node_idx = node_map[node_id]
            processed_features = process_node_features(features)
            node_features_tensor[node_idx] = torch.tensor(processed_features, dtype=torch.float)

    # Prepare edge index tensor
    edge_index_tensor = torch.tensor([[node_map[node] for node in edge] for edge in edge_list],
                                     dtype=torch.long).t().contiguous()

    graph_features_tensor = torch.tensor(graph_features, dtype=torch.float)

    # Expand and repeat graph-level features for each node
    expanded_graph_features = graph_features_tensor.unsqueeze(0).repeat(num_nodes, 1)

    # Concatenate graph-level features with node features
    result_tensor = torch.cat([node_features_tensor, expanded_graph_features], dim=1)

    current_size = result_tensor.size(-1)

    if current_size < max_feature_size:
        padding_needed = max_feature_size - current_size
        pad_tensor = torch.full((result_tensor.size(0), padding_needed), 0, dtype=result_tensor.dtype)
        result_tensor = torch.cat([result_tensor, pad_tensor], dim=-1)

    with torch.no_grad():
        prediction = model(Data(x=result_tensor, edge_index=edge_index_tensor))
        predicted_kappa_q_4 = prediction.item()


    print(predicted_kappa_q_4)

    return {'pred': predicted_kappa_q_4}

