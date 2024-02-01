import base64
import os
from io import BytesIO

import numpy as np
import pandas as pd
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import ast
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model():
    # Load the dataset
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory,"../data/Final_Dataset_for_Model_Train.csv/")
    data = pd.read_csv(file_path)

    # Function to convert string representations to Python objects
    def safe_ast_literal_eval(s):
        try:
            return ast.literal_eval(s)
        except ValueError:
            return s

    # Function to encode string features as numerical values
    def encode_features(features, feature_map):
        encoded_features = []
        for feature in features:
            encoded_feature = [feature_map.get(item, len(feature_map)) for item in feature]
            feature_map.update({item: i for i, item in enumerate(feature_map, start=len(feature_map)) if item not in feature_map})
            encoded_features.append(encoded_feature)
        return encoded_features

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

    feature_map = {}  # Dictionary to map strings to integers for node features
    node_map = {}     # Dictionary to map node identifiers to integers for edges
    unique_nodes = set()

    def get_unique_nodes(edge_list, node_features_list):
        unique_nodes = set()
        for edge in edge_list:
            unique_nodes.update(edge)
        for feature in node_features_list:
            unique_nodes.update(feature)
        return list(unique_nodes)

    data_list = []
    for _, row in data.iterrows():
        # Convert string representations to Python objects
        node_features_list = safe_ast_literal_eval(row['Node Features'])
        edge_list = safe_ast_literal_eval(row['Edge List'])
        graph_features = safe_ast_literal_eval(row['Graph-Level Features'])

        # Create a mapping for all unique nodes in both node features and edge list
        for edge in edge_list:
            unique_nodes.update(edge)
        for features in node_features_list:
            unique_nodes.add(features[0])  # Assuming first element of each feature list is the node identifier

        node_map = {node_id: i for i, node_id in enumerate(unique_nodes)}

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
        edge_index_tensor = torch.tensor([[node_map[node] for node in edge] for edge in edge_list], dtype=torch.long).t().contiguous()

        graph_features_tensor = torch.tensor(graph_features, dtype=torch.float)

        y = torch.tensor([row['Kappa (q^-4)']], dtype=torch.float)
        data_list.append(Data(x=node_features_tensor, edge_index=edge_index_tensor, y=y, graph_features=graph_features_tensor))

    # if 'D2B' not in node_map:
    #     print("'D2B' is not in the node_map. You might need to update your node_map or retrain the model.")

    # After creating feature_map and node_map during training
    torch.save(feature_map, '../models/feature_map.pth')
    torch.save(node_map, '../models/node_map.pth')

    # Split the data into training and validation sets
    train_data = data_list[:int(0.8 * len(data_list))]
    val_data = data_list[int(0.8 * len(data_list)):]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    # Define the GCN model
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

    # Adjust the model instantiation accordingly

    # Create model instance, optimizer, and loss function

    model = GCNPredictor(input_features=standard_feature_size, hidden_features=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    # Training loop
    for epoch in range(500):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                out = model(data)
                # print(out.items())
                val_loss += loss_fn(out, data.y).item()
            val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    # After your training loop
    # Assuming 'model' is your trained model instance
    torch.save(model, '../models/gcn_complete_model.pth')
    # Assume feature_map and node_map are created during training

    def evaluate_model(model, loader):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data in loader:
                out = model(data)
                predictions.append(out.numpy())
                actuals.append(data.y.numpy())
        return np.concatenate(predictions, axis=0), np.concatenate(actuals, axis=0)

    # Evaluate model on validation set
    predictions, actuals = evaluate_model(model, val_loader)

    # Calculate regression metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f'Mean Squared Error: {mse:.4f}')
    print(f'Root Mean Squared Error: {rmse:.4f}')
    print(f'Mean Absolute Error: {mae:.4f}')
    print(f'R-squared: {r2:.4f}')

    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Actuals')
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'k--', lw=3)
    # plt.show()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    result_json = {
        'Mean Squared Error': f'{mse:.4f}',
        'Root Mean Squared Error': f'{rmse:.4f}',
        'Mean Absolute Error': f'{mae:.4f}',
        'R-squared': f'{r2:.4f}',
        'graph':plot_data
    }
