import json
import os

import pandas as pd
import torch
from django.template.context_processors import csrf
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import ast
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from django.http import JsonResponse
from io import BytesIO
import base64

plt.switch_backend('agg')


def predict_value(data,df):
    # Feature Engineering
    # One-hot encoding for 'Lipid composition (molar)'
    encoder = OneHotEncoder(sparse=False)
    lipid_composition_encoded = encoder.fit_transform(df[['Lipid composition (molar)']])

    # # Standardizing numerical features
    # scaler = StandardScaler()
    # numerical_features = df[['Salt, M', 'Pressure, bar', 'Temperature, K']]
    # numerical_features_scaled = scaler.fit_transform(numerical_features)
    #
    # Standardizing numerical features
    scaler = StandardScaler()
    numerical_features = df[
        ['N_lipids/layer', 'N_water', 'Lz_0, nm', 'Performance, us/day', 'Lx_mean, nm', 'Lz_mean', 'Area/lipid, nm^2',
         'L/lipid, nm', 'Memb_thickness', 'kappa, kT (q^-4)', 'Time_to_reach_kappa, ns', 'q0, nm-1', 't_decorr(q0), ns',
         'Kappa  gamma_CU (direct)', 'Kappa  BW-DCF', 'kappa  binning, kT', 'Kappa gamma/binning', 'Kappa_rsf']]
    numerical_features_scaled = scaler.fit_transform(numerical_features)

    # # Combining transformed features with original dataframe
    # encoded_df = pd.DataFrame(lipid_composition_encoded, columns=encoder.get_feature_names_out())
    # scaled_numerical_df = pd.DataFrame(numerical_features_scaled, columns=numerical_features.columns)
    # df = pd.concat([encoded_df, scaled_numerical_df,
    #                 df.drop(['Lipid composition (molar)', 'Salt, M', 'Pressure, bar', 'Temperature, K'], axis=1)],
    #                axis=1)

    # Combining transformed features with original dataframe
    encoded_df = pd.DataFrame(lipid_composition_encoded, columns=encoder.get_feature_names_out())
    scaled_numerical_df = pd.DataFrame(numerical_features_scaled, columns=numerical_features.columns)
    df = pd.concat([encoded_df, scaled_numerical_df, df.drop(['Lipid composition (molar)'], axis=1)], axis=1)


    # Function to encode atom features
    def encode_atom_features(atom_features, feature_encodings, max_length):
        encoding = feature_encodings.get(atom_features)
        if encoding is None:
            encoding = len(feature_encodings)
            feature_encodings[atom_features] = encoding
        return [1 if i == encoding else 0 for i in range(max_length)]


    # Function to process each row of the dataset
    def process_row(row, feature_encodings, max_length):
        try:
            node_features_str = ast.literal_eval(row['node_features'])
            edges_str = ast.literal_eval(row['edge'])
        except ValueError as e:
            print("Error processing row:", row)
            raise ValueError(f"Error parsing literals: {e}")

        # Adjusting target extraction based on your dataset
        target = float(row['kappa, kT (q^-4)'])

        node_features = [encode_atom_features(atom_feature, feature_encodings, max_length) for atom_feature in
                         node_features_str]
        x = torch.tensor(node_features, dtype=torch.float)

        edge_index = []
        for edge_str in edges_str:
            if isinstance(edge_str, tuple) and len(edge_str) == 2:
                start_atom, end_atom = edge_str
            else:
                raise ValueError(f"Invalid edge format: {edge_str}")

            start_indices = [i for i, atom_feature in enumerate(node_features_str) if atom_feature[0] == start_atom]
            end_indices = [i for i, atom_feature in enumerate(node_features_str) if atom_feature[0] == end_atom]
            for start in start_indices:
                for end in end_indices:
                    edge_index.extend([[start, end], [end, start]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index, y=torch.tensor([target], dtype=torch.float))


    # Adjusting the script to use the new DataFrame structure
    feature_encodings = {}
    max_length = sum(1 for _, row in df.iterrows() for _ in ast.literal_eval(row['node_features']))
    data_list = [process_row(row, feature_encodings, max_length) for _, row in df.iterrows()]

    # Split the dataset
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


    # Define the GCN model
    class GCN(torch.nn.Module):
        def __init__(self, input_channels, hidden_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(input_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.out = Linear(hidden_channels, 1)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.out(x)


    # Model, optimizer, and criterion
    model = GCN(input_channels=max_length, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()


    # Training and evaluation functions (as in your original script)
    def train():
        model.train()
        total_loss = 0
        y_real = []
        y_pred = []
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            y_real.extend(data.y.view(-1).tolist())
            y_pred.extend(output.view(-1).tolist())
        r2 = r2_score(y_real, y_pred)
        return total_loss / len(train_loader), r2


    def test(loader):
        model.eval()
        y_real = []
        y_pred = []
        with torch.no_grad():
            for data in loader:
                output = model(data)
                y_real.extend(data.y.view(-1).tolist())
                y_pred.extend(output.view(-1).tolist())
        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2 = r2_score(y_real, y_pred)
        return mae, rmse, r2, y_real, y_pred


    # Training loop and evaluation
    train_losses = []
    train_r2_scores = []
    test_losses = []
    test_r2_scores = []

    for epoch in range(200):
        train_loss, train_r2 = train()
        test_loss, test_r2, mae, rmse, _ = test(test_loader)

        train_losses.append(train_loss)
        train_r2_scores.append(train_r2)
        test_losses.append(test_loss)
        test_r2_scores.append(test_r2)

        print(
            f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f}, Test Loss: {test_loss:.4f}, Test R²: {test_r2:.4f}')

    # Plotting loss and R² scores
    plt.figure(figsize=(15, 5))

    # Plotting training and test loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss per Epoch')
    plt.legend()

    # Plotting R² score
    plt.subplot(1, 2, 2)
    plt.plot(train_r2_scores, label='Train R² Score')
    plt.plot(test_r2_scores, label='Test R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('R² Score per Epoch')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    # # Plot R² scores
    # plt.figure(figsize=(12, 6))
    # plt.plot(train_r2_scores, label='Train R²')
    # plt.plot(test_r2_scores, label='Test R²')
    # plt.xlabel('Epoch')
    # plt.ylabel('R² Score')
    # plt.title('Train and Test R² Scores Over Epochs')
    # plt.legend()
    # Save the plot in memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    # Encode the plot data to base64
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')


    # Function to process a new molecule
    def process_new_molecule(new_molecule, feature_encodings, max_length):
        return process_row(new_molecule, feature_encodings, max_length)


    # Function to predict for a new molecule
    def predict_new_molecule(new_molecule_data):
        model.eval()
        predictions = []
        with torch.no_grad():
            for data in new_molecule_data:
                output = model(data)
                predictions.append(output.view(-1).tolist())
        return predictions


    # Function to clean tuples by removing white spaces
    def clean_tuples(tuples_list):
        return [(item[0].strip(), item[1].strip()) for item in tuples_list]


    # Function to get molecule data for a given lipid
    def get_molecule_data(lipid_name):
        filtered_df = df[
            df["Lipid composition (molar)"].str.contains(fr'\b{lipid_name}\b', regex=True, case=False, na=False)]

        if filtered_df.empty:
            return f"No data found for lipid: {lipid_name}"
        lipid_data = filtered_df.iloc[0]
        # Clean the node_features and edge data
        node_features = clean_tuples(ast.literal_eval(lipid_data['node_features']))
        edge = clean_tuples(ast.literal_eval(lipid_data['edge']))

        # Convert to string without additional quotes
        node_features_str = str(node_features)
        edge_str = str(edge)

        molecule_data = {
            'node_features': node_features_str,
            'edge': edge_str,
            'kappa, kT (q^-4)': 0  # This field is taken as is from the dataframe
        }
        return molecule_data


    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, 'moleculesEDited.csv')
    df = pd.read_csv(file_path)
    pressed = int(data.get('issingle'))
    prediction_value = None
    if pressed == 2:
        lipid_name = data.get('lipid_name')
        percentage = float(data.get('percentage'))
        lipid_name2 = data.get('lipid_name2')
        percentage2 = float(data.get('percentage2'))
        molecule_data = get_molecule_data(lipid_name)
        molecule_data2 = get_molecule_data(lipid_name2)
        processed_POPC = process_new_molecule(molecule_data, feature_encodings, max_length)
        processed_POPE = process_new_molecule(molecule_data2, feature_encodings, max_length)

        prediction_POPC = predict_new_molecule(DataLoader([processed_POPC], batch_size=1))
        prediction_POPE = predict_new_molecule(DataLoader([processed_POPE], batch_size=1))
        # if isinstance(molecule_data, str):
        #     print(molecule_data)
        #     print(molecule_data2)
        # else:
        #     print(f"{lipid_name} = {molecule_data}")
        #     print(f"{lipid_name2} = {molecule_data2}")
        prediction_value = (percentage / 100) * prediction_POPC[0][0] + (percentage2 / 100) * prediction_POPE[0][0]
        # print("Combined Prediction for", percentage, '% ', lipid_name, "and", percentage2, " %", lipid_name2, " : ",
        #       combined_prediction)
    else:
        lipid_name = data.get('lipid_name')
        molecule_data = get_molecule_data(lipid_name)
        # if isinstance(molecule_data, str):
        #     print(molecule_data)
        # else:
        #     print(f"{lipid_name} = {molecule_data}")
        processed_new_molecule = process_new_molecule(molecule_data, feature_encodings, max_length)
        new_molecule_loader = DataLoader([processed_new_molecule], batch_size=1)
        prediction_value = predict_new_molecule(new_molecule_loader)
        # print("individual prediction for ", lipid_name, predictions)
    return JsonResponse({'graph': plot_data,
                         'pred': prediction_value})
