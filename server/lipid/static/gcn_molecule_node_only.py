import os

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder

def gcn_node_prediction(lipid_comp):
    # Load the data
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, 'Lipid_Composition_Nodes_Only_final.csv')
    data = pd.read_csv(file_path)
    # Convert string representations of sets to actual sets
    def convert_to_set(row):
        return set(row.strip('{}').replace("'", "").split(', ')) if row.strip('{}') else set()

    data['All Node Features'] = data['All Node Features'].apply(convert_to_set)

    # Encoding 'All Node Features' as the target
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(data['All Node Features'])

    # One-hot encoding for 'Lipid Comp'
    lipid_comp_encoder = OneHotEncoder(sparse=False)
    X_lipid_comp = lipid_comp_encoder.fit_transform(data[['Lipid Comp']])

    # Combine all features
    X_tensor = torch.tensor(X_lipid_comp, dtype=torch.float)
    Y_tensor = torch.tensor(Y, dtype=torch.float)

    # Simplified Model Implementation
    class GCN(torch.nn.Module):
        def __init__(self, num_node_features):
            super(GCN, self).__init__()
            self.fc1 = torch.nn.Linear(num_node_features, 16)
            self.fc2 = torch.nn.Linear(16, len(mlb.classes_))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return x

    model = GCN(num_node_features=X_tensor.shape[1])

    # Training and Prediction
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = F.binary_cross_entropy_with_logits(out, Y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Function for Predicting Node Features
    def predict_node_features(node_data):
        model.eval()
        with torch.no_grad():
            node_data_tensor = torch.tensor(node_data, dtype=torch.float).unsqueeze(0)
            output = model(node_data_tensor)
            predicted_features = torch.sigmoid(output).squeeze(0)
            return predicted_features

    def predict_features_for_lipid_comp(lipid_comp):
        # Transform the input lipid composition
        input_data = pd.DataFrame({'Lipid Comp': [lipid_comp]})
        input_encoded = lipid_comp_encoder.transform(input_data)

        # Get the prediction from the model
        predicted_tensor = predict_node_features(input_encoded[0])
        predicted_features = (predicted_tensor > 0.5).int().numpy()

        # Convert the binary labels back to feature names
        predicted_features_array = np.array([predicted_features])  # Convert to a 2D NumPy array
        predicted_feature_names = mlb.inverse_transform(predicted_features_array)[0]
        return set(predicted_feature_names)

    # Example Usage

    predicted_node_features = predict_features_for_lipid_comp(lipid_comp)

    return list(predicted_node_features)


