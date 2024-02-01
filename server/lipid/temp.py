import json
import os

from django.shortcuts import render
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

from .gnn_kappa_prediction.src.train_model import train_model
# from .gnn_kappa_prediction.src.predict_model import predict_model

from .static.Predict_Value.Predict_value import predict_value
from .static.gnn_molecule_edge_only import edge_pred
from .gnn_kappa_prediction.src.extract_dataset import extract_dataset
plt.switch_backend('agg')


# Accessing file data
def make_dataset(request):
    adjacency_file = request.FILES.get('adjacencyFile')
    node_feature_file = request.FILES.get('nodeFeatureFile')
    # Accessing text and other data
    adjacency_text = request.POST.get('adjacencyText')
    node_feature_text = request.POST.get('nodeFeatureText')
    type = request.POST.get('type')
    print(type)
    compositions = request.POST.get('compositions')
    data = request.POST.get('data')
    compositions = json.loads(compositions)
    print(compositions)
    data = json.loads(data)
    comp_name = compositions["comp1"]["name"]
    if type=='single':
        comp_name_format=f'{compositions["comp1"]["percentage"]}% {compositions["comp1"]["name"]}'
    else:
        comp_name_format = f'{compositions["comp1"]["percentage"]}% {compositions["comp1"]["name"]},{compositions["comp2"]["percentage"]}% {compositions["comp2"]["name"]}'

    # Specify the desired path to save the adjacency_file
    current_directory = os.path.dirname(__file__)
    save_path = os.path.join(current_directory, f'./gcn_model/data/TextFiles/Adjacency_Matrix/{comp_name} .txt')
    # Save the adjacency_file to the specified path
    with open(save_path, 'wb') as destination:
        for chunk in adjacency_file.chunks():
            destination.write(chunk)
    save_path = os.path.join(current_directory,  f'./gcn_model/data/TextFiles/Node_Features/{comp_name}.txt')
    # Save the adjacency_file to the specified path
    with open(save_path, 'wb') as destination:
        for chunk in node_feature_file.chunks():
            destination.write(chunk)

    json_data = {"Composition":comp_name_format,
                 "N_water": data["Number of Water"],
                 "Temperature (K)":  data["Temperature"],
                 "N Lipids/Layer": data["Number of Lipid Per Layer"],
                 "Avg Membrane Thickness": data["Membrane Thickness"],
                 "Kappa (BW-DCF)":  data["Kappa BW DCF"],
                 "Kappa (RSF)": data["Kappa RSF"]}
    # Convert the JSON data to a DataFrame
    df_to_add = pd.DataFrame([json_data])

    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, './gcn_model/data/Merged_and_Sorted_Df.csv')
    existing_df = pd.read_csv(file_path)
    # Append new data to the existing DataFrame
    Merged_and_Sorted_Df = pd.concat([existing_df, df_to_add], ignore_index=True)
    # print("\nChanges:")
    # print(Merged_and_Sorted_Df[~existing_df.isin(Merged_and_Sorted_Df)].dropna())

    Merged_and_Sorted_Df.to_csv(file_path,index=False)
    print(Merged_and_Sorted_Df.tail())
    extract_dataset(Merged_and_Sorted_Df)

    train_model()

    # return JsonResponse({'result_json': train_model(),
    #                      # 'prediction': predict_model(comp_name_format, data, type),
    #                      })