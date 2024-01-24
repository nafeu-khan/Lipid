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

from .static.Predict_Value.Predict_value import predict_value
from .static.gnn_molecule_edge_only import edge_pred

plt.switch_backend('agg')

@api_view(['GET','POST'])
@csrf_exempt
# Create your views here.
def prediction(req):
    data=json.loads(req.body)
    # Load and preprocess the dataset
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, 'filtered_molecule_Edit.csv')
    df = pd.read_csv(file_path)

    return predict_value(data,df)
@api_view(['GET','POST'])
def pred_edge(req):
    data = json.loads(req.body)
    print(data)
    return edge_pred(data.get("mol_name"))