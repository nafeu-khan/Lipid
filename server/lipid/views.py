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

from .gcn_model.src.train_model import train_model
from .gcn_model.src.predict_model import predict_model

from .static.Predict_Value.Predict_value import predict_value
from .static.gnn_molecule_edge_only import edge_pred
from .gcn_model.src.extract_dataset import extract_dataset
from .temp import make_dataset

plt.switch_backend('agg')


@csrf_exempt
# Create your views here.
@api_view(['GET','POST'])
def get_data(request):
    return make_dataset(request)


@api_view(['GET','POST'])
def create_model(req):
    return train_model()
@api_view(['GET','POST'])
def predict_model(req):
    data=json.loads(req.body)
    predict_model(data)
@api_view(['GET','POST'])
def prediction(req):
    data=json.loads(req.body)
    # Load and preprocess the dataset
    current_directory = os.path.dirname(__file__)
    file_path = os.path.join(current_directory, 'final_dataset.csv')
    df = pd.read_csv(file_path)
    return predict_value(data,df)
@api_view(['GET','POST'])
def pred_edge(req):
    data = json.loads(req.body)
    print(data)
    return edge_pred(data.get("mol_name"))