import pandas as pd

from   d2l import torch as d2l
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from   torch.utils.data import TensorDataset, DataLoader, Dataset
from   torchsummary import summary
from   torchtext.data.utils import get_tokenizer
from   torchtext.datasets import AG_NEWS
from   transformers import AutoTokenizer
from   torch.nn.utils.rnn import pad_sequence
from   torch.utils.data import TensorDataset, DataLoader
from   transformers import AutoModel, AutoTokenizer
from   torch import nn

from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch.nn import LazyLinear, Transformer, Conv1d, Dropout, BatchNorm1d

from torch.nn import Embedding
from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader  # Correct import



from sklearn.metrics import accuracy_score
import psutil
from tqdm import tqdm
import scipy.sparse as sp

import sys
import pickle
import copy
import json
import os
import warnings
import csv
import h5py
import ast
import gc
import time

import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from datetime import datetime

import cProfile
import pstats


class Model1(torch.nn.Module):
    ''' 
    Second Model built with Simple Protein Convolutional Layrs and GCN for Molecular Graphs
    ''' 
    
    def __init__(self,
                 input_dim, 
                 hidden_dim = 256,
                 protein_size = 768, 
                 kernel_size = 10):
        super().__init__()
        
        padding_size      = kernel_size - 2
        
        # Layer Number Parameters
        self.input_dim    = input_dim
        self.hidden_dim   = hidden_dim
        self.dim_p1       = 512
        self.num_layers   = 4
        
        
        # Convolutional Number Parameters:
        self.protein_size = protein_size
        self.protein_dim1 = protein_size // 2
        self.protein_dim2 = protein_size // 4
        self.protein_dim3 = protein_size // 8
        self.concat_dim   = self.protein_dim2 + (hidden_dim * 2)
        
        
        # Molecular Graph Layers
        self.conv1        = GCNConv(input_dim, hidden_dim)
        self.bn1          = BatchNorm1d(hidden_dim)
        self.dropout1     = Dropout(0.4)
        self.conv2        = GCNConv(hidden_dim, hidden_dim)
        self.bn2          = BatchNorm1d(hidden_dim)
        self.dropout2     = Dropout(0.4)
        
        # Protein NN Layers
        self.prot_conv1   = Conv1d(protein_size, self.protein_dim1, kernel_size, padding = padding_size)
        self.prot_conv2   = Conv1d(self.protein_dim1, self.protein_dim2, kernel_size)
#         self.prot_conv3   = Conv1d(self.protein_dim2, self.protein_dim3, kernel_size)
        
        
        #         self.p_fc1        = LazyLinear(self.dim_p1)     
        #         self.p_fc2        = LazyLinear(64)
        
        

        # Joined NN Layers
#         self.tf1          = Transformer(d_model = self.concat_dim)
        self.fc1          = LazyLinear(256)
        self.fc2          = LazyLinear(128)
        self.fc3          = LazyLinear(64)
        self.out          = LazyLinear(1)
        
        
    def forward(self, data, protein_data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        
        # Molecular Graph
        x            = x.float()
        x            = self.dropout1(F.relu(self.bn1(self.conv1(x, edge_index))))
        x            = self.dropout2(F.relu(self.bn2(self.conv2(x, edge_index))))
#         x            = self.GAT1(x, edge_index)
        x            = global_mean_pool(x, batch)
        
        
        # Protein NN
#         protein_data = self.p_fc1(protein_data)
#         protein_data = self.p_fc2(protein_data)
        protein_data = protein_data.permute(0, 2, 1)
        
        protein_data = F.relu(self.prot_conv1(protein_data))
        protein_data = F.relu(self.prot_conv2(protein_data))
#         protein_data = F.relu(self.prot_conv3(protein_data))
        
        # FLatten to (batch_size, features)
        protein_data = protein_data.reshape(protein_data.size(0), -1)
        
        # Joined NN
        x            = torch.concatenate([x, protein_data], axis = 1)
#         x            = F.sigmoid(self.tf1(x, tgt = y))
        x            = F.relu(self.fc1(x))
        x            = F.relu(self.fc2(x))
        x            = F.relu(self.fc3(x))
        x            = F.sigmoid(self.out(x))
        
        return x
    
    
 # Example GCN model definition
class Model2(torch.nn.Module):
    '''
    First Model built Simple Fully Connected Protein parallel with GCN for Molecular Graphs
    ''' 
    def __init__(self,
                 input_dim, 
                 hidden_dim = 256,
                 protein_size = 1024, 
                 kernel_size = 10):
        
        super().__init__()
        
        padding_size      = kernel_size - 2
        
        # Layer Number Parameters
        self.input_dim    = input_dim
        self.hidden_dim   = hidden_dim
        self.dim_p1       = 256
        self.num_layers   = 4
        
        
        # Convolutional Number Parameters:
#         self.concat_dim   = self.protein_dim2 + (hidden_dim * 2)
        
        
        # Molecular Graph Layers
        self.conv1        = GCNConv(input_dim, hidden_dim)
        self.bn1          = BatchNorm1d(hidden_dim)
        self.dropout1     = Dropout(0.4)
        self.conv2        = GCNConv(hidden_dim, hidden_dim)
        self.bn2          = BatchNorm1d(hidden_dim)
        self.dropout2     = Dropout(0.4)
        
        # Protein NN Layers
        self.p_fc1        = LazyLinear(self.dim_p1)     
        self.p_fc2        = LazyLinear(64)

        # Joined NN Layers
        self.fc1          = LazyLinear(128)
        self.fc2          = LazyLinear(64)
        self.fc3          = LazyLinear(32)
        self.fc4          = LazyLinear(1)

    
        
        
    def forward(self, data, protein_data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        
        # Molecular Graph
        x            = x.float()
        x            = self.dropout1(F.relu(self.bn1(self.conv1(x, edge_index))))
        x            = self.dropout2(F.relu(self.bn2(self.conv2(x, edge_index))))
#         x            = self.GAT1(x, edge_index)
        x            = global_mean_pool(x, batch)
        
        
        # Protein NN
#         print(protein_data.size())
        protein_data = protein_data.reshape(protein_data.size(0), - 1)
        protein_data = F.relu(self.p_fc1(protein_data))
        protein_data = F.relu(self.p_fc2(protein_data))
        
        # Joined NN
#         print(x.shape, protein_data.shape)
        x            = torch.concatenate([x, protein_data], axis = 1)
        x            = F.relu(self.fc1(x))
        x            = F.relu(self.fc2(x))
        x            = F.relu(self.fc3(x))
        x            = F.sigmoid(self.fc4(x))
        
        return x
    

 # Example GCN model definition
class Model3(torch.nn.Module):
    '''
    First Model built Simple Fully Connected Protein parallel with GCN for Molecular Graphs
    ''' 
    def __init__(self,
                 input_dim, 
                 hidden_dim = 256,
                 protein_size = 1024, 
                 kernel_size = 10):
        
        super().__init__()
        
        padding_size      = kernel_size - 2
        
        # Layer Number Parameters
        self.input_dim    = input_dim
        self.hidden_dim   = hidden_dim
        self.dim_p1       = 256
        self.num_layers   = 4
        
        
        # Convolutional Number Parameters:
#         self.concat_dim   = self.protein_dim2 + (hidden_dim * 2)
        
        
        # Molecular Graph Layers
        self.conv1        = GCNConv(input_dim, hidden_dim)
        self.bn1          = BatchNorm1d(hidden_dim)
        self.dropout1     = Dropout(0.4)
        self.conv2        = GCNConv(hidden_dim, hidden_dim)
        self.bn2          = BatchNorm1d(hidden_dim)
        self.dropout2     = Dropout(0.4)
        
        # Protein NN Layers
        self.p_fc1        = LazyLinear(self.dim_p1)     
        self.p_fc2        = LazyLinear(64)

        # Joined NN Layers
        self.fc1          = LazyLinear(1)

    
        
        
    def forward(self, data, protein_data):
        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y
        
        # Molecular Graph
        x            = x.float()
        x            = x
        x            = self.dropout1(F.relu(self.bn1(self.conv1(x, edge_index))))
        x            = self.dropout2(F.relu(self.bn2(self.conv2(x, edge_index))))
#         x            = self.GAT1(x, edge_index)
        x            = global_mean_pool(x, batch)
        
        
        # Protein NN
        protein_data = protein_data.reshape(protein_data.size(0), - 1)
        protein_data = self.p_fc1(protein_data)
        protein_data = self.p_fc2(protein_data)
        
        # Joined NN
        x            = torch.concatenate([x, protein_data], axis = 1)
        x            = F.sigmoid(self.fc1(x))
        
        return x
    
    
class MorganModel(nn.Module):
    
    def __init__(self,
                 hidden_dim = 256,
                 protein_size = 1024, 
                 kernel_size = 10):
           
        super().__init__()
        
        padding_size      = kernel_size - 2
        
        # Layer Number Parameters
        self.hidden_dim   = hidden_dim
        self.dim_p1       = 512
        self.num_layers   = 4
        
        # Fully Connected Morgan
        self.mfc1         = nn.LazyLinear(256)
        self.mfc2         = nn.LazyLinear(128)
        self.mfc3         = nn.LazyLinear(64)
        
        # Convolutional Number Parameters:
        self.protein_size = protein_size
        self.protein_dim1 = protein_size // 2
        self.protein_dim2 = protein_size // 4
        self.protein_dim3 = protein_size // 8
        self.concat_dim   = self.protein_dim2 + (hidden_dim * 2)
        
        # Protein NN Layers
        self.prot_conv1   = Conv1d(protein_size, self.protein_dim1, kernel_size, padding = padding_size)
        self.prot_conv2   = Conv1d(self.protein_dim1, self.protein_dim2, kernel_size)
        
        # Joined NN Layers
#         self.tf1          = Transformer(d_model = self.concat_dim)
        
        self.fc1          = LazyLinear(256)
        self.fc2          = LazyLinear(128)
        self.fc3          = LazyLinear(64)
        self.out          = LazyLinear(1)
        
    
    def forward(self, x, protein_data):
        x            = x.float()
        x            = self.mfc1(x)
        x            = self.mfc2(x)
        x            = self.mfc3(x)
        
        protein_data = protein_data.permute(0, 2, 1)
        protein_data = F.relu(self.prot_conv1(protein_data))
        protein_data = F.relu(self.prot_conv2(protein_data))
        
        # FLatten to (batch_size, features)
        protein_data = protein_data.reshape(protein_data.size(0), -1)
        x            = torch.concatenate([x, protein_data], axis = 1)
        x            = F.sigmoid(self.out(x))
        
        return x
        
        
class ResidualBlock(nn.Module):
    def __init__(self, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.LazyLinear(out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.LazyLinear(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        out += residual
        out = torch.relu(out)
        return out

class ResModel(nn.Module):
    def __init__(self):
        super(ResModel, self).__init__()
        
        # Molecular Graph Fully Connected Layers
        self.m_fc1 = nn.LazyLinear(256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.m_fc2 = nn.LazyLinear(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        # Protein NN Fully Connected Layers
        self.p_fc1 = nn.LazyLinear(256)
        self.p_fc2 = nn.LazyLinear(128)
        
        # Joined Fully Connected Layers with Residual Blocks
        self.fc1 = nn.LazyLinear(256)
        self.res1 = ResidualBlock(256)
        self.fc2 = nn.LazyLinear(128)
        self.res2 = ResidualBlock(128)
        self.fc3 = nn.LazyLinear(64)
        self.fc4 = nn.LazyLinear(1)
    
    def forward(self, data, protein_data):
        # Molecular Graph
        x = data.float()
        x = self.dropout1(F.relu(self.bn1(self.m_fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.m_fc2(x))))
        
        # Protein NN
        protein_data = protein_data.reshape(protein_data.size(0), -1)
        protein_data = F.relu(self.p_fc1(protein_data))
        protein_data = F.relu(self.p_fc2(protein_data))
        
        # Joined NN
        x = torch.cat([x, protein_data], dim=1)
        x = F.relu(self.fc1(x))
        x = self.res1(x)
        x = F.relu(self.fc2(x))
        x = self.res2(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        
        return x
        
        

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        self.fc1 = nn.LazyLinear(1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.LazyLinear(512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.LazyLinear(256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = nn.LazyLinear(128)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.fc5 = nn.LazyLinear(64)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.fc6 = nn.LazyLinear(1)
        
    def forward(self, x, protein_data):
        x = x.float()
        protein_data = protein_data.reshape(protein_data.size(0), -1)
        x = torch.cat([x, protein_data], dim=1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = torch.sigmoid(self.fc6(x))
        return x
        
        
        
        
        
        
        
        
    
    