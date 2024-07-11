from zipfile import ZipFile
import pandas as pd
import copy
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from   torch.utils.data import TensorDataset, DataLoader, Dataset
from torch_geometric.data   import Data
from torch_geometric.loader import DataLoader  # Correct import

from sklearn.model_selection import train_test_split

from JunctionTree.utils import *
from JunctionTree.vocabulary import tokenize_dict

import cProfile
import pstats

import pyarrow.parquet as pq
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer



# with ZipFile('../zinc250k.zip') as zp:
#     data = pd.read_csv(zp.open('250k_rndm_zinc_drugs_clean_3.csv'))

# adjusted_smiles  = []
# for index, smile in enumerate(data['smiles']):
#   smile          = smile.replace('\n', '')
#   adjusted_smiles.append(smile)

# data['smiles'] = adjusted_smiles


class Datapreprocess:
    
    def __init__(self, data, targets = None, embedding_type = 'DeepChem'):
        
        self.data                                     = data
        
        if embedding_type == 'Self-Made':
            self.adjacency_substituted, self.graph_tokens = self.load_matricies(self.data)
        
        elif embedding_type == 'DeepChem':
            self.train_features, self.test_features       = self.deepchem_embeddings(self.data, targets = targets)
        
    def get_processed_data(self):
        return self.adjacency_substituted, self.graph_tokens
    
    def deepchem_embeddings(self, data, targets):
        featurizer            = MolGraphConvFeaturizer()
        
        if targets is not None:
            x_train, x_test, y_train, y_test = train_test_split(data, targets, stratify = targets)
            train_features                   = featurizer.featurize(x_train)
            test_features                    = featurizer.featurize(x_test)
            
        else:
            features                         = featurizer.featurize(data)
            
        # left off here figuring out how to split the datasets
        if targets is not None:
            y_train          = torch.tensor(y_train.values, dtype = torch.float32)
            y_test           = torch.tensor(y_test.values, dtype = torch.float32)
            
            train_features   = [self._extract_graphs(item, y_train[index]) for index, item in tqdm(enumerate(train_features), total = len(train_features))]
            train_features   = DataLoader(train_features, batch_size = 32)
            
            test_features    = [self._extract_graphs(item, y_test[index]) for index, item in tqdm(enumerate(test_features), total = len(test_features))]
            test_features    = DataLoader(test_features, batch_size = 32)
          
              
            return train_features, test_features

        else:
#             test_features        = dc.data.NumpyDataset(x = train_features)
#             test_features         = dc.data.NumpyDataset(X = features)
            
            return test_features
        
    
    def get_deepchem(self):
        return self.train_features, self.test_features 
    
    def pad_tensor(self, tensor, pad_value, max_size):
        """Pad a tensor to max_size with pad_value."""
        padding = torch.full((max_size - tensor.size(0),) + tensor.size()[1:], pad_value, dtype=tensor.dtype)
        return torch.cat([tensor, padding])

    def collate_fn(self, batch):
        node_features_list = [data.x for data in batch]
        edge_index_list = [data.edge_index for data in batch]
        targets = torch.stack([data.y for data in batch])

        max_num_nodes = max(node_features.size(0) for node_features in node_features_list)
        max_num_edges = max(edge_index.size(1) for edge_index in edge_index_list)

        node_features_padded = [self.pad_tensor(node_features, 0, max_num_nodes) for node_features in node_features_list]
        edge_indices_padded = [self.pad_tensor(edge_index.t(), 0, max_num_edges).t() for edge_index in edge_index_list]

        node_features_padded = torch.stack(node_features_padded)
        edge_indices_padded = torch.stack(edge_indices_padded)

        padded_batch = []
        for i in range(len(batch)):
            padded_batch.append(PyGData(x=node_features_padded[i], edge_index=edge_indices_padded[i], y=targets[i]))

        return Batch.from_data_list(padded_batch)

    
    def _extract_graphs(self, item, y):
        node_features = torch.tensor(item.node_features, dtype = torch.float32)
        edge_index    = torch.tensor(item.edge_index, dtype = torch.long)
        
        data_obj      = Data(x = node_features, edge_index = edge_index, y = y)
        return data_obj
        
        
    
    def load_unique_vocab(self, token_dictionary = None):
        ''' Load the unique vocab for our prebuitl dictionary
        if the string/vocab is not in the prebuilt vocab,
        then add it to the vocabulary dynamically

        Parameters:
            vocabulary: Vocabulary constructed from data using construct_vocabulary
            Token_dictionary: an empty initialized tokenization dictionary 

        Returns:
            token_dictionary: the prebuilt and additional tokens based on user data

        '''

        self.token_dictionary = tokenize_dict(None)

        return self.token_dictionary


    def load_matricies(self, data):
        ''' 
        Parameters:
            Data:

        Returns:
            adjacency_substituted: The users original molecules decomposed into a matrix representing the connectivity
                                   in a graph structure, as well as the vocabulary substituted into the connections
                                   [ 0 0 24 0 12 0] where 24 and 12 represent different tokens of molecular sturcutres
        '''

        # Convert Data Into Adjacency Matricies, Create Metadata Molecule Dataframes, and Junction Tree
        adjacency_matricies, molecule_df, junction_tree = smiles_to_matrix(data)
        

        # Pad Matricies to the same length (Max Length of the Longest Matrix)
#         padded_matricies                                = pad_matricies(adjacency_matricies)
        
        
        # Construct vocabulary 
        vocabulary, molecule_dictionary                 = construct_vocabulary(molecule_df, junction_tree, data)
        
#         print(self.data
        # Load the prebuilt vocabulary
        unique_vocab                                    = self.load_unique_vocab()

        token_dictionary, unique_vocab, valency_dict    = build_unique_vocabulary(vocabulary, data, unique_vocab, molecule_df)
        
        
#         adjacency_substituted                           = copy.copy(padded_matricies)
        graph_tokens                  = substitute_vocabulary(adjacency_matricies, token_dictionary, unique_vocab)

        return adjacency_matricies, graph_tokens
    

        
        

#     adjacency_substituted = load_matricies(data['smiles'])








