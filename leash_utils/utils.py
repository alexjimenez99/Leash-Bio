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

from JunctionTree.main import Datapreprocess
from transformers import EsmTokenizer, EsmModel

from tape import ProteinBertModel, TAPETokenizer

import pyarrow.parquet as pq
import warnings
import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer
from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

from datetime import datetime, timedelta
import pytz


# **First Protein Target from Competition Description**
# 1. Epoxide Hydrolase 2
# 2. High Blood pressure and diabetes target

def _load_proteins():
    
    EPHX2 = '''MTLRAAVFDLDGVLALPAVFGVLGRTEEALALPRGLLNDAFQKGGPEGATTRLMKGEI
    TLSQWIPLMEENCRKCSETAKVCLPKNFSIKEIFDKAISARKINRPMLQAALMLRKKGFTTA
    ILTNTWLDDRAERDGLAQLMCELKMHFDFLIESCQVGMVKPEPQIYKFLLDTLKASPSEVVF
    LDDIGANLKPARDLGMVTILVQDTDTALKELEKVTGIQLLNTPAPLPTSCNPSDMSHGYVTV
    KPRVRLHFVELGSGPAVCLCHGFPESWYSWRYQIPALAQAGYRVLAMDMKGYGESSAPPEIE
    EYCMEVLCKEMVTFLDKLGLSQAVFIGHDWGGMLVWYMALFYPERVRAVASLNTPFIPANPNM
    SPLESIKANPVFDYQLYFQEPGVAEAELEQNLSRTFKSLFRASDESVLSMHKVCEAGGLFVNS
    PEEPSLSRMVTEEEIQFYVQQFKKSGFRGPLNWYRNMERNWKWACKSLGRKILIPALMVTAEK
    DFVLVPQMSQHMEDWIPHLKRGHIEDCGHWTQMDKPTEVNQILIKWLDSDARNPPVVSKM'''

    # EPHX2 = EPHX2[1:554] # Commerically Purchased Variant is amino 2-555

    # **Second Protein Target from Competition Description**
    # 1. Bromo Domain 4
    # 2. Cancer disease progression (histone protein)

    BRD4 = '''MSAESGPGTRLRNLPVMGDGLETSQMSTTQAQAQPQPANAASTNPPPPETSNPNKPKRQTN
    QLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQEC
    IQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEETEIMIVQAKGRGRGRKETGTA
    KPGVSTVPNTTQASTPPQTQTPQPNPPPVQATPHPFPAVTPDLIVQTPVMTVVPPQPLQTPPPV
    PPQPQPPPAPAPQPVQSHPPIIAATPQPVKTKKGVKRKADTTTPTTIDPIHEPPSLPPEPKTTK
    LGQRRESSRPVKPPKKDVPDSQQHPAPEKSSKVSEQLKCCSGILKEMFAKKHAAYAWPFYKPVD
    VEALGLHDYCDIIKHPMDMSTIKSKLEAREYRDAQEFGADVRLMFSNCYKYNPPDHEVVAMARK
    LQDVFEMRFAKMPDEPEEPVVAVSSPAVPPPTKVVAPPSSSDSSSDSSSDSDSSTDDSEEERAQ
    RLAELQEQLKAVHEQLAALSQPQQNKPKKKEKDKKEKKKEKHKRKEEVEENKKSKAKEPPPKKT
    KKNNSSNSNVSKKEPAPMKSKPPPTYESEEEDKCKPMSYEEKRQLSLDINKLPGEKLGRVVHII
    QSREPSLKNSNPDEIEIDFETLKPSTLRELERYVTSCLRKKRKPQAEKVDVIAGSSKMKGFSSS
    ESESSSESSSSDSEDSETEMAPKSKKKGHPGREQKKHHHHHHQQMQQAPAPVPQQPPPPPQQPP
    PPPPPQQQQQPPPPPPPPSMPQQAAPAMKSSPPPFIATQVPVLEPQLPGSVFDPIGHFTQPILH
    LPQPELPPHLPQPPEHSTPPHLNQHAVVSPPALHNALPQQPSRPSNRAAALPPKPARPPAVSPA
    LTQTPLLPQPPMAQPPQVLLEDEEPPAPPLTSMQMQLYLQQLQKVQPPTPLLPSVKVQSQPPPP
    LPPPPHPSVQQQLQQQPPPPPPPQPQPPPQQQHQPPPRPVHLQPMQFSTHIQQPPPPQGQQPPH
    PPPGQQPPPPQPAKPQQVIQHHHSPRHHKSDPYSTGHLREAPSPLMIHSPQMSQFQSLTHQSPP
    QQNVQPKKQELRAASVVQPQPLVVVKEEKIHSPIIRSEPFSPSLRPEPPKHPESIKAPVHLPQR
    PEMKPVDVGRPVIRPPEQNAPPPGAPDKDKQKQEPKTPVAPKKDLKIKNMGSWASLVQKHPTTP
    SSTAKSSSDSFEQFRRAAREKEEREKALKAQAEHAEKEKERLRQERMRSREDEDALEQARRAHE
    EARRRQEQQQQQRQEQQQQQQQQAAAVAAAATPQAQSSQPQSMLDQQRELARKREQERRRREAM
    AATIDMNFQSDLLSIFEENLF'''


    BRD4   =   BRD4[3:459] # Positions 4-460



    # **Third Protein From Competition Description**
    # 1. Serum Albumin
    # 2. The most common protein in the blood

    P02768 = '''MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFE
    DHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECF
    LQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTE
    CCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVS
    KLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDE
    MPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCA
    AADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVS
    RNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALE
    VDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEK
    CCKADDKETCFAEEGKKLVAASQAALGL'''


    P02768 = P02768[24:608] # Positions 25-609
    
    return EPHX2, BRD4, P02768



def _load_protein_model(model_name):
    
    if model_name == 'Rostlab/prot_bert':
        tokenizer           = AutoTokenizer.from_pretrained(model_name)
        model               = AutoModel.from_pretrained(model_name)
    
    elif model_name == 'facebook/esm2_t33_650M_UR50D':
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model     = EsmModel.from_pretrained(model_name)
        
    elif model_name == 'ProteinBert':
        tokenizer = TAPETokenizer(vocab='iupac')
        model     = ProteinBertModel.from_pretrained('bert-base')
        
    
    return model, tokenizer


def create_protein_embeddings(model_name = 'Rostlab/prot_bert'):
    ''' 
    This function takes protein sequences and turns them into
    protein embeddings using the Rostlab ProtBert model
    
    Parameters:
    - 
    - 
    
    Returns
    - 
    - 
    
    ''' 
    
    EPHX2, BRD4, P02768 = _load_proteins()
    
    if model_name == 'factorize':
        amino_acid_dict = {
        "A": 1, "R": 2, "N": 3, "D": 4, "C": 5,
        "Q": 6, "E": 7, "G": 8, "H": 9, "I": 10,
        "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
        "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20
        }
        
        proteins            = {'sEH': EPHX2, 'BRD4': BRD4, 'HSA': P02768}
        embedded_sequences  = {}
        for name, amino_acid in proteins.items():
            amino_acid                = amino_acid.replace('\n', '')
            amino_acid                = amino_acid.replace(' ', '')
            embedded_sequences[name]  = [amino_acid_dict[aa] for aa in amino_acid]
            embedded_sequences[name]  = torch.tensor(embedded_sequences[name], dtype = torch.float32).unsqueeze(0)
        
        
        longest_padding    = max([i.size(1) for i in embedded_sequences.values()])
        embedded_sequences = {name: F.pad(tensor, (longest_padding - tensor.size(1), 1),
                                                   mode = 'constant', value = 0)
                                                   for name, tensor in embedded_sequences.items()}
        
      
    else:
        # Load Proteins and Model
        
        model, tokenizer    = _load_protein_model(model_name)

        # Initialize Protein Dictionaries
        embedded_sequences  = {}
        proteins            = {'sEH': EPHX2, 'BRD4': BRD4, 'HSA': P02768}

        for name, sequence in proteins.items():
            if model_name == 'ProteinBert':
                sequence        = sequence.replace('\n', '')
                sequence        = sequence.replace(' ', '')
                inputs          = torch.tensor([tokenizer.encode(sequence)], dtype = torch.long)

                with torch.no_grad():
                    print(inputs.size())
                    outputs    = model(inputs)
                    embeddings = outputs[0]
                    print(embeddings.size())

            else:
                inputs          = tokenizer(sequence, return_tensors="pt")


                with torch.no_grad():
                    outputs       = model(**inputs)
                    embeddings    = outputs.last_hidden_state
                    pooled_output = outputs.pooler_output


            # Reshape for pytorch pad function
            embedded_sequences[name] = embeddings.permute(0, 2, 1)

        # Add 0 Padding along embedding dim to homogenize shape of inputs
        longest_padding       = max([i.size(2) for i in embedded_sequences.values()])
        embedded_sequences    = {name: F.pad(tensor, (longest_padding - tensor.size(2), 0),
                                                   mode = 'constant', value = 0).permute(0, 2, 1)
                                       for name, tensor in embedded_sequences.items()}
        
    
    return embedded_sequences




def open_train_test(class_0_size = 1500,
                      test_file    = 'test_df.pckl',
                      train_file   = 'train_df_truncated.pckl'):
    ''' 
    Load the full test and partial train data
    
    Parameters:
    
    
    Returns:
    
    
    '''

    
    test_data             = pickle.load(open(test_file, 'rb'))
    train_truncated       = pickle.load(open(train_file, 'rb'))
    
    train_bind            = train_truncated[train_truncated['binds']== 1]
    train_no              = train_truncated[train_truncated['binds'] == 0][:class_0_size]
    
    train_data            = pd.concat([train_bind, train_no])
    train_data            = train_data.reset_index()
    
    train_data            = train_data.drop(columns = ['index'])
    
    return test_data, train_truncated



def create_graph_embeddings(test_data, output_name = 'test_processed_batched', batch_size = 100000, start_index = 0, write_json = False):
    ''' 
    Preprocess smiles in batches using JunctionTree folder functions
    JunctionTree decomposes molecular graphs
    
    Parameters:
    - output_name (str): Desired Export Prefix
    - batch_size (int): Size to batch data in
    
    Returns:
    - None
    
    '''
    smiles_group = {}
    batches      = math.ceil(len(test_data) / batch_size)
    train_smiles = list(test_data['molecule_smiles'])
    index        = start_index
    index_json   = start_index * batch_size
    
#     print(start_index)
    for batch in range(start_index, batches):
        print(f'Executing Batch {batch + 1}')
        
        if batch <= batches-1:
            train_batch = train_smiles[batch*batch_size:(batch + 1)*batch_size]   
        else:
            train_batch = train_smiles[batch*batch_size:]

        train_batch     = Datapreprocess(train_batch)
        train_adjacency, graph_tokens  = train_batch.get_processed_data()
        
        # JSON File are being dynamically updated now
#         token_dictionary = preprocess.token_dictionary

        
        pickle_file      = output_name + f'_Batch_{batch}.pckl'
        pickle_file      = pickle.dump(train_adjacency, open(pickle_file, 'wb'))
        
         
        if write_json:
            
            json_file        = output_name + '.json' 
            if os.path.exists(json_file):
                mode = 'a'
            else:
                mode = 'w'

    #         # Write Vocab Tokens to Json File to be loaded later
            with open(json_file, mode, buffering=1) as file:
                for key, value in graph_tokens.items():
                    # Serialize and write each key-value pair as a JSON string per line

                    json_str = json.dumps({index_json: value})
                    file.write(json_str + "\n")
                    index_json += 1
                print('Exited Json Writing')
               
    return 



def get_memory_usage():
    # Get the memory usage statistics
    memory           = psutil.virtual_memory()
    gb_conversion    = 1024 ** 3
    total_memory     = memory.total / gb_conversion
    used_memory      = memory.used / gb_conversion
    available_memory = memory.available / gb_conversion

    print(f"Available Memory: {available_memory:.2f} GB")
    
    return available_memory





def load_vocab_tokens(base_path):
    ''' 
    Returns vocabulary tokens for the training data set created by
    Processing all test data through JunctionTree algorithm
    -------------------------------------------------------------
    -------------------------------------------------------------
    
    Parameters:
    - base_path (str): Name of File without extension
    
    Returns:
    - vocab_tokens (list): Molecular graph tokens
    
    '''
    
    vocab_tokens  = {}
    base_path     = base_path + '.txt'
    
    with open(base_path, 'r') as txt:
        
        for index, line in tqdm(enumerate(txt)):
            dict_line           = ast.literal_eval(line.strip())
            vocab_tokens.update(dict_line)  
#             if index == (final_index - 1):
#                 break
    
    
    return vocab_tokens

    
    

def load_embeddings(base_path, index = None, mode = 'batch'):
    
    '''
    Opening the prepocessed files with memory statistics
    to understand the memory usage needed
    -------------------------------------
    -------------------------------------
    
    Parameters
    - base_path (str): Name of file without extension
    - index (int): Index of batch
    - mode (str): Opening Data Mode
    
    Returns:
    - data_batches (list): 
    
    '''
    global data_batches
    
    data_batches                = {}
    
    if mode == 'all':
        for index in tqdm(range(batches), total = batches):
            get_memory_usage()
            t1                  = time.time()
            batch_path          = base_path + f'_{index}.pckl'
            train_batch         = pickle.load(open(batch_path, 'rb'))
            data_batches[index] = train_batch
            
            t2                  = time.time()
        
        print('Time Elapsed:', t2 - t1)
        
    elif mode == 'batch' and index is not None:
            get_memory_usage()
            batch_path          = base_path + f'_{index}.pckl'
            data_batches        = pickle.load(open(batch_path, 'rb'))
        
    return data_batches





def prepare_gcn_data(train_adjacency, 
                     graph_tokens, 
                     batch_size = None,
                     batch_index = None, 
                     targets = None, 
                     mode = 'Evaluate'):
    
    '''
    Create the Data Object that is compatible with Graph Convolutional Networks
    ----------------------------------------------------------------
    ----------------------------------------------------------------
    
    
    Parameters
    - train_adjacency:
    - graph_tokens:
    - targets:
    - batch_size:
    - batch_index:
    - targets:
    - mode:
    
    
    Returns
    - final_data:
    
    '''
    
    graphs             = []
    feature_size       = 0
    if mode == 'Evaluate':
        for index in range(len(train_adjacency)):

            adj_matrix = train_adjacency[index]
            row, col   = np.where(adj_matrix > 0)

            # Bond information should flow both ways in graph
            bi_message = [[row, col], [col, row]]
            edge_index = torch.tensor(bi_message, dtype=torch.long)

            # Vocab tokens were exported as 1-1,500,000
            str_index  = str(index + (batch_size * batch_index))
            x_features = torch.tensor(graph_tokens[str_index], dtype = torch.long)
            x_features = x_features.unsqueeze(1)

            if targets is not None:
                y_data     = targets[index]
                graph      = Data(x=x_features, edge_index=edge_index, y = y_data)
            else:
                graph      = Data(x=x_features, edge_index=edge_index)

            graphs.append(graph)

    elif mode == 'Build':
        for index in range(len(train_adjacency)):

            adj_matrix   = train_adjacency[index]
            row, col     = np.where(adj_matrix > 0)
            # Bond information should flow both ways in graph
            bi_message   = [[row, col], [col, row]]
            edge_index   = torch.tensor(bi_message, dtype=torch.long)

            # Expand dim so that dim = 1 can be set in the model definition
            x_features   = torch.tensor(graph_tokens[index], dtype = torch.long)
            x_features   = x_features.unsqueeze(1)
 
            
            if targets is not None:
                y_data   = targets[index]
                graph    = Data(x=x_features, edge_index=edge_index, y = y_data)
            
            else:
                graph    = Data(x=x_features, edge_index=edge_index)
           
            graphs.append(graph)

    final_data = DataLoader(graphs, batch_size = 32)
    
    return final_data


def _embed_proteins(data, embedded_sequences, embed_type = 'torch'):
    '''
    '''
    
    if embed_type == 'numpy':
        embedded_sequences = {name:np.array(protein) for name, protein in embedded_sequences.items()}
        
    proteins        = data['protein_name']
    proteins        = list(proteins.apply(lambda x: embedded_sequences[x]))
   
    return proteins


def _prepare_data(data_batch,
                 vocab_tokens, 
                 embedded_sequences,
                 index      = None,
                 batch_size = None, 
                 targets    = None, 
                 mode       = 'Build', 
                 test       = False, 
                 data       = None, 
                 ):
    ''' 
    Create the protein dataset and the GCN dataset that is used for training
    --------------------------------------
    --------------------------------------
    
    Parameters:
    - data_batch:
    - batch_size:
    - index:
    - vocab_tokens:
    - mode:
    - test:
    - data:
    
    Returns:
    - final_data
    - proteins
    
    '''
    
   
    if mode == 'Build':
        proteins    = _embed_proteins(data, embedded_sequences)
#         proteins    = data['protein_name']
#         proteins    = list(proteins.apply(lambda x: embedded_sequences[x]))
#         proteins    = torch.concatenate([p for p in proteins], axis = 0)
        
        if not test:
            final_data  = prepare_gcn_data(data_batch, vocab_tokens, targets = targets, mode = mode)
        else:
            final_data  = prepare_gcn_data(data_batch, vocab_tokens, targets = targets, mode = mode)

    elif mode == 'Evaluate' and index is not None and batch_size is not None:
        
        proteins      = test_data['protein_name'][index*batch_size:(index + 1)*batch_size]
        proteins      = list(proteins.apply(lambda x: embedded_sequences[x]))
        proteins      = torch.concatenate([p.unsqueeze(0) for p in proteins], axis = 0)
        final_data    = prepare_gcn_data(data_batch, vocab_tokens, batch_size, index, mode = mode)
        
    else:
        raise ValueError(f'{mode} not compatibile when index argument is None')
        
    
    return final_data, proteins


def _reset_drop(data):
    ''' 
    Commonly used cleaning operation
    
    Parameters:
    - data (pd.DataFrame): Data Dataframe
    
    Returns:
    - data: Cleaned DataFrame
    
    '''
    
    data               = data.reset_index()
    data               = data.drop(columns = ['index'])
    
    return data

def smiles_to_maccs(smiles):
    """
    Converts a SMILES string to MACCS keys.

    Parameters:
    smiles (str): A string representing a molecule in SMILES notation.

    Returns:
    list: A list of MACCS keys.
    """
    
    maccs_list = []
    for smile in tqdm(smiles, total = smiles.shape[0]):
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            raise ValueError(f"Invalid SMILES string: {smile}")
            
        maccs_keys = np.array(MACCSkeys.GenMACCSKeys(molecule))
        
        maccs_list.append(maccs_keys)

    return np.array(maccs_list)

# Function to compute Morgan fingerprint
def compute_morgan_fingerprint(smiles, radius=3, n_bits=2048):
    
    generator   = rdFingerprintGenerator.GetMorganGenerator(radius = radius, fpSize = n_bits)
    smiles_list = []
    for smile in tqdm(smiles, total = len(smiles)):
        mol      = Chem.MolFromSmiles(smile)
        fp       = generator.GetFingerprint(mol).ToList()   
        smiles_list.append(fp)
    
    return np.array(smiles_list, dtype = np.uint8)



def preprocess_build(train_data,
                     embedded_sequences, 
                     batch_size = 100000,
                     batch_loading = True,
                     embedding_type = 'DeepChem',
                     protein_dtype = 'torch'):
    
    ''' 
    Preprocessing build funtion for batch preprocessing, loading,
    and evaluating of test_data
    
    Parameters:
    - train_data:
    - batch_size:
    - batch_loading:
    
    
    Returns:
    - final_train:
    - final_test:
    - x_train_p:
    - x_test_p:
    - y_test:
    
    
    '''
    index       = 0 
    batches     = math.ceil(len(train_data) / batch_size)
    
    if batch_loading:
        for batch in range(batches):

            batch_start  = index*batch_size
            batch_end    = (index + 1)*batch_size
            batch_subset = train_data[batch_start: batch_end]['']

            train_batch                    = Datapreprocess(batch_subset)
            train_adjacency, graph_tokens  = train_batch.get_processed_data()
            final_test, proteins           = _prepare_data(data_batch = data_batch,
                                             batch_size = batch_size, 
                                             index = index, 
                                             vocab_tokens = vocab_tokens,
                                             mode = 'Evaluate',
                                             embedded_sequences = embedded_sequences)
            
            break
#         data_batches    = load_embeddings(base_path = 'Batched Data/test_preprocessed_Batch', index = index)
            
    else:
        
        # 
        train_bind                           = train_data[train_data['binds'] == 1].dropna()
        train_no_bind                        = train_data[train_data['binds'] == 0].dropna()[:batch_size] 
        train_input                          = pd.concat([train_bind, train_no_bind], axis = 0)
        train_input                          = _reset_drop(train_input)

        targets                              = torch.tensor(train_input['binds'], dtype = torch.long)
 
        if embedding_type == 'Self-Made':
            train_batch                      = Datapreprocess(train_input['molecule_smiles'], embedding_type = embedding_type)
            data_batch, vocab_tokens         = train_batch.get_processed_data()
        
            # Split Train and Test Data
            x_train, x_test, y_train, y_test = train_test_split(data_batch, targets, stratify = targets, random_state = 42)
            train_input, test_input, _, _    = train_test_split(train_input, targets, stratify = targets, random_state = 42)
            train_token, test_token, _, _    = train_test_split(vocab_tokens, targets, stratify = targets, random_state = 42)

            train_input                      = _reset_drop(train_input)
            test_input                       = _reset_drop(test_input)       
       
            # Load Final Model Inputs
            final_train, x_train_p           = _prepare_data(data_batch      = x_train,
                                                             targets         = y_train,
                                                             vocab_tokens    = train_token,
                                                             mode            = 'Build',
                                                             data            = train_input,
                                                             embedded_sequences = embedded_sequences)

            final_test, x_test_p             = _prepare_data(data_batch      = x_test,
                                                             targets         = None,
                                                             vocab_tokens    = test_token,
                                                             mode            = 'Build',
                                                             test            = True,
                                                             data            = test_input,
                                                             embedded_sequences = embedded_sequences)
            
        elif embedding_type == 'DeepChem':
            train_batch                      = Datapreprocess(train_input['molecule_smiles'], 
                                                              targets = train_input['binds'], 
                                                              embedding_type = embedding_type)
            
            
            final_train, final_test          = train_batch.get_deepchem()
            train_input, test_input, y_train, y_test    = train_test_split(train_input, targets, stratify = targets, random_state = 42)

            train_input                      = _reset_drop(train_input)
            test_input                       = _reset_drop(test_input)       
            
            x_train_p                        = _embed_proteins(train_input, embedded_sequences)  
            x_test_p                         = _embed_proteins(test_input, embedded_sequences)
            
        elif embedding_type == 'MACCS':
            
            train_batch                                 = smiles_to_maccs(train_input['molecule_smiles'])
            final_train, final_test, y_train, y_test    = train_test_split(train_batch, targets, 
                                                                           stratify = targets, 
                                                                           random_state = 42)
            
            y_train = y_train.unsqueeze(1)
            y_test  = y_test.unsqueeze(1)
            
            train_input, test_input, _, _     = train_test_split(train_input, targets, stratify = targets, random_state = 42)
            
            train_input                       = _reset_drop(train_input)
            test_input                        = _reset_drop(test_input)       
            
            x_train_p                         = _embed_proteins(train_input, embedded_sequences, embed_type = protein_dtype)
            x_test_p                          = _embed_proteins(test_input, embedded_sequences, embed_type = protein_dtype)

        elif embedding_type == 'morgan_fingerprints':

            train_batch                       = compute_morgan_fingerprint(train_input['molecule_smiles'])
            final_train, final_test, y_train, y_test = train_test_split(train_batch, targets, 
                                                                           stratify = targets, 
                                                                           random_state = 42)
            y_train                           = y_train.unsqueeze(1)
            y_test                            = y_test.unsqueeze(1)
            
            train_input, test_input, _, _     = train_test_split(train_input, targets, stratify = targets, random_state = 42)
            
            train_input                       = _reset_drop(train_input)
            test_input                        = _reset_drop(test_input)   
           
            x_train_p                         = _embed_proteins(train_input, embedded_sequences, embed_type = protein_dtype)
            x_test_p                          = _embed_proteins(test_input, embedded_sequences, embed_type = protein_dtype)
       
    return final_train, final_test, x_train_p, x_test_p, y_train, y_test



def test_morgan(data):
    '''
    '''
    fingerprints = []
    smiles       = data['molecule_smiles']
    for smile in tqdm(smiles, total = len(smiles)):
        fingerprint = create_morgan_fingerprint(smile)
        fingerprints.append(fingerprint)
        
    return fingerprints
    

def morgan_batches(data, mode = 'write', batch_size = 100000):
    '''
    '''
    
    batches   = math.ceil(len(data) / batch_size)
    if mode == 'write':
        
        file_name = ''
        for batch in range(batches):
            get_memory_usage()

            if batch != (batches - 1):
                data_batch = data[batch*batch_size:(batch+1)*batch_size]
            else:
                data_batch = data[batch*batch_size:]

            fingerprints = data_batch['molecule_smiles'].apply(lambda x: create_morgan_fingerprint(x))
            data_batch   = torch.concatenate([fingerprint.unsqueeze(0) for fingerprint in fingerprints], axis = 0)
            pickle.dump(data_batch, open(f'Batched Morgan/test_morgan_{batch}.pckl', 'wb'))
            
            del fingerprints
            del data_batch
            gc.collect()
        
        
        fingerprints = None
        
    elif mode == 'read':
        
        fingerprints = []
        file_prefix = f'Batched Morgan'
        for batch in range(batches):
            get_memory_usage()
            
            file_name   = file_prefix + f'/test_morgan_{batch}.pckl'
            data_tensor = pickle.load(open(file_name, 'rb'))
            
            fingerprints.append(data_tensor)
            
        
        fingerprints = torch.concatenate([batch for batch in fingerprints], axis = 0)
    
    return fingerprints
    
    
    
def morgan_embeddings(train_data, class_0_size = None, track_progress = False):
    class1              = train_data[train_data['binds'] == 1].dropna()
    class0              = train_data[train_data['binds'] == 0].dropna()[:class_0_size]
    
    train_input         = pd.concat([class1, class0], axis = 0)
    targets             = torch.tensor(train_input['binds'], dtype = torch.long)
    
    if track_progress:
        smiles = train_input['molecule_smiles']
        fingerprints = []
        for smile in tqdm(smiles, total = len(smiles)):
            fingerprint = create_morgan_fingerprint(smile)
            fingerprints.append(fingerprint)
            
    else:
        fingerprints = train_input['molecule_smiles'].apply(lambda x: create_morgan_fingerprint(x))
    
    fingerprints = torch.concatenate([fingerprint.unsqueeze(0) for fingerprint in fingerprints], axis = 0)
    return fingerprints, targets, train_input


def create_morgan_fingerprint(smiles, radius=2, n_bits=1024):
    """
    Create a Morgan fingerprint for a given SMILES string.
    
    Parameters:
        smiles (str): SMILES representation of the molecule.
        radius (int): Radius of the Morgan fingerprint.
        n_bits (int): Number of bits in the fingerprint.
        
    Returns:
        np.array: The Morgan fingerprint as a numpy array.
    """
    mol         = Chem.MolFromSmiles(smiles) 
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(fingerprint, dtype = torch.long)