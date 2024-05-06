
from zipfile import ZipFile
import pandas as pd
import copy
import os
# print(os.getcwd())
# print(os.path.exists('JunctionTree'))
from JunctionTree.utils import *
from JunctionTree.vocabulary import tokenize_dict

# with ZipFile('../zinc250k.zip') as zp:
#     data = pd.read_csv(zp.open('250k_rndm_zinc_drugs_clean_3.csv'))

# adjusted_smiles  = []
# for index, smile in enumerate(data['smiles']):
#   smile          = smile.replace('\n', '')
#   adjusted_smiles.append(smile)

# data['smiles'] = adjusted_smiles


class Datapreprocess:
    
    def __init__(self, data):
        self.data                                     = data
        self.adjacency_substituted, self.graph_tokens = self.load_matricies(self.data)
        
    def get_processed_data(self):
        return self.adjacency_substituted, self.graph_tokens

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

        # Load the prebuilt vocabulary
        unique_vocab                                    = self.load_unique_vocab()

        token_dictionary, unique_vocab, valency_dict    = build_unique_vocabulary(vocabulary, data, unique_vocab, molecule_df)
        
        
#         adjacency_substituted                           = copy.copy(padded_matricies)
        graph_tokens                  = substitute_vocabulary(adjacency_matricies, token_dictionary, unique_vocab)

        return adjacency_matricies, graph_tokens

#     adjacency_substituted = load_matricies(data['smiles'])






