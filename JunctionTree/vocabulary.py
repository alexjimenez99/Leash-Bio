import json
import os

# Construct the path dynamically
current_dir = os.path.dirname(__file__)  # Current file's directory
parent_dir  = os.path.abspath(os.path.join(current_dir, '..'))  # Move up one level
json_path   = os.path.join(parent_dir, 'Vocabulary', 'zinc250_vocab.json')


def tokenize_dict(token, vocab_path = '../Vocabulary/zinc250_vocab.json'):
    '''
    Load the Tokenization Dictionary
    '''
    print(os.getcwd())
    # Original 369 Tokens Built using Kaggle Zinc250K dataset
    vocab    = json.load(open(json_path, 'r'))
    return vocab
    
def update_vocabulary(token, tokenize_dict, update_batch = True, update_frequency = 1):
    '''
    Update the json vocabulary for molecules if the tokens don't exist already
    
    Parameters
    ----------
    - token:
    - tokenize_dict:
    
    
    Returns:
    ---------
    - tokenize_dict
    
    '''
    
    vocab_token                = len(list(tokenize_dict.keys())) + 1
    tokenize_dict[token]       = vocab_token
    
    # Update every 5 tokens
    if update_batch and vocab_token % update_frequency == 0:
        with open(json_path, 'w') as file:
            json.dump(tokenize_dict, file, indent=4)
        
        
    return tokenize_dict