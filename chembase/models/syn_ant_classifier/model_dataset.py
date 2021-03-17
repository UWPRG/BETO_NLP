import os
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

#dataset creation
#train dataset

class Phase_I_Train_Dataset(Dataset):
    
    def __init__(self):
        
        data = pd.read_json('./data/Phase_I_Train.json', dtype = np.float32)
        self.len = data.shape[0]
        
        #creating a list of tuples where [w1,w2] and [ss, as]
        data_x = list(zip(data['word 1'], data['word 2']))
        data_y = list(data['label'])
        
        #split into x_data our features and y_data our targets
        #F.soft_max expects 'float' predictions and 'long' labels
        self.x_data = torch.tensor(data_x)
        self.x_data = self.x_data.type(torch.float)
        
        self.y_data = torch.tensor(data_y)
        self.y_data = self.y_data.type(torch.long)
        
    def __len__(self):
        
        return self.len
    
    def __getitem__(self, index):
        
        return self.x_data[index], self.y_data[index]


#test dataset

class Phase_I_Test_Dataset(Dataset):
    
    def __init__(self):
        
        data = pd.read_json('./data/Phase_I_Test.json', dtype = np.float32)
        self.len = data.shape[0]
        
        #creating a list of tuples where [w1,w2] and [ss, as]
        data_x = list(zip(data['word 1'], data['word 2']))
        data_y = list(data['label'])
            
        #split into x_data our features and y_data our targets
        #F.soft_max expects 'float' predictions and 'long' labels
        self.x_data = torch.tensor(data_x)
        self.x_data = self.x_data.type(torch.float)
        
        self.y_data = torch.tensor(data_y)
        self.y_data = self.y_data.type(torch.long)
        
    def __len__(self):
        
        return self.len
    
    def __getitem__(self, index):
        
        return self.x_data[index], self.y_data[index]


class DistillerDataset(Dataset):
    """
    Dataset class for Phase I (Distiller), which encodes word pairs into
    synonym and antonym subspaces. Takes in pandas.DataFrames() of the word
    pairs, their labels, and vocabulary indices. It then returns a 
    torch.utils.data.Dataset()
    """
    
    def __init__(self, word_pairs, labels, indices, path):
        
        super(DistillerDataset).__init__()
        
        if os.path.exists(path) == False: #first time using dataset or train-test-split
            
            self.word_pairs = word_pairs
            self.labels = labels.rename('labels')
            self.indices = indices
            self.index_pairs = pd.DataFrame(columns = ['word 1', 'word 2'])

            pbar = tqdm(total = len(self.word_pairs), position = 0)

            for i in range(len(self.word_pairs)):

                word1 = self.word_pairs['word 1'].iloc[i]
                word2 = self.word_pairs['word 2'].iloc[i]
                                
                index1 = self.get_index(word1)
                index2 = self.get_index(word2)

                self.index_pairs.loc[i] = pd.Series({'word 1':index1, 'word 2':index2})

                pbar.update()
                
            self.index_pairs.to_json(path)
            self.word_pairs.to_json(path[:-5]+'_words.json')
            self.labels.to_json(path[:-5]+'_labels.json')
                            
        else: #okay to use previously created train-test-split
            
            self.index_pairs = pd.read_json(path)
            self.word_pairs = pd.read_json(path[:-5]+'_words.json')
            
            with open(path[:-5]+'_labels.json', 'r') as f:
                j_labs = json.load(f)
                
            self.labels = pd.DataFrame({'labels':j_labs})
            
        self.x_data = torch.tensor(self.index_pairs[['word 1', 'word 2']].values.astype(float))
        self.x_data = self.x_data.type(torch.long)

        self.y_data = torch.tensor(self.labels.values.astype(float))
        self.y_data = self.y_data.type(torch.long)
        
        
    def __len__(self):
        
        self.len = self.index_pairs.shape[0]
        
        return self.len
    
    
    def __getitem__(self, key):
        
        
        return self.x_data[key], self.y_data[key]
    
    
    def get_index(self, word):
        
        index_word_pair = self.indices.query('word == @word')
        
        indx_list = index_word_pair['index'].values
        
        #because pd.DataFrame.values returns a list, need the single
        #element that is within it
        if len(indx_list) == 0:
            print(f'index-word mishap {word}')
            bad_row = self.indices.query('word == @word')
            print(bad_row)
            
        else:
            index = indx_list[0]
        
        return index
        

def generate_indices(word_pairs_df):
    """
    Helper function to generate a common set of indices for a
    given list of word pairs.
    """
    
    indices = pd.DataFrame(columns = ['index', 'word'])

    index = 0

    pbar = tqdm(total = len(word_pairs_df), position = 0)

    for i in range(len(word_pairs_df)):

        word1 = word_pairs_df['word 1'].iloc[i]
        word2 = word_pairs_df['word 2'].iloc[i]

        if word1 not in indices['word']:
            indices.loc[index] = pd.Series({'index':index, 'word':word1})
            index+=1
        else:
            pass

        if word2 not in indices['word']:
            indices.loc[index] = pd.Series({'index':index, 'word':word2})
            index+=1
        else:
            pass

        pbar.update()
        
    return indices