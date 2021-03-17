import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.model_selection import train_test_split

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import model_functions_PhaseI as functions

#model architecture

class Phase_I_NN(nn.Module):
    """
    This class contains the first of two neural networks to be used to determine synonymy,
    antonymy or irrelevance. Using w2v pre-trained embeddings that are then embedded into
    our NN using the nn.Embedding layer we are able to obtain the encoded embeddings of two
    words (pushed as a tuple) in synonym and antonym subspaces. These encodings are then used
    to calculate the synonymy and antonymy score of those two words. 
    
    This mimics the Distiller method described by Asif Ali et al.
    """

    def __init__(self, in_dims, vocab_length, common, w2v_model):
        super(Phase_I_NN, self).__init__()
        
        if common != None:
            #embedding layer
            self.embedded = functions.w2v_embedding_pre_trained_weights(common, w2v_model)
            
        else:
            #take in word index and generate random embedding
            self.embedded = nn.Embedding(num_embeddings = vocab_length,
                                         embedding_dim = in_dims)
        
        #hidden layers
        self.hidden_layers = nn.Sequential(
        nn.Linear(in_dims, 800), #expand
        nn.Linear(800, 1000),
#         nn.Softplus()
        )
        
        #synonym subspace branch
        self.S_branch = nn.Sequential(
#         nn.Dropout(0.1), #to limit overfitting
        nn.Linear(1000,500), #compress
        nn.Linear(500,60),
#         nn.Softplus()
        )
        
        #antonym subspace branch
        self.A_branch = nn.Sequential(
#         nn.Dropout(0.1), #to limit overfitting
        nn.Linear(1000,500), #compress
        nn.Linear(500,60),
#         nn.Softplus()
        )
        
    def forward(self, index_tuples):
                        
        #for every word pair in the training batch, get (pre-trained embeddings) from
        #the index. em_1 is tensor of first words, em_2 is tensor of second words
        em_1 = self.embedded(index_tuples[:,0])
        em_2 = self.embedded(index_tuples[:,1])
        
        #pass through hidden layers
        out_w1 = self.hidden_layers(em_1) 
        out_w2 = self.hidden_layers(em_2)
                
        #pass each embedded word-pair through each branch to be situated in subspaces
        S1_out = self.S_branch(out_w1)
        S2_out = self.S_branch(out_w2)
        A1_out = self.A_branch(out_w1)
        A2_out = self.A_branch(out_w2)
                
        #calculate synonymy and antonymy scores
        synonymy_score = F.cosine_similarity(S1_out, S2_out, dim = 1)
        synonymy_score = synonymy_score.view(-1, 1)
        antonymy_score = torch.max(F.cosine_similarity(A1_out, S2_out, dim = 1),
                                   F.cosine_similarity(A2_out, S1_out, dim = 1))
        antonymy_score = antonymy_score.view(-1, 1)
                                      
        return em_1, em_2, S1_out, S2_out, A1_out, A2_out, synonymy_score, antonymy_score



class Phase2XGBoost():
    """
    This XGBoost multiclass classifier takes in the synonymy and antonymy
    scores distilled by Phase_I_NN, as well as the pre-trained word embeddings
    and uses them to predict synonymy/antonymy classification for each word pair.
    
    0 = irrelevant pair
    1 = synonymous pair
    2 = antonymous pair
    """
    def __init__(self, io_path):
        """
        Sets XGBoost parameters. For training loops, io_path is used to save/overwrite
        the current model state with the result of training on the most recent loop's
        outputs. During testing loops, io_path is used to load the most recently saved
        XGBoost model state to make predictions using the test set
        """
        super(Phase2XGBoost).__init__()
        
        self.io_path = io_path
        
        #XGBoost parameters
        self.params = {
            'booster' : 'gbtree',
            'verbosity' : 0,
            'eta' : 0.5,
            'gamma' : 0,
            'max_depth' : 6,
            'objective' : 'multi:softmax',
            'num_class' : 3,
        }
        
        self.num_round = 5
        
    def train_save(self, dists, syn_scores, ant_scores, labels):
        """
        During training loops, this function takes in the outputs of Phase I and saves
        the model state for use in the testing loop
        """
        
        #convert lists to np.arrays
        np_dists = np.asarray(dists)
        np_syn_scs = np.asarray(syn_scores)
        np_ant_scs = np.asarray(ant_scores)
        self.labels = np.asarray(labels)
        
        #consolidate training features and split for validation set
        self.features = np.stack((np_dists, np_syn_scs, np_ant_scs), axis = 1)
        self.xtr, self.xts, self.ytr, self.yts = train_test_split(self.features, self.labels,
                                                                  test_size = 0.2, shuffle = True)
        
        #convert datasets to xgb.DMatrix (train, val, total)
        self.dtrain = xgb.DMatrix(self.xtr, label = self.ytr)
        self.dval = xgb.DMatrix(self.xts, label = self.yts)
        self.dtotal = xgb.DMatrix(self.features, label = self.labels)
        
        watchlist = [(self.dval, 'eval'), (self.dtrain, 'train')]
        
        bst = xgb.train(self.params, self.dtrain, self.num_round, watchlist)
        
        preds = bst.predict(self.dtotal)
        
        bst.save_model(self.io_path)
        
        return preds
    
    def test_pred(self, dists, syn_scores, ant_scores, labels):
        """
        During testing loops, this function loads the most recently trained XGBoost
        model and then uses the outputs of Phase I to make predictions
        """
        
        #load saved model
        bst = xgb.Booster(model_file = self.io_path)
        
        #convert torch.tensors to np.arrays
        np_dists = np.asarray(dists)
        np_syn_scs = np.asarray(syn_scores)
        np_ant_scs = np.asarray(ant_scores)
        self.labels = np.asarray(labels)
        
        #consolidate features for prediction
        self.features = np.stack((np_dists, np_syn_scs, np_ant_scs), axis = 1)
        
        #convert datasets to xgb.DMatrix
        self.dtest = xgb.DMatrix(self.features, label = self.labels)
                
        preds = bst.predict(self.dtest)
        
        return preds
    
    
    def accuracy(self, preds, labels):
        """
        This simple function takes in a list of labels and an np.array
        of the predictions produced by either self.train_save() or 
        self.test_pred() and returns a list of accuracy values where the
        elements are: [irrelevant_acc, syn_acc, ant_acc]
        """
        
        np_labels = np.asarray(labels)
        
        cor_syn = 0
        wrng_syn = 0
        cor_ant = 0
        wrng_ant = 0
        cor_irrel = 0
        wrng_irrel = 0

        for pred, label in zip(preds, np_labels):

            if label == 0: #irrels
                if pred == 0:
                    cor_irrel += 1
                else:
                    wrng_irrel += 1

            if label == 1: #syns
                if pred == 1:
                    cor_syn += 1
                else:
                    wrng_syn += 1

            if label == 2: #ants
                if pred == 2:
                    cor_ant += 1
                else:
                    wrng_ant += 1
        
        irrel_acc = (cor_irrel/(wrng_irrel+cor_irrel))*100
        syn_acc = (cor_syn/(wrng_syn+cor_syn))*100
        ant_acc = (cor_ant/(wrng_ant+cor_ant))*100
        
        return [irrel_acc, syn_acc, ant_acc]
        
    #TODO: add functions for precision and recall