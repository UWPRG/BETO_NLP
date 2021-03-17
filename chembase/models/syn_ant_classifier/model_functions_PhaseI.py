import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os



class Loss_Synonymy(nn.Module):
    """
    This class contains a loss function that uses the sum of ReLu loss to make predictions for the encoded embeddings
    in the synonym subspace. A lower and higher bound for synonymy are to be determined. Need to better understand the
    equation found in the Asif Ali et al. paper.
    """  
    def __init__ (self):
        super(Loss_Synonymy, self).__init__()
        
    def forward(self, S1_out, S2_out, labels):
        
        result_list = torch.zeros(S1_out.size(0))

        #x=synonymy_score, a=S1_out, b=S2_out
        for i, (x, a, b) in enumerate(zip(labels, S1_out, S2_out)):
                    
            error = torch.zeros(1,1)
            
            #synonymous pairs
            if x == 1:
                error = F.relu(torch.add(torch.tensor(1), torch.neg(torch.tanh(torch.dist(a, b, 2)))))
            
            #antonymous or irrelevant pairs
            else:
                error = F.relu(torch.add(torch.tensor(1), torch.tanh(torch.dist(a, b, 2))))                
        
            result_list[i] = error
        
        result = result_list.sum()
        
        return result



class Loss_Antonymy(nn.Module):
    """
    This class contains a loss function that uses the sum of ReLu loss to make predictions for the encoded embeddings
    in the antonym subspace. A lower and higher bound for antonymy are to be determined. Need to better understand the
    equation found in the Asif Ali et al. paper.
    """
    
    def __init__(self):
        super(Loss_Antonymy, self).__init__()
       
    def forward(self, S2_out, A1_out, labels): 
        
        result_list = torch.zeros(S2_out.size(0))
        
        #x=antonymy_score, a=A1_out, b=S2_out (to ensure trans-transitivity)
        for i, (x, a, b) in enumerate(zip(labels, A1_out, S2_out)):
            
            #error1 = antonymous pairs, error2 = non-antonymous pairs
            error = torch.zeros((1, 1))
            
            #antonymous pair
            if x == 2:
                error = F.relu(torch.add(torch.tensor(1), torch.neg(torch.tanh(torch.dist(a, b, 2)))))
            
            #synonymous or irrelevant pair
            else:
                error = F.relu(torch.add(torch.tensor(1), torch.tanh(torch.dist(a, b, 2))))
                 
            result_list[i] = error
        
        loss = result_list.sum()
        
        return loss



class Loss_Labels(nn.Module):
    """
    This class is the last portion (L_m) of the general loss function. Here the 
    predicted synonymy and antonymy scoresare concatenated and compared to the 
    concatenated labeled synonymy and antonymy scores
    """
    def __init__(self):
        super(Loss_Labels, self).__init__()
       
    def forward(self, synonymy_score, antonymy_score, labels):
        
        batch_size = labels.size(0)
        
        result_list = torch.zeros((batch_size, 2))
                
        for i, (x, a, b) in enumerate(zip(labels, synonymy_score, antonymy_score)):
            
            total_vec = torch.cat((a, b), dim = 0)            
            
            probs = F.log_softmax(total_vec, dim = 0) #class probability
            pred = torch.argmax(probs, dim = 0) #predicted class
            
            if x == 1:
                error = probs[0]
            
            if x == 2:
                error = probs[1]
                
            if x == 0: #phase 1 is not meant to distinguish irrelevant pairs
                error = 0
                
            result_list[i] = error
        
        loss = torch.neg(result_list.mean())
    
        return loss
    
    
class Phase1Accuracy(nn.Module):
    """
    This class takes in a batch of synonymy scores, antonymy scores, predictons, and labels
    to identify the accuracy for the batch in Phase 1 (Distiller)
    """
    
    def __init__(self):
        super(Phase1Accuracy, self).__init__()
        
    def forward(self, synonymy_scores, antonymy_scores, labels):
        
        correct_syn = 0
        wrong_syn = 0
        
        correct_ant = 0
        wrong_ant = 0
        
        correct_irrel = 0
        wrong_irrel = 0
        
        syn_size = 0
        ant_size = 0
        irrel_size = 0
                
        for label, syn_sc, ant_sc in zip(labels, synonymy_scores, antonymy_scores):
            
            total_vec = torch.cat((syn_sc, ant_sc), dim = 0)            
            
            probs = F.log_softmax(total_vec, dim = 0) #class probability
            pred = torch.argmax(probs, dim = 0) #predicted class
                
            if syn_sc <= 0.4 and ant_sc <= 0.4:
                pred = 2
            
            #word pair is synonymous
            if label == 1:
                syn_size += 1
                
                if pred == 0:
                    correct_syn += 1
                    
                else:
                    wrong_syn += 1
            
            #word pair is antonymous
            if label == 2:
                ant_size +=1
                
                if pred == 1:
                    correct_ant +=1
                    
                else:
                    wrong_ant += 1
            
            #word pair has no relationship
            if label == 0:
                irrel_size +=1
                
                if pred == 2:
                    correct_irrel += 1
                
                else:
                    wrong_irrel += 1
        
        #need to account for division by zero in training batches
        if syn_size == 0:
            syn_acc = 0
        else:
            syn_acc = (correct_syn/syn_size)*100
        
        if ant_size == 0:
            ant_acc = 0
        else:
            ant_acc = (correct_ant/ant_size)*100
        
        if irrel_size == 0:
            irrel_acc = 0
        else:
            irrel_acc = (correct_irrel/irrel_size)*100
        
        return [syn_acc, ant_acc, irrel_acc]
    
    
    def confusion(self, synonymy_scores, antonymy_scores, labels):
        """
        helper function to get lists of ground-truths and predictions for the
        creation of a confusion matrix
        """
        preds = np.ndarray((labels.size()[0], 2))
        truths = np.ndarray((labels.size()[0], 2))
        
        for i, (label, syn_sc, ant_sc) in enumerate(zip(labels, synonymy_scores, antonymy_scores)):
            
            preds[i, 0] = syn_sc.item()
            preds[i, 1] = ant_sc.item()
            
            if label == 1: #synonymous pair
                truths[i, 0] = 1
                truths[i, 1] = 0
            
            elif label == 2: #antonymous pair
                truths[i, 0] = 0
                truths[i, 1] = 1
                
            else: #irrelevant pair
                truths[i, 0] = 0
                truths[i, 1] = 0
            
        return preds, truths



#feeding the model pretrained weights

class w2v_embedding_pre_trained_weights(nn.Module):
    """
    This class contains the pre-training of the Phase_I_NN neural network weights using
    a list of words from which a list of weights can be obtained. It is then converted 
    that can then be embedded using the from_pretrained() function into the NN model
    """
    def __init__(self, words, model):
        super(w2v_embedding_pre_trained_weights, self).__init__()
    
        for i in range(len(words)):
            words[i] = model.wv.__getitem__(words[i]).tolist()
    
        weight = torch.tensor(words)
    
        self.embedding = nn.Embedding.from_pretrained(weight)
    
    def forward(self, index):
        
#         index_vector = self.embedding(torch.LongTensor(index))
        
        #Internal function to F.log_softmax not implemented for "Long"
        index_vector = self.embedding(index)
        
        return index_vector

    
    
class glove_embedding_pre_trained_weights(nn.Module):
    """
    This class contains the pre-training of the Phase_I_NN neural network weights using a list of words from which a list of weights can be obtained from a downloaded GloVe embedding dictionary 
    """
    def __init__(self, words):
        super(glove_embedding_pre_trained_weights, self).__init__()
        
        data = '/Users/wesleytatum/Desktop/post_doc/data/glove.6B'
        os.chdir(data)
        embeddings_dict = {}
        with open("glove.6B.50d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
    
        for i in range(len(words)):
            words[i] = embeddings_dict[words[i]].tolist()
    
        weight = torch.tensor(words)
    
        self.embedding = nn.Embedding.from_pretrained(weight)
    
    def forward(self, index):
        
#         index_vector = self.embedding(torch.LongTensor(index))
        
        #Internal function to F.log_softmax not implemented for "Long"
        index_vector = self.embedding(index)
        
        return index_vector


#Network Utilities

def init_weights(model):
    
    classname = model.__class__.__name__
    
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        
    elif classname.find('Conv2d') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)
        
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.zeros_(model.bias)

