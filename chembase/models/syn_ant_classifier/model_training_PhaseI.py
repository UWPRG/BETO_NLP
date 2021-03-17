import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)
import model_functions_PhaseI as functions
import model_morphology as momo


def Phase_I_train_model(model, training_data_set, optimizer):
    
    dists = []
    syn_scs = []
    ant_scs = []
    labs = []
    
    train_losses = []
    syn_train_losses = []
    ant_train_losses = []
    Lm_train_losses = []
    syn_train_acc_list = []
    ant_train_acc_list = []
    irrel_train_acc_list = []
    
    train_total = 0
    
    #switch model to training mode
    model.train()
    
    syn_criterion = functions.Loss_Synonymy()
    ant_criterion = functions.Loss_Antonymy()
    Lm_criterion = functions.Loss_Labels()
    
    for i, (words, labels) in enumerate(training_data_set):
        
        batch_syn_loss = torch.zeros(len(training_data_set))
        batch_ant_loss = torch.zeros(len(training_data_set))
        batch_Lm_loss = torch.zeros(len(training_data_set))
      
        model.zero_grad() #zero out any gradients from prior loops 
                                           
        #gather model predictions for this loop
        em1, em2, S1_out, S2_out, A1_out, A2_out, synonymy_score, antonymy_score = model(words)
        
        #gather cosine distances, scores, and labels for phase II
        cos_tens = F.cosine_similarity(em1, em2, dim = 1)
        
        dists.extend(cos_tens.tolist())
        syn_scs.extend(synonymy_score.squeeze().tolist())
        ant_scs.extend(antonymy_score.squeeze().tolist())
        labs.extend(labels.tolist())
                        
        #calculate error in the predictions. torch.mul() to weight a loss type
        syn_loss = syn_criterion(S1_out, S2_out, labels)
        ant_loss = ant_criterion(S2_out, A1_out, labels)
        Lm_loss = Lm_criterion(synonymy_score, antonymy_score, labels)
                
        total_loss = syn_loss + ant_loss + Lm_loss
            
        #save total loss for this batch
        train_losses.append(total_loss.item())
        train_total+=1
        
        #store batch losses in lists
        syn_train_losses.append(syn_loss)
        ant_train_losses.append(ant_loss)
        Lm_train_losses.append(Lm_loss)
        
        #BACKPROPAGATE LIKE A MF
        torch.autograd.backward([syn_loss, ant_loss, Lm_loss])
        optimizer.step()
        
        #accuracy function
        acc = functions.Phase1Accuracy()
        accuracies = acc(synonymy_score, antonymy_score, labels)
        syn_train_acc_list.append(accuracies[0])
        ant_train_acc_list.append(accuracies[1])
        irrel_train_acc_list.append(accuracies[2])
       
    #calculate total error for this epoch of training
    train_epoch_loss = sum(train_losses)/train_total
    syn_train_epoch_loss = sum(syn_train_losses)/train_total
    ant_train_epoch_loss = sum(ant_train_losses)/train_total
    Lm_train_epoch_loss = sum(Lm_train_losses)/train_total
    
    #rough phase 1 accuracy for comparison
    syn_epoch_acc = sum(syn_train_acc_list)/train_total
    ant_epoch_acc = sum(ant_train_acc_list)/train_total
    irrel_epoch_acc = sum(irrel_train_acc_list)/train_total
    
    #train, validate, and evaluate Phase II
    p2_path = '/Users/wesleytatum/Desktop/post_doc/data/phase2_xgb_model.model'
    
    p2 = momo.Phase2XGBoost(p2_path)
    preds = p2.train_save(dists, syn_scs, ant_scs, labs)
    
    p2_accs = p2.accuracy(preds, labs)
    
    return train_epoch_loss, syn_train_epoch_loss, ant_train_epoch_loss, Lm_train_epoch_loss, syn_epoch_acc, ant_epoch_acc, irrel_epoch_acc, p2_accs

