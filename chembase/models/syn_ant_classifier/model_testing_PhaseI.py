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


def Phase_I_eval_model(model, testing_data_set, optimizer):

    model.eval()
    
    dists = []
    syn_scs = []
    ant_scs = []
    labs = []
    
    syn_criterion = functions.Loss_Synonymy()
    ant_criterion = functions.Loss_Antonymy()
    Lm_criterion = functions.Loss_Labels()

    with torch.no_grad():
        test_losses = []
        syn_test_losses = []
        ant_test_losses = []
        Lm_test_losses = []
        
        syn_test_acc_list = []
        ant_test_acc_list = []
        irrel_test_acc_list = []
        
        test_total = 0
        ant_el_count = 0
        ant_correct = 0
        syn_el_count = 0
        syn_correct = 0
        
        #for confusion matrix
        syn_predictions = []
        ant_predictions = []
        syn_true = []
        ant_true = []

        for i, (inputs, labels) in enumerate(testing_data_set):
        
            inputs = torch.from_numpy(np.asarray(inputs)).long()
            labels = torch.from_numpy(np.asarray(labels)).long()
                    
            em1, em2, S1_out, S2_out, A1_out, A2_out, synonymy_score, antonymy_score = model(inputs)
            
            #gather cosine distances, scores, and labels for phase II
            cos_tens = F.cosine_similarity(em1, em2, dim = 1)

            dists.extend(cos_tens.tolist())
            syn_scs.extend(synonymy_score.squeeze().tolist())
            ant_scs.extend(antonymy_score.squeeze().tolist())
            labs.extend(labels.tolist())
                
            #calculate loss per batch of testing data
            syn_test_loss = syn_criterion(S1_out, S2_out, synonymy_score)
            ant_test_loss = ant_criterion(S2_out, A1_out, antonymy_score)
            Lm_test_loss = Lm_criterion(synonymy_score, antonymy_score, labels)
            
            test_loss = syn_test_loss + ant_test_loss + Lm_test_loss
            
            test_losses.append(test_loss.item())
            syn_test_losses.append(syn_test_loss.item())
            ant_test_losses.append(ant_test_loss.item())
            Lm_test_losses.append(Lm_test_loss.item())

            test_total += 1 
        
            #accuracy function
            acc = functions.Phase1Accuracy()
            accuracies = acc(synonymy_score, antonymy_score, labels)
            
            #TODO: add accuracy list for irrelevant pairs (accuracies[2])
            syn_test_acc_list.append(accuracies[0])
            ant_test_acc_list.append(accuracies[1])
            irrel_test_acc_list.append(accuracies[2])
            
            #get predictions and labels for confusion matrix
            preds, truths = acc.confusion(synonymy_score, antonymy_score, labels)
            
            syn_predictions.extend(preds[:,0].tolist())
            ant_predictions.extend(preds[:,1].tolist())
            syn_true.extend(truths[:,0].tolist())
            ant_true.extend(truths[:,1].tolist())

        test_epoch_loss = sum(test_losses)/test_total
        syn_test_epoch_loss = sum(syn_test_losses)/test_total
        ant_test_epoch_loss = sum(ant_test_losses)/test_total
        Lm_test_epoch_loss = sum(Lm_test_losses)/test_total

        syn_epoch_acc = sum(syn_test_acc_list)/test_total
        ant_epoch_acc = sum(ant_test_acc_list)/test_total
        irrel_epoch_acc = sum(irrel_test_acc_list)/test_total
        
        #test Phase II
        p2_path = '/Users/wesleytatum/Desktop/post_doc/data/phase2_xgb_model.model'

        p2 = momo.Phase2XGBoost(p2_path)
        preds = p2.test_pred(dists, syn_scs, ant_scs, labs)

        p2_accs = p2.accuracy(preds, labs)


#         print(f"Total Epoch Testing Loss is: {test_epoch_loss}")
#         print(f"Total Epoch Antonym Testing Accuracy is: {ant_epoch_acc}")
#         print(f"Total Epoch Synonym Testing Accuracy is: {syn_epoch_acc}")       
        
    
    return test_epoch_loss, syn_test_epoch_loss, ant_test_epoch_loss, Lm_test_epoch_loss, syn_epoch_acc, ant_epoch_acc, irrel_epoch_acc, syn_true, syn_predictions, ant_true, ant_predictions, p2_accs

