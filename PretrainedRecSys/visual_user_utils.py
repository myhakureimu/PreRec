
        
def from_asin2idx_to_idx2asin(asin2idx):
    idx2asin = {}
    for asin, idx in asin2idx.items():
        idx2asin[idx] = asin
    return idx2asin




#####################################################################################
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import pickle
from itertools import chain
from tqdm import tqdm

import multiprocess as mp
import matplotlib.pyplot as plt
import time


def train_valid(model, dataloader, confounder, item_debias, mode='train', loss_f=None, lr_scheduler=None, opt=None, method='our', pop_d=0, device='cpu'):
    if mode == 'train':
        model.train()
        forward = True
        calculate_loss = True
        backward = True
    elif mode == 'valid':
        model.eval()
        forward = True
        calculate_loss = True
        backward = False
    else:
        print('Wrong mode!!!')
        
    loss_sum = 0#torch.tensor(0.0).to(device)
    fake_loss_sum = 0#torch.tensor(0.0).to(device)
    num = 0
    loss_distribution = {}
    #time_6=0
    
    print(calculate_loss)
    for batch in tqdm(dataloader):
        #time.sleep(0.05)
        # historyEmb = batch['historyEmb']
        # historyL   = batch['historyL']
        # posEmb     = batch['posEmb']
        # negEmb     = batch['negEmb']
        # domainIdx  = batch['domainIdx']
        
        if method == 'our':
            if forward:
                prediction, label, historyL = model(batch, confounder, item_debias, pop_d, device)

            if calculate_loss:
                loss = loss_f(prediction, label)
                fake_loss = loss_f(torch.zeros_like(prediction), label)
        elif method == 'unisrec':
            interaction = {}
            interaction['item_id_list'] = batch['historyItemIdxTensor'].to(device)
            interaction['item_emb_list'] = batch['historyItemEmbTensor'].to(device)
            interaction['item_length'] = batch['historyLTensor'].to(device)
            interaction['item_id'] = batch['posItemIdxTensor'].to(device)
            interaction['pos_item_emb'] = batch['posItemEmbTensor'].squeeze(1).to(device)
            loss = model.pretrain(interaction)
        else:
            raise ValueError('not supported method.')
        
        if backward:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # if collect_loss_distribution:
        #     for i,seq in enumerate(seq_list):
        #         for j in np.arange(len(seq)):
        #             if j in loss_distribution.keys():
        #                 loss_distribution[j].append(loss_s[i][j])
        #             else:
        #                 loss_distribution[j] = [loss_s[i]]
        
        del batch
        if calculate_loss:
            loss_sum += loss.detach().item()
            if method == 'our':
               fake_loss_sum += fake_loss.detach().item()
            num += 1
        
    
    if lr_scheduler != None:
        lr_scheduler.step()
    if calculate_loss:
        with torch.no_grad():
            average_loss = (loss_sum/num)
            average_fake_loss = (fake_loss_sum/num)
            print(average_loss)
            print(average_fake_loss)
    
    return_dict = {
        'average_loss': average_loss,
        'average_fake_loss': average_fake_loss,
        'loss_distribution': loss_distribution,
    }
    
    return return_dict


#####################################################################################


import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import pickle
from itertools import chain
from tqdm import tqdm

import multiprocess as mp
import matplotlib.pyplot as plt
import time

from DataLoader import get_myDataLoader
from model_our import myGRU4Rec
import argparse


def RandomGuess_metric(N):
    return_dict = {}
    # ### original metric
    # # idea
    # iDCG = (1/np.log2(1+1))
    # # real
    # rank = np.arange(N)+1
    # score = 1/np.log2(rank+1)
    # # metric
    # rDCG = np.sum(score[:K])/N
    # R = K/N
    # return_dict = {
    #     'NDCG': rDCG/iDCG,
    #     'Recall':R
    # }
    
    ### revised metric
    a = 2
    b = 15000
    ratio=0.0004
    K = int(N*ratio)
    # idea
    iDCG = (1/np.log2(a))
    # real
    rank = np.arange(N)+1
    score = 1/np.log2(a+b*(rank-1)/N)
    # metric
    rDCG = np.sum(score[:K])/N
    R = K/N
    return_dict['rNDCG'] = rDCG/iDCG
    return_dict['rRecall'] = R
    
    return return_dict

def calculate_metric(param):
    item_idx = param[0]
    #seq_idx = param[2]
    #item_overlap = param[3]
    prediction = param[1]
    #ratio = param[2]
    return_dict = {}
    
#     ### original metric
#     # idea
#     iDCG = (1/np.log2(1+1))
#     # real
#     index_start = np.sum(prediction > prediction[item_idx])+1
#     num_same = np.sum(prediction == prediction[item_idx])
#     indexes = index_start + np.arange(num_same)
#     #print(index)
#     if np.min(indexes>K):
#         return_dict = {
#             'NDCG': 0,
#             'Recall': 0
#         }
#     else:
#         indexes = indexes[indexes<=K]

#         rDCG = np.sum(1/np.log2(indexes+1))/num_same
#         R = indexes.shape[0]/num_same

#         return_dict = {
#             'NDCG': rDCG/iDCG,
#             'Recall': R
#         }
    
    ### revised metric
    N = prediction.shape[0]
    a = 2
    b = 15000
    ratio=0.0004
    K = int(N*ratio)
    # idea
    iDCG = (1/np.log2(a))
    # real
    index_start = np.sum(prediction > prediction[item_idx])+1
    num_same = np.sum(prediction == prediction[item_idx])
    indexes = index_start + np.arange(num_same)
    #print(index)
    if np.min(indexes>K):
        return_dict['rNDCG'] = 0
        return_dict['rRecall'] = 0
    else:
        indexes = indexes[indexes<=K]

        rDCG = np.sum(1/np.log2(a+b*(indexes-1)/N))/num_same
        R = indexes.shape[0]/num_same
        
        return_dict['rNDCG'] = rDCG/iDCG
        return_dict['rRecall'] = R
    return return_dict

def get_item_rec(model, function_name, N_item_nlp, bs, device, verbose=False, method='our' , item_debias=None, domainInt=None):
    if verbose:
        print('get_item_rec start')
    N_item_rec = []
    N = N_item_nlp.shape[0]
    N_domainIdx = torch.ones([N, 1], dtype = torch.long)*domainInt
    N_itemIdx = torch.arange(N, dtype = torch.long).reshape([N, 1])
    
    J = int(np.ceil(N/bs))
    
    pbar = tqdm(np.arange(J))
    pbar.set_description('get_item_rec')
    for j in pbar:
        bs_item_nlp = N_item_nlp[j*bs:(j+1)*bs].type(torch.FloatTensor).to(device).unsqueeze(1)
        domainIdx = N_domainIdx[j*bs:(j+1)*bs]
        itemIdx = N_itemIdx[j*bs:(j+1)*bs]
        N_item_rec.append( getattr(model, function_name)(bs_item_nlp, item_debias=item_debias, domainIdx=domainIdx, itemIdx=itemIdx).cpu().squeeze(1) )
    N_item_rec = torch.cat(N_item_rec, dim=0)
    if verbose:
        print('N_item_rec.shape = ', N_item_rec.shape)
        print('get_item_rec end')
    print('N_item_rec', N_item_rec.shape)
    return N_item_rec # N * F

def get_pop_score(model, function_name, dayItemPop, bs_d, bs_n, device, verbose=False, method='our'):
    if verbose:
        print('get_pop_score start')
    D, N, _ = dayItemPop.shape
    DxN_pop_score = torch.zeros([D, N], dtype = torch.float)
    
    J = int(np.ceil(D/bs_d))
    K = int(np.ceil(N/bs_n))
    
    pbar = tqdm(np.arange(J))
    pbar.set_description('get_pop_score')
    for j in pbar:
        for k in np.arange(K):
            bs_pop = dayItemPop[j*bs_d:(j+1)*bs_d, k*bs_n:(k+1)*bs_n].to(device)
            DxN_pop_score[j*bs_d:(j+1)*bs_d, k*bs_n:(k+1)*bs_n] = getattr(model, function_name)(bs_pop).cpu()
    
    return DxN_pop_score
    
    
def get_many(model, function_name, test_dataloader, confounder, index2asin, seen_asin2_, seen_user2_, device, verbose=False, method='our', item_debias=None, domainInt=None):
    if verbose:
        print('get_UserRec_NextIdx_SeqIdx_InteractionIdx start')
    
    I_userRec = []
    I_historyL = []
    I_posIdx = []
    I_posDay = []
    I_seen = []
    
    pbar = tqdm(test_dataloader)
    pbar.set_description('get_many')
    for batch in pbar:
        List_userStr = batch['List_userStr']
        historyEmb = batch['historyItemEmbTensor']
        Tensor_posIdx = batch['posItemIdxTensor']
        Tensor_posDay = batch['posDayIdxTensor']
        Tensor_domainIdx = batch['posDomainIntTensor']
        Tensor_historyL = batch['historyLTensor']
        Tensor_historyPop = batch['historyPopEmbTensor']
        
        if method == 'our':
            print(confounder)
            userRec = getattr(model, function_name)(batch, confounder, item_debias)
        elif method == 'unisrec':
            interaction = {}
            interaction['item_id_list'] = batch['historyItemIdxTensor'].to(device)
            interaction['item_emb_list'] = batch['historyItemEmbTensor'].to(device)
            interaction['item_length'] = batch['historyLTensor'].to(device)
            interaction['item_id'] = batch['posItemIdxTensor'].to(device)
            interaction['pos_item_emb'] = batch['posItemEmbTensor'].squeeze(1).to(device)
            userRec = getattr(model, function_name)(interaction)
        I_userRec.append(userRec.cpu())
        
        I_historyL.append(Tensor_historyL)
        
        I_posIdx.append(Tensor_posIdx)
        I_posDay.append(Tensor_posDay)
        
        for i, Idx in enumerate(Tensor_posIdx):
            item_asin = index2asin[Idx.item()]
            if (item_asin in seen_asin2_.keys()) or (List_userStr[i] in seen_user2_.keys()):
                I_seen.append(True)
            else:
                I_seen.append(False)
                        
    # concatecate
    I_userRec = torch.cat(I_userRec, dim=0)
    I_historyL = torch.cat(I_historyL, dim=0)
    I_posIdx = torch.cat(I_posIdx, dim=0)
    I_posDay = torch.cat(I_posDay, dim=0)
    # I_seen = I_seen
    I_I_index = list(np.arange(I_userRec.shape[0]))
    
    if verbose:
        print('I_userRec.shape: ', I_userRec.shape, type(I_userRec))
        print('I_historyL.shape: ', I_historyL.shape, type(I_historyL))
        print('I_posIdx.shape: ', I_posIdx.shape, type(I_posIdx))
        print('len(I_seen): ', len(I_seen), type(I_seen))
        print('len(I_I_index): ', len(I_I_index), type(I_I_index))
        
    if verbose:
        print('get_UserRec_NextIdx_SeqIdx_InteractionIdx end')
    return_dict = {
        'I_userRec': I_userRec,
        'I_posDay': I_posDay,
        'I_historyL': I_historyL,
        'I_posIdx': I_posIdx,
        'I_seen': I_seen,
        'I_I_index': I_I_index
    }
    return return_dict

def get_metric(N_itemRec, DxN_pop_score, use_pop, I_many, bs, device, verbose=True):
    if verbose:
        print('***** calculate NDCG / Recall *****')
    N = N_itemRec.shape[0]
    I_userRec      = I_many['I_userRec']
    I_historyL     = I_many['I_historyL']
    I_posIdx       = I_many['I_posIdx']
    I_posDay       = I_many['I_posDay'].squeeze(-1)
    I_seen         = I_many['I_seen']
    I_I_index      = I_many['I_I_index']
    I = I_userRec.shape[0]
    
    J = int(np.ceil(I/bs))
    
    # initial needed metric
    overlap_dict = {
        't_count': 0,
        'f_count': 0,
    }
    # ndcg_seq_sum_dict = {}
    # ndcg_seq_count_dict = {}
    # ndcg_overlap_dict = {
    #     't_sum': 0,
    #     'f_sum': 0,
    # }
    #Rndcg_seq_sum_dict = {}
    #Rndcg_seq_count_dict = {}
    Rndcg_overlap_dict = {
        't_sum': 0,
        'f_sum': 0,
    }
    # recall_seq_sum_dict = {}
    # recall_seq_count_dict = {}
    # recall_overlap_dict = {
    #     't_sum': 0,
    #     'f_sum': 0,
    # }
    #Rrecall_seq_sum_dict = {}
    #Rrecall_seq_count_dict = {}
    Rrecall_overlap_dict = {
        't_sum': 0,
        'f_sum': 0,
    }
    
    pbar = tqdm(np.arange(J))
    pbar.set_description('get_metric')
    for j in pbar:
        bs_posDay = I_posDay[j*bs:(j+1)*bs]
        bs_userRec = I_userRec[j*bs:(j+1)*bs]
        
        bs_similarity = (bs_userRec @ N_itemRec.T).cpu() 
        if use_pop>0:
            bs_similarity = bs_similarity + DxN_pop_score[bs_posDay]
        bs_similarity = bs_similarity.numpy()
        
        bs_userRec = I_userRec [j*bs:(j+1)*bs]
        bs_posIdx = I_posIdx  [j*bs:(j+1)*bs]
        bs_seen    = I_seen    [j*bs:(j+1)*bs]
        bs_I_index = I_I_index [j*bs:(j+1)*bs]-j*bs
        
        zipped_info = [[I_index, posIdx, seen] for I_index, posIdx, seen in zip(bs_I_index, bs_posIdx, bs_seen)]
        
        for I_index, posIdx, seen in zipped_info:
            param = [posIdx, bs_similarity[I_index]]
            NDCG_Recall = calculate_metric(param)
            #ndcg = NDCG_Recall['NDCG']
            Rndcg = NDCG_Recall['rNDCG']
            #recall = NDCG_Recall['Recall']
            Rrecall = NDCG_Recall['rRecall']
            if seen == True:
                overlap_dict['t_count'] += 1
                #ndcg_overlap_dict['t_sum'] += ndcg
                Rndcg_overlap_dict['t_sum'] += Rndcg
                #recall_overlap_dict['t_sum'] += recall
                Rrecall_overlap_dict['t_sum'] += Rrecall
            elif seen == False:
                overlap_dict['f_count'] += 1
                #ndcg_overlap_dict['f_sum'] += ndcg
                Rndcg_overlap_dict['f_sum'] += Rndcg
                #recall_overlap_dict['f_sum'] += recall
                Rrecall_overlap_dict['f_sum'] += Rrecall
            else:
                print('------------------- Wrong -------------------')
            # if insession_idx in ndcg_seq_sum_dict.keys():
            #     ndcg_seq_sum_dict[insession_idx] += ndcg
            #     ndcg_seq_count_dict[insession_idx] += 1
            #     Rndcg_seq_sum_dict[insession_idx] += Rndcg
            #     Rndcg_seq_count_dict[insession_idx] += 1
            #     recall_seq_sum_dict[insession_idx] += recall
            #     recall_seq_count_dict[insession_idx] += 1
            #     Rrecall_seq_sum_dict[insession_idx] += Rrecall
            #     Rrecall_seq_count_dict[insession_idx] += 1
            # else:
            #     ndcg_seq_sum_dict[insession_idx] = ndcg
            #     ndcg_seq_count_dict[insession_idx] = 1
            #     Rndcg_seq_sum_dict[insession_idx] = Rndcg
            #     Rndcg_seq_count_dict[insession_idx] = 1
            #     recall_seq_sum_dict[insession_idx] = recall
            #     recall_seq_count_dict[insession_idx] = 1
            #     Rrecall_seq_sum_dict[insession_idx] = Rrecall
            #     Rrecall_seq_count_dict[insession_idx] = 1
    
    random_metric = RandomGuess_metric(N)
    #random_ndcg = random_metric['NDCG']
    random_Rndcg = random_metric['rNDCG']
    #random_recall = random_metric['Recall']
    random_Rrecall = random_metric['rRecall']
    
    lenT = overlap_dict['t_count']
    lenF = overlap_dict['f_count']
    
    #mean_ndcg = (ndcg_overlap_dict['t_sum']+ndcg_overlap_dict['f_sum']) / (lenT+lenF)
    mean_Rndcg = (Rndcg_overlap_dict['t_sum']+Rndcg_overlap_dict['f_sum']) / (lenT+lenF)
    #mean_recall = (recall_overlap_dict['t_sum']+recall_overlap_dict['f_sum']) / (lenT+lenF)
    mean_Rrecall = (Rrecall_overlap_dict['t_sum']+Rrecall_overlap_dict['f_sum']) / (lenT+lenF)
    
    #seen_ndcg = ndcg_overlap_dict['t_sum'] / lenT
    seen_Rndcg = Rndcg_overlap_dict['t_sum'] / lenT
    #seen_recall = recall_overlap_dict['t_sum'] / lenT
    seen_Rrecall = Rrecall_overlap_dict['t_sum'] / lenT
    
    #unseen_ndcg = ndcg_overlap_dict['f_sum'] / (lenF+10**-10)
    unseen_Rndcg = Rndcg_overlap_dict['f_sum'] / (lenF+10**-10)
    #unseen_recall = recall_overlap_dict['f_sum'] / (lenF+10**-10)
    unseen_Rrecall = Rrecall_overlap_dict['f_sum'] / (lenF+10**-10)
    
    #_random_ndcg    = str(round(random_ndcg   ,4))
    _random_Rndcg   = str(round(random_Rndcg  ,4))
    #_random_recall  = str(round(random_recall ,4))
    _random_Rrecall = str(round(random_Rrecall,4))
    
    #_mean_ndcg    = str(round(mean_ndcg   ,4))
    _mean_Rndcg   = str(round(mean_Rndcg  ,4))
    #_mean_recall  = str(round(mean_recall ,4))
    _mean_Rrecall = str(round(mean_Rrecall,4))
    
    #_seen_ndcg    = str(round(seen_ndcg   ,4))
    _seen_Rndcg   = str(round(seen_Rndcg  ,4))
    #_seen_recall  = str(round(seen_recall ,4))
    _seen_Rrecall = str(round(seen_Rrecall,4))
    
    #_unseen_ndcg    = str(round(unseen_ndcg   ,4))
    _unseen_Rndcg   = str(round(unseen_Rndcg  ,4))
    #_unseen_recall  = str(round(unseen_recall ,4))
    _unseen_Rrecall = str(round(unseen_Rrecall,4))
    
    if lenT == 0:
        _seen_num = str(0)
    if lenF == 0:
        _unseen_num = str(0)
    if verbose:
        print('seen: ', lenT)
        print('unseen: ', lenF)
#         print('random_ndcg: ', _random_ndcg)
#         print('mean_ndcg  : ', _mean_ndcg)
#         print('seen_ndcg  : ', _seen_ndcg)
#         print('unseen_ndcg: ', _unseen_ndcg)
#         print('random_recall: ', _random_recall)
#         print('mean_recall  : ', _mean_recall)
#         print('seen_recall  : ', _seen_recall)
#         print('unseen_recall: ', _unseen_recall)
        print('random_Rndcg: ', _random_Rndcg)
        print('mean_Rndcg  : ', _mean_Rndcg)
        print('seen_Rndcg  : ', _seen_Rndcg)
        print('unseen_Rndcg: ', _unseen_Rndcg)
        print('random_Rrecall: ', _random_Rrecall)
        print('mean_Rrecall  : ', _mean_Rrecall)
        print('seen_Rrecall  : ', _seen_Rrecall)
        print('unseen_Rrecall: ', _unseen_Rrecall)
    
    return_dict = {
        'random':{
            'count':  0,
            #'NDCG':    random_ndcg,
            'rNDCG':   random_Rndcg,
            #'Recall':  random_recall,
            'rRecall': random_Rrecall,
        },
        'mean':{
            'count':  lenT + lenF,
            #'NDCG':    mean_ndcg,
            'rNDCG':   mean_Rndcg,
            #'Recall':  mean_recall,
            'rRecall': mean_Rrecall,
        },
        'seen':{
            'count':  lenT,
            #'NDCG':    seen_ndcg,
            'rNDCG':   seen_Rndcg,
            #'Recall':  seen_recall,
            'rRecall': seen_Rrecall,
        },
        'unseen':{
            'count':  lenF,
            #'NDCG':    unseen_ndcg,
            'rNDCG':   unseen_Rndcg,
            #'Recall':  unseen_recall,
            'rRecall': unseen_Rrecall,
        },
    }
    return return_dict


def test(seen_asin2_,
         seen_user2_,
         model, confounder, item_debias, use_pop,
         domain_list,
         split_dataloader,
         I2B,
         method='our',
         device='cpu',
         metric_on='unseen',
         bs_n=2048, bs_i=2048, bs_d=1024,
         visual=False):#bs_n=256, bs_i=512, bs_d=1024):


            
    #dataloaders = get_myDataLoader(seen_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = 128, device = device)
    #seen_index2asin = dataloaders['fused_idx2asinDict']
    #seen_user = dataloaders['fused_user']
            
    #dataloaders = get_myDataLoader(test_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = 128, device = device)
    #split_test_dataloader = dataloaders['split_test__dataloader']
    #split_x_nlp = dataloaders['split_nlpTensor']
            
    #split_index2asin = dataloaders['split_idx2asinDict']
    #del dataloaders
            
    #seen_asin2_ = dict.fromkeys(seen_index2asin.values())
    #seen_user2_ = dict.fromkeys(seen_user)
            
    #print('***** test *****')
    model.eval()
            
    # initialize metric
    results = {}
    
    User_Emb = {}
    with torch.no_grad():
        for domainInt, domainStr in enumerate(domain_list):
            itemEmb = I2B[domainInt]['itemEmb']
            index2asin = I2B[domainInt]['I2A']
            dayItemPop = I2B[domainInt]['dayItemPop']
            if domainStr != I2B[domainInt]['domainStr']:
                raise Exception('domain wrong') 
            test_dataloader = split_dataloader[domainInt]
            
            # i -> domainInt
            # test_domain -> domainStr
            # test_dataloader
            # item_nlp -> itemEmb
            # index2asin -> index2asin
            # -> dayItemPop
            
            N_item_rec = get_item_rec(model, 'get_item_rec', itemEmb, bs_n, device, verbose=False, method=method, item_debias=item_debias, domainInt=domainInt)
            
            if use_pop>0:
                DxN_pop_score = get_pop_score(model, 'get_pop_score', dayItemPop, bs_d, bs_n, device, verbose=False, method=method)
                print(DxN_pop_score.shape)
            else:
                DxN_pop_score = None
            I_many = get_many(model, 'get_user_rec', test_dataloader, confounder, index2asin, seen_asin2_, seen_user2_, device, verbose=False, method=method, item_debias=item_debias, domainInt=domainInt)
            
            results[domainStr] = get_metric(N_item_rec, DxN_pop_score, use_pop, I_many, bs_i, device, verbose=False)
            
            A = len(domain_list)
            B = domainStr
            C = metric_on
            D = results[domainStr][metric_on]['count']
            E = results[domainStr][metric_on]['rRecall']
            F = results[domainStr][metric_on]['rNDCG']
            print(f'----- {domainInt+1} / {A} {B} ----- {C} count: {D} rRecall: {E:.4f} rNDCG: {F:.4f}')
            
            User_Emb[B] = I_many['I_userRec']
            
    random_Rndcg   = np.mean([results[key]['random']['rNDCG'] for key in results.keys()])
    random_Rrecall = np.mean([results[key]['random']['rRecall'] for key in results.keys()])
    
    mean_Rndcg   = np.mean([results[key]['mean']['rNDCG'] for key in results.keys()])
    mean_Rrecall = np.mean([results[key]['mean']['rRecall'] for key in results.keys()])
        
    seen_Rndcg   = np.mean([results[key]['seen']['rNDCG'] for key in results.keys()])
    seen_Rrecall = np.mean([results[key]['seen']['rRecall'] for key in results.keys()])
    
    unseen_Rndcg   = np.mean([results[key]['unseen']['rNDCG'] for key in results.keys()])
    unseen_Rrecall = np.mean([results[key]['unseen']['rRecall'] for key in results.keys()])
    
    mean_dict = {
        'random':{
            'count':   0,
            #'NDCG':    random_ndcg,
            'rNDCG':   random_Rndcg,
            #'Recall':  random_recall,
            'rRecall': random_Rrecall,
        },
        'mean':{
            'count':   [results[key]['mean']['count'] for key in results.keys()], #lenT + lenF,
            #'NDCG':    mean_ndcg,
            'rNDCG':   mean_Rndcg,
            #'Recall':  mean_recall,
            'rRecall': mean_Rrecall,
        },
        'seen':{
            'count':   None, #lenT,
            #'NDCG':    seen_ndcg,
            'rNDCG':   seen_Rndcg,
            #'Recall':  seen_recall,
            'rRecall': seen_Rrecall,
        },
        'unseen':{
            'count':   [results[key]['unseen']['count'] for key in results.keys()], #lenF,
            #'NDCG':    unseen_ndcg,
            'rNDCG':   unseen_Rndcg,
            #'Recall':  unseen_recall,
            'rRecall': unseen_Rrecall,
        },
    }
    
    results['mean'] = mean_dict
    if visual == False:
        return results
    else:
        return results, User_Emb