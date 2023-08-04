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
from utils import myGRU4Rec
import argparse

train_parser = argparse.ArgumentParser()
train_parser = argparse.ArgumentParser(description='Test')
train_parser.add_argument('-gpu', '--gpu', type=str, default='0',
                    help='gpu index -- example: 0')

train_parser.add_argument('-seendl', '--seen_domain_list', type=str, nargs='+', default=['cn'],
                    help='domain list -- example: jp cn')
train_parser.add_argument('-traindl', '--train_domain_list', type=str, nargs='+', default=['cn'],
                    help='domain list -- example: jp cn')
train_parser.add_argument('-testdl', '--test_domain_list', type=str, nargs='+', default=['cn'],
                    help='domain list -- example: jp cn')
train_parser.add_argument('-s2t', '--seen2test', type=str, default='cn:cn:cn:cn',
                    help='sequntial trained model folder -- example: cn:cn:cn:cn')

train_parser.add_argument('-nlp_f', '--nlp_feature', type=int, default=768,
                         help='nlp feature')
train_parser.add_argument('-rec_f', '--rec_feature', type=int, default=64,
                         help='rec feature')
train_parser.add_argument('-gru_l', '--gru_layer', type=int, default=2,
                         help='gru layer')


train_parser.add_argument('-c', '--confounder', type=int, default=2,
                         help='0: no confounder; 1: add first; 2: add all')
train_parser.add_argument('-db', '--domain_balance', type=int, default=1,
                         help='0: no domain balance; 1: domain balance')
train_parser.add_argument('-r', '--regularization', type=float, default=10.0,
                         help='regularization weights')

train_parser.add_argument('-lr', '--lr', type=float, default=0.001,
                         help='learning rate')
train_parser.add_argument('-bs', '--bs', type=int, default=256,
                         help='batch_size')

train_parser.add_argument('-method', '--method', type=str, default='our',
                          help= 'method')
train_parser.add_argument('-tf', '--test_file', type=str, default='cn:cn.txt',
                    help='test file -- example: cn:cn.txt')

train_parser = train_parser.parse_args()
print('gpu: ', train_parser.gpu)
#print('train domain list: ', train_parser.domain_list)

print('confounder = ', train_parser.confounder)
print('domain_balance = ', train_parser.domain_balance)

os.environ["CUDA_VISIBLE_DEVICES"] = train_parser.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('gpu or cpu?: ',device)


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

def get_item_rec(model, function_name, N_item_nlp, bs, device, verbose=False, method='our'):
    print('get_item_rec start')
    N_item_rec = []
    N = item_nlp.shape[0]
    J = int(np.ceil(N/bs))
    for j in tqdm(np.arange(J)):
        bs_item_nlp = N_item_nlp[j*bs:(j+1)*bs].type(torch.FloatTensor).to(device)
        N_item_rec.append( getattr(model,'get_item_rec')(bs_item_nlp).cpu() )
    N_item_rec = torch.cat(N_item_rec, dim=0)
    print('N_item_rec.shape = ', N_item_rec.shape)
    print('get_item_rec end')
    return N_item_rec # N * F

def get_many(model, function_name, test_dataloader, index2asin, seen_asin2_, seen_user2_, device, verbose=False, method='our'):
    print('get_UserRec_NextIdx_SeqIdx_InteractionIdx start')
    confounder = 0
    
    I_userRec = []
    I_historyL = []
    I_posIdx = []
    I_seen = []
    for batch in tqdm(test_dataloader):
        List_userStr = batch['userStr']
        historyEmb = batch['historyEmb']
        Tensor_posIdx = batch['posIdx']
        Tensor_domainIdx = batch['domainIdx']
        Tensor_historyL = batch['historyL']
        
        if method == 'our':
            userRec = getattr(model, function_name)(historyEmb, Tensor_historyL, confounder, Tensor_domainIdx)
        elif method == 'unisrec':
            interaction = {}
            interaction['item_id_list'] = batch['historyIdxPadded'].to(device)
            interaction['item_emb_list'] = batch['historyEmb'].to(device)
            interaction['item_length'] = batch['historyL'].to(device)
            interaction['item_id'] = batch['posIdx'].to(device)
            interaction['pos_item_emb'] = batch['posEmb'].squeeze(1).to(device)
            userRec = getattr(model, function_name)(interaction)
        I_userRec.append(userRec.cpu())
        
        I_historyL.append(Tensor_historyL)
        
        I_posIdx.append(Tensor_posIdx)
        
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
    # I_seen = I_seen
    I_I_index = list(np.arange(I_userRec.shape[0]))
    
    if verbose:
        print('I_userRec.shape: ', I_userRec.shape, type(I_userRec))
        print('I_historyL.shape: ', I_historyL.shape, type(I_historyL))
        print('I_posIdx.shape: ', I_posIdx.shape, type(I_posIdx))
        print('len(I_seen): ', len(I_seen), type(I_seen))
        print('len(I_I_index): ', len(I_I_index), type(I_I_index))
        
    print('get_UserRec_NextIdx_SeqIdx_InteractionIdx end')
    return_dict = {
        'I_userRec': I_userRec,
        'I_historyL': I_historyL,
        'I_posIdx': I_posIdx,
        'I_seen': I_seen,
        'I_I_index': I_I_index
    }
    return return_dict

def get_metric(N_itemRec, I_many, bs, K, device, verbose=True):
    print('***** calculate NDCG / Recall *****')
    N = N_itemRec.shape[0]
    I_userRec      = I_many['I_userRec']
    I_historyL     = I_many['I_historyL']
    I_posIdx       = I_many['I_posIdx']
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
    
    for j in tqdm(np.arange(J)):
        bs_userRec = I_userRec[j*bs:(j+1)*bs]
        
        bs_similarity = (bs_userRec @ N_itemRec.T).cpu().numpy()
        
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

if 1:
    if 1:
        if 1:
            method = train_parser.method
            
            F = train_parser.nlp_feature
            H = train_parser.rec_feature
            L = train_parser.gru_layer
            
            # method hyper
            confounder = train_parser.confounder
            db = train_parser.domain_balance
            r = train_parser.regularization
            
            lr = train_parser.lr
            #EP = 100
            bs = train_parser.bs
            bs_n = 256
            bs_i = 512 #8192 #65536 #32768
            
            K=20
            
            n_neg = 63
            truncate = 64

            seen_domain_list  = train_parser.seen_domain_list
            train_domain_list = train_parser.train_domain_list
            test_domain_list  = train_parser.test_domain_list
            
            #seen_domain_list .sort()
            #train_domain_list.sort()
            #test_domain_list .sort()

            train_prefix = ':'.join(train_domain_list)
            test_prefix  = ':'.join(test_domain_list )
            print('train prefix(unsorted alphabeta): ', train_prefix)
            print('test. prefix(unsorted alphabeta): ', test_prefix )
            
            confounder = train_parser.confounder
            db = train_parser.domain_balance
            
            experiment_main = train_parser.seen2test
            experiment_sub = 'c='+str(confounder)+'_lr='+str(lr)
            
            import setproctitle
            setproctitle.setproctitle(train_prefix+'_test:'+experiment_sub)
    
            train_folder = 'experiments/'+experiment_main+'/'+experiment_sub+'/'+train_prefix
            
            save_folder  = 'experiments/'+experiment_main+'/'+experiment_sub+'/'+train_prefix+'2'+test_prefix
            
            if method == 'our':
                model = myGRU4Rec(F,H,L,train_domain_list).to(device)
                print(model)
                print(model.init_domain.shape)
                print('trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
            else:
                from baseline_unisrec import UniSRec
                model = UniSRec(truncate, H).to(device)
                print(model) 
            
            if method == 'our':
                model.load_state_dict(torch.load(train_folder+'/our.pt'))
            if method == 'unisrec':
                model.load_state_dict(torch.load(train_folder+'/unisrec.pt'))
            
            isExist = os.path.exists(save_folder)
            if not isExist:
                os.makedirs(save_folder)
            
            dataloaders = get_myDataLoader(seen_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = 128, device = device)
            seen_index2asin = dataloaders['fused_idx2asinDict']
            seen_user = dataloaders['fused_user']
            
            dataloaders = get_myDataLoader(test_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = 128, device = device)
            split_test_dataloader = dataloaders['split_test__dataloader']
            split_x_nlp = dataloaders['split_nlpTensor']
            
            split_index2asin = dataloaders['split_idx2asinDict']
            del dataloaders
            
            seen_asin2_ = dict.fromkeys(seen_index2asin.values())
            seen_user2_ = dict.fromkeys(seen_user)
            
            print('***** test *****')
            model.eval()
            
            # initialize metric
            results = {}
            
            with torch.no_grad():
                for i, [test_domain, test_dataloader, item_nlp, index2asin] in enumerate(zip(test_domain_list, split_test_dataloader, split_x_nlp, split_index2asin)):
                    print('-------------------- ',i+1,'/',len(test_domain_list),'  ',test_domain,' start --------------------')
                    
                    N_item_rec = get_item_rec(model, 'get_item_rec', item_nlp, bs_n, device, verbose=True, method=method)
                    
                    I_many = get_many(model, 'get_user_rec', test_dataloader, index2asin, seen_asin2_, seen_user2_, device, verbose=True, method=method)
                    
                    results[test_domain] = get_metric(N_item_rec, I_many, bs_i, K, device, verbose=True)
                    
                    print('-------------------- ',i+1,'/',len(test_domain_list),'  ',test_domain,'   end --------------------')
            
            pkl_file = save_folder+'.pickle'
            
            with open(pkl_file, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('save pkl to: ',pkl_file)
            with open(pkl_file, 'rb') as handle:
                results = pickle.load(handle)
            
            # for key in results.keys():
            #     print(type(results[key]['random']['NDCG']))
            # print([results[key]['random']['NDCG'] for key in results.keys()])
            
            random_ndcg   = np.mean([results[key]['random']['rNDCG'] for key in results.keys()])
            random_recall = np.mean([results[key]['random']['rRecall'] for key in results.keys()])
            
            mean_ndcg   = np.mean([results[key]['mean']['rNDCG'] for key in results.keys()])
            mean_recall = np.mean([results[key]['mean']['rRecall'] for key in results.keys()])
            
            seen_ndcg   = np.mean([results[key]['seen']['rNDCG'] for key in results.keys()])
            seen_recall = np.mean([results[key]['seen']['rRecall'] for key in results.keys()])
            
            unseen_ndcg   = np.mean([results[key]['unseen']['rNDCG'] for key in results.keys()])
            unseen_recall = np.mean([results[key]['unseen']['rRecall'] for key in results.keys()])
            
            print(train_prefix+'     '+test_prefix+'     '+str(confounder)+'\n')
            print('         random / mean   / seen   / unseen \n')
            print('rNDCG  : ' + '{:.4f}'.format(random_ndcg  ) + ' / ' + '{:.4f}'.format(mean_ndcg  ) + ' / ' + '{:.4f}'.format(seen_ndcg  ) + ' / ' + '{:.4f}'.format(unseen_ndcg  ) + '\n')
            print('rRecall: ' + '{:.4f}'.format(random_recall) + ' / ' + '{:.4f}'.format(mean_recall) + ' / ' + '{:.4f}'.format(seen_recall) + ' / ' + '{:.4f}'.format(unseen_recall) + '\n')
            
            with open('experiments/'+experiment_main+'/'+experiment_sub+'/'+method+'_'+train_parser.test_file, 'a') as f:
                f.write(train_prefix+'     '+test_prefix+'     '+str(confounder)+'\n')
                f.write('         random / mean   / seen   / unseen \n')
                f.write('rNDCG  : ' + '{:.4f}'.format(random_ndcg  ) + ' / ' + '{:.4f}'.format(mean_ndcg  ) + ' / ' + '{:.4f}'.format(seen_ndcg  ) + ' / ' + '{:.4f}'.format(unseen_ndcg  ) + '\n')
                f.write('rRecall: ' + '{:.4f}'.format(random_recall) + ' / ' + '{:.4f}'.format(mean_recall) + ' / ' + '{:.4f}'.format(seen_recall) + ' / ' + '{:.4f}'.format(unseen_recall) + '\n')
            
            
