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

import wandb

train_parser = argparse.ArgumentParser()
train_parser = argparse.ArgumentParser(description='Train')
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

train_parser.add_argument('-ue', '--user_embedder', type=str, default='gru',
                          help= 'method')
train_parser.add_argument('-method', '--method', type=str, default='our',
                          help= 'method')

train_parser.add_argument('-pre', '--pretrain', type=int, default=0,
                          help= 'use pretrained model')
train_parser.add_argument('-ftpart', '--finetune_part', type=str, nargs='+', default=['all'],
                          help= 'use pretrained model')
train_parser.add_argument('-ftdl', '--finetune_domain_list', type=str, nargs='+', default=['cn'],
                          help= 'use pretrained model')

train_parser = train_parser.parse_args()
print('gpu: ', train_parser.gpu)
print('train domain list: ', train_parser.train_domain_list)

print('confounder = ', train_parser.confounder)
print('domain_balance = ', train_parser.domain_balance)

os.environ["CUDA_VISIBLE_DEVICES"] = train_parser.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('gpu or cpu?: ',device)


from utils import train_valid, test


if 1:
    use_pretrained = train_parser.pretrain
    finetune_part = train_parser.finetune_part
    ft_flag = [0,0,0,0,0]
    if 'init_domain' in finetune_part:
        ft_flag[0] = 1
    if 'init_item' in finetune_part:
        ft_flag[1] = 1
    if 'position_emb_lookup' in finetune_part:
        ft_flag[2] = 1
    if 'nlp2rec' in finetune_part:
        ft_flag[3] = 1
    if 'user_embedder' in finetune_part:
        ft_flag[4] = 1
                
    user_embedder = train_parser.user_embedder
    method = train_parser.method
    metric_on = 'unseen'
    
    # model hyper
    F = train_parser.nlp_feature
    H = train_parser.rec_feature
    L = train_parser.gru_layer
    
    # training hyper
    lr = train_parser.lr
    bs = train_parser.bs
    
    # method hyper
    confounder = train_parser.confounder
    db = train_parser.domain_balance
    r = train_parser.regularization
    
    # fixed
    EP = 50 
    max_patient = 3
    n_neg = 255
    truncate = 64
    
    
    experiment_main = train_parser.seen2test
    experiment_sub = 'db='+str(db)+'_c='+str(confounder)+'_r='+str(r)+'_ue='+user_embedder
    
    seen_domain_list  = train_parser.seen_domain_list
    train_domain_list = train_parser.train_domain_list
    zero_domain_list  = train_parser.test_domain_list
    finetune_domain_list  = train_parser.finetune_domain_list
    
    run = wandb.init(project = 'finetune '+':'.join(finetune_domain_list))
    wandb.config.update({
        'db': db,
        'method': method,
        'user_embedder': user_embedder,
        'c': confounder,
        'r': r,
        'use_pretrained': use_pretrained,
        'ft_flag': str(ft_flag)
        
    })

for cut in [0, 10, 100, 1000, 10000, 100000]:

    prefix = ':'.join(train_domain_list)
    import setproctitle
    setproctitle.setproctitle(prefix+'_ft:'+experiment_sub+'_pre:'+str(use_pretrained))
    
    print('prefix(unsorted alphabeta): ', prefix)
    
    save_folder = 'experiments/'+experiment_main+'/'+experiment_sub+'/'+prefix
    
    isExist = os.path.exists(save_folder)
    if not isExist:
        os.makedirs(save_folder)
    
    
    
    
#     dataloaders = get_myDataLoader(seen_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = 128, device = device)
#     seen_index2asin = dataloaders['fused_idx2asinDict']
#     seen_user = dataloaders['fused_user']
#     seen_asin2_ = dict.fromkeys(seen_index2asin.values()) # for test
#     seen_user2_ = dict.fromkeys(seen_user) # for test
#     del dataloaders
    
    N=0
#     dataloaders = get_myDataLoader(train_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = bs, device = device)
#     N = dataloaders['fused_nlpTensor'].shape[0]
#     train_dataloader = dataloaders['fused_train_dataloader']
#     split_valid_dataloader = dataloaders['split_valid_dataloader']
#     split_test_dataloader = dataloaders['split_test__dataloader'] # for test
#     split_train_x_nlp = dataloaders['split_nlpTensor'] # for test
#     split_train_index2asin = dataloaders['split_idx2asinDict'] # for test
#     del dataloaders
    

    
    dataloaders = get_myDataLoader(finetune_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = 128, train_cut = max(1,cut), validtest_cut = 50000, device = device)
    
    train_dataloader = dataloaders['fused_train_dataloader']
    
    split_valid_dataloader = dataloaders['split_valid_dataloader'] # for test
    split_zero_dataloader = dataloaders['split_test__dataloader'] # for test
    split_zero_x_nlp = dataloaders['split_nlpTensor'] # for test
    split_zero_index2asin = dataloaders['split_idx2asinDict'] # for test
    
    seen_index2asin = dataloaders['fused_idx2asinDict']
    seen_user = dataloaders['fused_user']
    seen_asin2_ = dict.fromkeys(seen_index2asin.values()) # for test
    seen_user2_ = dict.fromkeys(seen_user) # for test
    del dataloaders
    
    
    if method == 'our':
        if confounder not in [99,100]:
            N = None
        else:
            N = N
        model = myGRU4Rec(F,H,L,train_domain_list,N,user_embedder=user_embedder, max_historyL=truncate).to(device)
        print(model)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                try:
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    module.weight.data.normal_(mean=0.0, std=0.00)
                except:
                    pass
        model.apply(init_weights)
        #print(model.init_domain.shape)
        print('trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    else:
        from baseline_unisrec import UniSRec
        model = UniSRec(truncate, H).to(device)
    
    
    if use_pretrained: # finetune
        if method == 'our':
            model_dict = model.state_dict()
            pretrained_dict = torch.load(save_folder+'/our.pt')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
        elif method == 'unisrec':
            model.load_state_dict(torch.load(save_folder+'/unisrec.pt'))
        else:
            raise ValueError('not support this method: '+method)
    
    loss_f = nn.CrossEntropyLoss(reduction='mean')
    if method == 'our':
        params = []
        ft_flag = [0,0,0,0,0]
        #print(finetune_part)
        if 'init_domain' in finetune_part:
            params.append({'params': model.init_domain,    'lr': lr, 'weight_decay': r})
            ft_flag[0] = 1
        if 'init_item' in finetune_part:
            params.append({'params': model.init_item,    'lr': lr, 'weight_decay': r})
            ft_flag[1] = 1
        if 'position_emb_lookup' in finetune_part:
            params.append({'params': model.position_emb_lookup,    'lr': lr, 'weight_decay': 0.0})
            ft_flag[2] = 1
        if 'nlp2rec' in finetune_part:
            params.append({'params': model.nlp2rec.parameters(), 'lr': lr, 'weight_decay': 0.0})
            ft_flag[3] = 1
        if 'user_embedder' in finetune_part:
            params.append({'params': model.user_embedder.parameters(),     'lr': lr, 'weight_decay': 0.0})
            ft_flag[4] = 1
        print(ft_flag)
        #print(params[0])
        #print(params[1])
        #print(params[2])
        #print(params[3])
        opt = torch.optim.Adam(params)
    else:
        if use_pretrained:
            opt = torch.optim.Adam(model.moe_adaptor.parameters(), lr=lr)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        


    with torch.no_grad():
        ep = 0
        string = 'Valid'
        print(f'********** {string} **********')
        test_dict = test(seen_asin2_, seen_user2_, model, finetune_domain_list, split_valid_dataloader, split_zero_x_nlp, split_zero_index2asin, method=method, device=device, metric_on='mean')
        count = test_dict['mean']['mean']['count']
        rRecall = test_dict['mean']['mean']['rRecall']
        rNDCG   = test_dict['mean']['mean']['rNDCG'  ]
        print(string+'_count  : ', count)
        print(string+'_rRecall: ', rRecall)
        print(string+'_rNDCG  : ', rNDCG  )
        #wandb.log({f'{string} mean rRecall': rRecall}, step=ep)
        #wandb.log({f'{string} mean rNDCG'  : rNDCG  }, step=ep)
        valid_rNDCG = rNDCG
        
        string = 'Test'
        print(f'********** {string} **********')
        test_dict = test(seen_asin2_, seen_user2_, model, finetune_domain_list, split_zero_dataloader, split_zero_x_nlp, split_zero_index2asin, method=method, device=device, metric_on='mean')
        count = test_dict['mean']['mean']['count']
        rRecall = test_dict['mean']['mean']['rRecall']
        rNDCG   = test_dict['mean']['mean']['rNDCG'  ]
        print(string+'_count  : ', count)
        print(string+'_rRecall: ', rRecall)
        print(string+'_rNDCG  : ', rNDCG  )
        #wandb.log({f'{string} mean rRecall': rRecall}, step=ep)
        #wandb.log({f'{string} mean rNDCG'  : rNDCG  }, step=ep)
        test_rNDCG   = rNDCG
        test_rRecall = rRecall
        
    best_valid_rNDCG  = valid_rNDCG
    best_test_rNDCG   = test_rNDCG
    best_test_rRecall = test_rRecall
    curr_patient = max_patient
    
    
    for ep in (np.arange(EP)+1):
        print('EP = ', ep)
        
        print('***** train *****')
        train_dict = train_valid(model, train_dataloader, confounder, mode='train', loss_f=loss_f, opt=opt, method=method, device=device)
        train_loss = train_dict['average_loss']
        #wandb.log({f'Train Loss': train_loss}, step=ep)


        with torch.no_grad():
            string = 'Valid'
            print(f'********** {string} **********')
            test_dict = test(seen_asin2_, seen_user2_, model, finetune_domain_list, split_valid_dataloader, split_zero_x_nlp, split_zero_index2asin, method=method, device=device, metric_on='mean')
            count = test_dict['mean']['mean']['count']
            rRecall = test_dict['mean']['mean']['rRecall']
            rNDCG   = test_dict['mean']['mean']['rNDCG'  ]
            print(string+'_count  : ', count)
            print(string+'_rRecall: ', rRecall)
            print(string+'_rNDCG  : ', rNDCG  )
            #wandb.log({f'{string} mean rRecall': rRecall}, step=ep)
            #wandb.log({f'{string} mean rNDCG'  : rNDCG  }, step=ep)
            valid_rNDCG = rNDCG
            
            string = 'Test'
            print(f'********** {string} **********')
            test_dict = test(seen_asin2_, seen_user2_, model, finetune_domain_list, split_zero_dataloader, split_zero_x_nlp, split_zero_index2asin, method=method, device=device, metric_on='mean')
            count = test_dict['mean']['mean']['count']
            rRecall = test_dict['mean']['mean']['rRecall']
            rNDCG   = test_dict['mean']['mean']['rNDCG'  ]
            print(string+'_count  : ', count)
            print(string+'_rRecall: ', rRecall)
            print(string+'_rNDCG  : ', rNDCG  )
            #wandb.log({f'{string} mean rRecall': rRecall}, step=ep)
            #wandb.log({f'{string} mean rNDCG'  : rNDCG  }, step=ep)
            test_rNDCG   = rNDCG
            test_rRecall = rRecall
        
        # early stop
        if valid_rNDCG > best_valid_rNDCG:
            curr_patient = max_patient
            best_valid_rNDCG  = valid_rNDCG
            best_test_rNDCG   = test_rNDCG
            best_test_rRecall = test_rRecall
            print('loss updated !!!') # do not save model for finetune
#             if method == 'our':
#                 torch.save(model.state_dict(), save_folder+'/our.pt')
#             if method == 'unisrec':
#                 torch.save(model.state_dict(), save_folder+'/unisrec.pt')
        else:
            curr_patient = curr_patient - 1
            print('patient remain: ', curr_patient)
        
        if curr_patient == 0:
            break
    
    wandb.log({f'fintune mean rNDCG  ': best_test_rNDCG    }, step=max(1,cut))
    wandb.log({f'fintune mean rRecall': best_test_rRecall  }, step=max(1,cut))
    #run.finish()