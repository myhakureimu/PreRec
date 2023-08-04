import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle
from itertools import chain
from tqdm import tqdm

#import multiprocess as mp
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

# data section
train_parser.add_argument('-seendl', '--seen_domain_list', type=str, nargs='+', default=['cn'],
                    help='domain list -- example: jp cn')
train_parser.add_argument('-traindl', '--train_domain_list', type=str, nargs='+', default=['cn'],
                    help='domain list -- example: jp cn')
train_parser.add_argument('-testdl', '--test_domain_list', type=str, nargs='+', default=['cn'],
                    help='domain list -- example: jp cn')
train_parser.add_argument('-s2t', '--seen2test', type=str, default='cn:cn:cn:cn',
                    help='sequntial trained model folder -- example: cn:cn:cn:cn')
train_parser.add_argument('-db', '--domain_balance', type=int, default=0,
                         help='0: no domain balance; 1: domain balance')
train_parser.add_argument('-sn', '--self_neg', type=int, default=0,
                         help='0: no domain balance; 1: domain balance')
train_parser.add_argument('-ls', '--lr_schedule', type=int, default=0,
                         help='0: no lr_schedule')

# model section
train_parser.add_argument('-F', '--nlp_feature', type=int, default=768,
                         help='nlp feature')
train_parser.add_argument('-H', '--H', type=int, default=64,
                         help='rec feature')
train_parser.add_argument('-L', '--L', type=int, default=2,
                         help='gru layer')

train_parser.add_argument('-method', '--method', type=str, default='our',
                          help= 'method')
train_parser.add_argument('-ue', '--user_embedder', type=str, default='transformer',
                          help= 'method')
train_parser.add_argument('-head', '--att_head', type=int, default=1,
                          help= 'att_head')

train_parser.add_argument('-c', '--confounder', type=int, default=1,
                         help='0: no confounder; 1: add first; 2: add all')
train_parser.add_argument('-r', '--regularization', type=float, default=0.3,
                         help='regularization weights')

train_parser.add_argument('-pt', '--pop_type', type=str, default='pop1',
                          help= 'pop_type')
train_parser.add_argument('-pd', '--pop_d', type=int, default=3,
                          help= 'dimension of pop embedding')

train_parser.add_argument('-id', '--item_debias', type=int, default=1,
                         help='0: w/o item_debias; 1: w/ item_debias')
train_parser.add_argument('-id_r', '--id_regularization', type=float, default=100.0,
                         help='id_regularization')

# training section
train_parser.add_argument('-lr', '--lr', type=float, default=0.001,
                         help='learning rate')
train_parser.add_argument('-bs', '--bs', type=int, default=256,
                         help='batch_size')

# test/zero
train_parser.add_argument('-test_confounder', '--test_confounder', type=int, default=1,
                          help= 'test_confounder')
train_parser.add_argument('-zero_confounder', '--zero_confounder', type=int, default=0,
                          help= 'zero_confounder')
train_parser.add_argument('-test_item_debias', '--test_item_debias', type=int, default=1,
                          help= 'test_confounder')
train_parser.add_argument('-zero_item_debias', '--zero_item_debias', type=int, default=0,
                          help= 'zero_confounder')
train_parser.add_argument('-test_use_pop', '--test_use_pop', type=int, default=1,
                          help= 'test_use_pop')
train_parser.add_argument('-zero_use_pop', '--zero_use_pop', type=int, default=1,
                          help= 'zero_use_pop')

# for try different structure
train_parser.add_argument('-model_flag', '--model_flag', type=int, default=0,
                          help= 'model_flag')




train_parser = train_parser.parse_args()
print('gpu: ', train_parser.gpu)
print('train domain list: ', train_parser.train_domain_list)

print('confounder = ', train_parser.confounder)
print('domain_balance = ', train_parser.domain_balance)

os.environ["CUDA_VISIBLE_DEVICES"] = train_parser.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('gpu or cpu?: ',device)


from utils import train_valid, test
torch.manual_seed(2023)

if 1:
    user_embedder = train_parser.user_embedder
    n_head = train_parser.att_head
    
    method = train_parser.method
    metric_on = 'unseen'
    
    model_flag = train_parser.model_flag
    test_confounder = train_parser.test_confounder
    zero_confounder = train_parser.zero_confounder
    test_item_debias = train_parser.test_item_debias
    zero_item_debias = train_parser.zero_item_debias
    test_use_pop = train_parser.test_use_pop
    zero_use_pop = train_parser.zero_use_pop
    print('HERE')
    print('test_confounder', test_confounder)
    print('zero_confounder', zero_confounder)
    print('test_item_debias', test_item_debias)
    print('zero_item_debias', zero_item_debias)
    print('test_use_pop', test_use_pop)
    print('zero_use_pop', zero_use_pop)
    
    # model hyper
    F = train_parser.nlp_feature
    H = train_parser.H
    L = train_parser.L
    
    # training hyper
    lr = train_parser.lr
    bs = train_parser.bs
    
    # method hyper
    confounder = train_parser.confounder
    r = train_parser.regularization
    
    item_debias = train_parser.item_debias
    id_r = train_parser.id_regularization
    
    
    
    # fixed
    EP = 15
    max_patient = 3
    n_neg = 255
    truncate = 64
    
    pt = train_parser.pop_type
    pd = train_parser.pop_d
    print('pt = ', pt)
    print('pd = ', pd)
    
    db = train_parser.domain_balance
    self_neg = train_parser.self_neg
    lr_schedule = train_parser.lr_schedule
    
    experiment_main = train_parser.seen2test
    experiment_sub = \
                    'self_neg= '+str(self_neg) \
                    +'_lr_schedule= '+str(lr_schedule) \
                    +'_F='+ str(F) \
                    +'_H='+ str(H) \
                    +'_L='+ str(L) \
                    +'_method='+ str(method) \
                    +'_user_embedder='+ str(user_embedder) \
                    +'_n_head='+ str(n_head) \
                    +'_domain_debias='+ str(confounder) \
                    +'_r='+ str(r) \
                    +'_pop_d='+ str(pd) \
                    +'_item_debias='+ str(item_debias) \
                    +'_id_r='+ str(id_r) \
                    +'_model_flag='+ str(model_flag)
    
    seen_domain_list  = train_parser.seen_domain_list
    train_domain_list = train_parser.train_domain_list
    zero_domain_list  = train_parser.test_domain_list
     

    prefix = ':'.join(train_domain_list)
    import setproctitle
    setproctitle.setproctitle(prefix+'_train:'+experiment_sub)
    
    print('prefix(unsorted alphabeta): ', prefix)
    
    save_folder = 'experiments/'+experiment_main+'/'+experiment_sub+'/'+prefix
    
    isExist = os.path.exists(save_folder)
    if not isExist:
        os.makedirs(save_folder)
    
    
        
        #def init_weights(m):
        #    if isinstance(m, nn.Linear):
        #        torch.nn.init.xavier_uniform(m.weight)
        #        if m.bias != None:
        #            m.bias.data.fill_(0.0)
        # 
        #model.apply(init_weights)
        
        #print(model)
        '''
        'fused_train_dataloader': fused_train_dataloader,
        'split_valid_dataloader': split_valid_dataloader,
        'split_test__dataloader': split_test__dataloader,
        'I2B': I2B,
        'seen_user': seen_user,
        'seen_asin': seen_asin,
        '''
        
    # get seen from seen_domain_list
    dataloaders = get_myDataLoader(seen_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = 128, pop_type=pt, self_neg=self_neg, device = device, dataSplit=[40,30,30], seen_only=True, max_user_cut=1000000)
    seen_asin2_ = dict.fromkeys(dataloaders['seen_asin'])
    seen_user2_ = dict.fromkeys(dataloaders['seen_user'])
    itemNum_perDomain = dataloaders['itemNum_perDomain']
    del dataloaders
    
    # get train/valid dataloader from train_domain_list
    dataloaders = get_myDataLoader(train_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = bs, pop_type=pt, self_neg=self_neg, device = device, dataSplit=[40,30,30], max_user_cut=1000000)
    train_I2B = dataloaders['I2B']
    train_dataloader = dataloaders['fused_train_dataloader']
    split_test_dataloader = dataloaders['split_test__dataloader']
    del dataloaders
    
    # get test dataloader from zero_domain_list
    dataloaders = get_myDataLoader(zero_domain_list, db = db, n_neg = n_neg, truncate = truncate, bs = 128, pop_type=pt, self_neg=self_neg, device = device, dataSplit=[40,30,30], max_user_cut=1000000)
    zero_I2B = dataloaders['I2B']
    split_zero_dataloader = dataloaders['split_test__dataloader'] # for test
    del dataloaders

    model_param = {'F': F,
                   'H': H,
                   'L': L,
                   'method': method,
                   'user_embedder': user_embedder,
                   'n_head': n_head,
                   'domain_debias': confounder,
                   'domain_list': train_domain_list,
                   'pop_d': pd,
                   'item_debias': item_debias,
                   'numItem_PerDomain': itemNum_perDomain,
                   'max_historyL': truncate,
                   'model_flag': model_flag,
                   'device': device,
                  }
    
    if method == 'our':
        if confounder not in [99,100]:
            N = None
        else:
            N = N
        model = myGRU4Rec(model_param).to(device) #F,H,L,train_domain_list,N,user_embedder=user_embedder, att_head=1, max_historyL=truncate, pop_d=pd, model_flag=model_flag, device=device).to(device)
        #print(model)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                try:
                    m.weight.data.normal_(mean=0.0, std=0.02)
                    m.weight.bias.normal_(mean=0.0, std=0.00)
                except:
                    pass
        model.apply(init_weights)
        #print(model.init_domain.shape)
        print('trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    else:
        from model_unisrec import UniSRec
        model = UniSRec(truncate, H).to(device)
        
    loss_f = nn.CrossEntropyLoss(reduction='mean')
    if method == 'our':
        if user_embedder == 'transformer':
            params = [{'params': model.position_emb_lookup,        'lr': lr, 'weight_decay': 0.0},
                      {'params': model.nlp2rec.parameters(),       'lr': lr, 'weight_decay': 0.0},
                      {'params': model.user_embedder.parameters(), 'lr': lr, 'weight_decay': 0.0},
                     ]
        elif user_embedder == 'gru':
            params = [
                      {'params': model.nlp2rec.parameters(),       'lr': lr, 'weight_decay': 0.0},
                      {'params': model.user_embedder.parameters(), 'lr': lr, 'weight_decay': 0.0},
                     ]
        else:
            raise Exception('user_embedder wrong') 
        if confounder == 1:
            params.append({'params': model.init_domain,  'lr': lr, 'weight_decay': r})
        if item_debias == 1:
            params.append({'params': model.init_item,    'lr': lr, 'weight_decay': id_r})
        if pd > 0:
            params.append({'params': model.pop_embedder.parameters(), 'lr': lr, 'weight_decay': 0.0})
        opt = torch.optim.Adam(params)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    lr_scheduler = None
    if lr_schedule == 1:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.65)
    
    best_test_rNDCG = 0
    curr_patient = max_patient


    # feed to wandb
    wandb.init(project = 'E 323 pretrain (revise domainIDX from 0 to i)')
    wandb.config.update({
        'self_neg': self_neg,
        'lr_schedule': lr_schedule,
        
        'F': F,
        'H': H,
        'L': L,
        'method': method,
        'user_embedder': user_embedder,
        'n_head': n_head,
        
        'domain_debias': confounder,
        'r': r,
        'item_debias': item_debias,
        'id_r': id_r,
        'pop_debias': pd,
    })
    wandb.watch(model)
    
    for ep in np.arange(EP+1):
        print('EP = ', ep)
        if ep != 0:
            print('***** train *****')
            train_dict = train_valid(model, train_dataloader, confounder, item_debias, mode='train', loss_f=loss_f, lr_scheduler=lr_scheduler, opt=opt, method=method, pop_d=pd, device=device)
            train_loss = train_dict['average_loss']
            wandb.log({f'Train Loss': train_loss}, step=ep)

            # print('***** valid *****')
            # valid_loss_sum = 0
            # with torch.no_grad():
            #     for valid_dataloader in split_valid_dataloader:
            #         valid_dict = train_valid(model, valid_dataloader, confounder, mode='valid', loss_f=loss_f, method=method, device=device)
            #         valid_loss_sum += valid_dict['average_loss']
            # valid_loss = valid_loss_sum/len(split_valid_dataloader)
            # print('Valid_loss: ', valid_loss)
            # wandb.log({f'Valid Loss': valid_loss}, step=ep)
        
        
        with torch.no_grad():
            print('********** Test **********')
            test_confounder = test_confounder
            test_item_debias = test_item_debias
            test_use_pop = test_use_pop
            test_dict = test(seen_asin2_,
                             seen_user2_,
                             model, test_confounder, test_item_debias, test_use_pop,
                             train_domain_list,
                             split_test_dataloader,
                             train_I2B,
                             method=method,
                             device=device,
                             metric_on='mean')
            
            test_rNDCG = test_dict['mean']['mean']['rNDCG']
            print('Test_count  : ', test_dict['mean']['mean']['count'])
            print('Test_rRecall: ', test_dict['mean']['mean']['rRecall'])
            print('Test_rNDCG  : ', test_dict['mean']['mean']['rNDCG'  ])
            wandb.log({f'Test mean rRecall': test_dict['mean']['mean']['rRecall']}, step=ep)
            wandb.log({f'Test mean rNDCG'  : test_dict['mean']['mean']['rNDCG'  ]}, step=ep) 

            if ep%5 == 0:
                print('********** Zero **********')
                zero_confounder = zero_confounder
                zero_item_debias = zero_item_debias
                zero_use_pop = zero_use_pop
                test_dict = test(seen_asin2_,
                                 seen_user2_,
                                 model, zero_confounder, zero_item_debias, zero_use_pop,
                                 zero_domain_list,
                                 split_zero_dataloader,
                                 zero_I2B,
                                 method=method,
                                 device=device,
                                 metric_on=metric_on)
                print('Zero_count  : ', test_dict['mean'][metric_on]['count'])
                print('Zero_mean_rRecall: ', test_dict['mean']['mean']['rRecall'])
                print('Zero_mean_rNDCG  : ', test_dict['mean']['mean']['rNDCG'  ])
                print('Zero_unseen_rRecall: ', test_dict['mean']['unseen']['rRecall'])
                print('Zero_unseen_rNDCG  : ', test_dict['mean']['unseen']['rNDCG'  ])

                for domain in zero_domain_list:
                    wandb.log({f'{domain} mean rRecall': test_dict[domain]['mean']['rRecall']}, step=ep)
                    wandb.log({f'{domain} mean rNDCG'  : test_dict[domain]['mean']['rNDCG'  ]}, step=ep)
                    wandb.log({f'{domain} unseen rRecall': test_dict[domain]['unseen']['rRecall']}, step=ep)
                    wandb.log({f'{domain} unseen rNDCG'  : test_dict[domain]['unseen']['rNDCG'  ]}, step=ep)

                wandb.log({f'Zero mean rRecall': test_dict['mean']['mean']['rRecall']}, step=ep)
                wandb.log({f'Zero mean rNDCG'  : test_dict['mean']['mean']['rNDCG'  ]}, step=ep)
                wandb.log({f'Zero unseen rRecall': test_dict['mean']['unseen']['rRecall']}, step=ep)
                wandb.log({f'Zero unseen rNDCG'  : test_dict['mean']['unseen']['rNDCG'  ]}, step=ep)
        
        # early stop
        if (test_rNDCG > best_test_rNDCG) or (ep%5 == 0):
            print('HERE')
            print(ep)
            curr_patient = max_patient
            best_test_rNDCG = test_rNDCG
            print('loss updated !!!')
            if method == 'our':
                print('SAVE')
                print(save_folder+'/our'+str(ep)+'.pt')
                torch.save(model.state_dict(), save_folder+'/our'+str(ep)+'.pt')
            if method == 'unisrec':
                print('SAVE')
                print(save_folder+'/unisrec'+str(ep)+'.pt')
                torch.save(model.state_dict(), save_folder+'/unisrec'+str(ep)+'.pt')
            
        else:
            curr_patient = curr_patient - 1
            print('patient remain: ', curr_patient)
        
        if curr_patient == 0:
            break
    
    # if method == 'our':
    #     model.load_state_dict(torch.load(save_folder+'/our.pt'))
    # if method == 'unisrec':
    #     model.load_state_dict(torch.load(save_folder+'/unisrec.pt'))

        
    if 0:
        if 0:
            print('********** Zero **********')
            zero_confounder = zero_confounder
            zero_item_debias = zero_item_debias
            zero_use_pop = zero_use_pop
            test_dict = test(seen_asin2_,
                             seen_user2_,
                             model, zero_confounder, zero_item_debias, zero_use_pop,
                             zero_domain_list,
                             split_zero_dataloader,
                             zero_I2B,
                             method=method,
                             device=device,
                             metric_on=metric_on)
            print('Zero_count  : ', test_dict['mean'][metric_on]['count'])
            print('Zero_mean_rRecall: ', test_dict['mean']['mean']['rRecall'])
            print('Zero_mean_rNDCG  : ', test_dict['mean']['mean']['rNDCG'  ])
            print('Zero_unseen_rRecall: ', test_dict['mean']['unseen']['rRecall'])
            print('Zero_unseen_rNDCG  : ', test_dict['mean']['unseen']['rNDCG'  ])
            
            for domain in zero_domain_list:
                wandb.log({f'{domain} mean rRecall': test_dict[domain]['mean']['rRecall']}, step=ep)
                wandb.log({f'{domain} mean rNDCG'  : test_dict[domain]['mean']['rNDCG'  ]}, step=ep)
                wandb.log({f'{domain} unseen rRecall': test_dict[domain]['unseen']['rRecall']}, step=ep)
                wandb.log({f'{domain} unseen rNDCG'  : test_dict[domain]['unseen']['rNDCG'  ]}, step=ep)
            
            wandb.log({f'Zero mean rRecall': test_dict['mean']['mean']['rRecall']}, step=ep)
            wandb.log({f'Zero mean rNDCG'  : test_dict['mean']['mean']['rNDCG'  ]}, step=ep)
            wandb.log({f'Zero unseen rRecall': test_dict['mean']['unseen']['rRecall']}, step=ep)
            wandb.log({f'Zero unseen rNDCG'  : test_dict['mean']['unseen']['rNDCG'  ]}, step=ep)
    # plt.figure()
    # plt.plot(train_loss_list, label='train')
    # plt.plot(valid_loss_list, label='valid')
    # plt.axhline(y=random_loss, label='random', linestyle='dotted')
    # plt.legend()
    # plt.savefig(save_folder+'/'+prefix+'_ep_loss_comparation.png')
    # plt.figure()
    # plt.plot(train_loss_dict.keys(), [x[0] for x in train_loss_dict.values()], label='train')
    # plt.plot(valid_loss_dict.keys(), [x[0] for x in valid_loss_dict.values()], label='valid')
    # plt.axhline(y=random_loss, label='random', linestyle='dotted')
    # plt.legend()
    # plt.savefig(save_folder+'/'+prefix+'_frequency_loss_comparation_.png')
