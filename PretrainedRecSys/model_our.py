import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from itertools import chain
from tqdm import tqdm

import multiprocess as mp
import matplotlib.pyplot as plt
import time


class NLP2Rec(torch.nn.Module):
    def __init__(self, F, H):
        super(NLP2Rec, self).__init__()
        self.linear0 = nn.Linear(in_features = F,
                                 out_features = H,
                                 bias = False)
    
    def forward(self, x):
        #print(x.dtype)
        x = self.linear0(x)
        return x

class GRU2Rec(torch.nn.Module):
    def __init__(self, F, H):
        super(GRU2Rec, self).__init__()
        self.linear0 = nn.Linear(in_features = F,
                                 out_features = H,
                                 bias = False)
    
    def forward(self, x):
        #print(x.type)
        x = self.linear0(x)
        return x

class popEmbedder(torch.nn.Module):
    def __init__(self, Fi, H, Fo):
        super(popEmbedder, self).__init__()
        self.linear0 = nn.Linear(in_features = Fi, out_features = H, bias = True)
        self.act0 = nn.GELU()
        #self.bn = nn.BatchNorm1d(H)
        self.linear1 = nn.Linear(in_features = H, out_features = Fo, bias = True)
    
    def forward(self, x):
        x = self.linear0(x)
        x = self.act0(x)
        #x = self.bn(x)
        x = self.linear1(x)
        return x.squeeze(-1)
        

                   
class myGRU4Rec(torch.nn.Module):
    def __init__(self, model_param): #F, H, L, domain_list, N=None, user_embedder='gru', att_head=2, max_historyL=64, pop_d=0, drop=0.1, model_flag=0, device=None):
        super(myGRU4Rec, self).__init__()
        self.MF = model_param['model_flag']
        self.F = model_param['F']
        self.H = model_param['H']
        self.L = model_param['L']
        print('********************************************* ',self.MF,' *********************************************')
        self.device = model_param['device']
        
        self.user_embedder_name = model_param['user_embedder']
        
        if model_param['domain_debias'] == 1:
            self.init_domain = torch.nn.Parameter(torch.randn(len(model_param['domain_list']), self.H))
            self.init_domain.data.fill_(0.00)
            self.init_domain.requires_grad = True
        
        
        if model_param['item_debias'] == 1:
            self.numItem_PerDomain = model_param['numItem_PerDomain']
            print(self.numItem_PerDomain)
            self.start_index = torch.zeros(len(self.numItem_PerDomain), dtype=torch.long)
            for i in np.arange(len(self.numItem_PerDomain)-1)+1:
                self.start_index[i] = np.sum(self.numItem_PerDomain[:i])
            print(self.start_index)
            
            N = sum(self.numItem_PerDomain)
            self.init_item = torch.nn.Parameter(torch.randn(N, model_param['H']))
            self.init_item.data.fill_(0.00)
            self.init_domain.requires_grad = True
        
        if model_param['pop_d'] > 0:
            self.pop_embedder = popEmbedder(model_param['pop_d'], 512, 1)   
        
        self.nlp2rec = NLP2Rec(F=self.F, H=self.H)
        
        if self.user_embedder_name == 'gru':
            self.user_embedder = nn.GRU(input_size = self.H,
                              hidden_size = self.H,
                              num_layers = self.L,
                              bias = True,
                              batch_first = True,
                              dropout = 0.1,
                              bidirectional = False)
        elif self.user_embedder_name == 'transformer':
            self.n_head = model_param['n_head']
            self.max_historyL = model_param['max_historyL']
            self.position_emb_lookup = torch.nn.Parameter(torch.randn(self.max_historyL, self.H))
            self.position_emb_lookup.data.fill_(0.00)
            self.position_emb_lookup.requires_grad = True
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.H, 
                                                       dim_feedforward=self.H, 
                                                       nhead=self.n_head, 
                                                       batch_first=True)
            self.user_embedder = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                                       num_layers=self.L,
                                                       norm=None)
        
        # elif self.user_embedder_name == 'transformerWhole':
        #     self.att_head = att_head
        #     self.max_historyL = max_historyL
        #     self.position_emb_lookup = torch.nn.Parameter(torch.randn(max_historyL, H))
        #     self.position_emb_lookup.data.fill_(0.00)
        #     self.position_emb_lookup.requires_grad = True
        #     encoder_layer = nn.TransformerEncoderLayer(d_model=256, dim_feedforward=256, nhead=att_head, batch_first=True)
        #     self.user_embedder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=2, norm=None)
        else:
            raise Exception("not valid embedder")
        
        #self.gru2rec = GRU2Rec(F=H, H=H)
        #self.BN = nn.BatchNorm1d(H)
        
    def forward(self, batch, confounder, item_debias, pop_d, device):
        
        historyEmb = batch['historyItemEmbTensor'] # B, truncate, H
        historyIdx = batch['historyItemIdxTensor'] # B, truncate
        historyL   = batch['historyLTensor']       # B, 1
        historyDomainIdx = batch['historyDomainIntTensor'] # B, truncate
        
        #print('historyEmb: ',historyEmb.device)
        
        posEmb        = batch['posItemEmbTensor']   # B, 1, H
        posIdx        = batch['posItemIdxTensor']   # B, 1
        posDomainIdx  = batch['posDomainIntTensor'] # B, 1
        
        if pop_d>0:
            posPop = batch['posPopEmbTensor']    # B, 1,     pop_d
            negPop = batch['negPopEmbTensor']    # B, n_neg, pop_d
            
        negEmb       = batch['negItemEmbTensor']   # B, n_neg, H
        negIdx       = batch['negItemIdxTensor']   # B, n_neg
        negDomainIdx = batch['negDomainIntTensor'] # B, n_neg
        
        
    
        posRec = self.get_item_rec(posEmb, confounder=0, domainIdx=posDomainIdx, item_debias=item_debias, itemIdx=posIdx)
            
        negRec = self.get_item_rec(negEmb, confounder=0, domainIdx=negDomainIdx, item_debias=item_debias, itemIdx=negIdx)
        
        preRec = self.get_user_rec(batch, confounder, item_debias).unsqueeze(1)
            
        pos_dis = torch.sum(preRec * posRec, dim=2, keepdim=False)
        neg_dis = torch.sum(preRec * negRec, dim=2, keepdim=False)
        #print(pos_dis.shape)
        #print(neg_dis.shape)
        
        if pop_d > 0:
            pos_dis = pos_dis + self.pop_embedder(posPop)
            neg_dis = neg_dis + self.pop_embedder(negPop)
        
        prediction = torch.cat([pos_dis, neg_dis], dim=1)
        
        label = torch.zeros(prediction.shape[0], dtype=torch.long).to(device)
        
        return prediction, label, historyL
    
    def get_pop_score(self, pop):
        pop_score = self.pop_embedder(pop)
        
        return pop_score
        
    def get_item_rec(self, x, confounder=0, domainIdx=None, item_debias=None, itemIdx=None):
        #print(x.shape)
        x_rec = self.nlp2rec(x)
        #print('x_rec ',x_rec.shape)
        #print('domainIdx ', itemIdx_shift.shape)
        #print('HERE')
        #print(self.numItem_PerDomain)
        #print(self.start_index)
        if item_debias == 1:
            itemIdx_shift = domainIdx*0
            for i in np.arange(len(self.numItem_PerDomain)):
                itemIdx_shift[domainIdx == i] = self.start_index[i]
            #print(domainIdx[:2,:4])
            #print(itemIdx_shift[:2,:4])
            #print(itemIdx_shift.shape)
            #print(x_rec.shape)
            #print('itemIdx ', itemIdx.shape)
            #print('shift_indx ',itemIdx_shift.shape)
            #print('rec_shift ',self.init_item[itemIdx+itemIdx_shift].shape)
            x_rec = x_rec + self.init_item[itemIdx+itemIdx_shift]
        
        #print(x_rec.shape)
        return x_rec
        
    def get_user_rec(self, batch, confounder, item_debias):
        historyEmb = batch['historyItemEmbTensor'] # B, truncate, H
        historyIdx = batch['historyItemIdxTensor'] # B, truncate
        historyL   = batch['historyLTensor'] # B, 1
        domainIdx  = batch['historyDomainIntTensor'] # B, truncate
        
        
        historyEmb = self.get_item_rec(historyEmb, confounder, domainIdx, item_debias, historyIdx)
        
        if confounder in [0,2,100]:
            historyEmb = historyEmb
        elif confounder == 1:
            #print(self.init_domain[domainIdx])
            historyEmb = historyEmb +self.init_domain[domainIdx]
        elif confounder == 99:
            historyEmb = historyEmb +self.init_item[historyIdx]
        else:
            print('Wrong domain vector')
        
        if self.user_embedder_name == 'gru':
            preRec, _ = self.user_embedder(historyEmb)
        elif self.user_embedder_name == 'transformer':
            historyEmb = historyEmb + self.position_emb_lookup.unsqueeze(0).expand_as(historyEmb)
            mask = self.generate_square_subsequent_mask(self.max_historyL)
            preRec = self.user_embedder(historyEmb, mask = mask)
        
        preRec = preRec[[torch.arange(historyL.shape[0]),historyL-1]]
        
        return preRec
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1). \
            transpose(0, 1)
        mask = mask.float(). \
            masked_fill(mask == 0, float('-inf')). \
            masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)
