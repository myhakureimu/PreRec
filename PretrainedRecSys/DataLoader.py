import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

import random
import numpy as np
from numpy.random import choice
import pickle 
import gzip
import time
from sklearn.model_selection import train_test_split
import itertools

def from_asin2idx_to_idx2asin(asin2idx):
    idx2asin = {}
    for asin, idx in asin2idx.items():
        idx2asin[idx] = asin
    return idx2asin

def truncate_seq2sample(domainStr, domainInt, truncate, List_seqDict):
    truncate_List_InteractionDict = []
    for seqDict in List_seqDict:
        for i in np.arange(len(seqDict['seq'])-1)+1:
            if i <= truncate:
                truncate_List_InteractionDict.append({'userStr': seqDict['user'],
                                                      'posDomainInt': domainInt,
                                                      'domainStr': domainStr,
                                                      'historyItemIdxList': seqDict['seq'][:i],
                                                      'historyDayIdxList': seqDict['day'][:i],
                                                      'posItemIdxList': seqDict['seq'][i:i+1],
                                                      'posDayIdxList': seqDict['day'][i:i+1],
                                                     })
            else:
                truncate_List_InteractionDict.append({'userStr': seqDict['user'],
                                                      'posDomainInt': domainInt,
                                                      'domainStr': domainStr,
                                                      'historyItemIdxList': seqDict['seq'][i-truncate:i],
                                                      'historyDayIdxList': seqDict['day'][i-truncate:i],
                                                      'posItemIdxList': seqDict['seq'][i:i+1],
                                                      'posDayIdxList': seqDict['day'][i:i+1],
                                                     })
    return truncate_List_InteractionDict
                
class myDataset(Dataset):
    def __init__(self, name, I2B, List_sampleDict, n_neg, truncate, self_neg=0, transform=None):
        self.name = name
        self.I2B = I2B
        self.List_sampleDict = List_sampleDict
        
        print(self.name+' num of sample =', len(self.List_sampleDict))
        self.self_neg = self_neg
        self.n_neg = n_neg
        self.T = truncate
        
    def __len__(self):
        return len(self.List_sampleDict)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            print('??? ??? ???')
            idx = idx.tolist()
        
        # get interaction
        sampleDict = self.List_sampleDict[idx]
        # get user
        userStr = sampleDict['userStr']
        # get domain
        posDomainInt = sampleDict['posDomainInt']
        posDomainIntList = [posDomainInt]
        posDomainIntTensor = torch.tensor(posDomainIntList) # 1
        
        # get historyDomainIntTensor
        historyDomainIntTensor = posDomainIntTensor.repeat(self.T)
        
        # get historyL ?
        historyItemIdxList = sampleDict['historyItemIdxList']
        historyL = len(historyItemIdxList)
        # get historyItemIdxTensor T (truncate)
        historyItemIdxTensor = torch.tensor(historyItemIdxList)
        historyItemIdxTensorPad = torch.randint(0, self.I2B[posDomainInt]['numItem'], [self.T-historyL])
        historyItemIdxTensor = torch.cat([historyItemIdxTensor, 
                                          historyItemIdxTensorPad])
        # get historyItemEmbTensor T x F (nlp embedding)
        historyItemEmbTensor = self.I2B[posDomainInt]['itemEmb'][historyItemIdxTensor]
        # get historyDayIdxTensor T
        historyDayIdxList = sampleDict['historyDayIdxList']
        historyDayIdxTensor = torch.tensor(historyDayIdxList)
        historyDayIdxTensorPad = torch.ones([self.T-historyL], dtype=torch.long)*historyDayIdxList[-1] # using the last day to do pad
        historyDayIdxTensor = torch.cat([historyDayIdxTensor, 
                                         historyDayIdxTensorPad])
        # get historyPopEmbTensor T x k (pop feature)
        historyPopEmbTensor = self.I2B[posDomainInt]['dayItemPop'][historyDayIdxTensor, historyItemIdxTensor]
        
        # get posItemIdxTensor 1
        posItemIdxList = sampleDict['posItemIdxList']
        posItemIdxTensor = torch.tensor(posItemIdxList)
        # get posItemEmbTensor 1 x F
        posItemEmbTensor = self.I2B[posDomainInt]['itemEmb'][posItemIdxTensor]
        # get posDayIdxTensor 1
        posDayIdxList = sampleDict['posDayIdxList']
        posDayIdxTensor = torch.tensor(posDayIdxList)
        # get posPopEmbTensor 1 x k
        posPopEmbTensor = self.I2B[posDomainInt]['dayItemPop'][posDayIdxTensor, posItemIdxTensor]
        posPopEmbTensor = posPopEmbTensor.to(posItemEmbTensor.device)
        
        # get negItemEmbTensor n_neg x F
        # get negItemEmbTensor n_neg x F
        # get negPopEmbTensor n_neg x k
        negDomainIntTensor_list = []
        negItemIdxTensor_list = []
        negItemEmbTensor_list = []
        negPopEmbTensor_list = []
        if self.name == 'train':
            if self.self_neg == 0:
                nD = len(self.I2B)
                average_neg = int(self.n_neg/nD)
                negPerD_list = [average_neg]*(nD-1) + [self.n_neg-average_neg*(nD-1)]
                np.random.shuffle(negPerD_list)
            elif self.self_neg == 1:
                #print('HERE')
                nD = len(self.I2B)
                negPerD_list = [0 for i in np.arange(nD)]
                negPerD_list[posDomainInt] = self.n_neg
            else:
                raise Exception('self_neg wrong') 
        else:
            negPerD_list = [0 for i in self.I2B.keys()]
            negPerD_list[posDomainInt] = self.n_neg
        if 1:
            for i, negPerD in enumerate(negPerD_list):
                negDomainIntTensor_list.append(torch.ones(negPerD, dtype=torch.long)*i)
                
                negItemIdxTensor = torch.randint(0, self.I2B[i]['numItem'], [negPerD])
                negItemIdxTensor_list.append(negItemIdxTensor)
                
                negItemEmbTensor = self.I2B[i]['itemEmb'][negItemIdxTensor]
                negItemEmbTensor_list.append(negItemEmbTensor)
                
                if i == posDomainInt: # using the last day to do n_neg
                    negDayIdxTensor = torch.ones([negPerD], dtype=torch.long) *posDayIdxList[0]
                    negPopEmbTensor = self.I2B[i]['dayItemPop'][negDayIdxTensor, negItemIdxTensor]
                else: # just give pop = 0
                    negDayIdxTensor = torch.ones([negPerD], dtype=torch.long) *0
                    negPopEmbTensor = self.I2B[i]['dayItemPop'][negDayIdxTensor, negItemIdxTensor] *0
                negPopEmbTensor_list.append(negPopEmbTensor)
        negDomainIntTensor = torch.cat(negDomainIntTensor_list, dim=0)
        negItemIdxTensor   = torch.cat(negItemIdxTensor_list  , dim=0)
        negItemEmbTensor   = torch.cat(negItemEmbTensor_list  , dim=0)
        negPopEmbTensor    = torch.cat(negPopEmbTensor_list   , dim=0)
        negPopEmbTensor = negPopEmbTensor.to(posItemEmbTensor.device)
#         print('negDomainIntTensor ',negDomainIntTensor.shape)
#         print('negItemIdxTensor ',negItemIdxTensor.shape)
#         print('negItemEmbTensor ',negItemEmbTensor.shape)
#         print('negPopEmbTensor ',negPopEmbTensor.shape)
        
        sample = {
            'userStr': userStr,
            'posDomainIntTensor': posDomainIntTensor,
            'historyL': historyL,
            
            'historyDomainIntTensor': historyDomainIntTensor,
            
            'historyItemIdxTensor': historyItemIdxTensor,
            'historyItemEmbTensor': historyItemEmbTensor,
            #'historyDayIdxTensor': historyDayIdxTensor, # we do not need this
            'historyPopEmbTensor': historyPopEmbTensor,
            
            'posItemIdxTensor': posItemIdxTensor,
            'posItemEmbTensor': posItemEmbTensor,
            'posDayIdxTensor': posDayIdxTensor,
            'posPopEmbTensor': posPopEmbTensor,
            
            'negDomainIntTensor': negDomainIntTensor,
            
            'negItemIdxTensor': negItemIdxTensor,
            'negItemEmbTensor': negItemEmbTensor,
            #'negDayIdxTensor': negDayIdxTensor, # we do not has this
            'negPopEmbTensor': negPopEmbTensor,
            
        }
        return sample



def collate_fn_pad(batch):
    #time_0 = time.time()
    List_userStr       = [data['userStr']    for data in batch]
    posDomainIntTensor = torch.cat([data['posDomainIntTensor'].unsqueeze(0)  for data in batch]) # B x 1
    historyLTensor     = torch.tensor([data['historyL']    for data in batch]) # B
    
    historyDomainIntTensor = torch.cat([data['historyDomainIntTensor'].unsqueeze(0)    for data in batch]) # B x T
    
    historyItemIdxTensor = torch.cat([data['historyItemIdxTensor'].unsqueeze(0)    for data in batch]) # B x T
    historyItemEmbTensor = torch.cat([data['historyItemEmbTensor'].unsqueeze(0)    for data in batch]) # B x T x F
    historyPopEmbTensor  = torch.cat([data['historyPopEmbTensor' ].unsqueeze(0)    for data in batch]) # B x T x k

    posItemIdxTensor = torch.cat([data['posItemIdxTensor'].unsqueeze(0)    for data in batch]) # B x 1
    posItemEmbTensor = torch.cat([data['posItemEmbTensor'].unsqueeze(0)    for data in batch]) # B x 1 x F
    posDayIdxTensor  = torch.cat([data['posDayIdxTensor' ].unsqueeze(0)    for data in batch]) # B x 1
    posPopEmbTensor  = torch.cat([data['posPopEmbTensor' ].unsqueeze(0)    for data in batch]) # B x 1 x k
    
    negDomainIntTensor = torch.cat([data['negDomainIntTensor'].unsqueeze(0)    for data in batch]) # B x n_neg

    negItemIdxTensor = torch.cat([data['negItemIdxTensor'].unsqueeze(0)    for data in batch]) # B x n_neg
    negItemEmbTensor = torch.cat([data['negItemEmbTensor'].unsqueeze(0)    for data in batch]) # B x n_neg x F
    negPopEmbTensor  = torch.cat([data['negPopEmbTensor' ].unsqueeze(0)    for data in batch]) # B x n_neg x k
    
    batch_dict = {
        'List_userStr': List_userStr,
        'posDomainIntTensor': posDomainIntTensor,
        'historyLTensor': historyLTensor,
        
        'historyDomainIntTensor': historyDomainIntTensor,
        
        'historyItemIdxTensor': historyItemIdxTensor,
        'historyItemEmbTensor': historyItemEmbTensor,
        #'historyDayIdxTensor': historyDayIdxTensor, # we do not need this
        'historyPopEmbTensor': historyPopEmbTensor,
        
        'posItemIdxTensor': posItemIdxTensor,
        'posItemEmbTensor': posItemEmbTensor,
        'posDayIdxTensor': posDayIdxTensor, # we do not need this
        'posPopEmbTensor': posPopEmbTensor,
        
        'negDomainIntTensor': negDomainIntTensor,
        
        'negItemIdxTensor': negItemIdxTensor,
        'negItemEmbTensor': negItemEmbTensor,
         #'negDayIdxTensor': negDayIdxTensor, # we do not has this
        'negPopEmbTensor': negPopEmbTensor,
    }
    return batch_dict


def get_split(seq_list, random_seed=2023, split=[80,10,10]):
    train, valid, test = split[0], split[1], split[2]
    train_seq_list, valid_test_seq_list = train_test_split(seq_list, test_size = 1-train/(train+valid+test), random_state=2023)
    valid_seq_list,       test_seq_list = train_test_split(valid_test_seq_list, test_size = valid/(valid+test), random_state=2023)
    return train_seq_list, valid_seq_list, test_seq_list

# def reindex_seq_based_on_item_index(split_List_seqDict, List_startItemIdx, fuse=True):
#     fused_List_seqDict = []
#     for i, List_seqDict in enumerate(split_List_seqDict):
#         previous_count = List_startItemIdx[i]
#         this_seqDict = []
#         for seqDict in List_seqDict:
#             this_seqDict.append({'user': seqDict['user'], 
#                                  'seq': [item_idx+previous_count for item_idx in seqDict['seq']],
#                                  'day': seqDict['pop1'],
#                                  #'pop1': seqDict['pop1'],
#                                  #'pop2': seqDict['pop2'],
#                                 })
#         if fuse:
#             fused_List_seqDict.extend(this_seqDict)
#         else:
#             fused_List_seqDict.append(this_seqDict)
#     return fused_List_seqDict

def get_myDataLoader(domain_list, db, n_neg, truncate, bs=4, train_cut=9999999, validtest_cut=None, pop_type='pop1', self_neg=0, device=None, dataSplit=[80,10,10], seen_only=False, max_user_cut=None):
    print(validtest_cut)
    print('load data from multiple domains')
    
    fused_train_List_sampleDict = [] #         list of dict   # we train domains together but valid/test domains seperately
    split_valid_List_sampleDict = [] # list of list of dict
    split_test__List_sampleDict = [] # list of list of dict
    
    fused_train_sampleWeightScalar = [] # list: [weight of interaction 0, ...]
    
    itemNum_perDomain = []
    
    domainInt2Base = {} # used for save itemEmb and dayItemPop
    domainStr2domainInt = {}
    for domainInt, domainStr  in enumerate(domain_list):
        domainInt2Base[domainInt] = {}
        domainInt2Base[domainInt]['domainStr'] = domainStr
        domainStr2domainInt[domainStr] = domainInt
    I2B = domainInt2Base

    
    
    seen_user = [] # used for checking user overlap
    seen_asin = [] # used for checking asin overlap
    
    for domainInt, domainStr in enumerate(domain_list):
        print('**********  ',domainInt,' -- ',domainStr,'  **********')
        data_file = 'CrossMarket/processed/'+domainStr+'_data_filtered.pickle'
        dayItemPop_file = 'CrossMarket/processed/'+domainStr+'_day_item_'+pop_type+'.pt'
        interaction_file = 'CrossMarket/processed/'+domainStr+'_interaction_seq.pklz'
        
        with open(data_file, 'rb') as handle:
            itemInfo = pickle.load(handle)['training_info']['asin']
            
            itemEmb = torch.from_numpy(itemInfo['x']).type(torch.FloatTensor).to(device)
            I2B[domainInt]['itemEmb'] = itemEmb
            #print('itemEmb: ', I2B[domainInt]['itemEmb'].device)
            itemNum_perDomain.append(itemEmb.shape[0])
            
            itemAsin2itemIdx = itemInfo['mapping']
            itemIdx2itemAsin = from_asin2idx_to_idx2asin(itemAsin2itemIdx)
            I2B[domainInt]['A2I'] = itemAsin2itemIdx
            I2B[domainInt]['I2A'] = itemIdx2itemAsin
            
            seen_asin.extend(list(itemAsin2itemIdx.keys())) 
            
            numItem = len(itemAsin2itemIdx)
            print('# of item = ', numItem)
            I2B[domainInt]['numItem'] = numItem
        

        I2B[domainInt]['dayItemPop'] = torch.load(dayItemPop_file)#.to(device)

        with gzip.open(interaction_file, 'rb') as ifp:
            List_seqDict = pickle.load(ifp)
            if (max_user_cut != None) and (len(List_seqDict) > max_user_cut):
                List_seqDict = random.sample(List_seqDict, max_user_cut)
            
            seen_user.extend([seqDict['user'] for seqDict in List_seqDict])
            
            numUser = len(List_seqDict)
            print('# of user = ', numUser)
            
            if not seen_only:
                numTotalInteraction = np.sum([len(seqDict['seq']) for seqDict in List_seqDict])
                print('# of total interaction = ', numTotalInteraction)
                
                numTotalSample = np.sum([np.max([0, len(seqDict['seq'])-1]) for seqDict in List_seqDict])
                print('# of total sample = ', numTotalSample)

                train_List_seqDict, valid_List_seqDict, test__List_seqDict = get_split(List_seqDict, random_seed=2023, split=dataSplit)

                numTrainSample = np.sum([np.max([0, len(seqDict['seq'])-1]) for seqDict in train_List_seqDict])
                print('# of train sample = ', numTrainSample)

                train_List_sampleDict = truncate_seq2sample(domainStr, domainInt, truncate, train_List_seqDict)
                train_List_sampleDict = train_List_sampleDict[:train_cut]
                numTrainSample = len(train_List_sampleDict)
                fused_train_List_sampleDict.extend(train_List_sampleDict)
                
                valid_List_sampleDict = truncate_seq2sample(domainStr, domainInt, truncate, valid_List_seqDict)
                print('HERE ', validtest_cut)
                if (validtest_cut!=None) and (validtest_cut<len(valid_List_sampleDict)):
                    valid_List_sampleDict = valid_List_sampleDict[:validtest_cut]
                    print(len(valid_List_sampleDict))
                split_valid_List_sampleDict.append(valid_List_sampleDict)
                
                test__List_sampleDict = truncate_seq2sample(domainStr, domainInt, truncate, test__List_seqDict)
                if (validtest_cut!=None) and (validtest_cut<len(test__List_sampleDict)):
                    test__List_sampleDict = test__List_sampleDict[:validtest_cut]
                split_test__List_sampleDict.append(test__List_sampleDict)

                if len(train_List_sampleDict) != numTrainSample:
                    print(len(train_List_sampleDict))
                    print(numTrainSample)
                    raise Exception('numTrainSample wrong') 

                if db == 0:
                    raise Exception('please choose db(domain balance) = 1') 
                    fused_train_sampleWeightScalar.extend([1]*len(train_List_seqDict))
                elif db == 1:
                    fused_train_sampleWeightScalar.extend([1/(numTrainSample*len(domain_list))]*numTrainSample)
                else:
                    raise Exception('db(domain balance) type not supported')
    
    if seen_only:
        print('seen_only')
        dataloaders = {
            'seen_user': seen_user,
            'seen_asin': seen_asin,
            'itemNum_perDomain': itemNum_perDomain,
        }
        return dataloaders
    
    seen_asin = list(dict.fromkeys(seen_asin).keys())
    seen_user = list(dict.fromkeys(seen_user).keys())
    
    print('*********** convert interaction to dataset ***********')
    print(fused_train_List_sampleDict[0])
    fused_train_dataset = myDataset('train', I2B, fused_train_List_sampleDict, n_neg, truncate, self_neg)
    split_valid_dataset = [myDataset('valid', I2B, List_InteractionDict, n_neg, truncate, self_neg) \
                           for i, List_InteractionDict in enumerate(split_valid_List_sampleDict)]
    split_test__dataset = [myDataset('test_', I2B, List_InteractionDict, n_neg, truncate, self_neg) \
                           for i, List_InteractionDict in enumerate(split_test__List_sampleDict)]
    #print('??????', len(fused_train_sample_weight))
    fused_train_dataloader = DataLoader(fused_train_dataset, batch_size=bs,
                                        sampler = WeightedRandomSampler(fused_train_sampleWeightScalar, 
                                                                        len(fused_train_sampleWeightScalar), 
                                                                        replacement=True),
                                        collate_fn = collate_fn_pad,
                                        shuffle=False, num_workers=0,pin_memory=False)
    split_valid_dataloader = [DataLoader(dataset, batch_size=bs,
                                         collate_fn = collate_fn_pad,
                                         shuffle=False, num_workers=0,pin_memory=False) for dataset in split_valid_dataset]
    split_test__dataloader = [DataLoader(dataset ,  batch_size=bs,
                                         collate_fn = collate_fn_pad,
                                         shuffle=False, num_workers=0,pin_memory=False) for dataset in split_test__dataset ]
    
    dataloaders = {
        'fused_train_dataloader': fused_train_dataloader,
        'split_valid_dataloader': split_valid_dataloader,
        'split_test__dataloader': split_test__dataloader,
        'I2B': I2B,
        'seen_user': seen_user,
        'seen_asin': seen_asin,
        'itemNum_perDomain': itemNum_perDomain,
        'numTrainSample': numTrainSample,
    }
    return dataloaders

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('gpu or cpu?: ',device)

    all_domains = ['es','in','ca','mx','au','de']
    seendl = ['us']
    testdl = []
    
    dataloaders = get_myDataLoader(all_domains, db = 1, n_neg = 7, truncate = 4, bs = 2, pop_type='pop1', device = device, seen_only=False)
#     for batch in dataloaders['fused_train_dataloader']:
#         for key in batch:
#             print(key)
#             if key == 'List_userStr':
#                 print(len(batch[key]))
#             else:       
#                 print(batch[key].shape)
#         break
#     for batch in dataloaders['split_valid_dataloader'][0]:
#         for key in batch:
#             print(key)
#             if key == 'List_userStr':
#                 print(len(batch[key]))
#             else:       
#                 print(batch[key].shape)
#         break
    
    
