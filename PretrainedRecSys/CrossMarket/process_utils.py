import pandas as pd
import gzip
import json
from tqdm import tqdm
import os
import requests
import gzip
import shutil
import zipfile
import numpy as np
import math
import time
#import progressbar
import os
import time
import datetime
from collections import Counter
import torch

def check_ratio(csv_df, ITEM_ID, keys_filter):
    print('***** filter started *****')
    print('keys_filter = ',keys_filter)
            
    ### check dataset
    keys_total = {key:0 for key in keys_filter}
    keys_valid = {key:0 for key in keys_filter}
    keys_ratio = {key:0 for key in keys_filter}

    keys_df = pd.DataFrame(keys_filter, columns=['key'])
    
    for key in keys_filter:
        for data in csv_df[key]:
            keys_total[key] += 1
            if (type(data) != float) and (type(data) != int) and (len(data)>0): # the 'nan' input data should be a list instead of float
                keys_valid[key] += 1
    for key in keys_filter:
        keys_ratio[key] = keys_valid[key]/keys_total[key]
    
    keys_df['exists_ratio'] = list(keys_ratio.values())
    print(keys_df)
    print('num_total = ', keys_total[ITEM_ID])
    ### check end

    csv_df = csv_df.drop_duplicates(subset=[ITEM_ID])  #remove duplicate asin
    csv_df = csv_df.reset_index(drop=True)
    print('***** filter finished *****')
    #print(csv_df.iloc[:5])
    
    return csv_df



def process_training(nlp_model, DIM, csv_df, ITEM_ID, ITEM_INIT, keys_node):
    print('***** process training started *****')
    item_dim = DIM
    training_info = {}
    print('*** Process Node: ITEM_ID ***')
    training_info[ITEM_ID] = {}
    training_info[ITEM_ID]['mapping'] = {}
    training_info[ITEM_ID]['x'] = np.zeros([csv_df.shape[0], item_dim])
    ID_series  = csv_df[ITEM_ID]
    Des_series = csv_df[ITEM_INIT]
    for i in tqdm(np.arange(csv_df.shape[0])):
        training_info[ITEM_ID]['mapping'][ID_series[i]] = i
        # include title
        des0 = csv_df.iloc[i]['title']+' '
        # include decription('features')
        try:
            if (type(Des_series[i]) != float) and (len(Des_series[i])>0):
                des1 = ' '.join(Des_series[i])
            else:
                des1 = ''
                #print('start case1')
                #print(Des_series[i])
                #print('end')
        except:
            des1 = ''
            #print('start case2')
            #print(Des_series[i])
            #print('end')
        # include category
        categories = csv_df.iloc[i]['categories']
        des2 = ' This item belongs to the following categories: ' 
        des3 = ' '.join(categories)
        #print('**************************')
        #print(des0+des1+des2+des3)
        
        embeddings = nlp_model.encode(des0+des1+des2+des3)
        
        training_info[ITEM_ID]['x'][i] = embeddings
    
    for key in keys_node:
        print('*** Process Node: '+key+' ***')
        training_info[key] = {}
        training_info[key]['mapping'] = {}
        training_info[key]['x'] = None

        csv_key = csv_df[key]
        categoreis_list = []
        for i in np.arange(csv_df.shape[0]):
            if type(csv_key[i]) == str:
                categoreis_list += [csv_key[i]]
            elif type(csv_key[i]) == list:
                categoreis_list += csv_key[i]
            else:
                raise ValueError('not str or list.')
        categoreis_list = list(dict.fromkeys(categoreis_list))
        for i, category in enumerate(categoreis_list):
            training_info[key]['mapping'][category] = i
        
        training_info[key]['x'] = np.zeros([len(categoreis_list), item_dim])
        for i, category in tqdm(enumerate(categoreis_list)):
            embeddings = nlp_model.encode(category)
            training_info[key]['x'][i] = embeddings

    for key in keys_node:
        print('*** Process Edge: '+key+' ***') 
        triple0 = ITEM_ID+'-included_in_'+key+'-'+key
        triple1 = key+'-'+key+'_includes-'+ITEM_ID
        
        training_info[triple0] = {}
        training_info[triple1] = {}
        
        triple0_edges = []
        triple1_edges = []
        
        training_info[ITEM_ID]['y_'+key] = np.full([csv_df.shape[0], len(training_info[key]['mapping'])], False)
        
        list_df = []
        #print('brand: ', csv_df['brand'])
        #print(training_info['brand']['mapping'])
        #print('category: ', csv_df['category'])
        #print(training_info['category']['mapping'])
        from_node = csv_df[ITEM_ID]
        to_node_list = csv_df[key]
        #print('to_node_list: ',to_node_list)
        for i in tqdm(np.arange(csv_df.shape[0])):
            item_ID = training_info[ITEM_ID]['mapping'][from_node[i]]
            #print(item_ID == i)
            if type(to_node_list[i]) == str:
                to_nodes = [to_node_list[i]]
            else:
                to_nodes = to_node_list[i]
            for to_node in to_nodes:
                #print(key)
                #print(training_info[key]['mapping'])
                category_ID = training_info[key]['mapping'][to_node]
                triple0_edges.append(np.array([[item_ID],[category_ID]]))
                triple1_edges.append(np.array([[category_ID],[item_ID]]))
            
                training_info[ITEM_ID]['y_'+key][item_ID,category_ID] = True
                
        training_info[triple0]['edge_index'] = np.concatenate(triple0_edges, axis=1)
        training_info[triple1]['edge_index'] = np.concatenate(triple1_edges, axis=1)
        
    print('***** process training finished *****')
    return training_info



def cal_pop_k(item_num, past_df_pop, curr_df_pop, k=2):
    #print('TEST')
    
    item_list = list(past_df_pop['item'])
    
    item2count = dict(Counter(item_list))
    
    count = np.array(list(item2count.values()))
    stats = np.array([np.mean(count**i)**(1/i) for i in np.arange(k)+1])
    v = np.sum(count)
    
    #print(stats)
    mapping1 = {key:value/stats for key, value in item2count.items()}
    mapping2 = {key:np.array([value, v]) for key, value in item2count.items()}
    #mapping3_freq = {key:np.array([value]) for key, value in item2count.items()}
    #mapping4_occu = {key:np.array([divide_(value,v)]) for key, value in item2count.items()}
    
    curr_df_pop['pop1'] = curr_df_pop['item'].map(mapping1).fillna('NAN')
    curr_df_pop['pop2'] = curr_df_pop['item'].map(mapping2).fillna('NAN')
    #curr_df_pop['freq'] = curr_df_pop['item'].map(mapping3_freq).fillna('NAN')
    #curr_df_pop['occu'] = curr_df_pop['item'].map(mapping4_occu).fillna('NAN')
    #print(curr_df_pop['pop'])
    #curr_df_pop.loc[curr_df_pop['pop'] == 'NAN', 'pop'] = np.ones(k)*0.0
    #print(curr_df_pop)
    
    # process mapping1 to item_index and item_pop
    Numpy_item_pop1 = np.zeros([item_num, k], dtype=float)
    Numpy_item_pop1[list(mapping1.keys())] = list(mapping1.values())
    Tensor_item_pop1 = torch.from_numpy(Numpy_item_pop1)
    
    Numpy_item_pop2 = np.zeros([item_num, 2], dtype=float)
    Numpy_item_pop2[list(mapping2.keys())] = list(mapping2.values())
    Tensor_item_pop2 = torch.from_numpy(Numpy_item_pop2)
    
    return curr_df_pop, Tensor_item_pop1, Tensor_item_pop2
    
def process_pop_k(item_num, user_item_time, k, past_interval = 15):
    print('process_pop_k start')
    Aday = 86400
    #output
    #user_item_time_day_pop = []

    time_min = np.min(user_item_time['time'])
    time_max = np.max(user_item_time['time'])
    
    #print(time_min)
    #check whether start with 00:00:00
    #if time_min%Aday != 0:
    #    raise Exception("start unixtime no satisfy 00:00:00")
    day0 = time_min//Aday
    dayE = time_max//Aday
    Tensor_day_item_pop1 = torch.zeros([dayE-day0+1, item_num, k], dtype=torch.float)
    Tensor_day_item_pop2 = torch.zeros([dayE-day0+1, item_num, 2], dtype=torch.float)
    
    user_item_time_day_pop = []
    pbar = tqdm(range(day0, dayE+1, 1))
    for day in pbar:
        #print(day-day0)
        curr_df_pop = user_item_time[user_item_time.time//Aday == day].copy()
        curr_df_pop['day'] = day-day0
        curr_df_pop['pop1'] = 'NAN'
        curr_df_pop['pop2'] = 'NAN'
        interval = past_interval
        if (day != day0) and (len(curr_df_pop) != 0): # if it's the first day, we do not calculate pop for that day.
            past_df_pop = user_item_time[(user_item_time.time//Aday < day) & (user_item_time.time//Aday >= day-interval)]
            while (len(past_df_pop) == 0):
                interval = interval+1
                past_df_pop = user_item_time[(user_item_time.time//Aday < day) & (user_item_time.time//Aday >= day-interval)]
            
            pbar.set_description('curr = '+str(len(curr_df_pop))+' past = '+str(len(past_df_pop)))
            
            curr_df_pop, Tensor_item_pop1, Tensor_item_pop2 = cal_pop_k(item_num, past_df_pop, curr_df_pop, k)
            Tensor_day_item_pop1[day-day0] = Tensor_item_pop1
            Tensor_day_item_pop2[day-day0] = Tensor_item_pop2
        user_item_time_day_pop.append(curr_df_pop)
    print('process_pop_k end')
    return_df = pd.concat(user_item_time_day_pop, ignore_index=True)
    
    #print(return_df[:5])
    return_df['pop1'] = return_df['pop1'].apply(lambda x: np.ones(k)*0.0 if x == 'NAN' else x)
    return_df['pop2'] = return_df['pop2'].apply(lambda x: np.ones(2)*0.0 if x == 'NAN' else x)
    #return_df['freq'] = return_df['freq'].apply(lambda x: np.ones(1)*0.0 if x == 'NAN' else x)
    #return_df['occu'] = return_df['occu'].apply(lambda x: np.ones(1)*0.0 if x == 'NAN' else x)
    #print(return_df[:5])
    
    return return_df, Tensor_day_item_pop1, Tensor_day_item_pop2