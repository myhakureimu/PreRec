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
import pickle
import datetime
from utils import check_ratio, process_training, process_pop_k
from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from collections import Counter
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#https://openreview.net/forum?id=ATRbfIyW6sI
print('load model')
nlp_model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')
DIM = 768
print('load finish')

domain_list = ['in']#['jp']

caterory_list = [
                 'Arts_Crafts_and_Sewing',
                 'Automotive',
                 'Books',
                 'CDs_and_Vinyl',
                 'Cell_Phones_and_Accessories',
                 'Digital_Music',
                 'Electronics',
                 'Grocery_and_Gourmet_Food',
                 'Home_and_Kitchen',
                 'Industrial_and_Scientific',
                 'Kindle_Store',
                 'Movies_and_TV',
                 'Musical_Instruments',
                 'Office_Products',
                 'Sports_and_Outdoors',
                 'Toys_and_Games'
                 ]

head_list = ['ratings', 'reviews', 'metadata']
tail_list = ['txt',     'json',    'json']

k = 4
# N_perrow = 2+k
# fig, axs = plt.subplots( figsize=(90.0, 90.0) , nrows=len(domain_list), ncols=1, sharey=False) 
# for domain, ax in zip(domain_list, axs):
#     ax.set_ylabel(domain, fontsize=30)
#     ax._frameon = False
#     ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')

for KK, domain in enumerate(domain_list):
    check_list = [False, False, True]
    item_list = []
    review_list = []
    for category in caterory_list:
        for head, tail, check in zip(head_list,tail_list, check_list):
            if check:
                file = 'data/'+domain+'/'+category+'/'+head+'_'+domain+'_'+category+'.json'
                data_df = pd.read_json(file, lines=True)
                item_list.append(data_df)
    
    item_df = pd.concat(item_list)
    print('************************* ', len(item_df))
    ''' example in official website
    import gzip
    example_rev_file = file+'.gz'
    review_lines = []
    with gzip.open(example_rev_file, 'rt', encoding='utf8') as f:
        review_lines = f.readlines()
        
    print( eval(review_lines[1].strip())[0] )
    '''
    
    ITEM_ID = 'asin'
    ITEM_INIT = 'features'
    keys_ratio = ['asin', 'categories', 'features']
    keys_node = ['categories']
    
    data_df = check_ratio(item_df, ITEM_ID, keys_ratio) #filter out duplicated item
    
    if 1:
        training_info = process_training(nlp_model, DIM, data_df, ITEM_ID, ITEM_INIT, keys_node)
        
        evaluation_info = None
        
        data = {'data_df': data_df,
                'training_info': training_info,
                'evaluation_info': evaluation_info}
        
        
        with open('processed/'+domain+'_data_filtered.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            
    else:     
        with open('processed/'+domain+'_data_filtered.pickle', 'rb') as handle:
            data = pickle.load(handle)
        
    training_info = data['training_info']
    mapping = data['training_info']['asin']['mapping']
        
    

    check_list = [False, True, False]
    reviewer_dict = {}
    for category in caterory_list:
        for head, tail, check in zip(head_list,tail_list, check_list):
            if check:
                ### extract needed information for category head
                #print(domain, category, head)
                file = 'data/'+domain+'/'+category+'/'+head+'_'+domain+'_'+category+'.json'
                data_df = pd.read_json(file, lines=True)
                for index, row in data_df.iterrows():
                    for index, value in row.items():
                        if value is not None:
                            #print(value.keys())
                            reviewerID = value['reviewerID']
                            asin = value['asin']
                            dt = datetime.datetime.strptime(value['cleanReviewTime'], '%Y-%m-%d')
                            unixTime = time.mktime(dt.timetuple())
                            if asin in mapping.keys():
                                if reviewerID in reviewer_dict.keys():
                                    reviewer_dict[reviewerID].append([unixTime, asin])
                                else:
                                    reviewer_dict[reviewerID] = [[unixTime, asin]]
    print(domain)
    interaction_seq = []
    #interaction_pair = []
    for key, value in reviewer_dict.items():
        unixReviewTime_asin_sorted = sorted(value, key=lambda value: value[0])
        interaction_seq.append({'user': key, 'seq': [mapping[x[1]] for x in unixReviewTime_asin_sorted]})
        if (100<len(unixReviewTime_asin_sorted)) and (len(unixReviewTime_asin_sorted)<200):
            print([x[0] for x in unixReviewTime_asin_sorted])
            break
        # for i in range(len(unixReviewTime_asin_sorted)-1):
        #     curr_ = unixReviewTime_asin_sorted[i][1]
        #     next_ = unixReviewTime_asin_sorted[i+1][1]
        #     if (curr_ != next_) and (curr_ in mapping.keys()) and (next_ in mapping.keys()):
        #         curr_i = mapping[curr_]
        #         next_i = mapping[next_]
        #         interaction_pair.append([curr_i, next_i])


    
    
    # process
    timestamp_list = []
    item_id2time_list = {}
    for item_id in mapping.values():
        item_id2time_list[item_id] = []
        
    user_item_time = []
    for user, sequence in reviewer_dict.items():
        for i in range(len(sequence)):
            interaction_time = int(sequence[i][0])
            interaction_id   = mapping[sequence[i][1]]
            timestamp_list.append(interaction_time)
            item_id2time_list[interaction_id].append(interaction_time)
            user_item_time.append([interaction_time, interaction_id, user])
    user_item_time = pd.DataFrame(user_item_time, columns =['time', 'item', 'user']).sort_values(by=['time'])
    
    item_num = len(mapping)
    df_user_item_time_day_pop, Tensor_day_item_pop1, Tensor_day_item_pop2 = process_pop_k(item_num, user_item_time, k, 15)
    
    dict_reviewer2item_time_day_pop = {}
    for index, row in df_user_item_time_day_pop.iterrows():
        reviewerID = row['user']
        itemId = row['item']
        unixTime = row['time']
        day = row['day']
        pop1 = row['pop1']
        pop2 = row['pop2']
        if reviewerID in dict_reviewer2item_time_day_pop.keys():
            dict_reviewer2item_time_day_pop[reviewerID].append([itemId, unixTime, day, pop1, pop2])
        else:
            dict_reviewer2item_time_day_pop[reviewerID] = [[itemId, unixTime, day, pop1, pop2]]
    #print(dict_reviewer2item_time_day_pop[0])
    
    interaction_day_pop_seq = []
    #interaction_pair = []
    for key, value in dict_reviewer2item_time_day_pop.items():
        unixReviewTime_asin_sorted = sorted(value, key=lambda value: value[1])
        interaction_day_pop_seq.append({'user': key,
                                        'domain': domain,
                                        'day':  [x[2] for x in unixReviewTime_asin_sorted],
                                        'seq':  [x[0] for x in unixReviewTime_asin_sorted],
                                        'pop1': [x[3] for x in unixReviewTime_asin_sorted],
                                        'pop2': [x[4] for x in unixReviewTime_asin_sorted],
                                       })
        # for i in range(len(unixReviewTime_asin_sorted)-1):
        #     curr_ = unixReviewTime_asin_sorted[i][1]
        #     next_ = unixReviewTime_asin_sorted[i+1][1]
        #     if (curr_ != next_) and (curr_ in mapping.keys()) and (next_ in mapping.keys()):
        #         curr_i = mapping[curr_]
        #         next_i = mapping[next_]
        #         interaction_pair.append([curr_i, next_i])
    print(interaction_day_pop_seq[0])
    
    if 1:
        import pickle
        import gzip
        #with open('interaction_filtered_all.pickle', 'wb') as handle:
        #    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open('processed/'+domain+'_interaction_seq.pklz', 'wb') as ofp:
            print('processed/'+domain+'_interaction_seq.pklz')
            pickle.dump(interaction_day_pop_seq, ofp)
        
        print('SAVE FILE')
        torch.save(Tensor_day_item_pop1, 'processed/'+domain+'_day_item_pop1.pt')
        torch.save(Tensor_day_item_pop2, 'processed/'+domain+'_day_item_pop2.pt')
        # with gzip.open('processed/'+domain+'_interaction_pair.pklz', 'wb') as ofp:
        #     pickle.dump(interaction_pair, ofp)
    
#     item_list = list(user_item_time['item'])
#     items = list(dict.fromkeys(item_list))

#     item2count = dict(Counter(item_list))
#     count_count = Counter(list(item2count.values()))
    
#     ax = fig.add_subplot(len(domain_list), N_perrow, KK*N_perrow+1)
#     ax.scatter(count_count.keys(), count_count.values())
#     if KK == 0:
#         ax.set_title('the distribution of\n# of total buy')

    
#     day_list = list(df_user_item_time_day_pop['day'])
#     day2count = Counter(list(day_list))
    
#     xs = [day for day in range(min(day2count), max(day2count)+1)]
#     ys = [day2count[x] for x in xs]
    
#     ax = fig.add_subplot(len(domain_list), N_perrow, KK*N_perrow+2)
#     ax.plot(xs, ys)
#     if KK == 0:
#         ax.set_title('the # of total buy\nfor a day')

    
#     for ki in range(k):
#         pop_list = list([x[ki] for x in df_user_item_time_day_pop['pop1']])
#         pop_list = list(filter(lambda score: score != 0.0, pop_list))
        
#         ax = fig.add_subplot(len(domain_list), N_perrow, KK*N_perrow+2+ki+1)
#         ax.hist(pop_list)
#         if KK == 0:
#             ax.set_title('x**'+str(ki+1))
    
# plt.savefig('stat.png')
print('END')
