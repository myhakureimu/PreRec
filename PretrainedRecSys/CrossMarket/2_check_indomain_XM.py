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
import time
import os
import time
import datetime

domain_list = ['us','ca','cn']

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
check_list = [False, True, False]




for domain in domain_list:
    info_dict = {}
    for category in caterory_list:
        for head, tail, check in zip(head_list,tail_list, check_list):
            if check:
                ### extract needed information for category head
                print(domain, category, head)
                reviewerID_asin_unixTime = []
                file = domain+'/'+category+'/'+head+'_'+domain+'_'+category+'.json'
                data_df = pd.read_json(file, lines=True)
                for index, row in data_df.iterrows():
                    for index, value in row.items():
                        if value is not None:
                            #print(value.keys())
                            reviewerID = value['reviewerID']
                            asin = value['asin']
                            dt = datetime.datetime.strptime(value['cleanReviewTime'], '%Y-%m-%d')
                            unixTime = time.mktime(dt.timetuple())
                            reviewerID_asin_unixTime.append([reviewerID, asin, unixTime])
                            
                info_dict[domain+'-'+category+'-'+head] = reviewerID_asin_unixTime

    review_dict = {}
    item_dict = {}
    for key in info_dict.keys():
        review_dict[key] = list(dict.fromkeys([x[0] for x in info_dict[key]]))
        item_dict[key] = list(dict.fromkeys([x[1] for x in info_dict[key]]))
        
    reviewConfusionMatrix = np.zeros([len(review_dict), len(review_dict)])
    reviewSum = 0
    itemConfusionMatrix = np.zeros([len(review_dict), len(review_dict)])
    itemSum = 0
    for i, keyi in enumerate(review_dict.keys()):
        for j, keyj in enumerate(review_dict.keys()):
            reviewConfusionMatrix[i, j] = len(list(set(review_dict[keyi]) & set(review_dict[keyj])))
            itemConfusionMatrix[i, j] = len(list(set(item_dict[keyi]) & set(item_dict[keyj])))
        reviewSum += reviewConfusionMatrix[i,i]
        itemSum += itemConfusionMatrix[i,i]
    print('reviewSum = ', reviewSum)
    print('itemSum = ', itemSum)
    np.save(domain+'-reviewConfusionMatrix.npy', reviewConfusionMatrix)
    np.save(domain+'-itemConfusionMatrix.npy', itemConfusionMatrix)
