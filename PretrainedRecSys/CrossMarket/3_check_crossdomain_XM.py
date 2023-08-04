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

domain_list = ['cn','ca','us','ae','au','br','de','es','fr','in','it','jp','mx','nl','sa','sg','tr','uk']

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



reviewer_dict = {}
item_dict = {} 
for domain in domain_list:
    folder = 'data/'+domain
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)
    
    reviewer_list = []
    item_list = []
    for category in caterory_list:
        for head, tail, check in zip(head_list,tail_list, check_list):
            if check:
                ### extract needed information for category head
                print(domain, category, head)
                reviewerID_asin_unixTime = []
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
                            reviewer_list.append(reviewerID)
                            item_list.append(asin)
    
    reviewer_dict[domain] = list(dict.fromkeys(reviewer_list))
    item_dict[domain] = list(dict.fromkeys(item_list))
    
        
reviewerConfusionMatrix = np.zeros([len(reviewer_dict), len(reviewer_dict)])
reviewerSum = 0
itemConfusionMatrix = np.zeros([len(reviewer_dict), len(reviewer_dict)])
itemSum = 0
for i, keyi in enumerate(reviewer_dict.keys()):
    for j, keyj in enumerate(reviewer_dict.keys()):
        reviewerConfusionMatrix[i, j] = len(list(set(reviewer_dict[keyi]) & set(reviewer_dict[keyj])))
        itemConfusionMatrix[i, j] = len(list(set(item_dict[keyi]) & set(item_dict[keyj])))
    reviewerSum += reviewerConfusionMatrix[i,i]
    itemSum += itemConfusionMatrix[i,i]
print('reviewerSum = ', reviewerSum)
print('itemSum = ', itemSum)

np.save('ALL-reviewerConfusionMatrix.npy', reviewerConfusionMatrix)
np.save('ALL-itemConfusionMatrix.npy', itemConfusionMatrix)
