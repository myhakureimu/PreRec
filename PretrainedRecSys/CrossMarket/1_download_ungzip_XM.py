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
#import progressbar
import os

folder = 'data'
isExist = os.path.exists(folder)
if not isExist:
    os.makedirs(folder)

domain_list = ['in','es','ca','au','mx','de']#['es','jp','au','de','ca','mx','us','uk'] 

caterory_list = ['Arts_Crafts_and_Sewing',
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
for domain in domain_list:
    folder = 'data/'+domain
    isExist = os.path.exists(folder)
    if not isExist:
        os.makedirs(folder)
    
    for category in caterory_list:
        for head, tail in zip(head_list,tail_list):
            print(domain, category, head)
            url = 'https://ciir.cs.umass.edu/downloads/XMarket/FULL/'+domain+'/'+category+'/'+head+'_'+domain+'_'+category+'.'+tail+'.gz'
            r = requests.get(url, allow_redirects=True)
            folder = 'data/'+domain+'/'+category
            isExist = os.path.exists(folder)
            if not isExist:
                os.makedirs(folder)
                
            open('data/'+domain+'/'+category+'/'+head+'_'+domain+'_'+category+'.json.gz', 'wb').write(r.content)
            
            cmd = 'gzip -d data/'+domain+'/'+category+'/'+head+'_'+domain+'_'+category+'.json.gz'
            os.system(cmd)
