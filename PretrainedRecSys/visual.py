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


with open('I_userRec_c1.pickle', 'rb') as handle:
    I_userRec_c1 = pickle.load(handle)
    
with open('I_userRec_c0.pickle', 'rb') as handle:
    I_userRec_c0 = pickle.load(handle)
    
    
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import seaborn as sns

plt.figure()


num = 20
#count = [0]
userRec = []
for key in I_userRec_c1.keys():
    Emb0 = I_userRec_c0[key]
    Emb1 = I_userRec_c1[key]
    Emb_dis = np.sum(((Emb0-Emb1)**2).numpy(),axis=1)
    idx = np.argsort(Emb_dis)[-num:]
    #np.random.shuffle(Emb)
    #count.append(count[-1]+I_userRec_c1[key].shape[0])
    userRec.append(Emb1[idx])
    userRec.append(Emb0[idx])
userRec = np.concatenate(userRec, axis=0)
#userRec = np.concatenate(userRec, axis=1)
    
tsne = TSNE(n_components=2, verbose=1, perplexity=4, n_iter=300)
userRec = tsne.fit_transform(userRec)
    
plt.figure()
i=0
colors=['red','blue','green']
for key in list(I_userRec_c1.keys()):
    plt.scatter(userRec[num*i    :num*(i+1),0], userRec[num*i    :num*(i+1),1], s=32, facecolors='none', edgecolors=colors[i%3], label=key+' w/ confounder')
    plt.scatter(userRec[num*(i+1):num*(i+2),0], userRec[num*(i+1):num*(i+2),1], s=32, marker='x',                 c=colors[i%3], label=key+' w/o confounder')
    plt.legend()
    i += 2
plt.show()