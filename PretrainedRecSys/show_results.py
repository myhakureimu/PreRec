#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 21:44:37 2022

@author: linzq
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
'''
fig, axs = plt.subplots( figsize=(30.0, 60.0) , nrows=len(domain_list), ncols=1, sharey=False) 
for domain, ax in zip(domain_list, axs):
    ax.set_ylabel(domain, fontsize=30)
    ax._frameon = False
    ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
'''

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

import csv

domain_list = ['au','mx','de']
metric_list = ['recall', 'ndcg']
method_list = ['our', 'unisrec', 'transformer', 'gru', 'bert'][::-1]

domain2title = {
    'au': 'Australia',
    'mx': 'Mexico',
    'de': 'Germany'
    }

method2linestyle = {
    'our': 'solid',
    'unisrec': 'solid',
    'gru': 'dashed',
    'transformer': 'dashed',
    'bert': 'dotted',
    }
method2linecolor = {
    'our': 'red',
    'unisrec': '#DC582A',
    'gru': 'blue',
    'transformer': 'green',
    'bert': 'gray',
    }
method2marker = {
    'our': '*',
    'unisrec': 'p',
    'gru': 'P',
    'transformer': 'X',
    'bert': 'o',
    }

method2legend = {
    'our': 'PreRec',
    'unisrec': 'UniSRec',
    'gru': '$GRU4Rec^*$',
    'transformer': '$SASRec^*$',
    'bert': 'SBERT',
    }

metric2ylabel = {
    'recall': 'Recall@K%',
    'ndcg': 'r-NDCG@K%',
    }

ymax_list = [[0.12, 0.28, 0.44],
             [0.08, 0.2, 0.32]]

domainBertR = {
    'au': 0.0552,
    'mx': 0.1509,
    'de': 0.2737,
    }
domain2BertN = {
    'au': 0.0380,
    'mx': 0.1095,
    'de': 0.2103,
    }

metric2domain2bert={
    'recall': domainBertR,
    'ndcg': domain2BertN,
    }

domain2maxL={
    'au': 51569,
    'mx': 92284,
    'de': 338214,
    }

fig, axs = plt.subplots( figsize=(20.0, 10.0) , nrows=len(metric_list), ncols=len(domain_list), sharex=True, sharey=False) 

for R, metric in enumerate(metric_list):
    axs[R,0].set_ylabel(metric2ylabel[metric])
    for C, domain in enumerate(domain_list):
        ymax = ymax_list[R][C]
        #domain = 'au'
        #metric = 'recall'
        step_list = []
        
        method_metrics = {}
        map2index = {}
        
        metric2csv_title = {
            'recall': 'rRecall',
            'ndcg': 'rNDCG',
            }
        
        
        for method in method_list:
            if method != 'bert':
                print(method)
                method_metrics[method] = []
                
        with open(domain+'-'+metric+'.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if 'Step' in row:
                    index = row.index('Step')
                    map2index['Step'] = index
                    for method in method_list:
                        if method != 'bert':
                            index = row.index(method+' - FT '+domain+' '+metric2csv_title[metric])
                            map2index[method] = index
                else:
                    step_list.append(int(row[map2index['Step']]))
                    for method in method_list:
                        if method != 'bert':
                            method_metrics[method].append(float(row[map2index[method]]))
                
            step_list.insert(0, 1)
            for method in method_list:
                if method != 'bert':
                    method_metrics[method].insert(0, method_metrics[method][0])
        
            
        for method in method_list:
            if (((R+1)==1) and ((C+1)==2)):
                print('legend')
                if method != 'bert':
                    print('A')
                    axs[R,C].plot(step_list, method_metrics[method],
                                  linestyle = method2linestyle[method],
                                  color = method2linecolor[method],
                                  marker = method2marker[method],
                                  label = method2legend[method],
                                  markersize=13)
                else:
                    
                    step_list = []
                    for i in range(6):
                        if 10**i > domain2maxL[domain]:
                            step_list.append(domain2maxL[domain])
                            break
                        step_list.append(10**i)
                    axs[R,C].plot(step_list, [metric2domain2bert[metric][domain] for i in np.arange(len(step_list))],
                                  linestyle = method2linestyle[method],
                                  color = method2linecolor[method],
                                  marker = method2marker[method],
                                  label = method2legend[method],
                                  markersize=13)
            else:
                if method != 'bert':
                    axs[R,C].plot(step_list, method_metrics[method],
                                  linestyle = method2linestyle[method],
                                  color = method2linecolor[method],
                                  marker = method2marker[method],
                                  markersize=13)
                else:
                    
                    step_list = []
                    for i in range(7):
                        if 10**i > domain2maxL[domain]:
                            step_list.append(domain2maxL[domain])
                            break
                        step_list.append(10**i)
                    axs[R,C].plot(step_list, [metric2domain2bert[metric][domain] for i in np.arange(len(step_list))],
                                  linestyle = method2linestyle[method],
                                  color = method2linecolor[method],
                                  marker = method2marker[method],
                                  label = method2legend[method],
                                  markersize=13)
                
            
        axs[R,C].vlines(x=domain2maxL[domain], ymin=0, ymax=ymax, 
                        linestyle = '-.',
                        color = 'gray')
        axs[R,C].set_xlim([1, 500000])
        axs[R,C].set_xscale('log')
        axs[R,C].set_ylim([0, ymax])
        axs[R,C].set_xticks([1, 10, 100, 1000, 10000, 100000],
                            labels=[f'$0$']+[f'$10^{i}$' for i in np.arange(5)+1])
        axs[R,C].set_yticks([0, ymax*0.25, ymax*0.5, ymax*0.75, ymax])
        if R==0:
            axs[R,C].set_title(domain2title[domain])
        
fig.text(0.5, 0.04, 'Number of fine-tuning samples', ha='center')
#fig.text(0.04, 0.5, 'Metric', va='center', rotation='vertical')

axs[0,1].legend(ncol=5, loc='upper center', bbox_to_anchor=(0.35, 1.35))

plt.savefig('incremental.pdf')