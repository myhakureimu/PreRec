from DataLoader import get_myDataLoader, from_asin2idx_to_idx2asin
from tqdm import tqdm
import pickle
import gzip

seen_domain_list = ['in','es','ca']

zero_domain_list = ['au','mx','de']

dataloaders = get_myDataLoader(seen_domain_list, db = 1, n_neg = 7, truncate = 4, bs = 2, pop_type='pop1', device = 'cuda', seen_only=True)

seen_asin2_ = dict.fromkeys(dataloaders['seen_asin'])
print(len(seen_asin2_))
seen_user2_ = dict.fromkeys(dataloaders['seen_user'])

del dataloaders


def overlap_count(seen_asin2_, seen_user2_, domainStr):
    
    count_seen = 0
    count_unseen = 0
    
    data_file = 'CrossMarket/processed/'+domainStr+'_data_filtered.pickle'
    #dayItemPop_file = 'CrossMarket/processed/'+domainStr+'_day_item_'+pop_type+'.pt'
    interaction_file = 'CrossMarket/processed/'+domainStr+'_interaction_seq.pklz'
        
    with open(data_file, 'rb') as handle:
        itemInfo = pickle.load(handle)['training_info']['asin']
        itemAsin2itemIdx = itemInfo['mapping']
        itemIdx2itemAsin = from_asin2idx_to_idx2asin(itemAsin2itemIdx)
        A2I = itemAsin2itemIdx
        I2A = itemIdx2itemAsin

        
    with gzip.open(interaction_file, 'rb') as ifp:
        List_seqDict = pickle.load(ifp)
        
        for seqDict in tqdm(List_seqDict):
            user = seqDict['user'] 
            for index in seqDict['seq'][1:]:
                asin = I2A[index]
                if (user in seen_user2_) or (asin in seen_asin2_):
                    count_seen += 1
                else:
                    count_unseen += 1
    
    print(count_seen, count_unseen)
    
for domain in zero_domain_list:
    overlap_count(seen_asin2_, seen_user2_, domain)