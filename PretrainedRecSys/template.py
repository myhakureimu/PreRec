import numpy as np
import os

method = 'our' 
user_embedder = 'transformer'
att_head = 1
db = 1
self_neg = 0
lr_schedule = 0

confounder = 0
regularization = 0.3

item_debias = 0
id_regularization = 100.0

pop_d = 0
pop_type = 'pop1'

model_flag = 999

test_confounder = confounder
zero_confounder = 0
test_item_debias = item_debias
zero_item_debias = 0
test_use_pop = pop_d
zero_use_pop = pop_d

bs = 256
lr = 0.0001*(1+lr_schedule*2)

F = 768
H = 256
L = 2 #best

gpu=str(7)

# seendl  = ['jp','it','in','ca']
# traindl = ['jp','it','in','ca']
# testdl  = ['fr','au','de','uk']

seendl = ['es','in','ca']
traindl = seendl
testdl = ['mx','au','de']

seen2test = ':'.join(seendl)+'2'+':'.join(testdl)
test_file = ':'.join(testdl)+'.txt'

if 1:
    if 1: 
        cl = 'python train_model.py -gpu '+gpu \
            +' -seendl '+' '.join(seendl) \
            +' -traindl '+' '.join(traindl) \
            +' -testdl '+' '.join(testdl) \
            +' -s2t '+seen2test \
            +' -db '+str(db) \
            +' -sn '+str(self_neg) \
            +' -ls '+str(lr_schedule) \
            +' -c '+str(confounder) \
            +' -r '+str(regularization) \
            +' -id '+str(item_debias) \
            +' -id_r '+str(id_regularization) \
            +' -pt '+str(pop_type) \
            +' -pd '+str(pop_d) \
            +' -F '+str(F) \
            +' -L '+str(L) \
            +' -H '+str(H) \
            +' -lr '+str(lr) \
            +' -bs '+str(bs) \
            +' -method '+method \
            +' -ue '+user_embedder \
            +' -head '+str(att_head) \
            +' -model_flag '+str(model_flag) \
            +' -test_confounder '+str(test_confounder) \
            +' -zero_confounder '+str(zero_confounder) \
            +' -test_item_debias '+str(test_item_debias) \
            +' -zero_item_debias '+str(zero_item_debias) \
            +' -test_use_pop '+str(test_use_pop) \
            +' -zero_use_pop '+str(zero_use_pop)
        print(cl)
        os.system(cl)