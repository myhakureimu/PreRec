# Step 1: download/process dataset
cd to PretrainedRecSys/CrossMarket
python 1_download_ungzip_XM.py
python 4_process_XM0.py

# Step 2: pre-train PreRec
cd to PretrainedRecSys.py
python exp_pretrain_zero.py