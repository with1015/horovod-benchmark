#!/bin/bash

data_name='movielens20M'
data_path='/ssd_dataset2/dataset/movielens_20M/ratings.csv'

#data_name='criteo'
#data_path='/ssd_dataset2/dataset/criteo_v2.1/criteo-research-uplift-v2.1.csv'

model='dfm'

export CUDA_VISIBLE_DEVICES=0,1,2,3

NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 horovodrun -np 4 -H shark6:4 -p 51234 \
python3 main.py \
  --dataset_name $data_name \
  --dataset_path $data_path \
  --model_name $model \
  --epoch 1 \
  --batch_size 4096
