#!/bin/bash

num_process=$1
hosts=$2

batch_size=32
model=vgg19
visible_gpus=0,1,2,3

echo "batch size: "$batch_size
echo "model: "$model
echo "GPU ID: "$visible_gpus

echo 'Storage:' $storage

data_dir='/ssd_dataset2/dataset/pytorch_imagenet/imagenet_2012/'
num_minibatches=100

log_file=""
#if [ -z $log_file ]; then
#  log_file="${model}_${batch_size}_${hostname}_step${num_minibatches}.txt"
#fi
#echo "log file:"${log_file}

NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 horovodrun -np $num_process -H $hosts -p 51234\
  python main.py -a $model \
  -b $batch_size \
  --num-minibatches $num_minibatches \
  --no-validate \
  $data_dir
  
  #$data_dir 2>&1 | tee -a -i $log_file 
