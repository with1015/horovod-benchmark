#/bin/bash

np=$1
hosts=$2

data_dir=/ssd_dataset2/dataset/pytorch_wmt16_en_de
batch_size=32


NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 \
  horovodrun -np $np -H $hosts -p 51234 python train.py \
  --dataset-dir $data_dir \
  --math fp32 \
  --seed 2 \
  --num-layers 4 \
  --train-batch-size $batch_size \
  --num-minibatches 100 \
  --epochs 1
