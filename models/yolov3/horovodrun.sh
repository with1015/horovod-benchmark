#!/bin/bash

num_process=$1
hosts=$2

batch_size=8
visible_gpus=0,1,2,3
num_minibatches=100
echo "batch size : "$batch_size
echo "GPU ID: "$visible_gpus
echo 'Storage:' $storage


pretrained_weights=/ssd_dataset2/dataset/pytorch_yolo/pretrained_weights/darknet53.conv.74
dataset_dir=/ssd_dataset2/dataset/coco


NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 horovodrun -np $num_process -H $hosts -p 51234 \
 python train.py \
  --weights $pretrained_weights \
  --batch-size $batch_size \
  --cfg cfg/yolov3.cfg \
  --data-config /home/plum/yolov3_myrepo/data/coco2014.data \
  --num-minibatches $num_minibatches \
  --notest \
  $dataset_dir
