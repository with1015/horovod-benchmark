#!/bin/bash

np=$1
hosts=$2
batch_size=32

echo "batch size: "$batch_size

dataset_dir='/ssd_dataset2/dataset/pytorch_wmt14_en_de'

visible_gpus='0,1,2,3'

# For test BLEU
#  --online-eval \
# For ckpt
#  --save-dir $save_dir \
# For profile
#  --profile-dir 'transformer_base_profile' \
# For batch size in terms of tokens
#  --max-tokens 5120 \
export PYTHONWARNINGS="ignore"  # To disable Userwarning

NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 horovodrun -np $np -H $hosts -p 51234 \
  python train.py \
    $dataset_dir \
    --arch transformer_wmt_en_de \
    --share-all-embeddings \
    --optimizer adam \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 0.0 \
    --warmup-updates 4000 \
    --lr 0.0007 \
    --min-lr 0.0 \
    --dropout 0.1 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-update 100 \
    --batch-size $batch_size \
    --sentence-avg \
    --seed 1 \
    --no-save \
    --online-eval \
    --no-validate
