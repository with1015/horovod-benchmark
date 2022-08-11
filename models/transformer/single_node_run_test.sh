#!/bin/bash

batch_size=$1

if [ $# -ne 1 ]; then
  echo "[USAGE] [Local batch size]"
  exit 1
fi
echo "batch size: "$batch_size

ret=`source ~/pytorch-benchmarks/scripts/get_cluster_info.sh`
ret_arr=($ret)
storage=${ret_arr[0]}
master=${ret_arr[1]}
echo 'Storage:' $storage
dataset_dir=${storage}'/pytorch_wmt14_en_de'

visible_gpus='0,1,2,3'
ckpt_dir=${storage}'/transformer_base_ckpt'

# For test BLEU
#  --online-eval \

# For profile
#  --profile-dir 'transformer_base_profile' \

# For batch size in terms of tokens
#  --max-tokens 5120 \

# For ckpt
#  --no-save \
#  --save-dir $ckpt_dir \
#  --save-interval 10 \
export PYTHONWARNINGS="ignore"  # To disable Userwarning
CUDA_VISIBLE_DEVICES=$visible_gpus python train.py \
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
  --max-update 104 \
  --batch-size $batch_size \
  --sentence-avg \
  --online-eval \
  --do-sanity-check \
  --seed 1 \
  --dist-url 'tcp://127.0.0.1:20000' \
  --distributed-world-size 1 \
  --distributed-rank 0 
