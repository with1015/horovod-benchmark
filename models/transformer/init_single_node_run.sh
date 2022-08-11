#!/bin/bash 

ret=`source ~/pytorch-benchmarks/scripts/get_cluster_info.sh`
ret_arr=($ret)
storage=${ret_arr[0]}
master=${ret_arr[1]}
echo 'Storage:' $storage
dataset_dir=${storage}'/pytorch_wmt14_en_de'

visible_gpus='0,1,2,3'
batch_size=64
save_dir=transformer_64_4
batch_size=128
save_dir=transformer_base_128_4
master=`hostname`

#  --arch transformer_wmt_en_de_big_t2t \
#  --amp \
#  --amp-level O2 \
#  --fuse-layer-norm \ # --> for amp
#  --optimizer adam \
#  --adam-betas '(0.9, 0.997)' \
#  --adam-eps "1e-9" \

# For test BLEU
#  --online-eval \
# For ckpt
#  --save-dir $save_dir \
# For profile
#  --profile-dir 'transformer_base_profile' \
#  --max-tokens 5120 \
python -m torch.distributed.launch --nproc_per_node 4 train.py \
  $dataset_dir \
  --arch transformer_wmt_en_de \
  --share-all-embeddings \
  --optimizer adam \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr 0.0 \
  --warmup-updates 4000 \
  --lr 0.0006 \
  --min-lr 0.0 \
  --dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --max-update 100 \
  --batch-size $batch_size \
  --seed 1 \
  --no-save \
  --distributed-init-method env://
  #2>&1 | tee -a -i $log_file

  #--distributed-rank 0

