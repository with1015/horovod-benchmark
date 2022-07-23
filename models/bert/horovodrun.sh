#!/usr/bin/env bash

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ret=`source ~/pytorch-benchmarks/scripts/get_cluster_info.sh`
ret_arr=($ret)
storage=${ret_arr[0]}
master=${ret_arr[1]}
echo 'Storage:' $storage

rank=0
world_size=1
master=`hostname`
visible_gpus='0,1,2,3'

batch_size=$1
echo "batch size: "$batch_size

DEBUG=True #False #True
SAVE_CKPT=False
FINAL_EVAL=False

BERT_PATH=/home/plum/graduate_gazuaaa/hipress/src/CaSync/horovod-torch/bert
BERT_PREP_WORKING_DIR=/ssd_dataset2/dataset/pytorch_bert
echo "BERT PATH: "$BERT_PATH
echo "BERT PREPROCESSING PATH: "$BERT_PREP_WORKING_DIR
model=bert-base-uncased

init_checkpoint=${3:-"$BERT_PREP_WORKING_DIR/checkpoints/bert_base.pt"}  # pretrained model
epochs=${4:-"2.0"}
batch_size=$batch_size #${3:-"20"}  # batch size per GPU
learning_rate=${5:-"3e-5"}
precision=${6:-""}  # No fp16 precision
num_gpu=${7:-"1"}  # DEPRECATED
seed=${8:-"1"}
squad_dir=${9:-"$BERT_PREP_WORKING_DIR/squad/v2.0"}
# L-12, H-768: BERT-base
vocab_file=${10:-"$BERT_PREP_WORKING_DIR/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt"}
OUT_DIR=${11:-"$BERT_PATH/results/SQuAD"}
mode=${12:-"train"} # train eval
CONFIG_FILE=${13:-"$BERT_PATH/bert_config.json"}
max_steps=${14:-"-1"}

LOGFILE=$OUT_DIR/logfile.txt

echo "output dir is $OUT_DIR"
mkdir -p $OUT_DIR  # To use tee at the end of script
if [ ! -d "$OUT_DIR" ]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [ "$precision" = "fp16" ] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

##### Execution command #####
export CUDA_VISIBLE_DEVICES=$visible_gpus
CMD="python run_squad.py "
if [[ $DEBUG == "True" ]]; then
  echo "RUN DEBUG MODE"
  CMD+="--debug "
fi

if [[ $SAVE_CKPT == "True" ]]; then
  CMD+="--save-ckpt "
fi

if [[ $FINAL_EVAL == "True" ]]; then
  CMD+="--only-final-eval "
fi

if [[ $PROFILE_DIR != "" ]]; then
  echo "RUN PROFILER MODE"
  CMD+="--profile-dir $PROFILE_DIR "
fi

#CMD+="--dist-url=tcp://${master}:8000 "
#CMD+="--rank $rank "
#CMD+="--world-size $world_size "

CMD+="--init_checkpoint=$init_checkpoint "
if [ "$mode" = "train" ] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v2.0.json "
  CMD+="--train_batch_size=$batch_size "
elif [ "$mode" = "eval" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v2.0.json "
  CMD+="--predict_batch_size=$batch_size "
elif [ "$mode" = "prediction" ] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v2.0.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v2.0.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v2.0.json "
  CMD+="--predict_batch_size=$batch_size "
fi
CMD+=" --version_2_with_negative"  # For SQUAD v2.0
CMD+=" --do_lower_case "
# CMD+=" --old "
# CMD+=" --loss_scale=128 "
CMD+=" --bert_model=$model "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+=" --check-param-sparsity "

CMD+=" $use_fp16"

echo "$CMD"

CUDA_LAUNCH_BLOCKING=0 NCCL_DEBUG=INFO NCCL_TREE_THRESHOLD=0 NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 \
horovodrun --verbose -np 1 -H shark7:1 -p 51234 $CMD
wait
