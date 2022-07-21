#!/bin/bash

available_disk_size=$((`df . | grep / | awk -F" " '{print $4}'` / 1024 / 1024))
echo "Available Disk size (GB):" $available_disk_size
if [[ $((available_disk_size)) -lt 12 ]]; then
  echo "[ERROR] No space on device for download."
  echo "[ERROR] At least disk of 12 GB is required."
else
  if [[ ! -d checkpoints/ ]]; then
    echo "Make ckpt dir: checkpoints/"
    mkdir checkpoints/
  fi
  cd checkpoints
  echo "current path: "$pwd

  # bert base uncased -> zip:1.5G, unzip: 1.8G
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_base_pretraining_amp_lamb/versions/19.09.0/zip -O bert_pyt_ckpt_base_pretraining_amp_lamb_19.09.0.zip

  unzip bert_pyt_ckpt_base_pretraining_amp_lamb_19.09.0.zip

  # bert large uncased
  #wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/bert_pyt_ckpt_large_pretraining_amp_lamb/versions/19.07.0/zip -O bert_pyt_ckpt_large_pretraining_amp_lamb_19.07.0.zip

  #unzip bert_pyt_ckpt_large_pretraining_amp_lamb_19.07.0.zip


  cd ../data/
  echo "current path: "$pwd
  ./download_squad_vocab.sh
fi
