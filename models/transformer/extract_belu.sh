#!/bin/bash 

file=$1
bleu_only=$2
if [ $# -ne 2 ]; then
  echo "[USAGE] [log file] [1 - BLEU only | 0 - all log]"
  exit 1
fi 

#cat $file | grep BLEU4 | awk -F" " '{print$8}' | sed -e 's/,//'
if [ $bleu_only -eq 1 ]; then
  cat $file | grep 'Test BLEU' | uniq | awk -F" " '{print$6}'
else
  cat $file | grep 'Test BLEU' | uniq
fi
