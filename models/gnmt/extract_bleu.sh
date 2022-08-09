#!/bin/bash

log_file=$1

if [ -z $log_file ]; then
  echo "[USAGE] [log file]"
  exit 1
fi

if [ ! -f $log_file ]; then
  echo "No such log file: "$log_file
  exit 1
fi

cat $log_file | grep BLEU | awk -F" " '{print $19}'
