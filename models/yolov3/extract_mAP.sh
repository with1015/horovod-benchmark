#!/bin/bash

file=$1
if [ $# -ne 1 ]; then
  echo "[USAGE] [log file]"
  exit 1
fi 

#cat $file | grep all | uniq | awk -F" " '{print $6}'
cat $file | grep all | awk -F" " '{print $6}'
