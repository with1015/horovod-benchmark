#!/bin/bash

file_name=$1
dest=$2

if [ -z $dest ]
then
  echo "[Error] USAGE: scpall.sh [file or dir] [dest path]"
  exit
fi

hostname=`hostname`

port=51234
if [[ $hostname =~ 'shark' ]]; then
  for i in 1 3 4 5 6 7 8; do
    scp -P $port -r $file_name shark${i}:$dest &
    sleep 0.0001
    echo $file_name "shark"${i}" done"
  done

elif [[ $hostname =~ 'tino' ]]; then
  for i in 1 2 3 4 5 6 7 8; do
    scp -r $file_name tino1${i}0:$dest &
    sleep 0.001
    echo $file_name "tino1"${i}"0 done"
  done
fi

echo "successfully copyed!"
