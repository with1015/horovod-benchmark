#!/bin/bash

file_name=$1
dest=$2

if [ -z $dest ]
then
  echo "[USAGE]: scpall_two_cluster.sh [file or dir] [dest path]"
  exit
fi

port=51234

hostname=`hostname`
if [[ $hostname =~ 'tino' ]]; then
  for i in 1 2 3 4 5 6 7 8; do
    scp -P 22 -r $file_name tino1${i}0:$dest &
    sleep 0.001
    echo $file_name "tino1"${i}"0 done"
  done
  dest=${dest/next/plum}
  for i in 1 3 4 5 6 7 8 ; do
    scp -P $port -r $file_name plum@shark0${i}:$dest &
    sleep 0.001
    echo $file_name "shark"${i}" done"
  done
elif [[ $hostname =~ 'shark' ]]; then
  for i in 1 3 4 5 6 7 8 ; do
    scp -P $port -r $file_name shark${i}:$dest &
    sleep 0.001
    echo $file_name "shark"${i}" done"
  done
  dest=${dest/plum/next}
  for i in 1 2 3 4 5 6 7 8; do
    scp -P 22 -r $file_name next@tino1${i}0:$dest &
    sleep 0.001
    echo $file_name "tino1"${i}"0 done"
  done
fi
