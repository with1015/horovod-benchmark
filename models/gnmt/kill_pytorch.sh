#!/bin/bash

cur_path=`pwd`
param_num=$#
param_host_range=$@

###################### CONFIG #####################
# host name e.g, "shark", "tino", "gold"
#HOST="tino"
HOST="tino"
# host range e.g, (1 2 3)
#RANGE=(110 120 130 140)
RANGE=(120 130)
###################################################
if [ "$HOST" == "" ] ||
   [ "$RANGE" == "" ]; then
    echo "[Important] You must set up config [HOST], [RANGE] before executation."
    exit 1
fi

if [ "$1" = "true" ] ; then
# echo -9 `ps -ef | grep -v grep | grep -E "tf_cnn_benchmarks.py|nvidia-smi|nvprof|dstat" | awk '{print $2}' `
# kill -9 `ps -ef | grep -v grep | grep -E "tf_cnn_benchmarks.py|nvidia-smi|nvprof|dstat" | awk '{print $2}' `
  sudo kill -9 `ps -ef | grep -v grep | grep -E "python" | awk '{print $2}' `
  echo "killed $(uname -n)"
  exit 0;
fi

function kill_pytorch() {
  hostname=`hostname`
  local target_host=$1
  shift
  local all_host_range=($@)

  if [[ $hostname =~ $target_host ]]; then
    if [ $param_num -gt 0 ]
    then
      for i in ${param_host_range}; do
        ssh ${target_host}${i} 'bash -s' < $cur_path/kill_pytorch.sh true &
      done
    else
      for i in ${all_host_range[@]}; do
        ssh ${target_host}${i} 'bash -s' < $cur_path/kill_pytorch.sh true &
      done
    fi
  else
    echo "[MESSAGE] the host ("${hostname}") doesn't have "$target_host
  fi
}

############# main #############
kill_pytorch $HOST ${RANGE[@]}
