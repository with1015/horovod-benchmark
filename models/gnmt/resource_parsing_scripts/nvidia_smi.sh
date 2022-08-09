#!/bin/bash

result_path=$1
if [ -z $result_path ]
then
  result_path=`pwd`
fi

#date=`date | tr ':' ' ' | tr -s ' ' | cut -d' ' -f2,3,4,5 | tr ' ' '_'`
date=`date '+%Y-%m-%d-%H-%M-%S'`
#echo "date: "$date

hostname=`hostname`
resource=gpu
result_path=$result_path/$hostname/$resource
if [ ! -d $result_path ]
then
  mkdir -p $result_path
fi

file=${date}_${hostname}_gpu_stat

( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.total,memory.used,memory.free,utilization.memory --format=csv -i 0 -lms 1000 >&2 ) 2> ${result_path}/${file}_device0.csv &
( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.total,memory.used,memory.free,utilization.memory --format=csv -i 1 -lms 1000 >&2 ) 2> ${result_path}/${file}_device1.csv &
( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.total,memory.used,memory.free,utilization.memory --format=csv -i 2 -lms 1000 >&2 ) 2> ${result_path}/${file}_device2.csv &
( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.total,memory.used,memory.free,utilization.memory --format=csv -i 3 -lms 1000 >&2 ) 2> ${result_path}/${file}_device3.csv
