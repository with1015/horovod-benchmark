#!/bin/bash

hostname=`hostname`

if [[ $hostname =~ 'shark' ]]; then
  user=plum
fi

if [[ $hostname =~ 'tino' ]]; then
  user=next
fi

kill -9 `ps -ef | grep -v grep | grep "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.total,memory.used,memory.free,utilization.memory --format=csv -i" | grep $user | awk '{print $2}' `
echo "nvidia-smi is killed."
