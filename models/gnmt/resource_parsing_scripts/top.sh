#!/bin/bash

result_path=$1
if [ -z $result_path ]
then
  result_path=`pwd`
fi

date=`date | tr ':' ' ' | tr -s ' ' | cut -d' ' -f2,3,4,5 | tr ' ' '_'`
resource=cpu_mem
hostname=`hostname`
result_path=$result_path/$hostname/$resource
if [ ! -d $result_path ]
then
  mkdir -p $result_path
fi

echo "date: "$date
file=${date}_${hostname}_top.csv
echo "top file: "$file

top -d 1 -bic > ${result_path}/${file}
