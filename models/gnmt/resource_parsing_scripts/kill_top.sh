
result_path=$1
master=$2

if [ -z $result_path ]
then
  result_path=`pwd`
fi
if [ ! -d $result_path ]
then
  mkdir -p $result_path
fi

hostname=`hostname`
resource=cpu_mem
root_result_dir=$result_path/$hostname
result_path=$result_path/$hostname/$resource

kill -9 `ps -ef | grep -v grep | grep "top -d 1 -bic" | grep "plum" | awk '{print $2}' `
echo "top is killed."
sleep 5
scp -r $result_path $master:$root_result_dir
echo "Collect "$resource" result dir to master"
