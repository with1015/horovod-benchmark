pytorch=`pwd`

#./scpall.sh ./kill_pytorch.sh $pytorch
#./scpall.sh ./multi_node_run.sh $pytorch
#./scpall.sh ./train.py $pytorch
./scpall.sh ./filecopy.sh $pytorch
#./scpall.sh ./scpall.sh $pytorch
#./scpall.sh ./single_node_run.sh $pytorch
#./scpall.sh ./scripts/docker/interactive.sh $pytorch/scripts/docker
#./scpall.sh ./seq2seq/inference/translator.py $pytorch/seq2seq/inference
#./scpall.sh ./seq2seq/inference/beam_search.py $pytorch/seq2seq/inference
#./scpall.sh ./seq2seq/train/trainer.py $pytorch/seq2seq/train
#./scpall.sh ./seq2seq/train/fp_optimizers.py $pytorch/seq2seq/train
#./scpall.sh ./seq2seq/utils.py $pytorch/seq2seq
#./scpall.sh ./seq2seq/data/dataset.py $pytorch/seq2seq/data
#./scpall.sh ./train.py $pytorch
./scpall.sh ./single_gpu_run.sh $pytorch
./scpall.sh ./single_gpu_profiler.sh $pytorch
./scpall.sh ./single_node_run.sh $pytorch
./scpall.sh ./single_node_run_test.sh $pytorch
./scpall.sh ./multi_node_run.sh $pytorch
./scpall.sh ./multi_node_run_test.sh $pytorch

#./scpall_two_cluster.sh ./train.py $pytorch
#./scpall_two_cluster.sh ./seq2seq/inference/beam_search.py $pytorch/seq2seq/inference
#./scpall_two_cluster.sh ./seq2seq/train/trainer.py $pytorch/seq2seq/train
#./scpall_two_cluster.sh ./seq2seq/utils.py $pytorch/seq2seq
#./scpall_two_cluster.sh ./seq2seq/train/fp_optimizers.py $pytorch/seq2seq/train
#./scpall_two_cluster.sh ./kill_python.sh $pytorch
#./scpall_two_cluster.sh ./single_node_run_test.sh $pytorch
#./scpall_two_cluster.sh ./seq2seq/inference/translator.py $pytorch/seq2seq/inference
#./scpall_two_cluster.sh ./filecopy.sh $pytorch

#./scpall_two_cluster.sh ./train.py $pytorch
#./scpall_two_cluster.sh ./seq2seq/train/trainer.py $pytorch/seq2seq/train
#./scpall_two_cluster.sh ./single_gpu_profiler.sh $pytorch
#./scpall_two_cluster.sh ./single_gpu_seq_profiler.sh $pytorch
#./scpall_two_cluster.sh ./single_gpu_run.sh $pytorch
#./scpall_two_cluster.sh ./single_gpu_seq_run.sh $pytorch
#./scpall_two_cluster.sh ./single_node_run_test.sh $pytorch
