dataset=$1
method=$2
backbone=$3
gpu=$4


CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset $dataset \
       --method $method \
       --backbone $backbone \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 1 \
       --overwrite_result False 