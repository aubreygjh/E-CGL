dataset=CoraFull-CL
method=$1
backbone=GCN
gpu=$2


CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset $dataset \
       --method $method \
       --backbone $backbone \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 1 \
       --overwrite_result False \
       --my_args " 'diversity_ratio': [0.1]; 'sample_budget': [1000]; 'random_sample': False"