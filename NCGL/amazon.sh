method=$1
gpu=$2


CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method $method \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 1 \
       --overwrite_result True

CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method bare \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 10 \
       --overwrite_result False 
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method ewc \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 10 \
       --overwrite_result False 
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method mas \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 10 \
       --overwrite_result False 
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method lwf \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 10 \
       --overwrite_result False 
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method gem \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 10 \
       --overwrite_result False 
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method twp \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 10 \
       --overwrite_result False 
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method ergnn \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 10 \
       --overwrite_result False 
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method joint \
       --backbone GCN \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 10 \
       --overwrite_result False 