method=$1
gpu=$2


CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
       --method $method \
       --backbone MLP \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 2 \
       --repeats 1 \
       --overwrite_result True

# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
#        --method bare \
#        --backbone GAT \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
#        --method ewc \
#        --backbone GAT \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
#        --method mas \
#        --backbone GAT \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
#        --method lwf \
#        --backbone GAT \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
#        --method gem \
#        --backbone GAT \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
#        --method twp \
#        --backbone GAT \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
#        --method ergnn \
#        --backbone GAT \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Amazon \
#        --method joint \
#        --backbone GAT \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result True 