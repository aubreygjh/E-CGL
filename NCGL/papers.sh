method=$1
gpu=$2


CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
       --method $method \
       --backbone MLP \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch True \
       --n_cls_per_task 10 \
       --repeats 1 \
       --overwrite_result True

# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
#        --method bare \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
#        --method ewc \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 10 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
#        --method mas \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 10 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
#        --method lwf \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 10 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
#        --method gem \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 10 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
#        --method twp \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 10 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
#        --method ergnn \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 10 \
#        --repeats 1 \
#        --overwrite_result True 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Papers100M \
#        --method joint \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 10 \
#        --repeats 1 \
#        --overwrite_result True 