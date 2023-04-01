method=$1
gpu=$2


CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
       --method $method \
       --backbone MLP \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 5 \
       --repeats 1 \
       --overwrite_result True

# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method bare \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method ewc \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method mas \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method lwf \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method gem \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method twp \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method ergnn \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method joint \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 10 \
#        --overwrite_result False 