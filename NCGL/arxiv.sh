method=$1
gpu=$2


CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
       --method my \
       --backbone MLP \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 4 \
       --repeats 1 \
       --overwrite_result True \
       --my_args " 'random_ratio': [0.25]; 'sample_budget': [2000]; 'con_weight': [0]" 

# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method bare \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method lwf \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method ewc \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method mas \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method gem \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method twp \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method ergnn \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method joint \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --epochs 1000 \
#        --lr 0.001 \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False