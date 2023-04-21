method=$1
gpu=$2


CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
       --method my \
       --backbone MLP \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch True \
       --n_cls_per_task 2 \
       --repeats 5 \
       --overwrite_result True \
       --my_args " 'random_ratio': [0,0.25,0.5,0.75,1]; 'sample_budget': [1000,2000,5000]; 'con_weight': [1]" 

# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method bare \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method ewc \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method mas \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method lwf \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method gem \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 10 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method twp \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method ergnn \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
#        --overwrite_result False
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method joint \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
#        --overwrite_result False