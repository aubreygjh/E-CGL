method=$1
gpu=$2


# main exp
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 2 \
#        --repeats 1 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [5000]; 'random_sample': False"

# ablation:-mlp,+gcn
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method my \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [5000]; 'random_sample': False"

# ablation:random sampling
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
       --method my \
       --backbone MLP \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --epochs 10 \
       --n_cls_per_task 2 \
       --repeats 5 \
       --overwrite_result False \
       --my_args " 'diversity_ratio': [0]; 'sample_budget': [5000]; 'random_sample': True"

# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method bare \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method lwf \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method ewc \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method mas \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Products-CL \
#        --method gem \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 2 \
#        --repeats 5 \
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
#        --overwrite_result False \
#        --ergnn_args " 'budget': [2000]; 'd': [0.5]; 'sampler': ['CM']" 
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