method=$1
gpu=$2


#0,0.25,0.5,0.75,1
# main exp
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.1]; 'sample_budget': [1000]; 'random_sample': False" 

# ablation:-mlp,+gcn
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method my \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.1]; 'sample_budget': [1000]; 'random_sample': False"  

# ablation:random sampling
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
       --method my \
       --backbone MLP \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --epochs 9 \
       --n_cls_per_task 5 \
       --repeats 5 \
       --overwrite_result False \
       --my_args " 'diversity_ratio': [0]; 'sample_budget': [1000]; 'random_sample': True"  

# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method bare \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method lwf \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method ewc \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method mas \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method gem \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method twp \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method ergnn \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False \
#        --ergnn_args " 'budget': [500]; 'd': [0.5]; 'sampler': ['MF']"
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset CoraFull-CL \
#        --method joint \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 5 \
#        --repeats 5 \
#        --overwrite_result False 