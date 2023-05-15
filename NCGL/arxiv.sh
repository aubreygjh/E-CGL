method=$1
gpu=$2


# main exp
CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
       --method my \
       --backbone MLP \
       --gpu 0 \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch False \
       --n_cls_per_task 4 \
       --repeats 5 \
       --overwrite_result True \
       --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [3000]; 'random_sample': False" 

# ablation:-mlp,+gcn
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method my \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [3000]; 'random_sample': False"  

# ablation:random sampling
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Arxiv-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --weight-decay 5e-5 \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0]; 'sample_budget': [3000]; 'random_sample': True" 

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
#        --overwrite_result False \
#        --ergnn_args " 'budget': [3000]; 'd': [0.5]; 'sampler': ['MF']" 
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