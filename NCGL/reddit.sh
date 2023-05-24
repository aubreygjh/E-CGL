method=$1
gpu=$2


# main exp
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 1 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [5000]; 'random_sample': False" 

# ablation:-mlp,+gcn
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method my \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [5000]; 'random_sample': False" 

# ablation:random sampling
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --epochs 4 \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0]; 'sample_budget': [5000]; 'random_sample': True" 

# parameter sensitivity: diversity ratio
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]; 'sample_budget': [5000]; 'random_sample': False"

# parameter sensitivity: sample budget
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [100,500,1000,2000,3000,5000]; 'random_sample': False"

# parameter sensitivity: loss weight
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [5000]; 'random_sample': False; 'lambda_replay':[0.0,0.1,0.5,1.0,5.0,10.0]" 

# classIL
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method my \
#        --backbone MLP \
#        --gpu 0 \
#        --ILmode classIL \
#        --inter-task-edges False \
#        --minibatch False \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --my_args " 'diversity_ratio': [0.25]; 'sample_budget': [5000]; 'random_sample': False" 


####################################################################################################################################


# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method bare \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method lwf \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method ewc \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method mas \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method gem \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method twp \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method ergnn \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False \
#        --ergnn_args " 'budget': [5000]; 'd': [0.5]; 'sampler': ['MF']" 
# CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset Reddit-CL \
#        --method joint \
#        --backbone GCN \
#        --gpu 0 \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch True \
#        --n_cls_per_task 4 \
#        --repeats 5 \
#        --overwrite_result False 