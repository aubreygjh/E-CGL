dataset=$1
method=$2
gpu=$3
minibatch=$4

python train.py --dataset $dataset \
       --method $method \
       --backbone GCN \
       --gpu $gpu \
       --ILmode taskIL \
       --inter-task-edges False \
       --minibatch $minibatch \

# python train.py --dataset corafull \
#        --method $method \
#        --backbone GCN \
#        --gpu $gpu \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch $minibatch 

# python train.py --dataset Arxiv-CL \
#        --method $method \
#        --backbone GCN \
#        --gpu $gpu \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch $minibatch \

# python train.py --dataset Reddit-CL \
#        --method $method \
#        --backbone GCN \
#        --gpu $gpu \
#        --ILmode taskIL \
#        --inter-task-edges False \
#        --minibatch $minibatch \