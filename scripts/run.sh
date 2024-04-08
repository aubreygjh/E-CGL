#!/bin/bash

dataset=$1
method=$2
backbone=$3
gpu=$4

if [[ $dataset == 'cora' ]]; then
    CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset cora \
           --method $method \
           --backbone $backbone \
           --gpu 0 \
           --ILmode taskIL \
           --inter-task-edges False \
           --minibatch False \
           --n_cls_per_task 5 \
           --repeats 1 \
           --overwrite_result False 
elif [[ $dataset == 'arxiv' ]]; then
    CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset arxiv \
           --method $method \
           --backbone $backbone \
           --gpu 0 \
           --ILmode taskIL \
           --inter-task-edges False \
           --minibatch False \
           --n_cls_per_task 4 \
           --repeats 1 \
           --overwrite_result False 
elif [[ $dataset == 'reddit' ]]; then
    CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset reddit \
           --method $method \
           --backbone $backbone \
           --gpu 0 \
           --ILmode taskIL \
           --inter-task-edges False \
           --minibatch False \
           --n_cls_per_task 4 \
           --repeats 1 \
           --overwrite_result False 
elif [[ $dataset == 'products' ]]; then
    CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset products \
           --method $method \
           --backbone $backbone \
           --gpu 0 \
           --ILmode taskIL \
           --inter-task-edges False \
           --minibatch False \
           --n_cls_per_task 2 \
           --repeats 1 \
           --overwrite_result False          
fi
