method=$1
gpu=$2

CUDA_VISIBLE_DEVICES=$gpu python train.py --dataset SIDER-tIL \
       --method $method \
       --backbone GCN \
       --gpu 0 \
       --clsIL False