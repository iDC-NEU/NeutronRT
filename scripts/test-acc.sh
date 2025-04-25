#!/bin/bash


# gpu=1

# for dataset in ogbn-arxiv GDELT
# do 

# echo 'part node'
# python3 layer-part-infer.py --dataset $dataset --num-epochs 10 --task node --gpu $gpu --use-ratio

# echo 'all node'
# python3 layer-part-once.py --dataset $dataset --num-epochs 10 --task node --gpu $gpu

# done


# for dataset in ogbl-collab GDELT reddit-body
# do 

# echo 'part link'
# python3 layer-part-infer.py --dataset $dataset --num-epochs 10 --task link_pred --gpu $gpu --use-ratio

# echo 'all link'
# python3 layer-part-once.py --dataset $dataset --num-epochs 10 --task link_pred --gpu $gpu

# done


for dataset in ogbn-arxiv ogbl-collab reddit-body GDELT
do 

echo $dataset

python3 lib/dataset.py $dataset

echo ''

done
