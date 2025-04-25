#!/bin/bash

date=$(date +"%Y-%m-%d")


run_fixed_or_batch() {
    
    dataset=$1
    batch_size=$2
    model=$3
    
    echo "Dataset: $dataset, Batch Size: $batch_size, Model: $model"
    echo ""

    echo '+ basic'
    # python3 layer-wise-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
    #  --emb-host-size '0,0' --emb-cache-size '0,0'
    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
     --emb-host-size '0,0' --emb-cache-size '0,0' \
     --aggr-host-size '0,0' --aggr-cache-size '0,0'

    echo '+ cache'
    # python3 layer-wise-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
    #  --emb-host-size '0,12' --emb-cache-size '1,1'
    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
     --emb-host-size '0,12' --emb-cache-size '1,1' \
     --aggr-host-size '0,0' --aggr-cache-size '0,0'

    echo '+ more cache'
    # python3 layer-wise-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
    #  --emb-host-size '0,32' --emb-cache-size '2,2'
    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
     --emb-host-size '0,32' --emb-cache-size '2,2' \
     --aggr-host-size '0,0' --aggr-cache-size '0,0'

    echo '+ partly'
    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
     --emb-host-size '0,12' --emb-cache-size '1,1' \
     --aggr-host-size '10,10' --aggr-cache-size '1,1'

    echo ""
}


run_ratio() {
    dataset=$1
    host_ratio=$2
    model=$3
    batch_size=1000

    more_ratio=$(echo "$host_ratio" | awk -F',' '{print $1*2","$2*2}')
    
    echo "Dataset: $dataset, Batch Size: $batch_size, Host ratio: $host_ratio", model: $model
    echo ""

    echo '+ cache'
    # python3 layer-wise-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
    #  --emb-host-ratio $host_ratio --emb-cache-ratio '0.01,0.01' --use-ratio
    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
     --emb-host-ratio $host_ratio --emb-cache-ratio '0.01,0.01' \
     --aggr-host-ratio '0,0' --aggr-cache-ratio '0,0' --use-ratio

    echo '+ more cache'
    # python3 layer-wise-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
    #  --emb-host-ratio $more_ratio --emb-cache-ratio '0.02,0.02' --use-ratio
    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
     --emb-host-ratio $more_ratio --emb-cache-ratio '0.02,0.02' \
     --aggr-host-ratio '0,0' --aggr-cache-ratio '0,0' --use-ratio

    echo '+ partly'
    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size --num-epochs 10 --model $model \
     --emb-host-ratio $host_ratio --emb-cache-ratio '0.01,0.01' \
     --aggr-host-ratio $host_ratio --aggr-cache-ratio '0.01,0.01' --use-ratio

    echo ""
}


# ogbn-papers100M
# reddit ogbn-products twitter 
# ogbn-arxiv ogbl-collab reddit-body reddit-title GDELT
for dataset in ogbn-arxiv ogbl-collab reddit-body reddit-title GDELT
do
for model in gcn gin sage
do
    for emb in 0,0 0.01,0.01 0.1,0.1 0.2,0.2 0.5,0.5 1,1
    do

    run_ratio $dataset $emb $model

    done
done
done

# 5 datasets: large * 2 + steaming (real)
# fixed size: 4G (GPU for all caches) 32G (Host for all caches except x0 and x_n) 
# basic / +cache / +partly compute
# large => adjust ratio 10% ... 

# ogbn-papers100M
# reddit ogbn-products twitter 
# ogbn-arxiv ogbl-collab reddit-body reddit-title GDELT
# batch_size=1000
# for dataset in twitter ogbn-arxiv ogbl-collab reddit-body reddit-title GDELT
# do
# for model in gcn gin sage
# do 
# # for batch_size in 100 10000
# # do

#     run_fixed_or_batch $dataset $batch_size $model

# done
# done
# done
