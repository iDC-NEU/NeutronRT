# GNNInfer

## Installation and Running


### Debug:

Debug code of cuda:

```sh
cuda-gdb

break /project_path/lib/cpp_extension/pma_graph.cu:149
file /home_path/anaconda3/envs/gnn/bin/python
r
exec(open('/project_path/test-infer-layer.py').read())
```



### Installation

1. Clone our library

    ```shell
    git clone 
    ```

2. Create enviroment

    cuda = 11.8

    ```sh
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    
    conda install pyg -c pyg
    pip install pyg-lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0%2Bcu118.html


    pip install ogb
    conda install pandas scipy
    ```


3. Install Ninja
    
    Ninja is used for JIT. 

    ```shell
    wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
    unzip ninja-linux.zip -d /usr/local/bin/
    update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
    ```

### Running

1. Prepare dataset
    
    ```shell
    unzip dataset.zip
    mkdir data 
    mv dataset data/
    ```

2. Run

    ```shell
    # --emb-host-size x,y: Cache size on *CPU* memory for *input* embeddings of layer 1 and layer 2 is x GB and y GB, respectively.
    # --emb-cache-size x,y: Cache size on *GPU* memory for *input* embeddings of layer 1 and layer 2 is x GB and y GB, respectively.
    # --aggr-host-size x,y: Cache size on *CPU* memory for *aggregated* embeddings of layer 1 and layer 2 is x GB and y GB, respectively.
    # --aggr-cache-size x,y: Cache size on *GPU* memory for *aggregated* embeddings of layer 1 and layer 2 is x GB and y GB, respectively.

    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size \
     --num-epochs 10 --model $model \
     --emb-host-size '0,12' --emb-cache-size '1,1' \
     --aggr-host-size '10,10' --aggr-cache-size '1,1'


    # --aa-bb-ratio x,y: Cache only store x*100% and y*100% of vertice's embeddings for layer 1 and layer 2, respectively. 

    python3 layer-part-infer.py --dataset $dataset --batch-size $batch_size \
     --num-epochs 10 --model $model \
     --emb-host-ratio '0.1,0.1' --emb-cache-ratio '0.01,0.01' \
     --aggr-host-ratio '0.1,0.1' --aggr-cache-ratio '0.01,0.01' --use-ratio
    ```

3. Process results

    ```sh
    python3 get-data.py $filename
    ```
