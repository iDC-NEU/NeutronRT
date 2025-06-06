# NeutronRT

This is the repo of NeutronRT.

## Prepare

It is advised to use `python -m venv <your-virtual-environment>` to create a virtual environment.

``` bash
# cuda version needed by PyTorch, PyG and DGL is recommanded to be the same
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install ogb pandas
```

## Usage

Enter python virtual environment with activate script.

Use `make sample` for RTEC-FULL, `make sample55` for RTEC-NS, `make eco` for RTEC-UER and `make part` for RTEC-INC.

For large datasets, use `make big-eco` for RTEC-UER and `make big-part` for RTEC-INC.

Parameters in the Makefile which is located in the main folder could be edited to adjust the system to run the script.

You could use `python part-infer.py --help` to get the usage of the parameters.
