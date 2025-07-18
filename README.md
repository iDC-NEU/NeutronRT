# NeutronRT

This is the repo of NeutronRT.

## Installation & Setup

**Prerequisites**

- Python 3.8+
- CUDA 12.4 (ensure NVIDIA drivers are compatible)
- Virtual environment recommended (e.g., venv or conda)

**Install Dependencies**

The following commands install PyTorch 2.4.0 with CUDA 12.4 support, along with required libraries:

``` bash
# cuda version needed by PyTorch, PyG and DGL is recommanded to be the same
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install ogb pandas numpy
```

## Usage

Enter python virtual environment with activate script.

Edit configuration in 'config/default.toml' for different type of task. Run:

``` bash
python main.py
```

All inference methods will run automatically based on the configuration.
