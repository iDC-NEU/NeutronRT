FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
	libibverbs-dev rdma-core infiniband-diags \
	&& rm -rf /var/lib/apt/lists/*

# The base image supplies Python and the matching CUDA PyTorch build.
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

COPY requirements.txt /opt/neutronrt/
RUN python -m pip install -r /opt/neutronrt/requirements.txt

COPY . /opt/neutronrt
WORKDIR /opt/neutronrt
RUN python -m pip install --no-deps .

ENTRYPOINT ["python", "-m", "scripts.simple"]
