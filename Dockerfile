FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
	libibverbs-dev rdma-core infiniband-diags \
	&& rm -rf /var/lib/apt/lists/*

# The base image supplies Python and the matching CUDA PyTorch build.
RUN python -m pip install --upgrade pip setuptools wheel
RUN conda install --yes -c dglteam/label/th24_cu124 dgl=2.4.0.th24.cu124 \
	&& conda clean --all --yes

COPY requirements.txt /opt/neutronrt/
RUN python -m pip install -r /opt/neutronrt/requirements.txt

COPY . /opt/neutronrt
WORKDIR /opt/neutronrt
RUN python -m pip install --no-deps .

ENTRYPOINT ["python", "-m", "scripts.simple"]
