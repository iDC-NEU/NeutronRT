free-gpus	:=2
root		:=/home/hdd/lhr/dataset # ./dataset # 
workspace	:=/home/lhr/gnn-inference
model		:=gcn
dataset		:=ogbn-products
A			?=
T			?=

.PHONY: clean-cache
.PHONY: part big sample test

# pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# pip install torch_geometric
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
# pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
# pip install ogb pandas

# sample

sample:
	python sample-infer-lite.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset)

sample55:
	python sample-infer-lite.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--sample 5,5

count-sample:
	python sample-infer-count.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset)

count-sample55:
	python sample-infer-count.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--sample 5,5

# embedding cache only
eco:
	python eco-infer-lite.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--embd-host-ratio 0,0 \
	--embd-cuda-ratio 1,1

count-eco:
	python eco-infer-count.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--embd-host-ratio 0,0 \
	--embd-cuda-ratio 1,1

big-eco:
	python eco-infer-lite.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--embd-host-ratio 1,1 \
	--embd-cuda-ratio 0,0

count-big-eco:
	python eco-infer-count.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--embd-host-ratio 1,1 \
	--embd-cuda-ratio 0,0

part:
	python part-infer.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--embd-cuda-ratio 1,1 \
	--aggr-cuda-ratio 1,1

big-part:
	python part-infer.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--embd-cuda-ratio 0,0 \
	--aggr-cuda-ratio 0,0

	
# test:
# 	PYTHONPATH=$(workspace) python $(T)

clean-cache:
	rm -rf $(strip $(root))/.gnn-stream