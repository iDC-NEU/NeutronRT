free-gpus	:=0
root		:=$(HOME)/.dataset
model		:=gcn
dataset		:=ogbn-arxiv
A			?=

small:
	python main.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--num-epochs 1