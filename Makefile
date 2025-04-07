free-gpus	:=0
root		:=$(HOME)/.dataset
model		:=gcn
dataset		:=twitter
A			?=

run:
	python main.py $(A) \
	--gpu $(free-gpus) --root $(root) \
	--model $(model) --dataset $(dataset) \
	--num-epochs 1

clean-logs:
	rm logs/NeutronRT.log
	rm logs/NeutronRT.perf.log