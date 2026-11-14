"""Conventional DGL full-neighbor and neighbor-sampled inference baselines.

Use DGL's multilayer sampler directly so the dependency blocks come from one
sampling pass. No incremental engine or project-specific sampler is involved.
"""

import logging
from time import perf_counter
from typing import Iterator

import dgl
import torch
from torch import Tensor

from lib.models import GNNModel


def infer_batches(
	graph,
	features: Tensor,
	model: GNNModel,
	nodes: Tensor,
	*,
	fanout: int | None,
	batch_size: int,
	sampler=None,
) -> Iterator[tuple[Tensor, Tensor]]:
	"""Yield output IDs and embeddings on the model device, without a dense cache.

	``None`` selects full neighborhoods; a positive fanout applies at every layer.
	Topology and features remain on CPU. DGL samples dependency blocks there, then
	the model device receives the selected features and blocks. The caller owns
	output placement and must include its synchronization/storage costs in timings.
	A supplied native sampler is retained across batches and graph edge updates.
	"""
	if fanout is not None and (type(fanout) is not int or fanout < 1):
		raise ValueError("Neighbor-sampling fanout must be positive")
	if batch_size < 1 or model.training:
		raise ValueError("Baseline inference requires a positive batch size and model.eval()")
	if graph.device.type != "cpu" or features.device.type != "cpu" or nodes.device.type != "cpu":
		raise ValueError("Baseline topology, features and output node IDs must be on CPU")
	if features.ndim != 2 or features.shape[0] != graph.num_nodes() or nodes.ndim != 1 or nodes.dtype != torch.int64:
		raise ValueError("Features and output IDs must match the graph")
	device = next(model.parameters()).device
	fanouts = [fanout if fanout is not None else -1] * model.num_layers
	if sampler is None:
		sampler = dgl.dataloading.NeighborSampler(fanouts)
	elif sampler.fanouts != fanouts:
		raise ValueError("Retained sampler fanouts differ from the requested baseline")
	loader = dgl.dataloading.DataLoader(
		graph, nodes, sampler, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0
	)
	for input_nodes, output_nodes, blocks in loader:
		with torch.inference_mode():
			h = features[input_nodes].to(device)
			for layer, block in enumerate(blocks):
				h = model(layer, block.to(device), h)
		yield output_nodes, h


class StreamingRecompute:
	"""Basic affected-output recomputation using native DGL topology operations."""

	def __init__(self, graph, features, model, *, fanout: int | None, batch_size: int, output_device):
		if model.model_name != "sage":
			raise ValueError("The streaming DGL reference currently supports mean GraphSAGE")
		self.graph, self.features, self.model = graph, features, model
		self.fanout, self.batch_size = fanout, batch_size
		self.sampler = dgl.dataloading.NeighborSampler([fanout if fanout is not None else -1] * model.num_layers)
		self.outputs = torch.empty(graph.num_nodes(), model.num_classes, device=output_device)
		self.ready = torch.zeros(graph.num_nodes(), dtype=torch.bool)

	def synchronize(self):
		device = next(self.model.parameters()).device
		if device.type == "cuda":
			torch.cuda.synchronize(device)

	def compute(self, nodes, *, progress=False):
		completed = 0
		for output_nodes, values in infer_batches(
			self.graph,
			self.features,
			self.model,
			nodes,
			fanout=self.fanout,
			batch_size=self.batch_size,
			sampler=self.sampler,
		):
			self.outputs[output_nodes.to(self.outputs.device)] = values.to(self.outputs.device)
			self.ready[output_nodes] = True
			completed += output_nodes.numel()
			if progress and (
				completed == output_nodes.numel()
				or completed // self.batch_size % 128 == 0
				or completed == nodes.numel()
			):
				logging.info("DGL initialization: rows=%d/%d", completed, nodes.numel())

	def query(self, nodes):
		missing = nodes[~self.ready[nodes]].unique()
		if missing.numel():
			self.compute(missing)
		return self.outputs[nodes.to(self.outputs.device)].cpu()

	def update(self, request):
		self.synchronize()
		started = perf_counter()
		first, second = request.first, request.second
		if request.kind == "features":
			self.features[first] = second
			targets, hops = first.unique(), self.model.num_layers
		elif request.kind == "insert":
			self.graph.add_edges(first, second)
			targets, hops = second.unique(), self.model.num_layers - 1
		elif request.kind == "delete":
			# Match the service contract: delete all parallel edges for each pair.
			eids = self.graph.edge_ids(first, second, return_uv=True)[2].unique()
			self.graph.remove_edges(eids)
			targets, hops = second.unique(), self.model.num_layers - 1
		else:
			raise ValueError("Unsupported recorded update kind")
		mutated = perf_counter()
		for _ in range(hops):
			targets = torch.cat((targets, self.graph.out_edges(targets)[1])).unique()
		analyzed = perf_counter()
		self.compute(targets)
		self.synchronize()
		finished = perf_counter()
		return {
			"kind": request.kind,
			"changes": first.numel(),
			"affected_outputs": targets.numel(),
			"topology_seconds": mutated - started,
			"analysis_seconds": analyzed - mutated,
			"compute_and_store_seconds": finished - analyzed,
			"response_seconds": finished - started,
		}
