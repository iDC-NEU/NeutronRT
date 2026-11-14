"""Neighborhood selection and intermediate-reuse policies for recomputation."""

import hashlib

import dgl
import torch
from attrs import define

from lib.graph.recompute import compile_block


@define(frozen=True)
class NeighborhoodPolicy:
	reuse_intermediates: bool = True
	fanout: int = -1
	seed: int = 0

	def __attrs_post_init__(self):
		if self.fanout != -1 and self.fanout < 1:
			raise ValueError("Fanout must be -1 (all neighbors) or positive")

	@property
	def approximate(self):
		return self.fanout != -1

	def block(self, graph, targets, layer, version):
		if not self.approximate:
			return compile_block(graph, targets)
		# Sampling reproducibility also requires a fixed DGL/OpenMP thread layout.
		# The workload generator has its own torch.Generator, independent of DGL.
		digest = hashlib.blake2b(digest_size=8)
		digest.update(str((self.seed, layer, version)).encode())
		digest.update(targets.numpy().tobytes())
		seed = int.from_bytes(digest.digest(), "little") % (2**31 - 1)
		sampler = getattr(graph, "sample_block", None)
		if sampler is not None:
			return sampler(targets, fanout=self.fanout, seed=seed)
		if not isinstance(graph, dgl.DGLGraph):
			raise NotImplementedError("Topology backend does not implement neighbor sampling")
		dgl.seed(seed)
		frontier = dgl.sampling.sample_neighbors(graph, targets, self.fanout, edge_dir="in", replace=False)
		return compile_block(frontier, targets)


def task_payload_bytes(task, input_width):
	"""Conservative packed-input estimate, including routes and edge attributes."""
	attributes = sum(
		task.block.edata[name].numel() * task.block.edata[name].element_size() for name in task.block.edata
	)
	return (
		4096
		+ task.sources.numel() * (4 * input_width + 64)
		+ task.targets.numel() * 16
		+ task.block.num_edges() * 16
		+ attributes
	)


def split_task(task, batch_size, *, input_width=0, max_bytes=None):
	"""Bound output rows and payload bytes without resampling selected edges.

	Operate entirely in the compiled block's local ID space. Large global vertex
	IDs must never cause allocation of a graph with that many vertices here.
	"""
	from lib.graph.recompute import LayerPlan, PlanTooLarge

	if batch_size < 1 or input_width < 0 or max_bytes is not None and max_bytes < 1:
		raise ValueError("Task row/byte limits must be positive and width nonnegative")
	if task.targets.numel() <= batch_size and (max_bytes is None or task_payload_bytes(task, input_width) <= max_bytes):
		yield task
		return
	u, v = task.block.edges()
	order = v.argsort(stable=True)
	offsets = torch.cat((torch.zeros(1, dtype=torch.int64), v.bincount(minlength=task.targets.numel()).cumsum(0)))

	def partition(start, stop):
		positions = order[int(offsets[start]) : int(offsets[stop])]
		neighbors = u[positions].unique()
		rows = torch.cat((torch.arange(start, stop), neighbors[(neighbors < start) | (neighbors >= stop)]))
		lookup = torch.empty(task.sources.numel(), dtype=torch.int64)
		lookup[rows] = torch.arange(rows.numel())
		block = dgl.create_block(
			(lookup[u[positions]], v[positions] - start), num_src_nodes=rows.numel(), num_dst_nodes=stop - start
		)
		sources, targets = task.sources[rows], task.targets[start:stop]
		block.srcdata[dgl.NID], block.dstdata[dgl.NID] = sources, targets
		for name in task.block.edata:
			block.edata[name] = task.block.edata[name][positions]
		degrees = task.source_out_degrees[rows] if task.source_out_degrees is not None else None
		return LayerPlan(task.layer, block, sources, targets, degrees)

	def bounded(start, stop):
		block = partition(start, stop)
		if max_bytes is None or task_payload_bytes(block, input_width) <= max_bytes:
			yield block
			return
		if stop - start == 1:
			raise PlanTooLarge("A single destination exceeds the bounded task payload budget")
		del block
		middle = (start + stop) // 2
		yield from bounded(start, middle)
		yield from bounded(middle, stop)

	for start in range(0, task.targets.numel(), batch_size):
		yield from bounded(start, min(start + batch_size, task.targets.numel()))
