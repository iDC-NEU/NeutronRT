"""Producer-local projected sums and outputs with paired completion metadata."""

import logging
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable, Optional, Union

import dgl
import torch
from torch import Tensor

from lib.graph.recompute import LayerPlan
from lib.runtime.aggregates.delta import DeltaHeader
from lib.runtime.aggregates.protocol import aggregate_identity
from lib.runtime.aggregates.recipe import RecipeHeader
from lib.runtime.aggregates.reconstruction import validate_recipe
from lib.runtime.aggregates.tables import receive_tables, resolve_tables
from lib.runtime.protocol import COMPUTE_WITH_AGGREGATE, ERROR, READY, ComputeHeader

if TYPE_CHECKING:
	from lib.runtime.worker import OwnedWorker


@dataclass(frozen=True)
class _Result:
	embedding: Union[ComputeHeader, RecipeHeader, DeltaHeader]
	aggregate: Union[ComputeHeader, RecipeHeader, DeltaHeader]
	version: int
	embedding_chunk: int
	aggregate_chunk: int
	embedding_handle: int
	aggregate_handle: int
	width: int


class AggregateTasks:
	def __init__(self, worker: "OwnedWorker") -> None:
		self.worker = worker
		self._results: dict[int, _Result] = {}
		self._handles: dict[int, tuple[int, int]] = {}

	def owns_request(self, request: int) -> bool:
		return request in self._results

	def retired(self, handle: int) -> None:
		requests = self._handles.pop(handle, ())
		for request in requests:
			self._results.pop(request, None)

	def _send(self, result: _Result, *, error: int = 0) -> None:
		worker = self.worker
		opcode = ERROR if error else READY
		worker.comm.send(
			torch.cat(
				(
					worker._reply_fields(
						opcode,
						result.embedding,
						error or result.embedding_handle,
						result.version,
						result.embedding_chunk,
						result.width,
					),
					worker._reply_fields(
						opcode,
						result.aggregate,
						error or result.aggregate_handle,
						result.version,
						result.aggregate_chunk,
						result.width,
					),
				)
			),
			0,
			epoch=worker.control_epoch,
		)

	def _store(self, output: Tensor, sums: Tensor) -> tuple[int, int]:
		store = self.worker.store
		handles = []
		try:
			for value in (output, sums):
				handles.append(store.put(value))
			with ExitStack() as stack:
				for handle in handles:
					stack.enter_context(store.acquire(handle, wait=True))
				store.mark_published_many(handles)
		except BaseException:
			for handle in handles:
				store.abandon(handle)
			raise
		return handles[0], handles[1]

	def _previous(self, identity: _Result) -> Optional[_Result]:
		header, aggregate = identity.embedding, identity.aggregate
		if header.request in self.worker._attempts or aggregate.request in self.worker._attempts:
			raise ValueError("A normal compute attempt cannot be reused for paired outputs")
		previous, companion = self._results.get(header.request), self._results.get(aggregate.request)
		if previous is None and companion is None:
			return None
		if (
			previous is None
			or previous is not companion
			or replace(previous, embedding_handle=0, aggregate_handle=0) != identity
		):
			raise ValueError("Paired output retry changed its state identity")
		return previous

	def _execute(self, identity: _Result, apply: Callable[[], tuple[Tensor, Tensor]]) -> None:
		worker, header = self.worker, identity.embedding
		try:
			result = self._previous(identity)
			if result is None:
				output, sums = apply()
				if any(
					value.shape != (header.targets, identity.width) or value.dtype != torch.float32
					for value in (output, sums)
				):
					raise ValueError("Paired output shapes/dtypes do not match the task")
				worker._drain()
				try:
					h, a = self._store(output, sums)
				except MemoryError:
					worker._drain(all_=True)
					h, a = self._store(output, sums)
				result = replace(identity, embedding_handle=h, aggregate_handle=a)
		except Exception as error:
			logging.exception(
				"Paired aggregate execution failed: rank=%s, attempt=%s", worker.owner.rank, header.request
			)
			self._send(identity, error=1 if isinstance(error, MemoryError) else 2)
			return
		requests = (header.request, identity.aggregate.request)
		for request in requests:
			self._results[request] = result
		for handle in (result.embedding_handle, result.aggregate_handle):
			self._handles[handle] = requests
		self._send(result)

	@torch.inference_mode()
	def compute(self, command: Tensor) -> None:
		worker = self.worker
		if worker._prepared is not None:
			raise ValueError("A paired task cannot replace retained prepared inputs")
		header = ComputeHeader.decode(command, opcode=COMPUTE_WITH_AGGREGATE)
		fields = worker._receive_fields(header, with_aggregate=True)
		aggregate, metadata = aggregate_identity(header, fields)
		version, chunk, _ = fields[0].cpu().tolist()
		width = worker.model.num_classes if header.layer == worker.model.num_layers else worker.model.num_hiddens
		identity = _Result(header, aggregate, version, chunk, int(metadata[1]), 0, 0, width)
		with ExitStack() as stack:
			inputs = worker._inputs(header, fields, stack)

			def apply() -> tuple[Tensor, Tensor]:
				block = dgl.create_block(
					(fields[1], fields[2]),
					num_src_nodes=header.sources,
					num_dst_nodes=header.targets,
					device=worker.comm.device,
				)
				return worker._apply_with_aggregate(LayerPlan(header.layer, block, fields[3], fields[4]), inputs)

			self._execute(identity, apply)

	@torch.inference_mode()
	def compute_recipe(self, command: Tensor) -> None:
		worker = self.worker
		if worker._prepared is not None:
			raise ValueError("An aggregate recipe cannot replace retained prepared inputs")
		header = RecipeHeader.decode(command)
		if header.generation != worker.owner.generation or header.layer > worker.model.num_layers:
			raise ValueError("Aggregate recipe refers to an invalid generation or layer")
		fields, tail = receive_tables(worker, header.nbytes, header.layer, extra_specs=header.tail_specs())
		version, chunk, request, aggregate_chunk = tail[0].cpu().tolist()
		if min(version, chunk, aggregate_chunk) < 0 or request < 1 or request == header.request:
			raise ValueError("Invalid paired aggregate recipe identity")
		widths = (
			worker.model.num_features,
			*((worker.model.num_hiddens,) * (worker.model.num_layers - 1)),
			worker.model.num_classes,
		)
		identity = _Result(
			header, replace(header, request=request), version, chunk, aggregate_chunk, 0, 0, widths[header.layer]
		)
		with ExitStack() as stack:
			tables = resolve_tables(worker, fields, stack)

			def apply() -> tuple[Tensor, Tensor]:
				validate_recipe(header.reconstruction(), fields, widths)
				inputs = worker._reconstruct_sage(0, tables, tail[5])
				block = dgl.create_block(
					(tail[1], tail[2]),
					num_src_nodes=header.sources,
					num_dst_nodes=header.targets,
					device=worker.comm.device,
				)
				return worker._apply_with_aggregate(LayerPlan(header.layer, block, tail[3], tail[4]), inputs)

			self._execute(identity, apply)
