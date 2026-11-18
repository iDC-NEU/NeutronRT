"""Worker execution that retains published embeddings on their producing device."""

import logging
from collections import deque
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union, cast

import dgl
import torch
from torch import Tensor

from lib.dist.comm_manager import CommManager, CommRequest
from lib.graph.deltas import NeighborUpdate, RoutedRows
from lib.graph.ownership import Owner
from lib.graph.recompute import LayerPlan
from lib.graph.resident import ResidentTensors, TensorLease
from lib.graph.serving import select_rows
from lib.models import MODEL_NAMES, GNNModel
from lib.models.aggregates import delta_sage_state, full_sage_state, reconstruct_sage
from lib.models.operators import full_neighbor_forward
from lib.runtime.protocol import (
	CANCEL_PREPARED,
	COMPUTE,
	COMPUTE_FROM_AGGREGATES,
	COMPUTE_WITH_AGGREGATE,
	DELTA_COMPUTE,
	ERROR,
	PREPARE,
	PREPARED,
	READ,
	READ_MANY,
	READY,
	RECONSTRUCT,
	RETIRE,
	RUN_PREPARED,
	STOP,
	ComputeHeader,
)

if TYPE_CHECKING:
	from lib.runtime.aggregates.delta import DeltaHeader
	from lib.runtime.aggregates.recipe import RecipeHeader
	from lib.runtime.aggregates.reconstruction import ReconstructionHeader
	from lib.runtime.aggregates.worker import AggregateTasks

REPLY_FIELDS = 10


@dataclass
class _PreparedTask:
	header: ComputeHeader
	fields: Tuple[Tensor, ...]
	inputs: Tensor
	leases: ExitStack


class OwnedWorker:
	"""Use the existing worker model and manager with a separate control epoch.

	The storage coordinator compiles every block. Normal completion returns only a
	handle and version metadata. Explicit reads transfer selected rows; their leases
	remain held until communication completes. Owned state is separate from caches.
	"""

	def __init__(
		self,
		comm: CommManager,
		model: GNNModel,
		owner: Owner,
		*,
		max_bytes: int,
		control_epoch: int,
		coalesced: bool = True,
		gather_inputs: bool = False,
		operator_incremental: bool = False,
		mapped_host_reads: bool = False,
		history_bytes: int = 256 * 1024**2,
		feature_cache_bytes: int = 0,
		refresh_steps: int = 32,
	) -> None:
		if owner.rank != comm.rank or owner.generation < 1:
			raise ValueError("Owned worker identity does not match its communicator")
		if model.model_name not in MODEL_NAMES:
			raise ValueError(f"Unsupported worker model: {model.model_name}")
		if type(feature_cache_bytes) is not int or feature_cache_bytes < 0:
			raise ValueError("Feature-row cache budget must be a nonnegative integer")
		if feature_cache_bytes:
			raise NotImplementedError(
				"Feature-row reuse is fail-closed until the wire protocol can avoid sending cache hits"
			)
		self.comm, self.model, self.owner = comm, model.eval(), owner
		self.control_epoch = control_epoch
		self.coalesced = coalesced
		self.gather_inputs = gather_inputs
		self.incremental = None
		self.feature_cache = None
		if operator_incremental:
			from lib.models.incremental import IncrementalLayers

			self.incremental = IncrementalLayers(
				model, max_bytes=history_bytes, refresh_steps=refresh_steps, mapped_host_reads=mapped_host_reads
			)
		self.store = ResidentTensors(max_bytes)
		self._attempts: Dict[int, Tuple[int, int, int, int]] = {}
		self._handles: Dict[int, int] = {}
		self._transfers: deque[Tuple[CommRequest, Union[TensorLease, ExitStack]]] = deque()
		self._prepared: Optional[_PreparedTask] = None
		self._aggregate_tasks: Optional[AggregateTasks] = None
		self._copy_stream = torch.cuda.Stream(device=comm.device) if comm.device.type == "cuda" else None
		for layer in model.layers:
			if hasattr(layer, "set_allow_zero_in_degree"):
				layer.set_allow_zero_in_degree(True)

	def _drain(self, *, all_: bool = False) -> None:
		while self._transfers:
			request, lease = self._transfers[0]
			if not all_ and len(self._transfers) <= 4 and not request.is_completed():
				break
			request.wait()
			lease.close()
			self._transfers.popleft()

	def _width(self, layer: int) -> int:
		return self.model.num_features if layer == 1 else self.model.num_hiddens

	def _inputs(
		self, header: ComputeHeader, fields: Tuple[Tensor, ...], stack: ExitStack, *, width: Optional[int] = None
	) -> Tensor:
		if header.layer == 1:
			return fields[9]
		if self.gather_inputs and header.groups > 1:
			return self._grouped_inputs(header, fields, stack, width=width)
		groups = fields[6].cpu().tolist()
		rows, positions = fields[7], fields[8]
		width = self._width(header.layer) if width is None else width
		output = None
		offset = 0
		for rank, generation, handle, count in groups:
			selected_rows = rows[offset : offset + count]
			selected_positions = positions[offset : offset + count]
			if rank == self.owner.rank:
				if generation != self.owner.generation:
					raise ValueError("Input refers to an older worker generation")
				values = stack.enter_context(self.store.acquire(handle, wait=True))
				selected = select_rows(values, selected_rows)
			else:
				selected = self.comm.recv_tensor((count, width), src=rank)
			if header.groups == 1 and torch.equal(
				selected_positions, torch.arange(header.sources, device=selected_positions.device)
			):
				return selected
			if output is None:
				output = self.comm.empty((header.sources, width))
			output.index_copy_(0, selected_positions, selected)
			offset += count
		if offset != header.sources:
			raise ValueError("Input chunk rows do not cover the compiled block")
		return cast(Tensor, output if output is not None else self.comm.empty((0, width)))

	def _grouped_inputs(
		self, header: ComputeHeader, fields: Tuple[Tensor, ...], stack: ExitStack, *, width: Optional[int] = None
	) -> Tensor:
		by_sender: Dict[int, list] = {}
		offset = 0
		for rank, generation, handle, count in fields[6].cpu().tolist():
			by_sender.setdefault(rank, []).append((generation, handle, count, offset))
			offset += count
		if offset != header.sources:
			raise ValueError("Input groups do not cover the compiled block")
		width = self._width(header.layer) if width is None else width
		output: Tensor = self.comm.empty((header.sources, width))
		for rank, groups in by_sender.items():
			received = (
				None
				if rank == self.owner.rank
				else self.comm.recv_tensor((sum(item[2] for item in groups), width), src=rank)
			)
			incoming = 0
			for generation, handle, count, offset in groups:
				if rank == self.owner.rank:
					if generation != self.owner.generation:
						raise ValueError("Input refers to an older worker generation")
					values = stack.enter_context(self.store.acquire(handle, wait=True))
					selected = select_rows(values, fields[7][offset : offset + count])
				else:
					assert received is not None
					selected = received.narrow(0, incoming, count)
					incoming += count
				output.index_copy_(0, fields[8][offset : offset + count], selected)
		return output

	def _reply_fields(
		self,
		opcode: int,
		header: Union[ComputeHeader, "ReconstructionHeader", "RecipeHeader", "DeltaHeader"],
		handle: int,
		version: int,
		chunk: int,
		width: int,
	) -> Tensor:
		return torch.tensor(
			[
				opcode,
				header.request,
				self.owner.generation,
				handle,
				version,
				header.layer,
				chunk,
				header.targets,
				width,
				self.store.bytes_used,
			],
			dtype=torch.int64,
		)

	def _reply(
		self,
		opcode: int,
		header: Union[ComputeHeader, "ReconstructionHeader", "RecipeHeader", "DeltaHeader"],
		handle: int,
		version: int,
		chunk: int,
		width: int,
	) -> None:
		self.comm.send(self._reply_fields(opcode, header, handle, version, chunk, width), 0, epoch=self.control_epoch)

	def _apply(self, task: LayerPlan, inputs: Tensor) -> Tensor:
		if self.incremental is not None:
			return self.incremental.apply(task, inputs)
		values = full_neighbor_forward(self.model.layers[task.layer - 1], task, inputs)
		if values.ndim == 3:
			values = values.mean(1)
		return values.relu() if task.layer < self.model.num_layers else values

	def _apply_with_aggregate(self, task: LayerPlan, inputs: Tensor) -> Tuple[Tensor, Tensor]:
		return full_sage_state(self.model, task, inputs)

	def _apply_delta(self, layer: int, tables: Tuple[Tensor, ...], tail: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor]:
		output, updated = delta_sage_state(
			self.model,
			layer,
			tables[1],
			NeighborUpdate(tables[0], tail[1]),
			RoutedRows(tables[2] - tables[3], tail[2], tail[3]),
			RoutedRows(tables[4], tail[4], tail[5], tail[6]),
		)
		return output, updated.sums

	def _reconstruct_sage(self, start_layer: int, tables: Tuple[Tensor, ...], degrees: Tensor) -> Tensor:
		return reconstruct_sage(
			self.model,
			tables[0],
			[NeighborUpdate(sums, degrees) for sums in tables[1:]],
			projected=True,
			start_layer=start_layer,
		)

	def reconstruct(self, command: Tensor) -> None:
		from lib.runtime.aggregates.reconstruction_worker import reconstruct

		reconstruct(self, command)

	def _receive_fields(self, header: ComputeHeader, *, with_aggregate: bool = False) -> Tuple[Tensor, ...]:
		if header.generation != self.owner.generation or header.layer > self.model.num_layers:
			raise ValueError("Task refers to an invalid worker generation or layer")
		specs = header.specs(self._width(header.layer))
		if with_aggregate:
			from lib.runtime.aggregates.protocol import aggregate_specs

			specs = aggregate_specs(header, self._width(header.layer))
		fields = (
			self.comm.recv_tensors(specs, 0)
			if self.coalesced
			else tuple(
				self.comm.recv_tensor(spec.shape, src=0, dtype=spec.dtype, name=f"field-{index}")
				if prod(spec.shape)
				else self.comm.empty(spec.shape, dtype=spec.dtype)
				for index, spec in enumerate(specs)
			)
		)
		version, chunk, has_degrees = fields[0].cpu().tolist()
		if min(version, chunk) < 0 or bool(has_degrees) != header.has_degrees:
			raise ValueError("Invalid compiled task identity or source degree metadata")
		return fields

	def compute_with_aggregate(self, header: Tensor) -> None:
		from lib.runtime.aggregates.worker import AggregateTasks

		if self._aggregate_tasks is None:
			self._aggregate_tasks = AggregateTasks(self)
		self._aggregate_tasks.compute(header)

	def compute_from_aggregates(self, header: Tensor) -> None:
		from lib.runtime.aggregates.worker import AggregateTasks

		if self._aggregate_tasks is None:
			self._aggregate_tasks = AggregateTasks(self)
		self._aggregate_tasks.compute_recipe(header)

	def compute_delta(self, command: Tensor) -> None:
		from lib.runtime.aggregates.delta_worker import compute_delta

		compute_delta(self, command)

	@torch.inference_mode()
	def compute(self, header_tensor: Tensor) -> None:
		if self._prepared is not None:
			raise ValueError("A worker cannot compute while a prepared task is retained")
		header = ComputeHeader.decode(header_tensor)
		fields = self._receive_fields(header)
		with ExitStack() as stack:
			self._execute(header, fields, self._inputs(header, fields, stack))

	def _execute(self, header: ComputeHeader, fields: Tuple[Tensor, ...], inputs: Tensor) -> None:
		version, chunk, _ = fields[0].cpu().tolist()
		width = self.model.num_classes if header.layer == self.model.num_layers else self.model.num_hiddens
		if self._aggregate_tasks is not None and self._aggregate_tasks.owns_request(header.request):
			self._reply(ERROR, header, 2, version, chunk, width)
			return
		previous = self._attempts.get(header.request)
		if previous is not None:
			if previous[:3] != (version, header.layer, chunk):
				raise ValueError("A compute attempt cannot name two state keys")
			self._reply(READY, header, previous[3], version, chunk, width)
			return
		block = dgl.create_block(
			(fields[1], fields[2]),
			num_src_nodes=header.sources,
			num_dst_nodes=header.targets,
			device=self.comm.device,
		)
		task = LayerPlan(header.layer, block, fields[3], fields[4], fields[5] if header.has_degrees else None)
		if header.edge_width:
			block.edata["nrt"] = fields[10]
		try:
			handle = None
			values = self._apply(task, inputs)
			if values.shape != (header.targets, width) or values.dtype != torch.float32:
				raise ValueError("Model output does not match the compiled owned task")
			self._drain()
			try:
				handle = self.store.put(values)
			except MemoryError:
				# Retired results may still be pinned by completed/in-flight reads.
				# Drain their consumers before asking storage for another buffer.
				self._drain(all_=True)
				handle = self.store.put(values)
			with self.store.acquire(handle, wait=True):
				self.store.mark_published(handle)
		except Exception as error:
			if handle is not None:
				self.store.abandon(handle)
			logging.exception("Owned computation failed: rank=%s, attempt=%s", self.owner.rank, header.request)
			self._reply(ERROR, header, 1 if isinstance(error, MemoryError) else 2, version, chunk, width)
			return
		self._attempts[header.request] = version, header.layer, chunk, handle
		self._handles[handle] = header.request
		self._reply(READY, header, handle, version, chunk, width)

	@torch.inference_mode()
	def prepare(self, header_tensor: Tensor) -> None:
		if self._prepared is not None:
			raise ValueError("Only one prepared task is allowed per worker")
		header = ComputeHeader.decode(header_tensor, opcode=PREPARE)
		fields = self._receive_fields(header)
		with ExitStack() as stack:
			inputs = self._inputs(header, fields, stack)
			version, chunk, _ = fields[0].cpu().tolist()
			width = self.model.num_classes if header.layer == self.model.num_layers else self.model.num_hiddens
			self._prepared = _PreparedTask(header, fields, inputs, stack.pop_all())
			try:
				self._reply(PREPARED, header, 0, version, chunk, width)
			except BaseException:
				self._discard_prepared()
				raise

	def _discard_prepared(self) -> None:
		prepared, self._prepared = self._prepared, None
		if prepared is not None:
			prepared.leases.close()

	@torch.inference_mode()
	def run_prepared(self, command: Tensor) -> None:
		prepared = self._prepared
		if prepared is None or command[1:3].tolist() != [prepared.header.request, self.owner.generation]:
			raise ValueError("Run command does not match the prepared task")
		self._prepared = None
		with prepared.leases:
			self._execute(prepared.header, prepared.fields, prepared.inputs)

	def cancel_prepared(self, command: Tensor) -> None:
		if int(command[2]) != self.owner.generation:
			raise ValueError("Cancellation refers to another worker generation")
		if self._prepared is not None and int(command[1]) != self._prepared.header.request:
			raise ValueError("Cancellation does not match the prepared task")
		self._discard_prepared()

	@torch.inference_mode()
	def read(self, header: Tensor) -> None:
		_, _, generation, handle, target, count, width, _ = header.tolist()
		if generation != self.owner.generation or count < 0 or width < 1:
			raise ValueError("Invalid owned read identity or dimensions")
		rows = self.comm.recv_tensor(count, dtype=torch.int64, src=0, device="cpu")
		lease = self.store.acquire(handle, wait=True)
		try:
			context = torch.cuda.stream(self._copy_stream) if self._copy_stream is not None else nullcontext()
			with context:
				selected = select_rows(lease.tensor, rows)
				if selected.ndim != 2 or selected.shape != (count, width):
					raise ValueError("Requested rows do not match the resident embedding width")
				request = self.comm.isend(selected, target, name="owned-read")
			self._transfers.append((request, lease))
		except BaseException:
			lease.close()
			raise
		self._drain()

	def retire(self, header: Tensor) -> None:
		_, _, generation, handle, _, _, _, _ = header.tolist()
		if generation != self.owner.generation:
			raise ValueError("Retirement refers to another worker generation")
		self.store.retire(handle)
		request = self._handles.pop(handle, None)
		if request is not None:
			self._attempts.pop(request, None)
		if self._aggregate_tasks is not None:
			self._aggregate_tasks.retired(handle)

	@torch.inference_mode()
	def read_many(self, header: Tensor) -> None:
		_, _, generation, target, groups, count, width, _ = header.tolist()
		if generation != self.owner.generation or min(groups, width) < 1 or count < 0 or target < 0:
			raise ValueError("Invalid grouped read identity or dimensions")
		metadata = self.comm.recv_tensor(
			(groups, 2), dtype=torch.int64, src=0, device="cpu", name="read-many-meta"
		).tolist()
		rows = self.comm.recv_tensor(count, dtype=torch.int64, src=0, name="read-many-indices")
		if any(handle < 0 or size < 0 for handle, size in metadata) or sum(size for _, size in metadata) != count:
			raise ValueError("Grouped read rows do not match its header")
		with ExitStack() as leases:
			# Indices are received on the manager's stream, then read by the copy
			# stream. Retain their allocation through the transfer completion.
			leases.callback(lambda _indices: None, rows)
			context = torch.cuda.stream(self._copy_stream) if self._copy_stream is not None else nullcontext()
			with context:
				output = self.comm.empty((count, width))
				offset = 0
				for handle, size in metadata:
					values = leases.enter_context(self.store.acquire(handle, wait=True))
					if values.ndim != 2 or values.shape[1] != width:
						raise ValueError("Grouped read embedding width mismatch")
					torch.index_select(values, 0, rows.narrow(0, offset, size), out=output.narrow(0, offset, size))
					offset += size
				request = self.comm.isend(output, target, name="owned-gather")
			self._transfers.append((request, leases.pop_all()))
		self._drain()

	def handle_extra(self, header: Tensor) -> bool:
		return False

	def profile(self) -> dict:
		return {
			"owner": {"rank": self.owner.rank, "generation": self.owner.generation},
			"feature_cache": {
				"enabled": False,
				"budget_bytes": 0,
				"status": "disabled",
			},
		}

	def run(self) -> None:
		try:
			while True:
				self._drain()
				header = self.comm.recv_tensor(
					8, src=0, dtype=torch.int64, device="cpu", name="control-header", epoch=self.control_epoch
				)
				opcode = int(header[0])
				if opcode == STOP:
					if int(header[2]) != self.owner.generation:
						raise ValueError("Stop refers to another worker generation")
					break
				if opcode == COMPUTE:
					self.compute(header)
				elif opcode == COMPUTE_WITH_AGGREGATE:
					self.compute_with_aggregate(header)
				elif opcode == RECONSTRUCT:
					self.reconstruct(header)
				elif opcode == COMPUTE_FROM_AGGREGATES:
					self.compute_from_aggregates(header)
				elif opcode == DELTA_COMPUTE:
					self.compute_delta(header)
				elif opcode == PREPARE:
					self.prepare(header)
				elif opcode == RUN_PREPARED:
					self.run_prepared(header)
				elif opcode == CANCEL_PREPARED:
					self.cancel_prepared(header)
				elif opcode == READ:
					self.read(header)
				elif opcode == READ_MANY:
					self.read_many(header)
				elif opcode == RETIRE:
					self.retire(header)
				elif not self.handle_extra(header):
					raise ValueError("Unsupported owned control opcode or wire version")
			self._drain(all_=True)
		finally:
			self._discard_prepared()
			self.comm.destroy()
