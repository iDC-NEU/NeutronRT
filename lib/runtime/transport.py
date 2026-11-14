"""Storage-side control/data exchange with process-owned embedding workers."""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Callable, Dict, Sequence, Tuple

import torch
from torch import Tensor

from lib.dist.comm_manager import CommManager, CommRequest
from lib.runtime.protocol import (
	CANCEL_PREPARED,
	ERROR,
	PREPARE,
	PREPARED,
	READ,
	READ_MANY,
	READY,
	RETIRE,
	RUN_PREPARED,
	STOP,
	ComputePacket,
)
from lib.runtime.worker import REPLY_FIELDS

if TYPE_CHECKING:
	from lib.runtime.aggregates.delta import DeltaPacket
	from lib.runtime.aggregates.protocol import AggregatePacket
	from lib.runtime.aggregates.recipe import RecipePacket
	from lib.runtime.aggregates.reconstruction import ReconstructionPacket
from lib.graph.ownership import Location, MissingState, Owner


class TransportFailure(RuntimeError):
	pass


class ModelExecutionError(RuntimeError):
	pass


@contextmanager
def _io():
	try:
		yield
	except (RuntimeError, OSError) as error:
		raise TransportFailure("Owned worker transport failed") from error


@dataclass
class RemoteEndpoint:
	owner: Owner
	device: torch.device
	max_bytes: int
	bytes_used: int = 0


@dataclass(frozen=True)
class ReadyResult:
	handle: int
	version: int
	layer: int
	chunk: int
	shape: Tuple[int, int]
	bytes_used: int


@dataclass(frozen=True)
class PendingRead:
	request: CommRequest
	value: Tensor

	def result(self) -> Tensor:
		with _io():
			self.request.wait()
		return self.value


@dataclass(frozen=True)
class PreparedCompute:
	owner: Owner
	packet: ComputePacket
	output_width: int


@dataclass(frozen=True)
class PendingCompute:
	request: CommRequest
	decode: Callable[[], ReadyResult]

	def result(self) -> ReadyResult:
		with _io():
			self.request.wait()
		return self.decode()


class OwnedTransport:
	"""A CPU coordinator receives metadata; explicit reads receive selected values.

	Per-worker posting locks keep command headers and data bodies in FIFO order.
	Compute inputs are supplied in the same order the target consumes them. Control
	completion uses a separate group, so an unmatched writeback cannot block it.
	"""

	def __init__(
		self, comm: CommManager, control_epoch: int, endpoints: Sequence[RemoteEndpoint], *, coalesced: bool = True
	) -> None:
		if comm.rank != 0 or comm.device.type != "cpu":
			raise ValueError("Owned transport requires a CPU storage coordinator at rank zero")
		if len({node.owner.rank for node in endpoints}) != len(endpoints):
			raise ValueError("Compute ranks must be distinct")
		self.comm, self.control_epoch = comm, control_epoch
		self.coalesced = coalesced
		self.endpoints = {node.owner: node for node in endpoints}
		self._locks: Dict[Owner, RLock] = {node.owner: RLock() for node in endpoints}

	def _endpoint(self, owner: Owner) -> RemoteEndpoint:
		try:
			return self.endpoints[owner]
		except KeyError as error:
			raise MissingState(owner) from error

	def locked(self, owner: Owner) -> RLock:
		self._endpoint(owner)
		return self._locks[owner]

	def compute(
		self,
		owner: Owner,
		packet: ComputePacket,
		*,
		output_width: int,
		supply_inputs: Callable[[], None],
	) -> ReadyResult:
		self._endpoint(owner)
		if packet.header.generation != owner.generation:
			raise ValueError("Task belongs to another worker generation")
		with _io(), self.locked(owner):
			self.comm.send(packet.header.tensor(), owner.rank, epoch=self.control_epoch)
			if self.coalesced:
				self.comm.isend_tensors(packet.fields, owner.rank).wait()
			else:
				self.comm.wait_all(
					[
						self.comm.isend(value, owner.rank, name=f"field-{index}")
						for index, value in enumerate(packet.fields)
						if value.numel()
					]
				)
			reply = torch.empty(REPLY_FIELDS, dtype=torch.int64)
			request = self.comm.irecv(reply, owner.rank, epoch=self.control_epoch)
			supply_inputs()
		with _io():
			request.wait()
		return self._completion(owner, packet, reply, output_width)

	def _completion(
		self, owner: Owner, packet: ComputePacket, reply: Tensor, output_width: int, *, opcode: int = READY
	) -> ReadyResult:
		version, chunk, _ = packet.fields[0].tolist()
		return self._ready(
			owner,
			reply,
			request=packet.header.request,
			version=version,
			layer=packet.header.layer,
			chunk=chunk,
			rows=packet.header.targets,
			width=output_width,
			opcode=opcode,
		)

	def _ready(
		self,
		owner: Owner,
		reply: Tensor,
		*,
		request: int,
		version: int,
		layer: int,
		chunk: int,
		rows: int,
		width: int,
		opcode: int = READY,
	) -> ReadyResult:
		node = self._endpoint(owner)
		fields = reply.tolist()
		if fields[1:3] != [request, owner.generation] or fields[4:7] != [version, layer, chunk]:
			raise RuntimeError("Owned completion does not match the active attempt")
		if fields[0] == ERROR:
			if fields[3] == 1:
				raise MemoryError(f"Worker {owner.rank} exhausted its owned-state budget")
			raise ModelExecutionError(f"Owned model execution failed on worker {owner.rank}; inspect its task log")
		if fields[0] != opcode or fields[3] < 0 or fields[7:9] != [rows, width] or fields[9] < 0:
			raise RuntimeError("Invalid owned completion shape, handle, or resident byte count")
		node.bytes_used = fields[9]
		return ReadyResult(fields[3], version, layer, chunk, (fields[7], fields[8]), fields[9])

	def reconstruct(
		self, owner: Owner, packet: "ReconstructionPacket", *, output_width: int, supply_inputs: Callable[[], None]
	) -> ReadyResult:
		from lib.runtime.aggregates.reconstruction_transport import reconstruct

		return reconstruct(self, owner, packet, output_width=output_width, supply_inputs=supply_inputs)

	def compute_with_aggregate(
		self, owner: Owner, packet: "AggregatePacket", *, output_width: int, supply_inputs: Callable[[], None]
	) -> Tuple[ReadyResult, ReadyResult]:
		from lib.runtime.aggregates.transport import compute

		return compute(self, owner, packet, output_width=output_width, supply_inputs=supply_inputs)

	def compute_from_aggregates(
		self, owner: Owner, packet: "RecipePacket", *, output_width: int, supply_inputs: Callable[[], None]
	) -> Tuple[ReadyResult, ReadyResult]:
		from lib.runtime.aggregates.transport import compute

		return compute(self, owner, packet, output_width=output_width, supply_inputs=supply_inputs)

	def compute_delta(
		self, owner: Owner, packet: "DeltaPacket", *, output_width: int, supply_inputs: Callable[[], None]
	) -> Tuple[ReadyResult, ReadyResult]:
		from lib.runtime.aggregates.transport import compute

		return compute(self, owner, packet, output_width=output_width, supply_inputs=supply_inputs)

	def prepare(
		self, owner: Owner, packet: ComputePacket, *, output_width: int, supply_inputs: Callable[[], None]
	) -> PreparedCompute:
		self._endpoint(owner)
		if packet.header.generation != owner.generation:
			raise ValueError("Prepared task belongs to another worker generation")
		with _io(), self.locked(owner):
			self.comm.send(packet.header.tensor(opcode=PREPARE), owner.rank, epoch=self.control_epoch)
			if self.coalesced:
				self.comm.isend_tensors(packet.fields, owner.rank).wait()
			else:
				self.comm.wait_all(
					[
						self.comm.isend(value, owner.rank, name=f"field-{index}")
						for index, value in enumerate(packet.fields)
						if value.numel()
					]
				)
			reply = torch.empty(REPLY_FIELDS, dtype=torch.int64)
			request = self.comm.irecv(reply, owner.rank, epoch=self.control_epoch)
			supply_inputs()
		with _io():
			request.wait()
		self._completion(owner, packet, reply, output_width, opcode=PREPARED)
		return PreparedCompute(owner, packet, output_width)

	def launch(self, prepared: PreparedCompute) -> PendingCompute:
		owner, packet = prepared.owner, prepared.packet
		self._endpoint(owner)
		reply = torch.empty(REPLY_FIELDS, dtype=torch.int64)
		with _io(), self.locked(owner):
			request = self.comm.irecv(reply, owner.rank, epoch=self.control_epoch)
			self.comm.send(
				torch.tensor([RUN_PREPARED, packet.header.request, owner.generation, 0, 0, 0, 0, 0]),
				owner.rank,
				epoch=self.control_epoch,
			)
		return PendingCompute(request, lambda: self._completion(owner, packet, reply, prepared.output_width))

	def cancel(self, prepared: PreparedCompute) -> None:
		owner = prepared.owner
		with _io(), self.locked(owner):
			self.comm.send(
				torch.tensor([CANCEL_PREPARED, prepared.packet.header.request, owner.generation, 0, 0, 0, 0, 0]),
				owner.rank,
				epoch=self.control_epoch,
			)

	def read(self, location: Location, rows: Tensor, width: int) -> PendingRead:
		with _io(), self.locked(location.owner):
			self._issue_read(location, rows, width, target=0)
			values = self.comm.empty((rows.numel(), width), dtype=torch.float32)
			request = self.comm.irecv(values, location.owner.rank, name="owned-rows")
		return PendingRead(request, values)

	def fetch_peer(self, location: Location, rows: Tensor, width: int, target: Owner) -> None:
		self._endpoint(target)
		if target == location.owner:
			raise ValueError("Local inputs must retain the worker's resident tensor")
		with _io(), self.locked(location.owner):
			self._issue_read(location, rows, width, target=target.rank)

	def read_many(self, requests: Sequence[Tuple[Location, Tensor]], width: int) -> PendingRead:
		owner = self._read_many_owner(requests, width)
		if len(requests) == 1:
			return self.read(*requests[0], width)
		with _io(), self.locked(owner):
			self._issue_read_many(requests, width, target=0)
			values = self.comm.empty((sum(rows.numel() for _, rows in requests), width), dtype=torch.float32)
			request = self.comm.irecv(values, owner.rank, name="owned-gather")
		return PendingRead(request, values)

	def fetch_peer_many(self, requests: Sequence[Tuple[Location, Tensor]], width: int, target: Owner) -> None:
		owner = self._read_many_owner(requests, width)
		self._endpoint(target)
		if owner == target:
			raise ValueError("Local inputs must retain the worker's resident tensor")
		if len(requests) == 1:
			return self.fetch_peer(*requests[0], width, target)
		with _io(), self.locked(owner):
			self._issue_read_many(requests, width, target=target.rank)

	@staticmethod
	def _read_many_owner(requests: Sequence[Tuple[Location, Tensor]], width: int) -> Owner:
		if not requests or width < 1:
			raise ValueError("Grouped reads need a source and a positive width")
		owner = requests[0][0].owner
		for location, rows in requests:
			if location.owner != owner or location.handle < 0:
				raise ValueError("Grouped reads must have one source generation and valid handles")
			if (
				rows.device.type != "cpu"
				or rows.dtype != torch.int64
				or rows.ndim != 1
				or rows.numel()
				and rows.min() < 0
			):
				raise ValueError("Grouped read offsets must be nonnegative CPU int64 vectors")
		return owner

	def _issue_read_many(self, requests: Sequence[Tuple[Location, Tensor]], width: int, *, target: int) -> None:
		owner = requests[0][0].owner
		metadata = torch.tensor([[location.handle, rows.numel()] for location, rows in requests], dtype=torch.int64)
		rows = torch.cat([rows for _, rows in requests])
		self.comm.send(
			torch.tensor([READ_MANY, 0, owner.generation, target, len(requests), rows.numel(), width, 0]),
			owner.rank,
			epoch=self.control_epoch,
		)
		self.comm.wait_all(
			[
				self.comm.isend(metadata, owner.rank, name="read-many-meta"),
				self.comm.isend(rows, owner.rank, name="read-many-indices"),
			]
		)

	def _issue_read(self, location: Location, rows: Tensor, width: int, *, target: int) -> None:
		if rows.device.type != "cpu" or rows.dtype != torch.int64 or rows.ndim != 1 or width < 1:
			raise ValueError("Owned read metadata must use CPU row IDs and a positive width")
		if rows.numel() and rows.min() < 0:
			raise ValueError("Owned read offsets must be nonnegative")
		self.comm.send(
			torch.tensor(
				[
					READ,
					0,
					location.owner.generation,
					location.handle,
					target,
					rows.numel(),
					width,
					0,
				]
			),
			location.owner.rank,
			epoch=self.control_epoch,
		)
		self.comm.send(rows, location.owner.rank)

	def send_cpu_input(self, owner: Owner, values: Tensor) -> None:
		if values.device.type != "cpu" or values.dtype != torch.float32 or values.ndim != 2:
			raise ValueError("Storage input rows must be CPU float32")
		# Called under the target's posting lock from compute.supply_inputs.
		self._endpoint(owner)
		with _io():
			self.comm.send(values, owner.rank)

	def retire(self, location: Location) -> None:
		with _io(), self.locked(location.owner):
			self.comm.send(
				torch.tensor(
					[
						RETIRE,
						0,
						location.owner.generation,
						location.handle,
						0,
						0,
						0,
						0,
					]
				),
				location.owner.rank,
				epoch=self.control_epoch,
			)

	def stop(self) -> None:
		for owner in self.endpoints:
			with _io(), self.locked(owner):
				self.comm.send(
					torch.tensor([STOP, 0, owner.generation, 0, 0, 0, 0, 0]), owner.rank, epoch=self.control_epoch
				)
