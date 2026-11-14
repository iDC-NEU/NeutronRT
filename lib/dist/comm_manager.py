"""Own device staging, completion, buffer lifetimes, and transport selection."""

import logging
import socket
from dataclasses import dataclass
from datetime import timedelta
from threading import RLock
from typing import Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import Work

from lib.dist.buffers import PackedLayout, TensorSpec


class CommRequest:
	"""Completion includes device copies; buffers stay owned until CUDA is ready."""

	def __init__(
		self,
		work: Work,
		*,
		finalize: Optional[Callable[[], None]] = None,
		keepalive: Optional[List[Tensor]] = None,
		name: str = "",
		tensor: Optional[Tensor] = None,
		cuda_device: Optional[torch.device] = None,
		epoch: Optional[int] = None,
	):
		self.work = work
		self.finalize = finalize
		self.keepalive = keepalive if keepalive is not None else []
		self.name = name
		self.tensor = tensor
		self.tensors = None
		self._cuda_device = cuda_device
		self.epoch = epoch
		self._prepared = False
		self._done = False
		self._event = None
		self._lock = RLock()

	def _prepare(self):
		if self._prepared:
			return
		if self._cuda_device is None:
			self.work.wait()
			if self.finalize is not None:
				self.finalize()
		else:
			with torch.cuda.device(self._cuda_device):
				# NCCL polling alone does not establish a consumer-stream dependency.
				self.work.wait()
				if self.finalize is not None:
					self.finalize()
				self._event = torch.cuda.Event()
				self._event.record(torch.cuda.current_stream(self._cuda_device))
		self.finalize = None
		self._prepared = True

	def _complete(self) -> bool:
		self._done = True
		self.keepalive.clear()
		return True

	def wait(self):
		with self._lock:
			return self._wait()

	def _wait(self):
		if self._done:
			return True
		self._prepare()
		if self._event is not None:
			self._event.synchronize()
		return self._complete()

	def is_completed(self) -> bool:
		with self._lock:
			return self._is_completed()

	def _is_completed(self) -> bool:
		if self._done:
			return True
		if not self._prepared:
			if not self.work.is_completed():
				return False
			self._prepare()
		if self._event is not None and not self._event.query():
			return False
		return self._complete()


@dataclass(frozen=True)
class _GroupEpoch:
	members: Tuple[int, ...]
	backend: str
	handle: object


@dataclass(frozen=True)
class RowExchange:
	data: Tensor
	counts: Tuple[int, ...]
	ranks: Tuple[int, ...]


class CommManager:
	def __init__(self, world_size: int, rank: int, group=None, *, device=None, peer_devices=None, cuda_group=None):
		if world_size < 1 or not 0 <= rank < world_size:
			raise ValueError("Invalid communication world size or local rank")
		self.world_size = world_size
		self.rank = rank
		self.device = torch.device("cpu" if device is None else device)
		if self.device.type not in ("cpu", "cuda"):
			raise ValueError("CommManager supports CPU and CUDA tensors")
		if self.device.type == "cuda" and self.device.index is None:
			self.device = torch.device("cuda", torch.cuda.current_device())
		self._group = group
		self._cuda_group = cuda_group
		self._peer_devices = tuple(peer_devices) if peer_devices is not None else None
		if self._peer_devices is not None and len(self._peer_devices) != world_size:
			raise ValueError("Need one device type per global rank")
		self._owned_groups: List[dist.ProcessGroup] = []
		self._owns_default_group = group is None
		self._pending: List[CommRequest] = []
		self._closed = False
		self._seq = 0
		self._posting = RLock()
		self._epochs: Dict[int, _GroupEpoch] = {}
		self._retiring: set[int] = set()
		self._epoch_sequence = 0

	@classmethod
	def initialize(
		cls,
		world_size,
		rank,
		*,
		device="cpu",
		backend="auto",
		init_method=None,
		timeout=timedelta(minutes=3),
		rendezvous_prefix=None,
	):
		"""Collective startup; every global rank enters in the same order.

		Routing uses endpoint topology, never a local tensor's dtype/placement.
		NCCL selects its available GPU/interconnect/network implementation.
		"""
		device = torch.device(device)
		if device.type == "cuda":
			torch.cuda.set_device(device)
		kwargs = {
			"backend": "gloo" if backend == "auto" else backend,
			"world_size": world_size,
			"rank": rank,
			"timeout": timeout,
		}
		if rendezvous_prefix is not None:
			store, _, _ = next(
				dist.rendezvous(init_method or "env://", rank=rank, world_size=world_size, timeout=timeout)
			)
			kwargs["store"] = dist.PrefixStore(rendezvous_prefix, store)
		elif init_method is not None:
			kwargs["init_method"] = init_method
		dist.init_process_group(**kwargs)
		try:
			if backend != "auto":
				return cls(world_size, rank, device=device)
			local = {"device": device.type, "host": socket.gethostname(), "nccl": dist.is_nccl_available()}
			topology = [None] * world_size
			dist.all_gather_object(topology, local)
			gpu_ranks = [r for r, peer in enumerate(topology) if peer["device"] == "cuda"]
			cuda_group = None
			if len(gpu_ranks) >= 2 and all(topology[r]["nccl"] for r in gpu_ranks):
				# All global ranks enter new_group, including nonmembers.
				cuda_group = dist.new_group(ranks=gpu_ranks, backend="nccl", timeout=timeout)
			manager = cls(
				world_size,
				rank,
				device=device,
				peer_devices=[peer["device"] for peer in topology],
				cuda_group=cuda_group,
			)
			if rank in gpu_ranks and cuda_group is not None:
				manager._owned_groups.append(cuda_group)
			logging.info(
				"Communication topology: rank=%s, device=%s, peers=%s, gpu_ranks=%s",
				rank,
				device,
				manager._peer_devices,
				gpu_ranks,
			)
			return manager
		except Exception:
			dist.destroy_process_group()
			raise

	@property
	def backend(self) -> str:
		return dist.get_backend(self._group)

	@property
	def is_nccl(self) -> bool:
		return self.backend == "nccl"

	@property
	def is_gloo(self) -> bool:
		return self.backend == "gloo"

	def _epoch(self, epoch: int) -> _GroupEpoch:
		if epoch in self._retiring or epoch not in self._epochs:
			raise ValueError("Communication epoch is unknown or retiring")
		return self._epochs[epoch]

	def create_epoch(self, members, *, timeout=timedelta(minutes=3)) -> int:
		"""Collectively create a logical group without replacing existing traffic.

		Every world rank calls this in the same order, including nonmembers. The
		manager selects a compatible CCL. Epoch IDs travel in the control protocol;
		both endpoints must select the same epoch for each message.
		"""
		with self._posting:
			if self._closed:
				raise RuntimeError("Communication manager is closed")
			specification = self._epoch_sequence + 1, tuple(members)
			proposals = [None] * self.world_size
			dist.all_gather_object(proposals, specification, group=self._group)
			if any(proposal != specification for proposal in proposals):
				raise ValueError("Ranks disagree on communication epoch membership or creation order")
			epoch, ranks = specification
			if (
				not ranks
				or any(not isinstance(rank, int) or isinstance(rank, bool) for rank in ranks)
				or len(set(ranks)) != len(ranks)
				or any(not 0 <= rank < self.world_size for rank in ranks)
			):
				raise ValueError("Group membership must contain distinct valid global ranks")
			ranks = tuple(sorted(ranks))
			backend = self.backend
			if self._peer_devices is not None:
				backend = (
					"nccl"
					if all(self._peer_devices[rank] == "cuda" for rank in ranks) and self._cuda_group is not None
					else "gloo"
				)
			handle = dist.new_group(ranks=list(ranks), backend=backend, timeout=timeout)
			self._epochs[epoch] = _GroupEpoch(ranks, backend, handle)
			self._epoch_sequence = epoch
			if self.rank in ranks:
				self._owned_groups.append(handle)
			return epoch

	def epoch_members(self, epoch: int) -> Tuple[int, ...]:
		return self._epoch(epoch).members

	def retire_epoch(self, epoch: int) -> None:
		"""Fence new postings, drain this epoch, then collectively release its group.

		All world ranks retire epochs in the same order. Other epochs remain usable;
		physical process joins/leaves still require a separate rendezvous protocol.
		"""
		with self._posting:
			if self._closed:
				raise RuntimeError("Communication manager is closed")
			record = self._epoch(epoch)
			self._retiring.add(epoch)
			pending = [request for request in self._pending if request.epoch == epoch]
		if self.rank in record.members:
			for request in pending:
				request.wait()
			dist.barrier(group=record.handle)
			dist.destroy_process_group(cast(dist.ProcessGroup, record.handle))
		with self._posting:
			self._owned_groups = [group for group in self._owned_groups if group is not record.handle]
			del self._epochs[epoch]
			self._retiring.remove(epoch)

	def _peer_group(self, peer, epoch=None):
		if self._closed:
			raise RuntimeError("Communication manager is closed")
		if not 0 <= peer < self.world_size:
			raise ValueError(f"Invalid peer rank {peer}")
		if epoch is not None:
			record = self._epoch(epoch)
			if self.rank not in record.members or peer not in record.members:
				raise ValueError("Both message endpoints must belong to the communication epoch")
			return record.handle
		if (
			self._cuda_group is not None
			and self._peer_devices is not None
			and self._peer_devices[self.rank] == self._peer_devices[peer] == "cuda"
		):
			return self._cuda_group
		return self._group

	def _wire_device(self, group):
		if dist.get_backend(group) == "nccl":
			if self.device.type == "cuda":
				return self.device
			if torch.cuda.is_available():
				return torch.device("cuda", torch.cuda.current_device())
			raise RuntimeError("NCCL transport requires an accessible CUDA device")
		return torch.device("cpu")

	def empty(self, shape, *, dtype=torch.float32, device=None):
		"""Allocate application storage; wire storage is an internal decision."""
		return torch.empty(shape, dtype=dtype, device=self.device if device is None else device)

	def recv_tensor(self, shape, *, src, dtype=torch.float32, device=None, name="", epoch=None):
		tensor = self.empty(shape, dtype=dtype, device=device)
		self.irecv(tensor, src=src, name=name, epoch=epoch).wait()
		return tensor

	def isend_tensors(self, tensors, dst, *, name="packed", epoch=None):
		"""Send all payload dtypes in one byte P2P; sizes are known by the protocol."""
		group = self._peer_group(dst, epoch)
		wire_device = self._wire_device(group)
		tensors = tuple(tensors)
		layout = PackedLayout(TensorSpec.of(tensor) for tensor in tensors)
		if layout.nbytes > 1024**3:
			raise ValueError("Packed payload exceeds the 1 GiB allocation limit")
		pinned = wire_device.type == "cpu" and any(tensor.is_cuda for tensor in tensors)
		buffer = torch.empty(layout.nbytes, dtype=torch.uint8, device=wire_device, pin_memory=pinned)
		for begin, end in layout.padding:
			buffer[begin:end].zero_()
		for tensor, (begin, end) in zip(tensors, layout.slices, strict=True):
			buffer[begin:end].copy_(tensor.contiguous().reshape(-1).view(torch.uint8), non_blocking=True)
		if wire_device.type == "cpu":
			for device in {tensor.device for tensor in tensors if tensor.is_cuda}:
				torch.cuda.current_stream(device).synchronize()
		request = self.isend(buffer, dst=dst, name=name, epoch=epoch)
		request.keepalive.extend(tensors)
		return request

	def irecv_tensors(self, specs, src, *, device=None, name="packed", epoch=None):
		layout = PackedLayout(specs)
		if layout.nbytes > 1024**3:
			raise ValueError("Packed payload exceeds the 1 GiB allocation limit")
		buffer = self.empty(layout.nbytes, dtype=torch.uint8, device=device)
		request = self.irecv(buffer, src=src, name=name, epoch=epoch)
		request.tensors = layout.views(buffer)
		return request

	def recv_tensors(self, specs, src, *, device=None, name="packed", epoch=None):
		request = self.irecv_tensors(specs, src, device=device, name=name, epoch=epoch)
		request.wait()
		return request.tensors

	def _wire_empty(self, tensor, device):
		return torch.empty(
			tensor.shape, dtype=tensor.dtype, device=device, pin_memory=device.type == "cpu" and tensor.is_cuda
		)

	def _stage_send(self, tensor, wire_device):
		if tensor.device == wire_device:
			return tensor.contiguous()
		wire = self._wire_empty(tensor, wire_device)
		wire.copy_(tensor, non_blocking=True)
		if wire_device.type == "cpu":
			# Gloo may read CPU bytes immediately; D2H must have landed first.
			torch.cuda.current_stream(tensor.device).synchronize()
		return wire

	def _track(self, request: CommRequest) -> CommRequest:
		self._pending = [r for r in self._pending if not r._done]
		self._pending.append(request)
		return request

	def _log_req(self, op, peer, tensor, group):
		self._seq += 1
		logging.debug(
			"COMM[%s] %s: rank=%s, peer=%s, shape=%s, dtype=%s, device=%s, backend=%s",
			self._seq,
			op,
			self.rank,
			peer,
			tuple(tensor.shape),
			tensor.dtype,
			tensor.device,
			dist.get_backend(group),
		)

	def send(self, tensor: Tensor, dst: int, *, epoch=None):
		self.isend(tensor, dst=dst, epoch=epoch).wait()

	def recv(self, tensor: Tensor, src: int, *, epoch=None):
		self.irecv(tensor, src=src, epoch=epoch).wait()
		return tensor

	def isend(self, tensor: Tensor, dst: int, *, name: str = "", epoch=None) -> CommRequest:
		with self._posting:
			return self._isend(tensor, dst, name=name, epoch=epoch)

	def _isend(self, tensor: Tensor, dst: int, *, name: str, epoch) -> CommRequest:
		group = self._peer_group(dst, epoch)
		wire_device = self._wire_device(group)
		wire = self._stage_send(tensor, wire_device)
		self._log_req("isend", dst, wire, group)
		work = dist.isend(wire, dst=dst, group=group)
		if work is None:
			raise RuntimeError("Sender is not a member of the selected communication group")
		return self._track(
			CommRequest(
				work,
				keepalive=[tensor, wire],
				name=name,
				cuda_device=wire_device if wire_device.type == "cuda" else None,
				epoch=epoch,
			)
		)

	def irecv(self, tensor: Tensor, src: int, *, name: str = "", epoch=None) -> CommRequest:
		with self._posting:
			return self._irecv(tensor, src, name=name, epoch=epoch)

	def _irecv(self, tensor: Tensor, src: int, *, name: str, epoch) -> CommRequest:
		group = self._peer_group(src, epoch)
		wire_device = self._wire_device(group)
		wire = (
			tensor if tensor.device == wire_device and tensor.is_contiguous() else self._wire_empty(tensor, wire_device)
		)
		self._log_req("irecv", src, wire, group)
		work = dist.irecv(wire, src=src, group=group)
		if work is None:
			raise RuntimeError("Receiver is not a member of the selected communication group")
		finalize = None
		if wire is not tensor:

			def finalize():
				tensor.copy_(wire, non_blocking=False)

		cuda_device = tensor.device if tensor.is_cuda else wire_device if wire_device.type == "cuda" else None
		return self._track(
			CommRequest(
				work,
				finalize=finalize,
				keepalive=[tensor, wire],
				name=name,
				tensor=tensor,
				cuda_device=cuda_device,
				epoch=epoch,
			)
		)

	def wait_all(self, reqs: List[CommRequest]):
		for request in reqs:
			request.wait()

	def _collective(self, tensor, peer, *, reduce_op=None):
		# World collectives always select the shared control group on every rank.
		if self._closed:
			raise RuntimeError("Communication manager is closed")
		wire_device = self._wire_device(self._group)
		wire = self._stage_send(tensor, wire_device)
		if reduce_op is None:
			work = dist.broadcast(wire, src=peer, group=self._group, async_op=True)
		else:
			work = dist.reduce(wire, dst=peer, op=reduce_op, group=self._group, async_op=True)
		finalize = None
		if wire is not tensor and (reduce_op is None or self.rank == peer):

			def finalize():
				tensor.copy_(wire, non_blocking=False)

		cuda_device = tensor.device if tensor.is_cuda else wire_device if wire_device.type == "cuda" else None
		CommRequest(work, finalize=finalize, keepalive=[tensor, wire], cuda_device=cuda_device).wait()
		return tensor

	def broadcast(self, tensor: Tensor, src: int):
		return self._collective(tensor, src)

	def reduce(self, tensor: Tensor, dst: int, op=dist.ReduceOp.SUM):
		return self._collective(tensor, dst, reduce_op=op)

	def exchange_rows(self, rows: Tensor, send_counts, *, epoch=None, device=None) -> RowExchange:
		"""Exchange variable row counts and return ready data grouped by source rank.

		Input rows are grouped by destination in epoch membership order (world order
		by default). All members call with the same width and dtype. CPU/GPU staging,
		count exchange, byte transport, and consumer readiness remain internal.
		"""
		with self._posting:
			if self._closed:
				raise RuntimeError("Communication manager is closed")
			if epoch is None:
				group, ranks = self._group, tuple(range(self.world_size))
			else:
				record = self._epoch(epoch)
				if self.rank not in record.members:
					raise ValueError("Local rank is not a member of the communication epoch")
				group, ranks = record.handle, record.members
			counts = tuple(send_counts)
			if (
				len(counts) != len(ranks)
				or any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts)
				or rows.ndim != 2
				or rows.layout != torch.strided
				or sum(counts) != rows.shape[0]
			):
				raise ValueError("Row counts must partition a strided matrix in group rank order")
			dtypes = (
				torch.bool,
				torch.uint8,
				torch.int8,
				torch.int16,
				torch.int32,
				torch.int64,
				torch.float16,
				torch.bfloat16,
				torch.float32,
				torch.float64,
				torch.complex64,
				torch.complex128,
			)
			if rows.dtype not in dtypes:
				raise ValueError("Unsupported row exchange dtype")
			wire_device = self._wire_device(group)
			target_device = self.device if device is None else torch.device(device)
			if target_device.type not in ("cpu", "cuda"):
				raise ValueError("Row destination must be CPU or CUDA")
			schema = torch.tensor([rows.shape[1], dtypes.index(rows.dtype)], dtype=torch.int64, device=wire_device)
			schemas = [torch.empty_like(schema) for _ in ranks]
			dist.all_gather(schemas, schema, group=group)
			if any(not torch.equal(other, schema) for other in schemas):
				raise ValueError("Row exchange schema differs across group members")
			send_sizes = torch.tensor(counts, dtype=torch.int64, device=wire_device)
			recv_sizes = torch.empty_like(send_sizes)
			dist.all_to_all_single(recv_sizes, send_sizes, group=group)
			received = tuple(recv_sizes.cpu().tolist())
			row_bytes = rows.shape[1] * rows.element_size()
			wire_rows = self._stage_send(rows, wire_device)
			wire_input = wire_rows.reshape(-1).view(torch.uint8)
			wire_output = torch.empty(
				sum(received) * row_bytes,
				dtype=torch.uint8,
				device=wire_device,
				pin_memory=wire_device.type == "cpu" and target_device.type == "cuda",
			)
			work = dist.all_to_all_single(
				wire_output,
				wire_input,
				output_split_sizes=[count * row_bytes for count in received],
				input_split_sizes=[count * row_bytes for count in counts],
				group=group,
				async_op=True,
			)
			if work is None:
				raise RuntimeError("Local rank is not in the selected exchange group")
			wire_values = wire_output.view(rows.dtype).reshape(sum(received), rows.shape[1])
			result = wire_values if target_device == wire_device else self._wire_empty(wire_values, target_device)
			finalize = None
			if result is not wire_values:

				def finalize():
					result.copy_(wire_values, non_blocking=False)

			cuda_device = result.device if result.is_cuda else wire_device if wire_device.type == "cuda" else None
			request = self._track(
				CommRequest(
					work,
					finalize=finalize,
					keepalive=[rows, wire_rows, wire_input, wire_output, result],
					tensor=result,
					cuda_device=cuda_device,
					epoch=epoch,
					name="exchange-rows",
				)
			)
		request.wait()
		return RowExchange(result, received, ranks)

	def destroy(self):
		if self._closed:
			return
		for request in self._pending:
			request.wait()
		self._pending.clear()
		for group in reversed(self._owned_groups):
			dist.destroy_process_group(group)
		if self._owns_default_group:
			dist.destroy_process_group()
		self._closed = True

	def abort(self) -> None:
		"""Release owned transport groups without waiting on failed requests.

		The runtime first stops its worker processes. Retain request buffers until
		backend teardown has finished; retries use a fresh rendezvous and manager.
		"""
		with self._posting:
			if self._closed:
				return
			self._closed = True
			if self._owns_default_group:
				if dist.is_initialized():
					dist.destroy_process_group()
			else:
				for group in reversed(self._owned_groups):
					dist.destroy_process_group(group)
			self._owned_groups.clear()
			self._epochs.clear()
			self._retiring.clear()
			self._pending.clear()
