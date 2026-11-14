"""Opt-in P2P payload, packing, staging, and wait accounting."""

from typing import Optional

import torch.distributed as dist
from torch import Tensor

from lib.dist.comm_manager import CommManager, CommRequest
from lib.profiling.recorder import Recorder


class _MeasuredRequest(CommRequest):
	def __init__(self, request: CommRequest, recorder: Recorder) -> None:
		super().__init__(
			request.work,
			finalize=request.finalize,
			keepalive=request.keepalive,
			name=request.name,
			tensor=request.tensor,
			cuda_device=request._cuda_device,
			epoch=request.epoch,
		)
		self.recorder = recorder

	def wait(self):
		with self._lock:
			if self._done:
				return True
			channel = "control" if self.epoch is not None else "data"
			with (
				self.recorder.span("communication.wait"),
				self.recorder.span(f"communication.wait.{channel}.{self.name or 'unnamed'}"),
			):
				return super().wait()

	def is_completed(self) -> bool:
		with self._lock:
			if self._done:
				return True
			with self.recorder.span("communication.poll"):
				return super().is_completed()


class ProfiledCommManager(CommManager):
	"""Count successful P2P posts, with no extra data copy or synchronization.

	Bytes are application tensor payloads, including packed alignment padding.
	They are not NIC wire bytes/packets. Startup collectives are outside the window;
	this observer currently measures P2P operations rather than collective traffic.
	Send and receive counters describe the same traffic from different endpoints.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.recorder = Recorder()

	def _track(self, request: CommRequest) -> CommRequest:
		return super()._track(_MeasuredRequest(request, self.recorder))

	def note_p2p(self, direction: str, peer: int, tensor: Tensor, epoch: Optional[int], *, amount: int = 1) -> None:
		if not self.recorder.active:
			return
		backend = dist.get_backend(self._peer_group(peer, epoch))
		channel = "default" if epoch is None else f"epoch{epoch}"
		key = f"p2p.{direction}.{backend}.peer{peer}.{channel}"
		self.recorder.count(f"{key}.operations", amount)
		self.recorder.count(f"{key}.payload_bytes", amount * tensor.numel() * tensor.element_size())

	def _isend(self, tensor: Tensor, dst: int, *, name: str, epoch) -> CommRequest:
		with self.recorder.span("communication.send_post"):
			request = super()._isend(tensor, dst, name=name, epoch=epoch)
		self.note_p2p("send", dst, tensor, epoch)
		return request

	def _irecv(self, tensor: Tensor, src: int, *, name: str, epoch) -> CommRequest:
		with self.recorder.span("communication.recv_post"):
			request = super()._irecv(tensor, src, name=name, epoch=epoch)
		self.note_p2p("recv", src, tensor, epoch)
		wire = request.keepalive[1]
		if wire is not tensor:
			self.recorder.count("receive.copy_bytes", tensor.numel() * tensor.element_size())
			if wire.device != tensor.device:
				self.recorder.count("staging.recv_bytes", tensor.numel() * tensor.element_size())
		return request

	def _stage_send(self, tensor, wire_device):
		with self.recorder.span("communication.send_staging"):
			wire = super()._stage_send(tensor, wire_device)
		if wire is not tensor:
			self.recorder.count("send.copy_bytes", tensor.numel() * tensor.element_size())
			if wire.device != tensor.device:
				self.recorder.count("staging.send_bytes", tensor.numel() * tensor.element_size())
		return wire

	def isend_tensors(self, tensors, dst, *, name="packed", epoch=None):
		tensors = tuple(tensors)
		with self.recorder.span("communication.pack_and_post"):
			request = super().isend_tensors(tensors, dst, name=name, epoch=epoch)
		wire = request.keepalive[0]
		logical = sum(value.numel() * value.element_size() for value in tensors)
		self.recorder.count("pack.source_bytes", logical)
		self.recorder.count("pack.padding_bytes", wire.numel() - logical)
		self.recorder.count("pack.buffer_bytes", wire.numel())
		self.recorder.count(
			"pack.temporary_copy_bytes",
			sum(value.numel() * value.element_size() for value in tensors if not value.is_contiguous()),
		)
		self.recorder.count(
			"staging.pack_bytes",
			sum(value.numel() * value.element_size() for value in tensors if value.device != wire.device),
		)
		return request
