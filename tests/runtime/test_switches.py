"""Exercise both wire layouts through the real worker without opening sockets."""

from types import SimpleNamespace

import pytest
import torch

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import ComputeAttempt, Owner, StateKey
from lib.models.spec import ModelSpec
from lib.runtime.protocol import pack_compute
from lib.runtime.transport import OwnedTransport, RemoteEndpoint
from lib.runtime.worker import OwnedWorker
from tests.runtime.test_protocol import _task


class _MemoryComm:
	device = torch.device("cpu")

	def __init__(self, rank):
		self.rank = rank
		self.fields = []
		self.messages = 0

	def empty(self, shape, dtype=torch.float32):
		return torch.empty(shape, dtype=dtype)

	def wait_all(self, requests):
		for request in requests:
			request.wait()

	def isend(self, value, *_args, **_kwargs):
		self.fields.append(value)
		self.messages += 1
		return SimpleNamespace(wait=lambda: None)

	def isend_tensors(self, values, *_args, **_kwargs):
		layout = PackedLayout(tuple(TensorSpec.of(value) for value in values))
		buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
		self.fields = list(layout.views(buffer))
		for destination, value in zip(self.fields, values, strict=True):
			destination.copy_(value)
		self.messages += 1
		return SimpleNamespace(wait=lambda: None)

	def recv_tensors(self, specs, *_args, **_kwargs):
		assert len(specs) == len(self.fields)
		return tuple(self.fields)

	def recv_tensor(self, shape, **kwargs):
		value = self.fields.pop(0)
		assert tuple(value.shape) == tuple(shape) and value.dtype == kwargs["dtype"]
		return value


@pytest.mark.parametrize("name", ["sage", "gat", "gcn"])
@pytest.mark.parametrize("coalesced", [False, True])
def test_typed_and_coalesced_paths_run_identical_model_and_skip_empty_fields(name, coalesced):
	owner = Owner(1, 1)
	root, remote = _MemoryComm(0), _MemoryComm(1)
	worker = OwnedWorker(
		remote, ModelSpec(name, 4, 4, 3).build(), owner, max_bytes=4096, control_epoch=9, coalesced=coalesced
	)
	transport = OwnedTransport(root, 9, [RemoteEndpoint(owner, remote.device, 4096)], coalesced=coalesced)
	task = _task(1, degrees=name == "gcn")
	packet = pack_compute(
		task,
		ComputeAttempt(StateKey(0, 1, 1), owner, 1),
		width=4,
		features=torch.arange(12, dtype=torch.float32).reshape(3, 4),
	)
	headers = []
	root.send = lambda value, *_args, **_kwargs: headers.append(value)

	def execute(reply, *_args, **_kwargs):
		remote.fields = root.fields.copy()
		remote.send = lambda value, *_args, **_kwargs: reply.copy_(value)
		worker.compute(headers.pop(0))
		return SimpleNamespace(wait=lambda: None)

	root.irecv = execute
	result = transport.compute(owner, packet, output_width=4, supply_inputs=lambda: None)
	with worker.store.acquire(result.handle, wait=True) as actual:
		expected = worker._apply(task, packet.fields[9])
		torch.testing.assert_close(actual, expected)
	assert root.messages == (1 if coalesced else sum(bool(field.numel()) for field in packet.fields))
	assert not headers
