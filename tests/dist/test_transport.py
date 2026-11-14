"""Request ownership and real transport round trips, including CUDA when visible."""

from datetime import timedelta
from time import monotonic
from unittest.mock import Mock

import pytest
import torch
import torch.multiprocessing as mp

from lib.dist.comm_manager import CommManager, CommRequest


def test_request_finalizes_once_and_retains_buffers_until_done():
	work = Mock()
	work.is_completed.side_effect = [False, True]
	finalize = Mock()
	buffer = torch.ones(4)
	request = CommRequest(work, finalize=finalize, keepalive=[buffer])
	assert not request.is_completed()
	assert request.keepalive == [buffer]
	assert request.is_completed()
	assert request.wait()
	assert request.is_completed()
	work.wait.assert_called_once()
	finalize.assert_called_once()
	assert not request.keepalive


def test_request_wait_failure_retains_buffers():
	work = Mock()
	work.wait.side_effect = RuntimeError("transport failed")
	finalize = Mock()
	request = CommRequest(work, finalize=finalize, keepalive=[torch.ones(4)])
	with pytest.raises(RuntimeError, match="transport failed"):
		request.wait()
	assert request.keepalive
	finalize.assert_not_called()


def test_cuda_poll_waits_for_copy_event(monkeypatch):
	work = Mock()
	work.is_completed.return_value = True
	event = Mock()
	event.query.side_effect = [False, True]
	monkeypatch.setattr(torch.cuda, "device", lambda _: __import__("contextlib").nullcontext())
	monkeypatch.setattr(torch.cuda, "Event", lambda: event)
	monkeypatch.setattr(torch.cuda, "current_stream", lambda _: "consumer-stream")
	request = CommRequest(work, keepalive=[torch.ones(1)], cuda_device=torch.device("cuda:0"))
	assert not request.is_completed()
	assert request.keepalive
	assert request.is_completed()
	work.wait.assert_called_once()
	event.record.assert_called_once_with("consumer-stream")
	assert not request.keepalive


def test_endpoint_routing_does_not_depend_on_message_placement():
	for rank in range(3):
		manager = CommManager(
			3, rank, group="cpu", device="cpu", peer_devices=["cpu", "cuda", "cuda"], cuda_group="gpu"
		)
		for peer in range(3):
			expected = "gpu" if rank in (1, 2) and peer in (1, 2) else "cpu"
			assert manager._peer_group(peer) == expected
		with pytest.raises(ValueError, match="Invalid peer"):
			manager._peer_group(3)


def _join_processes(context, timeout=60):
	deadline = monotonic() + timeout
	try:
		while monotonic() < deadline:
			if context.join(timeout=1):
				return
		raise TimeoutError("Distributed test exceeded its deadline")
	finally:
		for process in context.processes:
			if process.is_alive():
				process.terminate()
		for process in context.processes:
			process.join(timeout=5)


def _cpu_roundtrip(rank, rendezvous, backend):
	torch.set_num_threads(1)
	manager = CommManager.initialize(
		2, rank, backend=backend, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=20)
	)
	try:
		for dtype in (torch.int64, torch.float32, torch.bool):
			source = (torch.arange(24).reshape(3, 8) % 3).to(dtype)[:, ::2]
			if rank == 0:
				manager.wait_all([
					manager.isend(torch.tensor([3, 4]), dst=1),
					manager.isend(source, dst=1),
				])
				result = manager.recv_tensor((3, 4), dtype=dtype, src=1)
				torch.testing.assert_close(result, source)
			else:
				header = manager.recv_tensor(2, dtype=torch.int64, src=0, device="cpu")
				assert header.tolist() == [3, 4]
				target = manager.empty((4, 3), dtype=dtype).t()
				request = manager.irecv(target, src=0)
				assert request.tensor is target
				request.wait()
				assert request.is_completed()
				torch.testing.assert_close(target, source)
				manager.send(target, dst=0)
		if rank == 0:
			manager.send(torch.empty(0), dst=1)
		else:
			assert manager.recv_tensor(0, src=0).numel() == 0
		value = torch.tensor([7 if rank == 0 else 0])
		manager.broadcast(value, src=0)
		assert value.item() == 7
		value = torch.tensor([rank + 1.0])
		manager.reduce(value, dst=0)
		if rank == 0:
			assert value.item() == 3
	finally:
		manager.destroy()
	manager.destroy()
	with pytest.raises(RuntimeError, match="closed"):
		manager.send(torch.ones(1), dst=1 - rank)


@pytest.mark.parametrize("backend", ["auto", "gloo"])
def test_cpu_transport_roundtrip(tmp_path, backend):
	context = mp.spawn(_cpu_roundtrip, args=(str(tmp_path / "store"), backend), nprocs=2, join=False)
	_join_processes(context)


def _gpu_roundtrip(rank, rendezvous):
	torch.set_num_threads(1)
	device = torch.device("cpu") if rank == 0 else torch.device("cuda", rank - 1)
	manager = CommManager.initialize(
		3, rank, device=device, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=30)
	)
	try:
		if rank == 0:
			for peer in (1, 2):
				manager.send(torch.arange(32, dtype=torch.float32), dst=peer)
				result = manager.recv_tensor(32, src=peer)
				torch.testing.assert_close(result, torch.arange(32, dtype=torch.float32) + peer)
		else:
			result = manager.recv_tensor(32, src=0)
			assert result.device == device
			manager.send(result + rank, dst=0)
		if rank == 1:
			# GPU peers select NCCL even when the application message is on CPU.
			manager.send(torch.arange(32, dtype=torch.float32), dst=2)
			stream = torch.cuda.Stream(device=device)
			with torch.cuda.stream(stream):
				manager.send(torch.arange(32, device=device, dtype=torch.float32) + 10, dst=2)
		elif rank == 2:
			assert manager._peer_group(1) is manager._cuda_group
			result = manager.recv_tensor(32, src=1)
			torch.testing.assert_close(result.cpu(), torch.arange(32, dtype=torch.float32))
			result = manager.recv_tensor(32, src=1, device="cpu")
			torch.testing.assert_close(result, torch.arange(32, dtype=torch.float32) + 10)
		value = manager.empty(1)
		value.fill_(12 if rank == 0 else 0)
		manager.broadcast(value, src=0)
		assert value.item() == 12
	finally:
		manager.destroy()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two visible GPUs")
def test_cpu_gpu_and_gpu_peer_transport(tmp_path):
	context = mp.spawn(_gpu_roundtrip, args=(str(tmp_path / "store"),), nprocs=3, join=False)
	_join_processes(context)


def _payload(device):
	return [
		torch.tensor([[2**60 + 7, -(2**60)], [13, 19]], dtype=torch.int64, device=device).t(),
		torch.tensor([[1.5, float("nan")], [-0.0, float("inf")]], device=device).t(),
		torch.tensor([True, False, True], device=device),
		torch.tensor(7.5, device=device),
		torch.empty((0, 3), device=device),
	]


def _check_payload(actual, device):
	for result, expected in zip(actual, _payload(device), strict=True):
		assert result.device == torch.device(device)
		assert result.shape == expected.shape and result.dtype == expected.dtype
		assert torch.equal(
			result.cpu().contiguous().reshape(-1).view(torch.uint8),
			expected.cpu().contiguous().reshape(-1).view(torch.uint8),
		)


def _packed_roundtrip(rank, rendezvous, use_cuda):
	from lib.dist.buffers import TensorSpec

	torch.set_num_threads(1)
	devices = ("cpu", "cuda:0", "cuda:1") if use_cuda else ("cpu", "cpu", "cpu")
	manager = CommManager.initialize(
		3, rank, device=devices[rank], init_method=f"file://{rendezvous}", timeout=timedelta(seconds=30)
	)
	try:
		specs = [TensorSpec.of(tensor) for tensor in _payload("cpu")]
		for source, target in ((0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)):
			if rank == source:
				manager.isend_tensors(_payload(devices[rank]), target).wait()
			elif rank == target:
				request = manager.irecv_tensors(specs, source)
				request.wait()
				_check_payload(request.tensors, devices[rank])
				addresses = [value.untyped_storage().data_ptr() for value in request.tensors]
				assert len(set(addresses)) == 1
			manager.broadcast(torch.tensor([1]), src=0)
	finally:
		manager.destroy()


@pytest.mark.parametrize("use_cuda", [False, True])
def test_single_p2p_payload_preserves_mixed_types_cpu_gpu_and_peers(tmp_path, use_cuda):
	if use_cuda and torch.cuda.device_count() < 2:
		pytest.skip("Requires two visible GPUs")
	context = mp.spawn(_packed_roundtrip, args=(str(tmp_path / "store"), use_cuda), nprocs=3, join=False)
	_join_processes(context)


def test_packing_issues_one_send_and_keeps_inputs_alive(monkeypatch):
	monkeypatch.setattr("torch.distributed.get_backend", lambda _: "gloo")
	send = Mock(return_value=Mock())
	monkeypatch.setattr("torch.distributed.isend", send)
	manager = CommManager(2, 0, group="test")
	values = _payload("cpu")
	request = manager.isend_tensors(values, 1)
	assert send.call_count == 1
	assert all(any(held is value for held in request.keepalive) for value in values)
	assert send.call_args.args[0].dtype == torch.uint8
	request.wait()
	assert not request.keepalive


def test_wait_and_poll_concurrently_finalize_once():
	from concurrent.futures import ThreadPoolExecutor

	work = Mock()
	work.is_completed.return_value = True
	finalize = Mock()
	request = CommRequest(work, finalize=finalize, keepalive=[torch.ones(1)])
	with ThreadPoolExecutor(max_workers=2) as pool:
		assert all(pool.map(lambda index: request.wait() if index % 2 else request.is_completed(), range(20)))
	finalize.assert_called_once()
	work.wait.assert_called_once()
