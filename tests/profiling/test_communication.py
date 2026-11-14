from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.profiling.communication import ProfiledCommManager
from tests.dist.test_transport import _join_processes


def _traffic(rank, rendezvous, use_cuda):
	torch.set_num_threads(1)
	manager = ProfiledCommManager.initialize(
		2,
		rank,
		device="cuda:0" if rank and use_cuda else "cpu",
		init_method=f"file://{rendezvous}",
		timeout=timedelta(seconds=15),
	)
	control = manager.create_epoch([0, 1], timeout=timedelta(seconds=15))
	values = (
		torch.tensor([2**60 + 3]),
		torch.tensor([True, False, True]),
		torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2],
	)
	specs = tuple(TensorSpec.of(value) for value in values)
	packed_bytes = PackedLayout(specs).nbytes
	try:
		if rank == 0:
			manager.send(torch.ones(1), 1)
		else:
			manager.recv_tensor(1, src=0)
		manager.recorder.start()
		dist.barrier()
		if rank == 0:
			request = manager.isend_tensors(values, 1)
			request.wait()
			request.wait()
			assert request.is_completed()
			assert manager.recv_tensor(1, src=1, dtype=torch.int64, epoch=control).item() == 17
		else:
			actual = manager.recv_tensors(specs, 0)
			for output, expected in zip(actual, values, strict=True):
				assert torch.equal(output.cpu(), expected)
			manager.send(torch.tensor([17]), 0, epoch=control)
		report = manager.recorder.stop()
		counters = report["counters"]
		body = f"p2p.{'send' if rank == 0 else 'recv'}.gloo.peer{1 - rank}.default"
		ack = f"p2p.{'recv' if rank == 0 else 'send'}.gloo.peer{1 - rank}.epoch{control}"
		assert counters[f"{body}.operations"] == 1 and counters[f"{body}.payload_bytes"] == packed_bytes
		assert counters[f"{ack}.operations"] == 1 and counters[f"{ack}.payload_bytes"] == 8
		assert report["timings"]["communication.wait"]["count"] == 2
		assert report["open_spans"] == 0
		if rank == 0:
			assert counters["pack.source_bytes"] == 27 and counters["pack.padding_bytes"] == 1
			assert counters["pack.temporary_copy_bytes"] == 16
			assert counters.get("staging.pack_bytes", 0) == 0
		elif use_cuda:
			assert counters["staging.recv_bytes"] == packed_bytes
		else:
			assert counters.get("staging.recv_bytes", 0) == 0
	finally:
		manager.destroy()


@pytest.mark.parametrize("use_cuda", [False, True])
def test_real_payload_counts_exclude_warmup_and_preserve_mixed_dtype_and_device_completion(tmp_path, use_cuda):
	if use_cuda and not torch.cuda.is_available():
		pytest.skip("Requires one visible GPU")
	context = mp.spawn(_traffic, args=(str(tmp_path / "store"), use_cuda), nprocs=2, join=False)
	_join_processes(context, timeout=60)
