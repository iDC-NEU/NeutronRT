import os
from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

from lib.dist.comm_manager import CommManager
from tests.dist.test_transport import _join_processes


def _crash(rank, rendezvous):
	torch.set_num_threads(1)
	manager = CommManager.initialize(2, rank, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=10))
	control = manager.create_epoch([0, 1], timeout=timedelta(seconds=10))
	if rank:
		manager.send(torch.tensor([17]), 0, epoch=control)
		os._exit(0)
	assert manager.recv_tensor(1, dtype=torch.int64, src=1, epoch=control).item() == 17
	with pytest.raises(RuntimeError):
		manager.irecv(torch.empty(1), 1).wait()
	manager.abort()
	assert manager._closed and not manager._pending
	manager.abort()
	fresh = CommManager.initialize(1, 0, init_method=f"file://{rendezvous}-fresh", timeout=timedelta(seconds=10))
	fresh.destroy()


def test_failed_worker_requests_can_be_abandoned_before_a_fresh_world(tmp_path):
	context = mp.spawn(_crash, args=(str(tmp_path / "store"),), nprocs=2, join=False)
	_join_processes(context, timeout=45)
