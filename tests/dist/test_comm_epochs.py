from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lib.dist.buffers import TensorSpec
from lib.dist.comm_manager import CommManager
from tests.dist.test_transport import _join_processes


def _epochs(rank, rendezvous, use_cuda):
	torch.set_num_threads(1)
	devices = ("cpu", "cuda:0", "cuda:1") if use_cuda else ("cpu", "cpu", "cpu")
	manager = CommManager.initialize(
		3, rank, device=devices[rank], init_method=f"file://{rendezvous}", timeout=timedelta(seconds=20)
	)
	try:
		old = manager.create_epoch([0, 1], timeout=timedelta(seconds=20))
		old_request = manager.isend(torch.tensor([17]), 1, epoch=old) if rank == 0 else None
		if old_request is not None:
			assert not old_request.work.is_completed()
		# The old receiver is deliberately absent until after group creation.
		larger = manager.create_epoch([0, 1, 2], timeout=timedelta(seconds=20))
		smaller = manager.create_epoch([1, 2], timeout=timedelta(seconds=20))
		assert (old, larger, smaller) == (1, 2, 3)
		assert manager.epoch_members(old) == (0, 1)
		assert manager.epoch_members(larger) == (0, 1, 2)
		if rank in (1, 2):
			assert dist.get_backend(manager._epoch(smaller).handle) == ("nccl" if use_cuda else "gloo")
		if rank == 2:
			with pytest.raises(ValueError, match="endpoints"):
				manager.isend(torch.ones(1), 1, epoch=old)
		# Each endpoint selects the epoch from its control protocol. A request on
		# the larger group remains pending while the old group is retired.
		new_request = peer_request = None
		values = [torch.tensor([2**60 + 11]), torch.tensor([True, False]), torch.tensor([-0.0, 1.5])]
		specs = [TensorSpec.of(value) for value in values]
		if rank == 0:
			new_request = manager.isend_tensors(values, 1, epoch=larger)
		elif rank == 1:
			old_output = manager.empty(1, dtype=torch.int64)
			old_request = manager.irecv(old_output, 0, epoch=old)
			new_request = manager.irecv_tensors(specs, 0, epoch=larger)
			peer_request = manager.isend(torch.tensor([31]), 2, epoch=smaller)
		else:
			peer_output = manager.empty(1, dtype=torch.int64)
			peer_request = manager.irecv(peer_output, 1, epoch=smaller)
		manager.retire_epoch(old)
		if old_request is not None:
			assert old_request._done
		if new_request is not None:
			assert not new_request._done
			assert new_request.keepalive
			new_request.wait()
		if rank == 1:
			assert old_output.item() == 17
			for actual, expected in zip(new_request.tensors, values, strict=True):
				assert actual.device == torch.device(devices[rank])
				assert torch.equal(actual.cpu().reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))
		if peer_request is not None:
			peer_request.wait()
		if rank == 2:
			assert peer_output.item() == 31
		with pytest.raises(ValueError, match="unknown"):
			manager._peer_group(1, old)
		# Disagreement and invalid membership are detected on all ranks before
		# anyone calls new_group, so the world can agree on another epoch later.
		with pytest.raises(ValueError, match="disagree"):
			manager.create_epoch([0, 1] if rank == 0 else [0, 2])
		with pytest.raises(ValueError, match="distinct"):
			manager.create_epoch([0, 0])
		next_epoch = manager.create_epoch([0, 1], timeout=timedelta(seconds=20))
		assert next_epoch == 4
		manager.retire_epoch(next_epoch)
		manager.retire_epoch(smaller)
		manager.retire_epoch(larger)
	finally:
		manager.destroy()


@pytest.mark.parametrize("use_cuda", [False, True])
def test_larger_and_smaller_groups_drain_old_traffic_without_stopping_new_epoch(tmp_path, use_cuda):
	if use_cuda and torch.cuda.device_count() < 2:
		pytest.skip("Requires two visible GPUs")
	context = mp.spawn(_epochs, args=(str(tmp_path / "store"), use_cuda), nprocs=3, join=False)
	_join_processes(context, timeout=90)
