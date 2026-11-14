from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp

from lib.dist.comm_manager import CommManager
from tests.dist.test_transport import _join_processes


def _rows(source, counts, dtype):
	values = [
		[source * 100 + target * 10 + row, source * 100 + target * 10 + row + 0.5]
		for target, count in enumerate(counts)
		for row in range(count)
	]
	result = torch.tensor(values, dtype=torch.float64).reshape(-1, 2)
	if dtype == torch.int64:
		result = result.long() + 2**60
	elif dtype == torch.bool:
		result = result.remainder(2) > 0
	else:
		result = result.to(dtype)
	return result


def _exchange(rank, rendezvous, use_cuda):
	torch.set_num_threads(1)
	devices = ("cpu", "cuda:0", "cuda:1") if use_cuda else ("cpu", "cpu", "cpu")
	manager = CommManager.initialize(
		3, rank, device=devices[rank], init_method=f"file://{rendezvous}", timeout=timedelta(seconds=20)
	)
	try:
		counts = [[0, 2, 3], [1, 0, 1], [0, 2, 0]]
		for dtype in (torch.float32, torch.int64, torch.bool):
			rows = _rows(rank, counts[rank], dtype).to(devices[rank]).t().contiguous().t()
			result = manager.exchange_rows(rows, counts[rank])
			expected = torch.cat(
				[
					_rows(source, counts[source], dtype)[sum(counts[source][:rank]) : sum(counts[source][: rank + 1])]
					for source in range(3)
				]
			)
			assert result.ranks == (0, 1, 2)
			assert result.counts == tuple(counts[source][rank] for source in range(3))
			assert result.data.device == torch.device(devices[rank])
			assert torch.equal(result.data.cpu(), expected)
		zero = manager.exchange_rows(torch.empty(0, 2, device=devices[rank]), [0, 0, 0])
		assert zero.data.shape == (0, 2)
		assert zero.counts == (0, 0, 0)
		with pytest.raises(ValueError, match="schema"):
			manager.exchange_rows(torch.empty(0, 3 if rank == 1 else 2, device=devices[rank]), [0, 0, 0])
		# A rejected schema does not issue a mismatched data collective.
		manager.exchange_rows(torch.empty(0, 2, device=devices[rank]), [0, 0, 0])
		epoch = manager.create_epoch([1, 2], timeout=timedelta(seconds=20))
		if rank in (1, 2):
			local = rank - 1
			counts = [[0, 2], [3, 0]]
			for target_device in (devices[rank], "cpu"):
				rows = _rows(local, counts[local], torch.int64).to(devices[rank])
				result = manager.exchange_rows(rows, counts[local], epoch=epoch, device=target_device)
				expected = torch.cat(
					[
						_rows(source, counts[source], torch.int64)[
							sum(counts[source][:local]) : sum(counts[source][: local + 1])
						]
						for source in range(2)
					]
				)
				assert result.ranks == (1, 2)
				assert result.data.device == torch.device(target_device)
				assert torch.equal(result.data.cpu(), expected)
		manager.broadcast(torch.tensor([1]), src=0)
		manager.retire_epoch(epoch)
	finally:
		manager.destroy()


@pytest.mark.parametrize("use_cuda", [False, True])
def test_variable_row_exchange_and_subgroup_staging(tmp_path, use_cuda):
	if use_cuda and torch.cuda.device_count() < 2:
		pytest.skip("Requires two visible GPUs")
	context = mp.spawn(_exchange, args=(str(tmp_path / "store"), use_cuda), nprocs=3, join=False)
	_join_processes(context, timeout=90)
