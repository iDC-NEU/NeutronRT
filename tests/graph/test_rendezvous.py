import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lib.dist.comm_manager import CommManager


def _repeat_world(rank, port):
	torch.set_num_threads(1)
	store = dist.TCPStore("127.0.0.1", port, 2, rank == 0, timeout=timedelta(seconds=20))
	os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "True"
	os.environ["TORCHELASTIC_RESTART_COUNT"] = "0"
	for iteration in range(3):
		manager = CommManager.initialize(
			2,
			rank,
			init_method=f"tcp://127.0.0.1:{port}",
			rendezvous_prefix=f"trial-{iteration}",
			timeout=timedelta(seconds=20),
		)
		try:
			if rank == 0:
				manager.send(torch.tensor([iteration]), 1)
			else:
				assert manager.recv_tensor(1, src=0, dtype=torch.int64).item() == iteration
		finally:
			manager.destroy()
	assert store is not None


@pytest.mark.host
def test_repeated_worlds_use_fresh_keys_in_a_persistent_launcher_store():
	with socket.socket() as listener:
		listener.bind(("127.0.0.1", 0))
		port = listener.getsockname()[1]
	mp.spawn(_repeat_world, args=(port,), nprocs=2, join=True)
