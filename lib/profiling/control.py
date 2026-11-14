"""Generation-fenced profile control, separate from application task completion."""

import json
from typing import List

import torch

from lib.graph.ownership import Owner
from lib.runtime.protocol import PROFILE
from lib.runtime.transport import OwnedTransport

_MAX_REPORT_BYTES = 8 * 1024**2


def _exchange(transport: OwnedTransport, owner: Owner, action: int, trace: bool, limit: int) -> dict:
	with transport.locked(owner):
		transport.comm.send(
			torch.tensor([PROFILE, action, owner.generation, int(trace), limit, 0, 0, 0]),
			owner.rank,
			epoch=transport.control_epoch,
		)
		header = transport.comm.recv_tensor(4, src=owner.rank, dtype=torch.int64, epoch=transport.control_epoch)
		opcode, operation, generation, size = header.tolist()
		if (opcode, operation, generation) != (PROFILE, action, owner.generation):
			raise RuntimeError("Worker profile response belongs to another operation or generation")
		if action == 1:
			if size:
				raise RuntimeError("Profile start acknowledgement cannot contain a report")
			return {}
		if not 0 < size <= _MAX_REPORT_BYTES:
			raise RuntimeError("Worker profile report exceeds its byte limit")
		payload = transport.comm.recv_tensor(size, src=owner.rank, dtype=torch.uint8, epoch=transport.control_epoch)
		report = json.loads(payload.numpy().tobytes())
		if not isinstance(report, dict):
			raise RuntimeError("Worker profile report must be a JSON object")
		if report.get("rank") != owner.rank or report.get("generation") != owner.generation:
			raise RuntimeError("Worker profile identity differs from its transport owner")
		return report


def start_workers(transport: OwnedTransport, *, trace: bool = False, trace_task_limit: int = 100) -> None:
	if not 1 <= trace_task_limit <= 1000:
		raise ValueError("Worker trace limit must be between 1 and 1000 model tasks")
	for owner in transport.endpoints:
		_exchange(transport, owner, 1, trace, trace_task_limit)


def stop_workers(transport: OwnedTransport) -> List[dict]:
	return [_exchange(transport, owner, 2, False, 0) for owner in transport.endpoints]
