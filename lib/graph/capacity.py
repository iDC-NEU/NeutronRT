"""Device-independent logical resident capacity admission."""

from dataclasses import dataclass
from typing import Callable

import torch


class CapacityAdmissionError(MemoryError):
	"""A logical resident envelope cannot be admitted."""


@dataclass
class CapacityPolicy:
	capacity_bytes: int
	device: torch.device
	reserved_bytes: int
	reserved_edge_count: int
	physical_checks: int = 0

	@classmethod
	def create(
		cls,
		capacity_bytes: int,
		device: torch.device,
		initial_bytes: int,
		initial_edges: int,
		estimate: Callable[[int], int],
	) -> "CapacityPolicy":
		if initial_bytes > capacity_bytes:
			raise CapacityAdmissionError(
				f"Resident admission needs {initial_bytes} bytes but capacity is {capacity_bytes}"
			)
		# A trace-independent headroom envelope. If the declared capacity cannot
		# cover it, the current state remains admissible and growth checks later.
		headroom = max(4096, initial_edges // 8)
		reserved_edges = initial_edges + headroom
		reserved = initial_bytes
		try:
			reserved = estimate(reserved_edges)
		except Exception:
			reserved_edges = initial_edges
		if reserved > capacity_bytes:
			reserved_edges = initial_edges
			reserved = initial_bytes
		policy = cls(capacity_bytes, device, reserved, reserved_edges)
		if device.type == "cuda":
			free, _ = torch.cuda.mem_get_info(device)
			policy.physical_checks = 1
			if initial_bytes > free:
				raise CapacityAdmissionError(
					f"Resident admission needs {initial_bytes} bytes but only {free} bytes are free on {device}"
				)
		return policy

	def admit(self, edge_count: int, estimate: Callable[[int], int]) -> int:
		logical = estimate(edge_count)
		if logical > self.capacity_bytes:
			raise CapacityAdmissionError(
				f"Resident transition needs {logical} bytes but capacity is {self.capacity_bytes}"
			)
		if edge_count <= self.reserved_edge_count and logical <= self.reserved_bytes:
			return logical
		if self.device.type == "cuda":
			free, _ = torch.cuda.mem_get_info(self.device)
			self.physical_checks += 1
			if logical > free:
				raise CapacityAdmissionError(
					f"Resident transition needs {logical} bytes but only {free} bytes are free on {self.device}"
				)
		self.reserved_edge_count = max(self.reserved_edge_count, edge_count)
		self.reserved_bytes = max(self.reserved_bytes, logical)
		return logical
