"""Aligned mixed-dtype payload layouts; integer and floating bits never convert."""

from dataclasses import dataclass
from math import prod

import torch


@dataclass(frozen=True)
class TensorSpec:
	shape: tuple
	dtype: torch.dtype

	@classmethod
	def of(cls, tensor):
		return cls(tuple(tensor.shape), tensor.dtype)


class PackedLayout:
	VERSION = 1

	def __init__(self, specs, *, version=VERSION):
		# The enclosing task/control protocol selects this schema version.
		if version != self.VERSION:
			raise ValueError("Unsupported packed payload schema version")
		self.version = version
		self.specs = tuple(specs)
		self.slices = []
		self.padding = []
		cursor = 0
		for spec in self.specs:
			if any(not isinstance(size, int) or isinstance(size, bool) for size in spec.shape):
				raise ValueError("Payload dimensions must be integers")
			if any(size < 0 for size in spec.shape):
				raise ValueError("Negative payload dimensions")
			size = torch.empty(0, dtype=spec.dtype).element_size()
			start = (cursor + size - 1) // size * size
			if start > cursor:
				self.padding.append((cursor, start))
			end = start + prod(spec.shape) * size
			self.slices.append((start, end))
			cursor = end
		self.nbytes = cursor

	def views(self, buffer):
		if (
			buffer.dtype != torch.uint8
			or buffer.ndim != 1
			or buffer.numel() != self.nbytes
			or not buffer.is_contiguous()
		):
			raise ValueError("Packed buffer does not match its layout")
		if any(buffer.storage_offset() % torch.empty(0, dtype=spec.dtype).element_size() for spec in self.specs):
			raise ValueError("Packed buffer base offset is not aligned")
		return [
			buffer[begin:end].view(spec.dtype).reshape(spec.shape)
			for spec, (begin, end) in zip(self.specs, self.slices, strict=True)
		]
