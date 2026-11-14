"""GPU-directed sparse reads from host rows, without replacing their storage."""

import torch
from dgl import ndarray
from dgl.utils import gather_pinned_tensor_rows


class PinnedHostRows:
	"""Register existing CPU rows; keep the tensor and DGL view alive until close.

	The caller fences mutations and waits for all GPU consumers before close.
	Registration uses the existing allocation, so authoritative CPU updates remain
	visible to subsequent gathers. This object never calls Tensor.pin_memory(),
	which would allocate a separate, potentially stale copy.
	"""

	def __init__(self, values):
		if values.device.type != "cpu" or values.ndim != 2 or not values.is_contiguous():
			raise ValueError("Pinned host rows require a contiguous CPU matrix")
		self.values = values
		self._view = ndarray.from_dlpack(torch.utils.dlpack.to_dlpack(values))
		self._owned_registration = not values.is_pinned()
		if self._owned_registration:
			self._view.pin_memory_()
		self.closed = False

	def take(self, rows):
		if self.closed:
			raise RuntimeError("Host row registration is closed")
		if rows.dtype != torch.int64 or rows.ndim != 1:
			raise ValueError("Row IDs must be an int64 vector")
		if rows.numel() and (rows.min() < 0 or rows.max() >= self.values.shape[0]):
			raise ValueError("Host row ID out of range")
		return gather_pinned_tensor_rows(self.values, rows) if rows.is_cuda else self.values[rows]

	def close(self):
		if not self.closed:
			if self._owned_registration:
				self._view.unpin_memory_()
			self.closed = True

	def __enter__(self):
		return self

	def __exit__(self, *args):
		self.close()
