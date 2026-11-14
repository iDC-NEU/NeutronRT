"""Resolve compact sum tables and publish only a newly reconstructed embedding."""

import logging
from contextlib import ExitStack
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lib.runtime.aggregates.reconstruction import ReconstructionHeader, validate_recipe
from lib.runtime.aggregates.tables import receive_tables, resolve_tables
from lib.runtime.protocol import ERROR, READY

if TYPE_CHECKING:
	from lib.runtime.worker import OwnedWorker


@torch.inference_mode()
def reconstruct(worker: "OwnedWorker", command: Tensor) -> None:
	if worker._prepared is not None:
		raise ValueError("Reconstruction cannot replace retained prepared inputs")
	header = ReconstructionHeader.decode(command)
	if header.generation != worker.owner.generation or header.layer > worker.model.num_layers:
		raise ValueError("Reconstruction refers to an invalid worker generation or model layer")
	fields, tail = receive_tables(worker, header.nbytes, header.count, extra_specs=header.tail_specs())
	version, chunk = tail[0].cpu().tolist()
	if min(version, chunk) < 0:
		raise ValueError("Invalid reconstruction state identity")
	widths = (
		worker.model.num_features,
		*((worker.model.num_hiddens,) * (worker.model.num_layers - 1)),
		worker.model.num_classes,
	)
	width = widths[header.layer]
	with ExitStack() as stack:
		# Drain routed inputs even for a retry, preserving each peer's data FIFO.
		tables = resolve_tables(worker, fields, stack)
		handle = None
		try:
			validate_recipe(header, fields, widths)
			if worker._aggregate_tasks is not None and worker._aggregate_tasks.owns_request(header.request):
				raise ValueError("A paired attempt cannot be reused for reconstruction")
			previous = worker._attempts.get(header.request)
			if previous is not None:
				if previous[:3] != (version, header.layer, chunk):
					raise ValueError("A reconstruction attempt cannot name two state keys")
				with worker.store.acquire(previous[3], wait=True) as values:
					if values.shape != (header.rows, width):
						raise ValueError("A reconstruction retry changed its output shape")
				worker._reply(READY, header, previous[3], version, chunk, width)
				return
			values = worker._reconstruct_sage(header.start_layer, tables, tail[1])
			if values.shape != (header.rows, width) or values.dtype != torch.float32:
				raise ValueError("Reconstructed output does not match the selected rows")
			worker._drain()
			try:
				handle = worker.store.put(values)
			except MemoryError:
				worker._drain(all_=True)
				handle = worker.store.put(values)
			with worker.store.acquire(handle, wait=True):
				worker.store.mark_published(handle)
		except Exception as error:
			if handle is not None:
				worker.store.abandon(handle)
			logging.exception("Aggregate reconstruction failed: rank=%s, attempt=%s", worker.owner.rank, header.request)
			worker._reply(ERROR, header, 1 if isinstance(error, MemoryError) else 2, version, chunk, width)
			return
		worker._attempts[header.request] = version, header.layer, chunk, handle
		worker._handles[handle] = header.request
		worker._reply(READY, header, handle, version, chunk, width)
