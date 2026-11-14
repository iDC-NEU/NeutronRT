"""Execute compact sparse updates through shared paired readiness and replay."""

from contextlib import ExitStack
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lib.runtime.aggregates.delta import DeltaHeader, validate_delta_tables
from lib.runtime.aggregates.tables import receive_tables, resolve_tables
from lib.runtime.aggregates.worker import AggregateTasks, _Result

if TYPE_CHECKING:
	from lib.runtime.worker import OwnedWorker


@torch.inference_mode()
def compute_delta(worker: "OwnedWorker", command: Tensor) -> None:
	if worker._prepared is not None:
		raise ValueError("A sparse delta cannot replace retained prepared inputs")
	header = DeltaHeader.decode(command)
	if header.generation != worker.owner.generation or header.layer > worker.model.num_layers:
		raise ValueError("Sparse delta refers to an invalid generation or model layer")
	fields, tail = receive_tables(worker, header.nbytes, 5, extra_specs=header.tail_specs())
	version, chunk, request, aggregate_chunk = tail[0].cpu().tolist()
	if min(version, chunk, aggregate_chunk) < 0 or request < 1 or request == header.request:
		raise ValueError("Invalid sparse paired output identity")
	widths = (
		worker.model.num_features,
		*((worker.model.num_hiddens,) * (worker.model.num_layers - 1)),
		worker.model.num_classes,
	)
	identity = _Result(
		header, replace(header, request=request), version, chunk, aggregate_chunk, 0, 0, widths[header.layer]
	)
	if worker._aggregate_tasks is None:
		worker._aggregate_tasks = AggregateTasks(worker)
	with ExitStack() as stack:
		tables = resolve_tables(worker, fields, stack)

		def apply() -> tuple[Tensor, Tensor]:
			validate_delta_tables(header, fields, widths)
			return worker._apply_delta(header.layer, tables, tail)

		worker._aggregate_tasks._execute(identity, apply)
