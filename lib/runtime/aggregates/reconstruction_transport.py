"""Submit a row-only recipe using the manager's existing typed or packed path."""

from typing import TYPE_CHECKING, Callable

import torch

from lib.graph.ownership import Owner
from lib.runtime.aggregates.reconstruction import ReconstructionPacket

if TYPE_CHECKING:
	from lib.runtime.transport import OwnedTransport, ReadyResult


def reconstruct(
	transport: "OwnedTransport",
	owner: Owner,
	packet: ReconstructionPacket,
	*,
	output_width: int,
	supply_inputs: Callable[[], None],
) -> "ReadyResult":
	from lib.runtime.transport import _io
	from lib.runtime.worker import REPLY_FIELDS

	transport._endpoint(owner)
	if packet.header.generation != owner.generation:
		raise ValueError("Reconstruction belongs to another worker generation")
	with _io(), transport.locked(owner):
		transport.comm.send(packet.header.tensor(), owner.rank, epoch=transport.control_epoch)
		if transport.coalesced:
			transport.comm.isend_tensors(packet.fields, owner.rank).wait()
		else:
			transport.comm.wait_all(
				[transport.comm.isend(value, owner.rank) for value in packet.fields if value.numel()]
			)
		reply = torch.empty(REPLY_FIELDS, dtype=torch.int64)
		request = transport.comm.irecv(reply, owner.rank, epoch=transport.control_epoch)
		supply_inputs()
	with _io():
		request.wait()
	version, chunk = packet.fields[-2].tolist()
	return transport._ready(
		owner,
		reply,
		request=packet.header.request,
		version=version,
		layer=packet.header.layer,
		chunk=chunk,
		rows=packet.header.rows,
		width=output_width,
	)
