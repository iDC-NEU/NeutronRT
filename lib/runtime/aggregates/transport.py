"""Post a paired task and drain both result identities in one completion frame."""

from typing import TYPE_CHECKING, Callable, Union

import torch

from lib.graph.ownership import Owner
from lib.runtime.aggregates.delta import DeltaPacket
from lib.runtime.aggregates.protocol import AggregatePacket
from lib.runtime.aggregates.recipe import RecipePacket
from lib.runtime.protocol import COMPUTE_WITH_AGGREGATE

if TYPE_CHECKING:
	from lib.runtime.transport import OwnedTransport, ReadyResult


def compute(
	transport: "OwnedTransport",
	owner: Owner,
	packet: Union[AggregatePacket, RecipePacket, DeltaPacket],
	*,
	output_width: int,
	supply_inputs: Callable[[], None],
) -> tuple["ReadyResult", "ReadyResult"]:
	from lib.runtime.transport import _io
	from lib.runtime.worker import REPLY_FIELDS

	transport._endpoint(owner)
	if packet.header.generation != owner.generation:
		raise ValueError("Paired task belongs to another worker generation")
	with _io(), transport.locked(owner):
		command = (
			packet.header.tensor(opcode=COMPUTE_WITH_AGGREGATE)
			if isinstance(packet, AggregatePacket)
			else packet.header.tensor()
		)
		transport.comm.send(command, owner.rank, epoch=transport.control_epoch)
		if transport.coalesced:
			transport.comm.isend_tensors(packet.fields, owner.rank).wait()
		else:
			transport.comm.wait_all(
				[
					transport.comm.isend(value, owner.rank, name=f"field-{index}")
					for index, value in enumerate(packet.fields)
					if value.numel()
				]
			)
		reply = torch.empty(2 * REPLY_FIELDS, dtype=torch.int64)
		request = transport.comm.irecv(reply, owner.rank, epoch=transport.control_epoch)
		supply_inputs()
	with _io():
		request.wait()
	if isinstance(packet, AggregatePacket):
		first, second = packet.result_packets()
		return (
			transport._completion(owner, first, reply[:REPLY_FIELDS], output_width),
			transport._completion(owner, second, reply[REPLY_FIELDS:], output_width),
		)
	version, chunk, request, aggregate_chunk = packet.fields[-6 if isinstance(packet, RecipePacket) else -7].tolist()
	return (
		transport._ready(
			owner,
			reply[:REPLY_FIELDS],
			request=packet.header.request,
			version=version,
			layer=packet.header.layer,
			chunk=chunk,
			rows=packet.header.targets,
			width=output_width,
		),
		transport._ready(
			owner,
			reply[REPLY_FIELDS:],
			request=request,
			version=version,
			layer=packet.header.layer,
			chunk=aggregate_chunk,
			rows=packet.header.targets,
			width=output_width,
		),
	)
