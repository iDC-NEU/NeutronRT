from dataclasses import replace

import dgl
import pytest
import torch

from lib.dist.buffers import PackedLayout, TensorSpec
from lib.graph.ownership import ComputeAttempt, Location, Owner, StateKey
from lib.graph.recompute import LayerPlan
from lib.runtime.protocol import COMPUTE, ComputeHeader, InputChunk, pack_compute


def _task(layer, *, degrees=False):
	block = dgl.create_block(([0, 2, 2], [1, 0, 0]), num_src_nodes=3, num_dst_nodes=2)
	return LayerPlan(
		layer,
		block,
		torch.tensor([40, 41, 42]),
		torch.tensor([40, 41]),
		torch.tensor([12, 2, 0]) if degrees else None,
	)


def _attempt(layer):
	return ComputeAttempt(StateKey(7, layer, 33), Owner(2, 3), 9)


def test_typed_edge_metadata_roundtrips_with_explicit_wire_dimensions():
	task = _task(1)
	task.block.edata["nrt"] = torch.tensor([[0, 1, 0.2], [2, 0.7, 0.1], [1, 1, -0.3]])
	packet = pack_compute(task, _attempt(1), width=4, features=torch.zeros(3, 4))
	assert packet.header.edge_width == 3
	assert ComputeHeader.decode(packet.header.tensor()) == packet.header
	assert packet.fields[10] is task.block.edata["nrt"]
	assert tuple(TensorSpec.of(value) for value in packet.fields) == packet.header.specs(4)


@pytest.mark.parametrize("degrees", [False, True])
def test_first_layer_packet_retains_storage_views_and_roundtrips_bits(degrees):
	task = _task(1, degrees=degrees)
	features = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
	packet = pack_compute(task, _attempt(1), width=4, features=features)
	assert ComputeHeader.decode(packet.header.tensor()) == packet.header
	assert packet.fields[3] is task.sources
	assert packet.fields[4] is task.targets
	assert packet.fields[9] is features
	assert packet.fields[5].numel() == (3 if degrees else 0)
	assert packet.header.has_degrees == degrees
	layout = PackedLayout(packet.header.specs(4))
	buffer = torch.empty(layout.nbytes, dtype=torch.uint8)
	for source, view in zip(packet.fields, layout.views(buffer), strict=True):
		view.copy_(source)
	for source, received in zip(packet.fields, layout.views(buffer), strict=True):
		assert torch.equal(source, received)
	assert tuple(TensorSpec.of(field) for field in packet.fields) == packet.header.specs(4)


def test_intermediate_packet_routes_only_chunk_offsets_and_positions():
	task = _task(2)
	inputs = [
		InputChunk(Location(Owner(1, 2), 7), torch.tensor([4, 2]), torch.tensor([2, 0])),
		InputChunk(Location(Owner(0, 1), 8), torch.tensor([1]), torch.tensor([1])),
	]
	packet = pack_compute(task, _attempt(2), width=4, inputs=inputs)
	assert packet.fields[6].tolist() == [[1, 2, 7, 2], [0, 1, 8, 1]]
	assert packet.fields[7].tolist() == [4, 2, 1]
	assert packet.fields[8].tolist() == [2, 0, 1]
	assert packet.fields[9].shape == (0, 4)
	assert packet.header.groups == 2
	assert ComputeHeader.decode(packet.header.tensor()) == packet.header
	bad = replace(inputs[0], positions=torch.tensor([1, 1]))
	with pytest.raises(ValueError, match="partition"):
		pack_compute(task, _attempt(2), width=4, inputs=[bad, inputs[1]])
	bad = replace(inputs[0], location=Location(Owner(1, 0), 7))
	with pytest.raises(ValueError, match="owner identity"):
		pack_compute(task, _attempt(2), width=4, inputs=[bad, inputs[1]])


@pytest.mark.parametrize("index,value", [(0, 1), (0, COMPUTE + 256), (1, 0), (2, 0), (3, 0), (4, -1), (7, -1)])
def test_owned_headers_reject_old_versions_invalid_identities_and_negative_sizes(index, value):
	header = ComputeHeader(1, 1, 1, 3, 2, 3, 0).tensor()
	header[index] = value
	with pytest.raises(ValueError):
		ComputeHeader.decode(header)


def test_owned_packet_rejects_layer_and_input_schema_mismatches():
	with pytest.raises(ValueError, match="attempt's layer"):
		pack_compute(_task(1), _attempt(2), width=4, features=torch.zeros(3, 4))
	with pytest.raises(ValueError, match="float32"):
		pack_compute(_task(1), _attempt(1), width=4, features=torch.zeros(3, 4, dtype=torch.float64))
	with pytest.raises(ValueError, match="match the layer"):
		pack_compute(_task(2), _attempt(2), width=4)
	with pytest.raises(ValueError, match="degrees"):
		pack_compute(
			replace(_task(1), source_out_degrees=torch.tensor([-1, 2, 3])),
			_attempt(1),
			width=4,
			features=torch.zeros(3, 4),
		)


def test_master_proxy_changes_sender_without_changing_source_identity_or_offsets():
	source = Location(Owner(1, 2), 77)
	item = InputChunk(source, torch.tensor([4, 2, 9]), torch.tensor([2, 0, 1]), sender=0)
	packet = pack_compute(_task(2), _attempt(2), width=4, inputs=[item])
	assert packet.fields[6].tolist() == [[0, 2, 77, 3]]
	assert packet.fields[7].tolist() == [4, 2, 9]
	assert item.location is source
	for sender in (1, 3, False):
		with pytest.raises(ValueError, match="proxy"):
			pack_compute(_task(2), _attempt(2), width=4, inputs=[replace(item, sender=sender)])
	with pytest.raises(ValueError, match="proxy"):
		pack_compute(_task(2), _attempt(2), width=4, inputs=[replace(item, location=Location(_attempt(2).owner, 77))])
