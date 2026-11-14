import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from google.protobuf import descriptor_pb2

from lib.rpc import inference_pb2 as wire
from lib.rpc.tensors import TensorPayload, decode_tensor, encode_tensor


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.bool])
def test_protobuf_tensor_uses_the_same_binary_schema_as_the_frontend_adapter(dtype):
	value = torch.arange(12).reshape(3, 4).to(dtype)
	payload = encode_tensor(value)
	message = wire.Tensor(dtype=cast(wire.Tensor.DType, payload.dtype), shape=payload.shape, data=payload.data)
	received = wire.Tensor.FromString(message.SerializeToString())
	actual = decode_tensor(TensorPayload(received.dtype, tuple(received.shape), received.data))
	assert torch.equal(actual, value)
	assert (wire.Tensor.INT64, wire.Tensor.FLOAT32, wire.Tensor.BOOL) == (1, 2, 3)


def test_tickets_keep_large_ids_and_queries_distinguish_default_output_from_features():
	ticket = wire.Ticket(session="live-session", number=2**60 + 3, request_id="one-update")
	assert wire.Ticket.FromString(ticket.SerializeToString()) == ticket
	query = wire.QueryRequest()
	assert not query.HasField("layer")
	query.layer = 0
	assert query.HasField("layer")
	assert list(wire.DESCRIPTOR.services_by_name["OwnedInference"].methods_by_name) == [
		"Sync",
		"Async",
		"Wait",
		"Release",
		"Query",
		"Info",
	]


def _strip_derived_names(message):
	for field in message.field:
		field.ClearField("json_name")
	for nested in message.nested_type:
		_strip_derived_names(nested)


def test_checked_in_descriptor_matches_the_proto_source(tmp_path):
	root = Path(__file__).resolve().parents[2]
	compiled = tmp_path / "schema.pb"
	subprocess.run(
		[
			sys.executable,
			"-m",
			"grpc_tools.protoc",
			"-I.",
			f"--descriptor_set_out={compiled}",
			"lib/rpc/inference.proto",
		],
		cwd=root,
		check=True,
	)
	expected = descriptor_pb2.FileDescriptorSet.FromString(compiled.read_bytes()).file[0]
	actual = descriptor_pb2.FileDescriptorProto()
	wire.DESCRIPTOR.CopyToProto(actual)
	for descriptor in (expected, actual):
		for message in descriptor.message_type:
			_strip_derived_names(message)
	assert actual == expected
