from base64 import b64encode
from dataclasses import replace
from pathlib import Path

import dgl
import pytest
import torch

from lib.models.spec import ModelSpec
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _to_wire


def workload(*, trained=False):
	request = wire.Update(
		kind=wire.Update.FEATURES,
		first=_to_wire(torch.tensor([2, 1])),
		second=_to_wire(torch.arange(8, dtype=torch.float32).reshape(2, 4)),
		request_id="warm-0",
	)
	insert = wire.Update(
		kind=wire.Update.INSERT,
		first=_to_wire(torch.tensor([0, 0])),
		second=_to_wire(torch.tensor([1, 1])),
		request_id="measured-0",
	)
	return RecordedWorkload(
		ModelSpec("sage", 4, 8, 3, checkpoint="local-checkpoint.pt" if trained else None),
		"input-content-hash",
		"producer-source-hash",
		(request,),
		((insert, _to_wire(torch.tensor([2, 0]))),),
		torch.tensor([0, 2]),
		"all",
		71,
		0,
		1e-6,
		1e-5,
	)


@pytest.mark.parametrize("trained", [False, True])
def test_recorded_requests_round_trip_without_checkpoint_paths(tmp_path, trained):
	original = workload(trained=trained)
	path = tmp_path / "workload.pt"
	original.save(path)
	assert b"local-checkpoint.pt" not in path.read_bytes()
	loaded = RecordedWorkload.load(path, checkpoint=Path("relocated.pt") if trained else None)
	assert loaded.digest == original.digest
	assert loaded.warmup == original.warmup and loaded.steps == original.steps
	assert torch.equal(loaded.verification_nodes, original.verification_nodes)
	assert loaded.input_identity == original.input_identity
	assert loaded.spec.checkpoint == ("relocated.pt" if trained else None)
	with pytest.raises(FileExistsError):
		original.save(path)
	with pytest.raises(ValueError, match="checkpoint"):
		RecordedWorkload.load(path, checkpoint=None if trained else Path("unexpected.pt"))


def test_trace_digest_rejects_a_changed_request_or_query(tmp_path):
	original = workload()
	path = tmp_path / "workload.pt"
	original.save(path)
	payload = torch.load(path, weights_only=True)
	request, _ = payload["steps"][0]
	query = _to_wire(torch.tensor([0, 2])).SerializeToString(deterministic=True)
	payload["steps"][0] = request, b64encode(query).decode("ascii")
	torch.save(payload, path)
	with pytest.raises(ValueError, match="digest"):
		RecordedWorkload.load(path)


def test_input_identity_depends_on_weights_and_configuration_rather_than_checkpoint_location():
	graph = dgl.graph(([0, 2], [1, 1]), num_nodes=3)
	features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
	spec = ModelSpec("sage", 4, 8, 3)
	model = spec.build()
	identity = inference_identity(graph, features, spec, model=model)
	assert identity == inference_identity(graph, features, replace(spec, checkpoint="moved.pt", seed=9), model=model)
	with torch.no_grad():
		next(model.parameters()).add_(0.1)
	assert identity != inference_identity(graph, features, spec, model=model)
