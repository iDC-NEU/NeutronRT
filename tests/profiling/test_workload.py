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


def test_workload_counts_keep_edge_pair_operations_separate_from_feature_rows_and_queries():
	original = workload()
	measured = original.counts["measured"]
	assert measured["insert_pairs"] == measured["edge_pair_operations"] == 2
	assert measured["feature_rows"] == measured["delete_pairs"] == 0
	assert measured["query_rows"] == 2
	assert original.counts["warmup"]["feature_rows"] == 2
	assert original.counts["total"]["input_changes"] == 4
	assert measured["requests_by_kind"]["insert"]["changes_per_request"] == [2]


def test_mixed_trace_counts_record_each_request_and_do_not_count_empty_queries():
	ids = _to_wire(torch.arange(128))
	values = _to_wire(torch.zeros(128, 4))
	kinds = (wire.Update.FEATURES, wire.Update.INSERT, wire.Update.DELETE)
	requests = tuple(wire.Update(kind=kind, first=ids, second=values if kind == kinds[0] else ids) for kind in kinds)
	recording = replace(
		workload(),
		warmup=requests,
		steps=tuple((requests[index % 3], _to_wire(torch.arange(64))) for index in range(20)),
	)
	counts = recording.counts
	assert counts["measured"]["update_requests"] == 20
	assert counts["measured"]["feature_rows"] == counts["measured"]["insert_pairs"] == 896
	assert counts["measured"]["delete_pairs"] == 768
	assert counts["measured"]["edge_pair_operations"] == 1664
	assert counts["measured"]["input_changes"] == 2560
	assert counts["measured"]["net_edge_pair_operations"] == 128
	assert counts["measured"]["query_requests"] == 20 and counts["measured"]["query_rows"] == 1280
	assert counts["total"]["edge_pair_operations"] == 1920
	assert counts["total"]["feature_rows"] == 1024
	empty = replace(recording, steps=((requests[0], _to_wire(torch.empty(0, dtype=torch.int64))),))
	assert empty.counts["measured"]["query_requests"] == 0
	assert empty.counts["measured"]["update_requests"] == 1


@pytest.mark.parametrize(
	"update",
	[
		wire.Update(kind=wire.Update.UNSPECIFIED),
		wire.Update(kind=wire.Update.INSERT, first=_to_wire(torch.zeros(1))),
		wire.Update(kind=wire.Update.INSERT, first=_to_wire(torch.tensor([1, 2])), second=_to_wire(torch.tensor([1]))),
	],
)
def test_workload_counts_reject_unknown_operations_or_malformed_row_counts(update):
	with pytest.raises(ValueError):
		_ = replace(workload(), warmup=(update,)).counts
