"""Opt-in CUDA correctness gate for the matched resident replay methods."""

import os

import dgl
import pytest
import torch

import benchmarks.streaming as streaming
import scripts.evaluate.resident as resident_replay
from lib.models.spec import ModelSpec
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _to_wire

pytestmark = pytest.mark.gpu

if os.environ.get("NEUTRONRT_RUN_GPU_TESTS") != "1":
	pytest.skip("CUDA tests are opt-in and must run through neu-sbox", allow_module_level=True)
if not torch.cuda.is_available():
	pytest.skip("CUDA is unavailable", allow_module_level=True)


def _request(kind, first, second, identity):
	return wire.Update(kind=kind, first=_to_wire(first), second=_to_wire(second), request_id=identity)


def _fixture():
	graph = dgl.graph(
		(
			torch.tensor([0, 0, 1, 2, 3, 4, 0]),
			torch.tensor([1, 1, 2, 3, 4, 0, 0]),
		),
		num_nodes=6,
	)
	features = torch.randn(6, 4, generator=torch.Generator().manual_seed(71))
	spec = ModelSpec("sage", 4, 8, 3, layers=2)
	updates = (
		_request(wire.Update.INSERT, torch.tensor([5]), torch.tensor([1]), "warm-0"),
		_request(wire.Update.FEATURES, torch.tensor([0]), torch.full((1, 4), 0.25), "measured-0"),
		_request(wire.Update.DELETE, torch.tensor([0]), torch.tensor([1]), "measured-1"),
		_request(wire.Update.INSERT, torch.tensor([3]), torch.tensor([5]), "measured-2"),
	)
	recording = RecordedWorkload(
		spec=spec,
		input_identity=inference_identity(graph, features, spec),
		producer_source="resident-gpu-test",
		warmup=(updates[0],),
		steps=tuple((update, _to_wire(torch.tensor([5, 0, 1]))) for update in updates[1:]),
		verification_nodes=torch.arange(6),
		initialization="all",
		seed=71,
		repetition=0,
		atol=1e-5,
		rtol=1e-5,
	)
	return graph, features, recording


@pytest.mark.parametrize("method", ["NrtInc", "RTEC-Full", "RTEC-NS", "RTEC-UER"])
def test_small_cuda_resident_replay_all_methods(tmp_path, method):
	graph, features, recording = _fixture()
	device = torch.device("cuda:0")
	output = tmp_path / method
	if method == "NrtInc":
		report = resident_replay._run_nrt(
			graph,
			features,
			recording,
			output,
			device=device,
			capacity_bytes=1 << 30,
			labels=None,
			test_mask=None,
		)
	else:
		report = streaming.run(
			graph,
			features,
			recording,
			output,
			fanout=5 if method == "RTEC-NS" else None,
			device=device,
			output_device=device,
			batch_size=2,
			output_budget_bytes=1 << 30,
			method=method,
			resident=True,
			resident_capacity_bytes=1 << 30,
		)
	assert report["valid"]
	assert report["input_identity"] == recording.input_identity
	assert report["trace_digest"] == recording.digest
	assert report["final_state_validation"]["edges_match"]
	assert report["final_state_validation"]["features_match"]
	assert report["query_count"] == 3
	assert report["query_rows"] == 9
	residency_device_key = "device" if method == "NrtInc" else "graph_device"
	assert report["residency"][residency_device_key] == "cuda:0"
	if method != "RTEC-NS":
		assert report["final_oracle"]["exact_embedding_equivalence"]
	for update in report["updates"]:
		assert abs(sum(update["phase_seconds"].values()) - update["response_seconds"]) < 1e-5
		assert update["service_seconds"] >= update["response_seconds"]
