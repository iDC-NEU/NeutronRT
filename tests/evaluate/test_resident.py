import dgl
import torch

import scripts.evaluate.resident as resident_replay
from lib.models.spec import ModelSpec
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _to_wire
from scripts.evaluate.resident import _run_nrt


def _request(kind, first, second, identity):
	return wire.Update(kind=kind, first=_to_wire(first), second=_to_wire(second), request_id=identity)


def test_recorded_resident_replay_decodes_queries_and_reports_final_oracle(tmp_path, monkeypatch):
	graph = dgl.graph(
		(torch.tensor([0, 0, 1, 2]), torch.tensor([1, 1, 2, 3])),
		num_nodes=5,
	)
	features = torch.randn(5, 4, generator=torch.Generator().manual_seed(41))
	spec = ModelSpec("sage", 4, 6, 3, layers=2)
	identity = inference_identity(graph, features, spec)
	steps = []
	for index in range(20):
		if index % 3 == 0:
			request = _request(
				wire.Update.FEATURES,
				torch.tensor([0]),
				torch.full((1, 4), 0.1 + index / 100),
				f"step-{index}",
			)
		elif index % 3 == 1:
			request = _request(wire.Update.INSERT, torch.tensor([4, 4]), torch.tensor([2, 2]), f"step-{index}")
		else:
			request = _request(wire.Update.DELETE, torch.tensor([4]), torch.tensor([2]), f"step-{index}")
		query = _to_wire(torch.tensor([0, 2, 4], dtype=torch.int64))
		steps.append((request, query))
	recording = RecordedWorkload(
		spec=spec,
		input_identity=identity,
		producer_source="resident-test",
		warmup=(),
		steps=tuple(steps),
		verification_nodes=torch.arange(5, dtype=torch.int64),
		initialization="all",
		seed=17,
		repetition=0,
		atol=1e-6,
		rtol=1e-5,
	)
	clock_value = 0.0

	def clock():
		nonlocal clock_value
		clock_value += 1.0
		return clock_value

	original_apply = resident_replay.apply_update
	original_update = resident_replay.ResidentGraphSAGE.update
	order = []

	def observed_update(self, request):
		order.append("service")
		return original_update(self, request)

	monkeypatch.setattr(resident_replay.ResidentGraphSAGE, "update", observed_update)

	def expensive_oracle_apply(graph_, features_, request_):
		nonlocal clock_value
		order.append("oracle")
		clock_value += 100.0
		return original_apply(graph_, features_, request_)

	monkeypatch.setattr(resident_replay, "perf_counter", clock)
	monkeypatch.setattr(resident_replay, "apply_update", expensive_oracle_apply)
	report = _run_nrt(
		graph,
		features,
		recording,
		tmp_path / "resident-replay",
		device=torch.device("cpu"),
		capacity_bytes=64 * 1024**2,
		labels=None,
		test_mask=None,
	)
	assert report["valid"]
	assert order == ["service"] * 20 + ["oracle"] * 20
	assert report["input_identity"] == identity
	assert report["trace_digest"] == recording.digest
	assert report["query_count"] == 20
	assert report["query_rows"] == 60
	assert report["window_seconds"] < 100.0
	assert report["harness_elapsed_seconds"] > report["window_seconds"] + 100.0
	assert len(report["updates"]) == 20
	assert report["final_quality"]["rows"] == 5
	assert report["final_oracle"]["exact_embedding_equivalence"]
	for update in report["updates"]:
		assert abs(sum(update["phase_seconds"].values()) - update["response_seconds"]) < 1e-6
		assert update["service_seconds"] >= update["response_seconds"]
