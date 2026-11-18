import inspect

import dgl
import torch

import benchmarks.streaming as streaming
import lib.graph.serving as serving
import lib.runtime.managed as managed
from lib.models.spec import ModelSpec
from lib.profiling.experiment import inference_identity
from lib.profiling.workload import RecordedWorkload
from lib.rpc import inference_pb2 as wire
from lib.rpc.service import _to_wire


def test_dgl_replay_boundary_contains_no_neutronrt_execution_entrypoint():
	source = inspect.getsource(streaming.run)
	for forbidden in ("ManagedRuntime", "EmbeddingService", "RecomputePlanner", "run_wave", "execute_recomputation"):
		assert forbidden not in source
	assert "comparison_group" in source


def test_tiny_replay_does_not_construct_runtime_execution_objects(tmp_path, monkeypatch):
	class Forbidden:
		def __init__(self, *args, **kwargs):
			raise AssertionError("dedicated DGL replay must not construct NeutronRT runtime objects")

	monkeypatch.setattr(managed, "ManagedRuntime", Forbidden)
	monkeypatch.setattr(serving, "EmbeddingService", Forbidden)
	graph = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])), num_nodes=3)
	features = torch.randn(3, 4, generator=torch.Generator().manual_seed(71))
	spec = ModelSpec("sage", 4, 8, 3, layers=2)
	request = wire.Update(
		kind=wire.Update.FEATURES,
		first=_to_wire(torch.tensor([1])),
		second=_to_wire(torch.full((1, 4), 0.25)),
		request_id="boundary",
	)
	workload = RecordedWorkload(
		spec,
		inference_identity(graph, features, spec),
		"boundary-source",
		(),
		((request, _to_wire(torch.arange(3))),),
		torch.arange(3),
		"all",
		71,
		0,
		1e-6,
		1e-5,
	)
	report = streaming.run(
		graph,
		features,
		workload,
		tmp_path / "trial",
		fanout=None,
		device=torch.device("cpu"),
		output_device=torch.device("cpu"),
		batch_size=2,
	)
	assert report["comparison_group"] == "dedicated_reference"
